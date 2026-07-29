from typing import Optional, Literal

import pandas as pd
import xarray as xr
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from siphon.catalog import TDSCatalog, Dataset

from surfer.agents.thredds_agent import (
    THREDDSConstraints, THREDDSDataset,
    THREDDSTimeseriesPlot, THREDDSProfilePlot, THREDDSMapPlot, THREDDSCustomPlot,
)
from surfer.agents.plotting import scatter_plot, trajectory_map, resolve_column, rename_lat_lon


def resolve_leaf_dataset(catalog_url: str, dataset_name: Optional[str] = None) -> Dataset | str:
    """Fetch the catalog at catalog_url and resolve a single leaf Dataset within it. If
    dataset_name is given, look it up (case-insensitive substring match). If not given and
    exactly one dataset exists at this level, use it. Returns an error string if not found,
    ambiguous, this level has no leaf datasets (only sub-catalogs), or catalog_url isn't a
    valid catalog URL at all (e.g. a dodsC/data-access URL passed by mistake instead of a
    catalog.xml URL)."""
    try:
        cat = TDSCatalog(catalog_url)
    except Exception as err:
        return (
            f"Could not fetch catalog at '{catalog_url}': {err}. dataset_id must be a "
            f"sub-catalog URL (catalog.xml), as returned by browse_catalog_tool or "
            f"find_datasets_tool's catalog_url field -- not a dataset's own data/download URL."
        )
    if not cat.datasets:
        return f"No datasets at {catalog_url} -- only sub-catalogs: {list(cat.catalog_refs.keys())}"

    if dataset_name is None:
        if len(cat.datasets) == 1:
            return next(iter(cat.datasets.values()))
        return f"Multiple datasets at {catalog_url}, specify dataset_name: {list(cat.datasets.keys())}"

    query = dataset_name.lower()
    matches = [name for name in cat.datasets if query in name.lower()]
    if not matches:
        return f"No dataset matching '{dataset_name}' at {catalog_url}. Available: {list(cat.datasets.keys())}"
    if len(matches) > 1 and dataset_name not in cat.datasets:
        return f"'{dataset_name}' is ambiguous at {catalog_url}. Did you mean one of: {matches}?"
    name = dataset_name if dataset_name in cat.datasets else matches[0]
    return cat.datasets[name]


def open_leaf_dataset(dataset: Dataset, use_service: str = "OpenDAP") -> xr.Dataset:
    """Resolve dataset.access_urls[use_service] and open it lazily via xr.open_dataset.
    Raises if use_service isn't exposed for this dataset. Deliberately resolves the URL and
    calls xr.open_dataset directly rather than Siphon's remote_access(use_xarray=True)
    wrapper -- keeps the Siphon dependency narrow (catalog navigation + URL resolution only)
    and keeps the data-opening call visible/swappable."""
    if use_service not in dataset.access_urls:
        raise ValueError(
            f"Service '{use_service}' not available for dataset '{dataset.name}'. "
            f"Available: {list(dataset.access_urls.keys())}"
        )
    return xr.open_dataset(dataset.access_urls[use_service], engine="netcdf4")


def describe_thredds_dataset(catalog_url: str, dataset_name: Optional[str] = None) -> dict | str:
    """Returns {"bounds": {...}, "variables": {...}}, matching describe_dataset()'s shape.
    Explicitly detects and rejects Grid datasets (gridded model output) -- the concrete
    v1/v2 boundary enforcement point in code, not just a prompt instruction. No ioos_category
    equivalent exists in CF/NetCDF attributes, so that key is simply absent here rather than
    fabricated."""
    dataset = resolve_leaf_dataset(catalog_url, dataset_name)
    if isinstance(dataset, str):
        return dataset

    try:
        ds = open_leaf_dataset(dataset)
    except Exception as err:
        return f"Could not open dataset '{dataset.name}': {err}"

    try:
        cdm_data_type = ds.attrs.get("cdm_data_type")
        if cdm_data_type == "Grid":
            return (
                f"Dataset '{dataset.name}' is a Grid dataset (gridded model output) -- not "
                f"supported in this version, which handles TrajectoryProfile (glider/trajectory) "
                f"datasets only."
            )

        def get_extreme(attr: str, default: float | str):
            val = ds.attrs.get(attr, default)
            try:
                return float(val)
            except (ValueError, TypeError):
                return val

        bounds = {
            "time>=": get_extreme("time_coverage_start", ""),
            "time<=": get_extreme("time_coverage_end", ""),
            "latitude>=": float(get_extreme("geospatial_lat_min", -90.0)),
            "latitude<=": float(get_extreme("geospatial_lat_max", 90.0)),
            "longitude>=": float(get_extreme("geospatial_lon_min", -180.0)),
            "longitude<=": float(get_extreme("geospatial_lon_max", 180.0)),
        }

        fields = ("standard_name", "long_name", "units")
        variables = {
            name: {key: var.attrs[key] for key in fields if key in var.attrs}
            for name, var in ds.variables.items()
        }
        return {"bounds": bounds, "variables": variables}
    finally:
        ds.close()


@tool
def describe_dataset_tool(runtime: ToolRuntime, catalog_url: str, dataset_name: Optional[str] = None) -> str:
    """
    Look up a dataset's time/lat/lon coverage and its exact variable names (with CF
    standard_name, long_name, and units), in one call. catalog_url is the catalog listing
    the dataset (as returned by browse_catalog_tool or find_datasets_tool); dataset_name
    disambiguates if that catalog lists more than one dataset. Rejects Grid (gridded model
    output) datasets with a clear message -- this agent supports trajectory/profile
    glider-style datasets only.
    """
    info = describe_thredds_dataset(catalog_url, dataset_name)
    if isinstance(info, str):
        return info

    c = info["bounds"]
    time_range = f"{c['time>=']} to {c['time<=']}" if c["time>="] else "not specified in metadata"
    lines = [
        f"Dataset at '{catalog_url}' covers:",
        f"Time: {time_range}",
        f"Latitude: {c['latitude>=']} to {c['latitude<=']}",
        f"Longitude: {c['longitude>=']} to {c['longitude<=']}",
        "",
        "Variables:",
    ]
    for name, attrs in info["variables"].items():
        detail = ", ".join(f"{k}: {v}" for k, v in attrs.items())
        lines.append(f"- `{name}`" + (f" ({detail})" if detail else ""))
    return "\n".join(lines)


def to_dataframe(ds: xr.Dataset, variables: list[str]) -> pd.DataFrame:
    """THREDDS-side analogue of fix_labels(). Known v1 limitation: assumes trajectory dim
    size 1 (one deployment per file); raises a clear error if violated rather than silently
    picking one."""
    if "trajectory" in ds.sizes and ds.sizes["trajectory"] > 1:
        raise ValueError(
            f"Dataset has {ds.sizes['trajectory']} trajectories in one file -- not supported "
            f"in this version, which assumes one deployment per file."
        )

    df = ds[variables].to_dataframe().reset_index()
    return rename_lat_lon(df)


def apply_constraints(ds: xr.Dataset, constraints: Optional[THREDDSConstraints]) -> xr.Dataset:
    """Client-side analogue of _apply_constraints() -- boolean-masks the lazily-open Dataset
    by time/lat/lon range rather than mutating a server-query constraints dict. Missing
    bounds pass through unfiltered. Unlike ERDDAP's server-side 404 on an inverted range,
    xarray silently selects nothing -- so inverted lat/lon bounds are swapped and an inverted
    time range is dropped entirely, matching ERDDAP's conventions with our own enforcement.
    Must be applied BEFORE to_dataframe(): an unconstrained .to_dataframe() on a large
    deployment can pull far more data client-side than an equivalent constrained ERDDAP
    request would, since ERDDAP subsets server-side before sending anything."""
    if not constraints:
        return ds

    mask = None

    def and_mask(new: xr.DataArray) -> None:
        nonlocal mask
        mask = new if mask is None else (mask & new)

    min_time, max_time = constraints.min_time, constraints.max_time
    if min_time and max_time:
        try:
            start, end = pd.to_datetime(min_time), pd.to_datetime(max_time)
        except (ValueError, TypeError):
            start, end = None, None
        if start is not None and end is not None and start > end:
            min_time = max_time = None  # inverted range: drop to unconstrained, like ERDDAP's time reset

    if min_time:
        and_mask(ds["time"] >= pd.to_datetime(min_time))
    if max_time:
        and_mask(ds["time"] <= pd.to_datetime(max_time))

    lat_min, lat_max = constraints.min_lat, constraints.max_lat
    if lat_min is not None and lat_max is not None and lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    if lat_min is not None:
        and_mask(ds["latitude"] >= lat_min)
    if lat_max is not None:
        and_mask(ds["latitude"] <= lat_max)

    lon_min, lon_max = constraints.min_lon, constraints.max_lon
    if lon_min is not None and lon_max is not None and lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    if lon_min is not None:
        and_mask(ds["longitude"] >= lon_min)
    if lon_max is not None:
        and_mask(ds["longitude"] <= lon_max)

    return ds.where(mask, drop=True) if mask is not None else ds


def dataset_metadata(ds: xr.Dataset, dataset_name: str) -> dict:
    """Same shape as _dataset_metadata(): {id, author, institution, platform, license}."""
    return {
        "id": dataset_name,
        "author": ds.attrs.get("contributor_name", ds.attrs.get("creator_name", "N/A")),
        "institution": ds.attrs.get("institution", "N/A"),
        "platform": ds.attrs.get("platform", "N/A"),
        "license": ds.attrs.get("license", "N/A"),
    }


def _load_dataset(catalog_url: str, dataset_name: Optional[str], variables: list[str]) -> tuple[xr.Dataset, Dataset] | str:
    """Shared resolve + open + constraint-application setup for the plot tools below."""
    dataset = resolve_leaf_dataset(catalog_url, dataset_name)
    if isinstance(dataset, str):
        return dataset
    try:
        ds = open_leaf_dataset(dataset)
    except Exception as err:
        return f"Could not open dataset '{dataset.name}': {err}"
    missing = [v for v in variables if v not in ds.variables]
    if missing:
        ds.close()
        return f"Variables {missing} not found. Available: {sorted(ds.variables)[:30]}"
    return ds, dataset


@tool(args_schema=THREDDSTimeseriesPlot)
def plot_timeseries(
    runtime: ToolRuntime,
    dataset_id: str,
    y_var: str,
    dataset_name: Optional[str] = None,
    color_var: Optional[str] = None,
    constraints: Optional[THREDDSConstraints] = None,
    step_size: int = 1,
    file_type: str = "png",
) -> str | Command:
    """Plot a variable against time."""
    needed = ["time", "latitude", "longitude", y_var] + ([color_var] if color_var else [])
    loaded = _load_dataset(dataset_id, dataset_name, needed)
    if isinstance(loaded, str):
        return loaded
    ds, dataset = loaded
    try:
        ds = apply_constraints(ds, constraints)
        df = to_dataframe(ds, needed)
        y_col = resolve_column(df, y_var)
        color_col = resolve_column(df, color_var) if color_var else None
        image_bytes = scatter_plot(
            df, x="time", y=y_col, color=color_col, step_size=step_size,
            title=f"{y_var} vs time", metadata=dataset_metadata(ds, dataset.name),
            filetype=file_type,
        )
    except Exception as err:
        return f"Plot failed for '{dataset.name}': {err}"
    finally:
        ds.close()

    return Command(update={
        "artifacts": [{"type": "image", "content": image_bytes, "title": f"{y_var} vs time"}],
        "messages": [ToolMessage(content="Timeseries plot generated.", tool_call_id=runtime.tool_call_id)],
    })


@tool(args_schema=THREDDSProfilePlot)
def plot_profile(
    runtime: ToolRuntime,
    dataset_id: str,
    x_var: str,
    dataset_name: Optional[str] = None,
    color_var: Optional[str] = None,
    constraints: Optional[THREDDSConstraints] = None,
    step_size: int = 1,
    file_type: str = "png",
) -> str | Command:
    """Plot a variable against depth."""
    needed = ["latitude", "longitude", x_var, "depth"] + ([color_var] if color_var else [])
    loaded = _load_dataset(dataset_id, dataset_name, needed)
    if isinstance(loaded, str):
        return loaded
    ds, dataset = loaded
    try:
        ds = apply_constraints(ds, constraints)
        df = to_dataframe(ds, needed)
        x_col = resolve_column(df, x_var)
        color_col = resolve_column(df, color_var) if color_var else None
        image_bytes = scatter_plot(
            df, x=x_col, y="depth", color=color_col, step_size=step_size, invert_y=True,
            title=f"{x_var} profile", metadata=dataset_metadata(ds, dataset.name),
            filetype=file_type,
        )
    except Exception as err:
        return f"Plot failed for '{dataset.name}': {err}"
    finally:
        ds.close()

    return Command(update={
        "artifacts": [{"type": "image", "content": image_bytes, "title": f"{x_var} profile"}],
        "messages": [ToolMessage(content="Profile plot generated.", tool_call_id=runtime.tool_call_id)],
    })


@tool(args_schema=THREDDSMapPlot)
def plot_map(
    runtime: ToolRuntime,
    dataset_id: str,
    dataset_name: Optional[str] = None,
    color_var: Optional[str] = None,
    constraints: Optional[THREDDSConstraints] = None,
    step_size: int = 1,
) -> str | Command:
    """Plot the dataset's trajectory (latitude/longitude), optionally colored by a variable."""
    needed = ["latitude", "longitude"] + ([color_var] if color_var else [])
    loaded = _load_dataset(dataset_id, dataset_name, needed)
    if isinstance(loaded, str):
        return loaded
    ds, dataset = loaded
    try:
        ds = apply_constraints(ds, constraints)
        df = to_dataframe(ds, needed).iloc[::step_size]
        color_col = resolve_column(df, color_var) if color_var else None
        cbar_label = None
        if color_var:
            units = ds[color_var].attrs.get("units")
            cbar_label = f"{color_var} ({units})" if units else color_var
        html = trajectory_map(df, color=color_col, cbar_label=cbar_label, title=f"Flight path: {dataset.name}")
    except Exception as err:
        return f"Plot failed for '{dataset.name}': {err}"
    finally:
        ds.close()

    return Command(update={
        "artifacts": [{"type": "html", "content": html, "title": f"Flight path: {dataset.name}"}],
        "messages": [ToolMessage(content="Trajectory map generated.", tool_call_id=runtime.tool_call_id)],
    })


@tool(args_schema=THREDDSCustomPlot)
def plot_custom(
    runtime: ToolRuntime,
    dataset_id: str,
    x_var: str,
    y_var: str = "time",
    dataset_name: Optional[str] = None,
    color_var: Optional[str] = None,
    constraints: Optional[THREDDSConstraints] = None,
    step_size: int = 1,
    file_type: str = "png",
) -> str | Command:
    """Plot any two variables against each other (e.g. a temperature-salinity diagram).
    Use plot_timeseries/plot_profile/plot_map instead for a variable vs. time, depth, or lat/lon."""
    needed = ["latitude", "longitude", x_var, y_var] + ([color_var] if color_var else [])
    loaded = _load_dataset(dataset_id, dataset_name, needed)
    if isinstance(loaded, str):
        return loaded
    ds, dataset = loaded
    try:
        ds = apply_constraints(ds, constraints)
        df = to_dataframe(ds, needed)
        x_col = resolve_column(df, x_var)
        y_col = resolve_column(df, y_var)
        color_col = resolve_column(df, color_var) if color_var else None
        image_bytes = scatter_plot(
            df, x=x_col, y=y_col, color=color_col, step_size=step_size,
            title=f"{x_var} vs {y_var}", metadata=dataset_metadata(ds, dataset.name),
            filetype=file_type,
        )
    except Exception as err:
        return f"Plot failed for '{dataset.name}': {err}"
    finally:
        ds.close()

    return Command(update={
        "artifacts": [{"type": "image", "content": image_bytes, "title": f"{x_var} vs {y_var}"}],
        "messages": [ToolMessage(content="Plot generated.", tool_call_id=runtime.tool_call_id)],
    })


@tool(args_schema=THREDDSDataset)
def get_dataset_download_tool(
    runtime: ToolRuntime,
    dataset_id: str,
    dataset_name: Optional[str] = None,
    variables: Optional[list[str]] = None,
    constraints: Optional[THREDDSConstraints] = None,
) -> str:
    """Get the direct download URL for a dataset without fetching the file through the agent.
    Returns a link the user can open or download directly. Defaults to a whole-file HTTPServer
    download, falling back to OpenDAP if HTTPServer isn't exposed for that dataset."""
    dataset = resolve_leaf_dataset(dataset_id, dataset_name)
    if isinstance(dataset, str):
        return dataset

    service: Literal["HTTPServer", "OpenDAP"] = "HTTPServer" if "HTTPServer" in dataset.access_urls else "OpenDAP"
    if service not in dataset.access_urls:
        return f"Neither HTTPServer nor OpenDAP is available for dataset '{dataset.name}'. Available: {list(dataset.access_urls.keys())}"
    return f"Download URL for dataset '{dataset.name}' ({service}): {dataset.access_urls[service]}"

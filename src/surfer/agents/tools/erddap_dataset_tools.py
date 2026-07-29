import re
import time
from typing import Optional, Literal

from erddapy import ERDDAP
from erddapy.core.url import quote_url
import pandas as pd
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

from surfer.agents.erddap_agent import (
    ERDDAPServer, ERDDAPConstraints, ERDDAPDataset,
    ERDDAPTimeseriesPlot, ERDDAPProfilePlot, ERDDAPMapPlot, ERDDAPCustomPlot,
    get_active_server, describe_erddap_error, is_qc_variable, ERDDAPRegionMapPlot
)
from surfer.agents.tools.erddap_server_tools import search_server
from surfer.agents.plotting import scatter_plot, trajectory_map, multi_trajectory_map, resolve_column, rename_lat_lon

_dataset_variables_cache: dict[tuple[str, str, str, str], tuple[float, dict]] = {}
_DATASET_CACHE_TTL_SECONDS = 60
_DATASET_CACHE_MAX_SIZE = 256


def _fetch_dataset_variables(server: str, dataset_id: str, protocol: str, response: str) -> dict:
    """Cached fetch of a dataset's variable+attribute metadata (erddapy's `_get_variables`).
    erddapy's own cache is per-instance, so this avoids re-fetching within one turn."""
    key = (server, dataset_id, protocol, response)
    cached = _dataset_variables_cache.get(key)
    if cached is not None and time.monotonic() - cached[0] < _DATASET_CACHE_TTL_SECONDS:
        return cached[1]

    e = ERDDAP(server=server, protocol=protocol, response=response)
    value = e._get_variables(dataset_id=dataset_id)

    if len(_dataset_variables_cache) >= _DATASET_CACHE_MAX_SIZE:
        _dataset_variables_cache.clear()  # short-TTL design
    _dataset_variables_cache[key] = (time.monotonic(), value)
    return value


def get_erddap(
    dataset_id: str,
    server: str,
    protocol: str = "tabledap",
    response: str = "csv",
    variables: list | None = None,
) -> ERDDAP:
    """Connects to ERDDAP server and sets up constraints via the dataset's metadata.
    Returns ERDDAP object filled with metadata."""
    e = ERDDAP(server=server, protocol=protocol, response=response)
    e.dataset_id = dataset_id
    e.variables = variables

    all_variables = _fetch_dataset_variables(server, dataset_id, protocol, response)
    if variables:
        available = sorted(set(all_variables) - {"NC_GLOBAL"})
        invalid = [v for v in variables if v not in available]
        if invalid:
            raise ValueError(f"Variables {invalid} not found. Available: {available[:30]}")

    global_attrs = all_variables.get("NC_GLOBAL", {})

    def get_extremes(var, default: float | str = "") -> float | str:
        if var not in global_attrs:
            return default
        val = global_attrs[var]
        try:
            return float(val)
        except (ValueError, TypeError):
            return val

    e.constraints = {
        "time>=": get_extremes("time_coverage_start"),
        "time<=": get_extremes("time_coverage_end"),
        "latitude>=": float(get_extremes("geospatial_lat_min", default=-90.0)),
        "latitude<=": float(get_extremes("geospatial_lat_max", default=90.0)),
        "longitude>=": float(get_extremes("geospatial_lon_min", default=-180.0)),
        "longitude<=": float(get_extremes("geospatial_lon_max", default=180.0)),
    }
    if variables:
        e.variables = variables
    return e


def describe_dataset(
    server: str,
    dataset_id: str,
    protocol: str = "tabledap",
    response: str = "csv",
) -> dict | str:
    """Fetch a dataset's time/lat/lon coverage and every variable's standard_name, long_name,
    units, and ioos_category, in one request. Variables still include QC/QARTOD flag columns
    -- pass to filter_dataset_variables to narrow. Returns {"bounds": {...}, "variables": {...}}
    or an error string if the dataset doesn't exist."""
    try:
        all_variables = _fetch_dataset_variables(server, dataset_id, protocol, response)
    except Exception as err:
        return f"Dataset '{dataset_id}' not found on {server}: {describe_erddap_error(err)}"

    global_attrs = all_variables.get("NC_GLOBAL", {})

    def get_extreme(attr: str, default: float | str):
        val = global_attrs.get(attr, default)
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

    fields = ("standard_name", "long_name", "units", "ioos_category")
    variables = {
        name: {key: attrs[key] for key in fields if key in attrs}
        for name, attrs in all_variables.items()
        if name != "NC_GLOBAL"
    }

    return {"bounds": bounds, "variables": variables}


def filter_dataset_variables(
    variables: dict[str, dict[str, str]],
    query: Optional[str] = None,
    exclude_qc: bool = True,
) -> dict[str, dict[str, str]]:
    """Drops QARTOD/QC flag variables by default. If `query` is given, keeps only variables
    whose name or attribute values contain it (case-insensitive)."""
    query_lower = query.lower() if query else None
    result = {}
    for name, attrs in variables.items():
        if exclude_qc and is_qc_variable(name):
            continue
        if query_lower:
            searchable_text = " ".join([name, *attrs.values()]).lower()
            if query_lower not in searchable_text:
                continue
        result[name] = attrs
    return result


@tool
def describe_dataset_tool(runtime: ToolRuntime, dataset_id: str, query: Optional[str] = None) -> str:
    """
    Look up a dataset's time/lat/lon coverage and its exact variable names (with CF
    standard_name, long_name, units, and ioos_category), in one call. Pass `query` to
    narrow the variable list on datasets with many variables.
    """
    server = get_active_server(runtime)
    info = describe_dataset(server.url, dataset_id, server.protocol, server.response)
    if isinstance(info, str):
        return info

    c = info["bounds"]
    time_range = f"{c['time>=']} to {c['time<=']}" if c["time>="] else "not specified in metadata"
    lines = [
        f"Dataset '{dataset_id}' covers:",
        f"Time: {time_range}",
        f"Latitude: {c['latitude>=']} to {c['latitude<=']}",
        f"Longitude: {c['longitude>=']} to {c['longitude<=']}",
        "",
    ]

    variables = filter_dataset_variables(info["variables"], query=query)
    if not variables:
        lines.append("No matching variables found.")
    else:
        lines.append("Variables:")
        for name, attrs in variables.items():
            detail = ", ".join(f"{k}: {v}" for k, v in attrs.items())
            lines.append(f"- `{name}`" + (f" ({detail})" if detail else ""))
    return "\n".join(lines)


def fix_labels(h):
    """Strips ERDDAP's "name (units)" suffix from column headers, and aliases
    latitude/longitude to the short lat/lon names the plotting functions expect."""
    h = h.rename(columns=lambda c: re.sub(r"\s*\([^)]*\)$", "", c))
    return rename_lat_lon(h)


def _load_dataset(
    dataset_id: str,
    server: ERDDAPServer,
    variables: Optional[list[str]] = None,
) -> ERDDAP | str:
    """Shared get_erddap() call + error handling for the tool functions below."""
    try:
        return get_erddap(
            dataset_id=dataset_id,
            server=server.url,
            protocol=server.protocol,
            response=server.response,
            variables=variables,
        )
    except Exception as err:
        return f"Could not load dataset '{dataset_id}': {describe_erddap_error(err)}"


def _apply_constraints(e: ERDDAP, constraints: Optional[ERDDAPConstraints]) -> ERDDAP:
    """Merge user-supplied time/lat/lon bounds into the dataset's own range (seeded by
    get_erddap). ERDDAP clips missing/out-of-range bounds itself; the one failure mode is
    an inverted range (min > max), which 404s -- lat/lon get swapped, time resets to full
    range instead."""
    if e.constraints is None:
        raise ValueError(
            f"ERDDAP object for dataset {e.dataset_id} has no constraints. "
        )
    if not constraints:
        return e
    c = e.constraints

    def pad_and_parse(value: str, end_of_day: bool):
        """Pad a bare date to a full day and parse it for the inversion check below.
        ERDDAP's relative time syntax (e.g. 'now-7days') never starts with a digit --
        left untouched, since the server interprets it at request time."""
        if not value or not value[0].isdigit():
            return value, None
        if "T" not in value:
            value = f"{value}T23:59:59Z" if end_of_day else f"{value}T00:00:00Z"
        try:
            return value, pd.to_datetime(value)
        except (ValueError, TypeError):
            return value, None

    db_start, db_end = c["time>="], c["time<="]
    start, start_dt = pad_and_parse(constraints.min_time or db_start, end_of_day=False)
    end, end_dt = pad_and_parse(constraints.max_time or db_end, end_of_day=True)
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        start, end = db_start, db_end  # inverted range 404s on the real server
    c["time>="], c["time<="] = start, end

    lat_min = constraints.min_lat if constraints.min_lat is not None else c["latitude>="]
    lat_max = constraints.max_lat if constraints.max_lat is not None else c["latitude<="]
    lon_min = constraints.min_lon if constraints.min_lon is not None else c["longitude>="]
    lon_max = constraints.max_lon if constraints.max_lon is not None else c["longitude<="]
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    c["latitude>="], c["latitude<="] = lat_min, lat_max
    c["longitude>="], c["longitude<="] = lon_min, lon_max

    return e


def _dataset_metadata(dataset_id: str, server: ERDDAPServer) -> dict:
    """Author/institution/platform/license labels for a plot's info box, read from the
    cached NC_GLOBAL attributes -- no extra HTTP request."""
    all_variables = _fetch_dataset_variables(server.url, dataset_id, server.protocol, server.response)
    global_attrs = all_variables.get("NC_GLOBAL", {})
    return {
        "id": dataset_id,
        "author": global_attrs.get("contributor_name", "N/A"),
        "institution": global_attrs.get("institution", "N/A"),
        "platform": global_attrs.get("platform", "N/A"),
        "license": global_attrs.get("license", "N/A"),
    }


def _variable_units(dataset_id: str, server: ERDDAPServer, var_name: str) -> Optional[str]:
    """Units string for one variable (e.g. 'degree_C'), from the cached variable fetch."""
    all_variables = _fetch_dataset_variables(server.url, dataset_id, server.protocol, server.response)
    return all_variables.get(var_name, {}).get("units")


def _graph_url(e: ERDDAP) -> str:
    """URL to ERDDAP's interactive Make-A-Graph page for the current dataset. quote_url
    escapes the raw '<'/'>' in constraint syntax (e.g. 'time<=2023-...'), which otherwise
    reads as an HTML tag to markdown renderers and truncates the link."""
    original_response = e.response
    e.response = "graph"
    try:
        return quote_url(e.get_download_url())
    finally:
        e.response = original_response


def _erddap_plot_url(e: ERDDAP, file_type: str) -> str:
    """URL to ERDDAP's own server-rendered plot image for this dataset's current
    variables/constraints -- lets the caller skip downloading data and rendering locally."""
    original_response = e.response
    e.response = file_type
    try:
        return quote_url(e.get_download_url())
    finally:
        e.response = original_response


@tool(args_schema=ERDDAPTimeseriesPlot)
def plot_timeseries(
    runtime: ToolRuntime,
    dataset_id: str,
    y_var: str,
    color_var: Optional[str] = None,
    constraints: Optional[ERDDAPConstraints] = None,
    engine: Literal["local", "erddap"] = "local",
    step_size: int = 1,
    file_type: str = "png",
) -> str | Command:
    """Plot a variable against time."""
    server = get_active_server(runtime)
    needed = ["time", y_var] + ([color_var] if color_var else [])
    e = _load_dataset(dataset_id, server, variables=needed)
    if isinstance(e, str):
        return e
    e = _apply_constraints(e, constraints)
    e.variables = needed  # order matters for _erddap_plot_url/_graph_url: x, y[, color]
    graph_url = _graph_url(e)

    if engine == "erddap":
        return f"Plot URL for dataset '{dataset_id}': {_erddap_plot_url(e, file_type)}\nCustomize it yourself: {graph_url}"

    try:
        df = fix_labels(e.to_pandas().reset_index())
        y_col = resolve_column(df, y_var)
        color_col = resolve_column(df, color_var) if color_var else None
        image_bytes = scatter_plot(
            df, x="time", y=y_col, color=color_col, step_size=step_size,
            title=f"{y_var} vs time", metadata=_dataset_metadata(dataset_id, server),
            filetype=file_type,
        )
    except Exception as err:
        return f"Plot failed for '{dataset_id}': {describe_erddap_error(err)}"

    return Command(update={
        "artifacts": [{"type": "image", "content": image_bytes, "title": f"{y_var} vs time"}],
        "messages": [ToolMessage(
            content=f"Timeseries plot generated.\nCustomize it yourself: {graph_url}",
            tool_call_id=runtime.tool_call_id,
        )],
    })


@tool(args_schema=ERDDAPProfilePlot)
def plot_profile(
    runtime: ToolRuntime,
    dataset_id: str,
    x_var: str,
    color_var: Optional[str] = None,
    constraints: Optional[ERDDAPConstraints] = None,
    engine: Literal["local", "erddap"] = "local",
    step_size: int = 1,
    file_type: str = "png",
) -> str | Command:
    """Plot a variable against depth."""
    server = get_active_server(runtime)
    needed = [x_var, "depth"] + ([color_var] if color_var else [])
    e = _load_dataset(dataset_id, server, variables=needed)
    if isinstance(e, str):
        return e
    e = _apply_constraints(e, constraints)
    e.variables = needed
    graph_url = _graph_url(e)

    if engine == "erddap":
        return f"Plot URL for dataset '{dataset_id}': {_erddap_plot_url(e, file_type)}\nCustomize it yourself: {graph_url}"

    try:
        df = fix_labels(e.to_pandas().reset_index())
        x_col = resolve_column(df, x_var)
        color_col = resolve_column(df, color_var) if color_var else None
        image_bytes = scatter_plot(
            df, x=x_col, y="depth", color=color_col, step_size=step_size, invert_y=True,
            title=f"{x_var} profile", metadata=_dataset_metadata(dataset_id, server),
            filetype=file_type,
        )
    except Exception as err:
        return f"Plot failed for '{dataset_id}': {describe_erddap_error(err)}"

    return Command(update={
        "artifacts": [{"type": "image", "content": image_bytes, "title": f"{x_var} profile"}],
        "messages": [ToolMessage(
            content=f"Profile plot generated.\nCustomize it yourself: {graph_url}",
            tool_call_id=runtime.tool_call_id,
        )],
    })


@tool(args_schema=ERDDAPMapPlot)
def plot_map(
    runtime: ToolRuntime,
    dataset_id: str,
    color_var: Optional[str] = None,
    constraints: Optional[ERDDAPConstraints] = None,
    engine: Literal["local", "erddap"] = "local",
    step_size: int = 1,
    file_type: str = "png",
) -> str | Command:
    """Plot the dataset's trajectory (latitude/longitude), optionally colored by a variable."""
    server = get_active_server(runtime)
    needed = ["longitude", "latitude"] + ([color_var] if color_var else [])
    e = _load_dataset(dataset_id, server, variables=needed)
    if isinstance(e, str):
        return e
    e = _apply_constraints(e, constraints)
    e.variables = needed
    graph_url = _graph_url(e)

    if engine == "erddap":
        return f"Map URL for dataset '{dataset_id}': {_erddap_plot_url(e, file_type)}\nCustomize it yourself: {graph_url}"

    try:
        df = fix_labels(e.to_pandas().reset_index()).iloc[::step_size]
        color_col = resolve_column(df, color_var) if color_var else None
        cbar_label = None
        if color_var:
            units = _variable_units(dataset_id, server, color_var)
            cbar_label = f"{color_var} ({units})" if units else color_var
        html = trajectory_map(df, color=color_col, cbar_label=cbar_label, title=f"Flight path: {dataset_id}")
    except Exception as err:
        return f"Plot failed for '{dataset_id}': {describe_erddap_error(err)}"

    return Command(update={
        "artifacts": [{"type": "html", "content": html, "title": f"Flight path: {dataset_id}"}],
        "messages": [ToolMessage(
            content=f"Trajectory map generated.\nCustomize it yourself: {graph_url}",
            tool_call_id=runtime.tool_call_id,
        )],
    })


@tool(args_schema=ERDDAPCustomPlot)
def plot_custom(
    runtime: ToolRuntime,
    dataset_id: str,
    x_var: str,
    y_var: str = "time",
    color_var: Optional[str] = None,
    constraints: Optional[ERDDAPConstraints] = None,
    engine: Literal["local", "erddap"] = "local",
    step_size: int = 1,
    file_type: str = "png",
) -> str | Command:
    """Plot any two variables against each other (e.g. a temperature-salinity diagram).
    Use plot_timeseries/plot_profile/plot_map instead for a variable vs. time, depth, or lat/lon."""
    server = get_active_server(runtime)
    needed = [x_var, y_var] + ([color_var] if color_var else [])
    e = _load_dataset(dataset_id, server, variables=needed)
    if isinstance(e, str):
        return e
    e = _apply_constraints(e, constraints)
    e.variables = needed
    graph_url = _graph_url(e)

    if engine == "erddap":
        return f"Plot URL for dataset '{dataset_id}': {_erddap_plot_url(e, file_type)}\nCustomize it yourself: {graph_url}"

    try:
        df = fix_labels(e.to_pandas().reset_index())
        x_col = resolve_column(df, x_var)
        y_col = resolve_column(df, y_var)
        color_col = resolve_column(df, color_var) if color_var else None
        image_bytes = scatter_plot(
            df, x=x_col, y=y_col, color=color_col, step_size=step_size,
            title=f"{x_var} vs {y_var}", metadata=_dataset_metadata(dataset_id, server),
            filetype=file_type,
        )
    except Exception as err:
        return f"Plot failed for '{dataset_id}': {describe_erddap_error(err)}"

    return Command(update={
        "artifacts": [{"type": "image", "content": image_bytes, "title": f"{x_var} vs {y_var}"}],
        "messages": [ToolMessage(
            content=f"Plot generated.\nCustomize it yourself: {graph_url}",
            tool_call_id=runtime.tool_call_id,
        )],
    })

@tool(args_schema=ERDDAPRegionMapPlot)
def plot_region_map(
    runtime: ToolRuntime,
    constraints: ERDDAPConstraints,
    search_for: str = "all",
    max_datasets: int = 10,
    step_size: int = 25,
) -> str | Command:
    """Map ALL datasets on the active server with data in a lat/lon region (and optional
    time window) on one combined trajectory map, one color per dataset. Use for questions
    like 'what data exists in this area' -- for one known dataset, use plot_map instead."""
    server = get_active_server(runtime)

    # 1) region search -- same path as search_server_tool, bbox included
    result = search_server(
        server=server.url,
        query=search_for,
        constraints=constraints.to_kwargs(),
        protocol=server.protocol,
        response=server.response,
    )
    if isinstance(result, str):
        return result
    if result.empty or "Dataset ID" not in result.columns:
        return f"No datasets found on {server.url} in that region."

    ids = result["Dataset ID"].drop_duplicates().tolist()
    total_found = len(ids)

    # 2) per-dataset lat/lon download, skipping failures instead of dying
    tracks: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for dataset_id in ids[:max_datasets]:
        e = _load_dataset(dataset_id, server, variables=["longitude", "latitude"])
        if isinstance(e, str):          # e.g. griddap dataset with no lat/lon columns
            skipped.append(dataset_id)
            continue
        e = _apply_constraints(e, constraints)
        try:
            tracks[dataset_id] = fix_labels(e.to_pandas().reset_index()).iloc[::step_size]
        except Exception:
            skipped.append(dataset_id)

    if not tracks:
        return (
            f"Found {total_found} dataset(s) in the region but none could be plotted "
            f"(skipped: {skipped[:10]})."
        )

    # 3) one composite map
    try:
        html = multi_trajectory_map(tracks, title=f"{len(tracks)} datasets on {server.url}")
    except ValueError as err:
        return f"Region map failed: {err}"

    summary = (
        f"Region map generated and already displayed to the user "
        f"({len(tracks)} of {total_found} dataset(s), one color per dataset)."
        )
    if total_found > max_datasets:
        summary += (
            f" Capped at max_datasets={max_datasets}; do NOT re-run with a higher cap "
            f"unless the user explicitly asks for more datasets."
        )

    return Command(update={
        "artifacts": [{"type": "html", "content": html, "title": "Region map"}],
        "messages": [ToolMessage(content=summary, tool_call_id=runtime.tool_call_id)],
    })

@tool(args_schema=ERDDAPDataset)
def get_dataset_download_tool(
    runtime: ToolRuntime,
    dataset_id: str,
    variables: Optional[list[str]] = None,
    constraints: Optional[ERDDAPConstraints] = None,
    distinct: bool = False,
) -> str:
    """Get the download URL for a dataset without fetching the file through the agent.
    Returns a link the user can open or download directly, rather than the file's contents."""
    server = get_active_server(runtime)
    e = _load_dataset(dataset_id, server, variables=variables)
    if isinstance(e, str):
        return e

    e = _apply_constraints(e, constraints)

    url = quote_url(e.get_download_url(distinct=distinct))
    return f"Download URL for dataset '{dataset_id}': {url}"

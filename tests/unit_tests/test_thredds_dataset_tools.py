from typing import Any, Callable
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import xarray as xr
from langchain.tools import ToolRuntime
from langgraph.types import Command

from surfer.agents.thredds_agent import THREDDSConstraints
from surfer.agents.tools.thredds_dataset_tools import (
    apply_constraints,
    dataset_metadata,
    describe_dataset_tool,
    describe_thredds_dataset,
    get_dataset_download_tool,
    open_leaf_dataset,
    plot_custom,
    plot_map,
    plot_profile,
    plot_timeseries,
    resolve_leaf_dataset,
    to_dataframe,
)


def _func(tool: Any) -> Callable[..., str]:
    return tool.func


def _runtime() -> ToolRuntime:
    return ToolRuntime(state={}, context=None, config={}, stream_writer=None, tool_call_id="fake", store=None)


def _make_dataset(
    trajectory_ids: list[str] | None = None,
    cdm_data_type: str = "TrajectoryProfile",
) -> xr.Dataset:
    trajectory_ids = trajectory_ids or ["glider1"]
    n_traj = len(trajectory_ids)
    ds = xr.Dataset(
        {
            "time": (("trajectory", "profile"), [[pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")]] * n_traj),
            "latitude": (("trajectory", "profile"), [[38.0, 38.5]] * n_traj),
            "longitude": (("trajectory", "profile"), [[-72.0, -72.5]] * n_traj),
            "depth": (("trajectory", "profile", "obs"), [[[0.0, 1.0], [0.0, 1.0]]] * n_traj),
            "temperature": (("trajectory", "profile", "obs"), [[[20.0, 19.5], [21.0, 20.5]]] * n_traj),
            "salinity": (("trajectory", "profile", "obs"), [[[35.0, 35.1], [35.2, 35.3]]] * n_traj),
        },
        coords={"trajectory": trajectory_ids},
        attrs={
            "cdm_data_type": cdm_data_type,
            "time_coverage_start": "2023-01-01T00:00:00Z",
            "time_coverage_end": "2023-01-02T00:00:00Z",
            "geospatial_lat_min": 38.0,
            "geospatial_lat_max": 38.5,
            "geospatial_lon_min": -72.5,
            "geospatial_lon_max": -72.0,
            "contributor_name": "Jane Doe",
            "institution": "Rutgers",
            "platform": "glider",
            "license": "public",
        },
    )
    ds["time"].attrs = {"standard_name": "time", "long_name": "Profile Time"}
    ds["latitude"].attrs = {"standard_name": "latitude", "units": "degrees_north"}
    ds["longitude"].attrs = {"standard_name": "longitude", "units": "degrees_east"}
    ds["depth"].attrs = {"standard_name": "depth", "units": "m"}
    ds["temperature"].attrs = {"standard_name": "sea_water_temperature", "long_name": "Sea Water Temperature", "units": "Celsius"}
    ds["salinity"].attrs = {"standard_name": "sea_water_salinity", "units": "1"}
    return ds


def _fake_dataset_obj(name: str, access_urls: dict[str, str]):
    return Mock(name=name, access_urls=access_urls)


# resolve_leaf_dataset ---------------------------------------------------------------

@patch("surfer.agents.tools.thredds_dataset_tools.TDSCatalog")
def test_resolve_leaf_dataset_single_implicit_dataset(mock_cls):
    ds_obj = Mock()
    mock_cls.return_value = Mock(datasets={"glider1.nc": ds_obj}, catalog_refs={})

    result = resolve_leaf_dataset("https://x/catalog.xml")

    assert result is ds_obj


@patch("surfer.agents.tools.thredds_dataset_tools.TDSCatalog")
def test_resolve_leaf_dataset_no_datasets_returns_error(mock_cls):
    mock_cls.return_value = Mock(datasets={}, catalog_refs={"a": Mock()})

    result = resolve_leaf_dataset("https://x/catalog.xml")

    assert isinstance(result, str)
    assert "No datasets" in result


@patch("surfer.agents.tools.thredds_dataset_tools.TDSCatalog")
def test_resolve_leaf_dataset_ambiguous_without_name_returns_error(mock_cls):
    mock_cls.return_value = Mock(datasets={"a.nc": Mock(), "b.nc": Mock()}, catalog_refs={})

    result = resolve_leaf_dataset("https://x/catalog.xml")

    assert isinstance(result, str)
    assert "Multiple datasets" in result


@patch("surfer.agents.tools.thredds_dataset_tools.TDSCatalog")
def test_resolve_leaf_dataset_exact_name_match(mock_cls):
    ds_obj = Mock()
    mock_cls.return_value = Mock(datasets={"a.nc": Mock(), "b.nc": ds_obj}, catalog_refs={})

    result = resolve_leaf_dataset("https://x/catalog.xml", dataset_name="b.nc")

    assert result is ds_obj


@patch("surfer.agents.tools.thredds_dataset_tools.TDSCatalog")
def test_resolve_leaf_dataset_no_name_match_returns_error(mock_cls):
    mock_cls.return_value = Mock(datasets={"a.nc": Mock()}, catalog_refs={})

    result = resolve_leaf_dataset("https://x/catalog.xml", dataset_name="zzz")

    assert isinstance(result, str)
    assert "No dataset matching" in result


@patch("surfer.agents.tools.thredds_dataset_tools.TDSCatalog", side_effect=Exception("400 Client Error"))
def test_resolve_leaf_dataset_returns_clean_error_on_invalid_catalog_url(mock_cls):
    """Direct regression test: a live 400 was seen when the model passed a dataset's own
    dodsC data-access URL as dataset_id instead of its containing catalog.xml URL --
    TDSCatalog() raised an uncaught requests.HTTPError that killed the whole tool node
    instead of surfacing a clean, actionable message."""
    result = resolve_leaf_dataset("https://x/thredds/dodsC/some/dataset.nc")

    assert isinstance(result, str)
    assert "Could not fetch catalog" in result
    assert "not a dataset's own data/download URL" in result


# open_leaf_dataset -------------------------------------------------------------------

@patch("surfer.agents.tools.thredds_dataset_tools.xr.open_dataset")
def test_open_leaf_dataset_resolves_opendap_url(mock_open):
    dataset = _fake_dataset_obj("glider1.nc", {"OpenDAP": "https://x/dodsC/glider1.nc"})

    open_leaf_dataset(dataset)

    mock_open.assert_called_once_with("https://x/dodsC/glider1.nc", engine="netcdf4")


def test_open_leaf_dataset_raises_when_service_unavailable():
    dataset = _fake_dataset_obj("glider1.nc", {"HTTPServer": "https://x/fileServer/glider1.nc"})

    with pytest.raises(ValueError, match="not available"):
        open_leaf_dataset(dataset, use_service="OpenDAP")


# describe_thredds_dataset ------------------------------------------------------------

@patch("surfer.agents.tools.thredds_dataset_tools.open_leaf_dataset")
@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_describe_thredds_dataset_returns_expected_shape(mock_resolve, mock_open):
    mock_resolve.return_value = _fake_dataset_obj("glider1.nc", {})
    mock_open.return_value = _make_dataset()

    result = describe_thredds_dataset("https://x/catalog.xml")

    assert result["bounds"]["latitude>="] == 38.0
    assert result["bounds"]["latitude<="] == 38.5
    assert "temperature" in result["variables"]
    assert result["variables"]["temperature"]["standard_name"] == "sea_water_temperature"


@patch("surfer.agents.tools.thredds_dataset_tools.open_leaf_dataset")
@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_describe_thredds_dataset_rejects_grid_datasets(mock_resolve, mock_open):
    mock_resolve.return_value = _fake_dataset_obj("model_output.nc", {})
    mock_open.return_value = _make_dataset(cdm_data_type="Grid")

    result = describe_thredds_dataset("https://x/catalog.xml")

    assert isinstance(result, str)
    assert "Grid dataset" in result
    assert "not supported" in result


@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_describe_thredds_dataset_propagates_resolve_error(mock_resolve):
    mock_resolve.return_value = "No datasets at https://x/catalog.xml"

    result = describe_thredds_dataset("https://x/catalog.xml")

    assert result == "No datasets at https://x/catalog.xml"


@patch("surfer.agents.tools.thredds_dataset_tools.describe_thredds_dataset")
def test_describe_dataset_tool_formats_result(mock_describe):
    mock_describe.return_value = {
        "bounds": {"time>=": "2023-01-01", "time<=": "2023-01-02", "latitude>=": 38.0, "latitude<=": 38.5,
                   "longitude>=": -72.5, "longitude<=": -72.0},
        "variables": {"temperature": {"standard_name": "sea_water_temperature", "units": "Celsius"}},
    }

    result = _func(describe_dataset_tool)(runtime=_runtime(), catalog_url="https://x/catalog.xml")

    assert "temperature" in result
    assert "sea_water_temperature" in result


# to_dataframe --------------------------------------------------------------------------

def test_to_dataframe_flattens_and_renames_lat_lon():
    ds = _make_dataset()
    df = to_dataframe(ds, ["time", "latitude", "longitude", "temperature"])

    assert "lat" in df.columns and "lon" in df.columns
    assert "latitude" not in df.columns and "longitude" not in df.columns
    assert len(df) == 4  # 1 trajectory * 2 profile * 2 obs


def test_to_dataframe_raises_on_multiple_trajectories():
    ds = _make_dataset(trajectory_ids=["glider1", "glider2"])

    with pytest.raises(ValueError, match="not supported"):
        to_dataframe(ds, ["time", "temperature"])


# apply_constraints -----------------------------------------------------------------------

def test_apply_constraints_none_returns_dataset_unchanged():
    ds = _make_dataset()
    result = apply_constraints(ds, None)
    assert result is ds


def test_apply_constraints_filters_by_lat_range():
    ds = _make_dataset()
    constraints = THREDDSConstraints(min_lat=38.2, max_lat=38.5)

    result = apply_constraints(ds, constraints)
    df = to_dataframe(result, ["time", "latitude", "temperature"])

    assert (df["lat"] >= 38.2).all()


def test_apply_constraints_swaps_inverted_lat_range():
    ds = _make_dataset()
    constraints = THREDDSConstraints(min_lat=38.5, max_lat=38.0)  # inverted

    result = apply_constraints(ds, constraints)
    df = to_dataframe(result, ["time", "latitude", "temperature"])

    # swapped to min=38.0/max=38.5 -- equivalent to no filtering on this dataset's range
    assert len(df) > 0


def test_apply_constraints_drops_inverted_time_range():
    ds = _make_dataset()
    constraints = THREDDSConstraints(min_time="2023-01-02", max_time="2023-01-01")  # inverted

    result = apply_constraints(ds, constraints)
    df = to_dataframe(result, ["time", "temperature"])

    assert len(df) == 4  # inverted range dropped -> unconstrained, full dataset


def test_apply_constraints_filters_by_time_range():
    ds = _make_dataset()
    constraints = THREDDSConstraints(min_time="2023-01-02", max_time="2023-01-02")

    result = apply_constraints(ds, constraints)
    df = to_dataframe(result, ["time", "temperature"])

    assert (df["time"] == pd.Timestamp("2023-01-02")).all()


# dataset_metadata --------------------------------------------------------------------------

def test_dataset_metadata_reads_global_attrs():
    ds = _make_dataset()
    result = dataset_metadata(ds, "glider1.nc")

    assert result["id"] == "glider1.nc"
    assert result["author"] == "Jane Doe"
    assert result["institution"] == "Rutgers"


def test_dataset_metadata_defaults_missing_fields_to_na():
    ds = xr.Dataset(attrs={})
    result = dataset_metadata(ds, "glider1.nc")

    assert result["author"] == "N/A"
    assert result["institution"] == "N/A"


# plot tools ----------------------------------------------------------------------------------

@patch("surfer.agents.tools.thredds_dataset_tools.scatter_plot")
@patch("surfer.agents.tools.thredds_dataset_tools.open_leaf_dataset")
@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_plot_timeseries_reuses_shared_scatter_plot(mock_resolve, mock_open, mock_scatter):
    mock_resolve.return_value = _fake_dataset_obj("glider1.nc", {})
    mock_open.return_value = _make_dataset()
    mock_scatter.return_value = b"fake-png-bytes"

    result = _func(plot_timeseries)(runtime=_runtime(), dataset_id="https://x/catalog.xml", y_var="temperature")

    assert isinstance(result, Command)
    assert result.update["artifacts"][0]["type"] == "image"
    assert result.update["artifacts"][0]["content"] == b"fake-png-bytes"
    mock_scatter.assert_called_once()


@patch("surfer.agents.tools.thredds_dataset_tools.scatter_plot")
@patch("surfer.agents.tools.thredds_dataset_tools.open_leaf_dataset")
@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_plot_timeseries_passes_dataset_name_to_resolve_leaf_dataset(mock_resolve, mock_open, mock_scatter):
    """Direct regression test: a catalog listing many datasets (e.g. a 'Gridded' folder)
    can only be disambiguated if dataset_name reaches resolve_leaf_dataset -- previously
    every plot tool hardcoded None here, so the LLM had no schema field to pass the
    filename it got from find_datasets_tool."""
    mock_resolve.return_value = _fake_dataset_obj("20130111T000000_20130514T000000_challenger_ru29.nc", {})
    mock_open.return_value = _make_dataset()
    mock_scatter.return_value = b"fake-png-bytes"

    _func(plot_timeseries)(
        runtime=_runtime(), dataset_id="https://x/catalog.xml",
        dataset_name="20130111T000000_20130514T000000_challenger_ru29.nc", y_var="temperature",
    )

    mock_resolve.assert_called_once_with(
        "https://x/catalog.xml", "20130111T000000_20130514T000000_challenger_ru29.nc"
    )


@patch("surfer.agents.tools.thredds_dataset_tools.trajectory_map")
@patch("surfer.agents.tools.thredds_dataset_tools.open_leaf_dataset")
@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_plot_map_reuses_shared_trajectory_map(mock_resolve, mock_open, mock_traj_map):
    mock_resolve.return_value = _fake_dataset_obj("glider1.nc", {})
    mock_open.return_value = _make_dataset()
    mock_traj_map.return_value = "<html>fake</html>"

    result = _func(plot_map)(runtime=_runtime(), dataset_id="https://x/catalog.xml")

    assert isinstance(result, Command)
    assert result.update["artifacts"][0]["type"] == "html"
    assert result.update["artifacts"][0]["content"] == "<html>fake</html>"


@patch("surfer.agents.tools.thredds_dataset_tools.trajectory_map")
@patch("surfer.agents.tools.thredds_dataset_tools.open_leaf_dataset")
@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_plot_map_passes_dataset_name_to_resolve_leaf_dataset(mock_resolve, mock_open, mock_traj_map):
    mock_resolve.return_value = _fake_dataset_obj("glider1.nc", {})
    mock_open.return_value = _make_dataset()
    mock_traj_map.return_value = "<html>fake</html>"

    _func(plot_map)(runtime=_runtime(), dataset_id="https://x/catalog.xml", dataset_name="glider1.nc")

    mock_resolve.assert_called_once_with("https://x/catalog.xml", "glider1.nc")


@patch("surfer.agents.tools.thredds_dataset_tools.scatter_plot")
@patch("surfer.agents.tools.thredds_dataset_tools.open_leaf_dataset")
@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_plot_profile_inverts_y_axis(mock_resolve, mock_open, mock_scatter):
    mock_resolve.return_value = _fake_dataset_obj("glider1.nc", {})
    mock_open.return_value = _make_dataset()
    mock_scatter.return_value = b"fake-png-bytes"

    _func(plot_profile)(runtime=_runtime(), dataset_id="https://x/catalog.xml", x_var="temperature")

    assert mock_scatter.call_args.kwargs["invert_y"] is True


@patch("surfer.agents.tools.thredds_dataset_tools.scatter_plot")
@patch("surfer.agents.tools.thredds_dataset_tools.open_leaf_dataset")
@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_plot_custom_plots_arbitrary_variable_pair(mock_resolve, mock_open, mock_scatter):
    mock_resolve.return_value = _fake_dataset_obj("glider1.nc", {})
    mock_open.return_value = _make_dataset()
    mock_scatter.return_value = b"fake-png-bytes"

    result = _func(plot_custom)(runtime=_runtime(), dataset_id="https://x/catalog.xml", x_var="salinity", y_var="temperature")

    assert isinstance(result, Command)
    assert mock_scatter.call_args.kwargs["x"] == "salinity"
    assert mock_scatter.call_args.kwargs["y"] == "temperature"


@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_plot_timeseries_returns_error_string_on_resolve_failure(mock_resolve):
    mock_resolve.return_value = "No datasets at https://x/catalog.xml"

    result = _func(plot_timeseries)(runtime=_runtime(), dataset_id="https://x/catalog.xml", y_var="temperature")

    assert result == "No datasets at https://x/catalog.xml"


@patch("surfer.agents.tools.thredds_dataset_tools.open_leaf_dataset")
@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_plot_timeseries_returns_error_on_missing_variable(mock_resolve, mock_open):
    mock_resolve.return_value = _fake_dataset_obj("glider1.nc", {})
    mock_open.return_value = _make_dataset()

    result = _func(plot_timeseries)(runtime=_runtime(), dataset_id="https://x/catalog.xml", y_var="bogus_var")

    assert isinstance(result, str)
    assert "not found" in result


# get_dataset_download_tool -----------------------------------------------------------------

@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_get_dataset_download_tool_defaults_to_httpserver(mock_resolve):
    mock_resolve.return_value = _fake_dataset_obj(
        "glider1.nc", {"HTTPServer": "https://x/fileServer/glider1.nc", "OpenDAP": "https://x/dodsC/glider1.nc"}
    )

    result = _func(get_dataset_download_tool)(runtime=_runtime(), dataset_id="https://x/catalog.xml")

    assert "https://x/fileServer/glider1.nc" in result
    assert "HTTPServer" in result


@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_get_dataset_download_tool_passes_dataset_name_to_resolve_leaf_dataset(mock_resolve):
    mock_resolve.return_value = _fake_dataset_obj("glider1.nc", {"HTTPServer": "https://x/fileServer/glider1.nc"})

    _func(get_dataset_download_tool)(runtime=_runtime(), dataset_id="https://x/catalog.xml", dataset_name="glider1.nc")

    mock_resolve.assert_called_once_with("https://x/catalog.xml", "glider1.nc")


@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_get_dataset_download_tool_falls_back_to_opendap(mock_resolve):
    mock_resolve.return_value = _fake_dataset_obj("glider1.nc", {"OpenDAP": "https://x/dodsC/glider1.nc"})

    result = _func(get_dataset_download_tool)(runtime=_runtime(), dataset_id="https://x/catalog.xml")

    assert "https://x/dodsC/glider1.nc" in result
    assert "OpenDAP" in result


@patch("surfer.agents.tools.thredds_dataset_tools.resolve_leaf_dataset")
def test_get_dataset_download_tool_reports_no_service_available(mock_resolve):
    mock_resolve.return_value = _fake_dataset_obj("glider1.nc", {"WMS": "https://x/wms/glider1.nc"})

    result = _func(get_dataset_download_tool)(runtime=_runtime(), dataset_id="https://x/catalog.xml")

    assert "Neither HTTPServer nor OpenDAP" in result

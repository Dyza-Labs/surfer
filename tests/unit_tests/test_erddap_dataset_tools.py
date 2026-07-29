from typing import Any, Callable
from unittest.mock import patch

import pandas as pd
import pytest
from erddapy import ERDDAP
from langchain.tools import ToolRuntime

from surfer.agents.erddap_agent import ERDDAPConstraints, ERDDAPServer
from surfer.agents.tools.erddap_dataset_tools import (
    _apply_constraints,
    _dataset_metadata,
    _erddap_plot_url,
    _graph_url,
    _variable_units,
    describe_dataset,
    fix_labels,
    filter_dataset_variables,
    get_dataset_download_tool,
    get_erddap,
    plot_custom,
    plot_map,
    plot_profile,
    plot_timeseries,
)


def _func(tool: Any) -> Callable[..., str]:
    """@tool-wrapped functions are BaseTool at the type level, which has no `.func`
    attribute -- it's only declared on the concrete StructuredTool subclass that the
    @tool decorator actually returns at runtime. Centralizes the cast in one place."""
    return tool.func


def make_fake_erddap(time_min="2016-09-01T00:00:00Z", time_max="2016-09-30T23:59:59Z",
                      lat_min=38.0, lat_max=41.0, lon_min=-72.0, lon_max=-69.0):

    with patch.object(ERDDAP, "__init__", lambda self, **kwargs: None):  # prevent server call for faster process
        e = ERDDAP.__new__(ERDDAP)

    e.server = "https://gliders.ioos.us/erddap"
    e.protocol = "tabledap"
    e.response = "csv"
    e.dataset_id = "test-glider-001"
    e.variables = ["depth", "latitude", "longitude", "salinity", "temperature", "time"]
    e.dim_names = None  # get_download_url() reads this even for tabledap (unused there, but must exist)
    e.constraints = {
        "time>=": time_min,
        "time<=": time_max,
        "latitude>=": lat_min,
        "latitude<=": lat_max,
        "longitude>=": lon_min,
        "longitude<=": lon_max,
    }
    return e


# _apply_constraints (merged time+coord bound handling) ----------------------

def test_apply_constraints_with_no_input():
    e = make_fake_erddap()
    result = _apply_constraints(e, None)

    assert result.constraints is not None
    constraints = result.constraints
    assert constraints["time>="] == "2016-09-01T00:00:00Z"
    assert constraints["time<="] == "2016-09-30T23:59:59Z"
    assert constraints["latitude>="] == 38.0
    assert constraints["latitude<="] == 41.0


def test_apply_constraints_time_only_start():
    e = make_fake_erddap()
    tb = ERDDAPConstraints(min_time="2016-09-10T00:00:00Z")
    result = _apply_constraints(e, tb)
    assert result.constraints is not None
    constraints = result.constraints
    assert constraints["time>="] == "2016-09-10T00:00:00Z"
    assert constraints["time<="] == "2016-09-30T23:59:59Z"


def test_apply_constraints_time_only_end():
    e = make_fake_erddap()
    tb = ERDDAPConstraints(max_time="2016-09-24")
    result = _apply_constraints(e, tb)
    assert result.constraints is not None
    constraints = result.constraints
    assert constraints["time>="] == "2016-09-01T00:00:00Z"
    assert constraints["time<="] == "2016-09-24T23:59:59Z"


def test_apply_constraints_time_appended():
    e = make_fake_erddap()
    tb = ERDDAPConstraints(min_time="2016-09-10", max_time="2016-09-15")
    result = _apply_constraints(e, tb)
    assert result.constraints is not None
    constraints = result.constraints
    assert constraints["time>="] == "2016-09-10T00:00:00Z"
    assert constraints["time<="] == "2016-09-15T23:59:59Z"


# bounds wider than the dataset's range pass through unclamped -- ERDDAP itself
# narrows an out-of-range request server-side (confirmed live).

def test_apply_constraints_time_wider_than_dataset_passes_through():
    e = make_fake_erddap()
    tb = ERDDAPConstraints(min_time="2016-08-01T00:00:00Z", max_time="2016-10-15T00:00:00Z")
    result = _apply_constraints(e, tb)
    assert result.constraints is not None
    constraints = result.constraints
    assert constraints["time>="] == "2016-08-01T00:00:00Z"
    assert constraints["time<="] == "2016-10-15T00:00:00Z"


# confirm start > end resets to the dataset's default range (no swap, unlike coord bounds)

def test_apply_constraints_time_inverted_resets_to_default():
    e = make_fake_erddap()
    tb = ERDDAPConstraints(min_time="2016-09-20T00:00:00Z", max_time="2016-09-05T00:00:00Z")
    result = _apply_constraints(e, tb)
    assert result.constraints is not None
    constraints = result.constraints
    assert constraints["time>="] == "2016-09-01T00:00:00Z"
    assert constraints["time<="] == "2016-09-30T23:59:59Z"


def test_apply_constraints_coord_partial():
    e = make_fake_erddap()
    cb = ERDDAPConstraints(min_lat=39.0)
    result = _apply_constraints(e, cb)
    assert result.constraints is not None
    constraints = result.constraints
    assert constraints["latitude>="] == 39.0
    assert constraints["latitude<="] == 41.0


def test_apply_constraints_coord_inverted():
    e = make_fake_erddap()
    cb = ERDDAPConstraints(min_lat=40.0, max_lat=39.0)
    result = _apply_constraints(e, cb)
    assert result.constraints is not None
    constraints = result.constraints
    assert constraints["latitude>="] == 39.0
    assert constraints["latitude<="] == 40.0


def test_apply_constraints_coord_wider_than_dataset_passes_through():
    e = make_fake_erddap()
    cb = ERDDAPConstraints(min_lat=10.0, max_lat=50.0)
    result = _apply_constraints(e, cb)
    assert result.constraints is not None
    constraints = result.constraints
    assert constraints["latitude>="] == 10.0
    assert constraints["latitude<="] == 50.0


def test_apply_constraints_coord_partial_lon():
    e = make_fake_erddap()
    cb = ERDDAPConstraints(min_lon=-71.0)
    result = _apply_constraints(e, cb)
    assert result.constraints is not None
    constraints = result.constraints
    assert constraints["longitude>="] == -71.0
    assert constraints["longitude<="] == -69.0


def test_apply_constraints_coord_inverted_lon():
    e = make_fake_erddap()
    cb = ERDDAPConstraints(min_lon=-69.5, max_lon=-71.0)
    result = _apply_constraints(e, cb)
    assert result.constraints is not None
    constraints = result.constraints
    assert constraints["longitude>="] == -71.0
    assert constraints["longitude<="] == -69.5


def test_apply_constraints_coord_wider_than_dataset_passes_through_lon():
    e = make_fake_erddap()
    cb = ERDDAPConstraints(min_lon=-80.0, max_lon=-60.0)
    result = _apply_constraints(e, cb)
    assert result.constraints is not None
    constraints = result.constraints
    assert constraints["longitude>="] == -80.0
    assert constraints["longitude<="] == -60.0


def test_apply_constraints_coord_both_lat_and_lon():
    e = make_fake_erddap()
    cb = ERDDAPConstraints(min_lat=39.0, max_lat=40.0, min_lon=-71.0, max_lon=-70.0)
    result = _apply_constraints(e, cb)
    assert result.constraints is not None
    constraints = result.constraints
    assert constraints["latitude>="] == 39.0
    assert constraints["latitude<="] == 40.0
    assert constraints["longitude>="] == -71.0
    assert constraints["longitude<="] == -70.0


def test_apply_constraints_time_and_coord_together():
    e = make_fake_erddap()
    c = ERDDAPConstraints(min_time="2016-09-10", max_time="2016-09-15", min_lat=39.0, max_lon=-70.0)
    result = _apply_constraints(e, c)
    assert result.constraints is not None
    constraints = result.constraints
    assert constraints["time>="] == "2016-09-10T00:00:00Z"
    assert constraints["time<="] == "2016-09-15T23:59:59Z"
    assert constraints["latitude>="] == 39.0
    assert constraints["longitude<="] == -70.0


# fix_labels -----------------------------------------------------------------

def test_fix_labels_strips_units_suffix():
    df = pd.DataFrame(columns=["time (UTC)", "temperature (Celsius)"])
    result = fix_labels(df)
    assert list(result.columns) == ["time", "temperature"]


def test_fix_labels_aliases_lat_lon():
    df = pd.DataFrame(columns=["latitude (degrees_north)", "longitude (degrees_east)"])
    result = fix_labels(df)
    assert list(result.columns) == ["lat", "lon"]


def test_fix_labels_no_units_suffix_passthrough():
    df = pd.DataFrame(columns=["depth", "salinity"])
    result = fix_labels(df)
    assert list(result.columns) == ["depth", "salinity"]


# filter_dataset_variables ----------------------------------------------------

def test_filter_dataset_variables_excludes_qc_by_default():
    variables = {
        "temperature": {"units": "Celsius"},
        "temperature_qc": {"units": "1"},
        "salinity_flag": {"units": "1"},
    }
    result = filter_dataset_variables(variables)
    assert list(result.keys()) == ["temperature"]


def test_filter_dataset_variables_can_keep_qc():
    variables = {"temperature": {"units": "Celsius"}, "temperature_qc": {"units": "1"}}
    result = filter_dataset_variables(variables, exclude_qc=False)
    assert set(result.keys()) == {"temperature", "temperature_qc"}


def test_filter_dataset_variables_query_matches_name():
    variables = {"temperature": {"units": "Celsius"}, "salinity": {"units": "1"}}
    result = filter_dataset_variables(variables, query="temp")
    assert list(result.keys()) == ["temperature"]


def test_filter_dataset_variables_query_matches_attribute_value():
    variables = {
        "temperature": {"standard_name": "sea_water_temperature"},
        "salinity": {"standard_name": "sea_water_salinity"},
    }
    result = filter_dataset_variables(variables, query="salinity")
    assert list(result.keys()) == ["salinity"]


def test_filter_dataset_variables_no_query_returns_all_non_qc():
    variables = {"temperature": {}, "salinity": {}, "temperature_qc": {}}
    result = filter_dataset_variables(variables)
    assert set(result.keys()) == {"temperature", "salinity"}


# describe_dataset -------------------------------------------------------------

@patch("surfer.agents.tools.erddap_dataset_tools._fetch_dataset_variables")
def test_describe_dataset_returns_bounds_and_variables(mock_fetch):
    mock_fetch.return_value = {
        "NC_GLOBAL": {
            "time_coverage_start": "2016-09-01T00:00:00Z",
            "time_coverage_end": "2016-09-30T00:00:00Z",
            "geospatial_lat_min": "38.0",
            "geospatial_lat_max": "41.0",
            "geospatial_lon_min": "-72.0",
            "geospatial_lon_max": "-69.0",
        },
        "temperature": {"standard_name": "sea_water_temperature", "units": "Celsius"},
    }
    result = describe_dataset("https://gliders.ioos.us/erddap", "whoi_406-2016")
    assert result["bounds"]["time>="] == "2016-09-01T00:00:00Z"
    assert result["bounds"]["latitude<="] == 41.0
    assert result["variables"]["temperature"]["units"] == "Celsius"
    assert "NC_GLOBAL" not in result["variables"]


@patch("surfer.agents.tools.erddap_dataset_tools._fetch_dataset_variables", side_effect=ValueError("404"))
def test_describe_dataset_returns_error_string_on_failure(mock_fetch):
    result = describe_dataset("https://gliders.ioos.us/erddap", "bogus-id")
    assert isinstance(result, str)
    assert "not found" in result


# get_erddap -------------------------------------------------------------------

@patch("surfer.agents.tools.erddap_dataset_tools._fetch_dataset_variables")
def test_get_erddap_seeds_constraints_from_nc_global(mock_fetch):
    mock_fetch.return_value = {
        "NC_GLOBAL": {
            "time_coverage_start": "2016-09-01T00:00:00Z",
            "time_coverage_end": "2016-09-30T00:00:00Z",
            "geospatial_lat_min": "38.0",
            "geospatial_lat_max": "41.0",
            "geospatial_lon_min": "-72.0",
            "geospatial_lon_max": "-69.0",
        },
        "temperature": {},
    }
    e = get_erddap("whoi_406-2016", "https://gliders.ioos.us/erddap")
    assert e.constraints is not None
    constraints = e.constraints
    assert constraints["time>="] == "2016-09-01T00:00:00Z"
    assert constraints["latitude>="] == 38.0
    assert constraints["longitude<="] == -69.0


@patch("surfer.agents.tools.erddap_dataset_tools._fetch_dataset_variables")
def test_get_erddap_raises_on_invalid_variable(mock_fetch):
    mock_fetch.return_value = {"NC_GLOBAL": {}, "temperature": {}}
    with pytest.raises(ValueError, match="not found"):
        get_erddap("whoi_406-2016", "https://gliders.ioos.us/erddap", variables=["bogus"])


# _dataset_metadata / _variable_units -------------------------------------------

@patch("surfer.agents.tools.erddap_dataset_tools._fetch_dataset_variables")
def test_dataset_metadata_reads_nc_global(mock_fetch):
    mock_fetch.return_value = {
        "NC_GLOBAL": {
            "contributor_name": "Jane Doe",
            "institution": "Rutgers",
            "platform": "glider",
            "license": "CC-BY",
        }
    }
    server = ERDDAPServer(url="https://gliders.ioos.us/erddap")
    result = _dataset_metadata("whoi_406-2016", server)
    assert result["author"] == "Jane Doe"
    assert result["institution"] == "Rutgers"


@patch("surfer.agents.tools.erddap_dataset_tools._fetch_dataset_variables")
def test_dataset_metadata_defaults_missing_attrs(mock_fetch):
    mock_fetch.return_value = {"NC_GLOBAL": {}}
    server = ERDDAPServer(url="https://gliders.ioos.us/erddap")
    result = _dataset_metadata("whoi_406-2016", server)
    assert result["author"] == "N/A"


@patch("surfer.agents.tools.erddap_dataset_tools._fetch_dataset_variables")
def test_variable_units_returns_units_string(mock_fetch):
    mock_fetch.return_value = {"temperature": {"units": "degree_C"}}
    server = ERDDAPServer(url="https://gliders.ioos.us/erddap")
    assert _variable_units("whoi_406-2016", server, "temperature") == "degree_C"


@patch("surfer.agents.tools.erddap_dataset_tools._fetch_dataset_variables")
def test_variable_units_returns_none_when_missing(mock_fetch):
    mock_fetch.return_value = {"temperature": {}}
    server = ERDDAPServer(url="https://gliders.ioos.us/erddap")
    assert _variable_units("whoi_406-2016", server, "temperature") is None


# _graph_url / _erddap_plot_url -- quote_url escaping regression --------------
# Raw '<'/'>' in ERDDAP constraint syntax (e.g. 'time<=2023-...') reads as an HTML tag
# opening to markdown renderers and truncates the link -- confirm escaping is applied.

def test_graph_url_escapes_angle_brackets():
    e = make_fake_erddap()
    url = _graph_url(e)
    assert "<" not in url
    assert ">" not in url
    assert "%3C" in url or "%3E" in url


def test_graph_url_restores_original_response():
    e = make_fake_erddap()
    e.response = "csv"
    _graph_url(e)
    assert e.response == "csv"


def test_erddap_plot_url_escapes_angle_brackets():
    e = make_fake_erddap()
    url = _erddap_plot_url(e, "png")
    assert "<" not in url
    assert ">" not in url
    assert "%3C" in url or "%3E" in url


def test_erddap_plot_url_restores_original_response():
    e = make_fake_erddap()
    e.response = "csv"
    _erddap_plot_url(e, "png")
    assert e.response == "csv"


def test_erddap_plot_url_uses_requested_file_type():
    e = make_fake_erddap()
    url = _erddap_plot_url(e, "pdf")
    assert "test-glider-001.pdf" in url


# plot_timeseries / plot_profile / plot_map / plot_custom tools ---------------

@pytest.fixture  # just change url for different tests
def server():
    return ERDDAPServer(
        url="https://gliders.ioos.us/erddap/index.html",
        protocol="tabledap",
        response="csv",
    )


@pytest.fixture
def runtime(server):
    """Hand-built ToolRuntime -- these tools read the active server via
    get_active_server(runtime) instead of taking a server field directly, so a real
    ToolRuntime (not a mock) is needed for get_active_server's runtime.state lookups."""
    return ToolRuntime(
        state={"active_server_url": server.url, "servers": {server.url: server.model_dump()}},
        context=None, config={}, stream_writer=None, tool_call_id="fake", store=None,
    )


# NOTE: patch targets are fixed to match the actual import locations used inside
# erddap_dataset_tools.py (get_erddap/_fetch_dataset_variables are defined there;
# scatter_plot/trajectory_map/resolve_column are imported from surfer.agents.plotting).
@patch("surfer.agents.tools.erddap_dataset_tools._fetch_dataset_variables", return_value={"NC_GLOBAL": {}})
@patch("surfer.agents.tools.erddap_dataset_tools.get_erddap")
def test_plot_timeseries_local_engine(mock_get_erddap, mock_metadata, runtime):
    fake_e = make_fake_erddap()
    fake_e.to_pandas = lambda requests_kwargs=None, **kw: pd.DataFrame({
        "time (UTC)": ["2016-09-10T00:00:00Z"], "temperature (Celsius)": [12.3],
    })
    mock_get_erddap.return_value = fake_e

    with patch("surfer.agents.tools.erddap_dataset_tools.scatter_plot", return_value=b"fake-png-bytes") as mock_scatter:
        res = _func(plot_timeseries)(
            runtime=runtime, dataset_id="whoi_406-2016", y_var="temperature", step_size=5,
        )

    mock_get_erddap.assert_called_once()
    mock_scatter.assert_called_once()
    called_kwargs = mock_scatter.call_args.kwargs
    assert called_kwargs["x"] == "time"
    assert called_kwargs["y"] == "temperature"
    assert called_kwargs["step_size"] == 5

    artifacts = res.update["artifacts"]
    assert artifacts == [{"type": "image", "content": b"fake-png-bytes", "title": "temperature vs time"}]
    assert "Customize it yourself" in res.update["messages"][0].content


@patch("surfer.agents.tools.erddap_dataset_tools.get_erddap")
def test_plot_timeseries_erddap_engine(mock_get_erddap, runtime):
    fake_e = make_fake_erddap()
    mock_get_erddap.return_value = fake_e

    res = _func(plot_timeseries)(
        runtime=runtime, dataset_id="whoi_406-2016", y_var="temperature", engine="erddap",
    )

    mock_get_erddap.assert_called_once()
    # engine="erddap" never downloads data -- just builds URLs off the ERDDAP object
    assert "test-glider-001.png" in res
    assert "test-glider-001.graph" in res


@patch("surfer.agents.tools.erddap_dataset_tools._fetch_dataset_variables", return_value={"NC_GLOBAL": {}})
@patch("surfer.agents.tools.erddap_dataset_tools.get_erddap")
def test_plot_profile_local_engine(mock_get_erddap, mock_metadata, runtime):
    fake_e = make_fake_erddap()
    fake_e.to_pandas = lambda requests_kwargs=None, **kw: pd.DataFrame({
        "depth (m)": [1.0], "salinity (1)": [35.1],
    })
    mock_get_erddap.return_value = fake_e

    with patch("surfer.agents.tools.erddap_dataset_tools.scatter_plot", return_value=b"fake-png-bytes") as mock_scatter:
        res = _func(plot_profile)(runtime=runtime, dataset_id="whoi_406-2016", x_var="salinity")

    called_kwargs = mock_scatter.call_args.kwargs
    assert called_kwargs["x"] == "salinity"
    assert called_kwargs["y"] == "depth"
    assert called_kwargs["invert_y"] is True

    artifacts = res.update["artifacts"]
    assert artifacts == [{"type": "image", "content": b"fake-png-bytes", "title": "salinity profile"}]


@patch("surfer.agents.tools.erddap_dataset_tools.get_erddap")
def test_plot_map_erddap_engine(mock_get_erddap, runtime):
    fake_e = make_fake_erddap()
    mock_get_erddap.return_value = fake_e

    res = _func(plot_map)(runtime=runtime, dataset_id="whoi_406-2016", engine="erddap")

    assert "test-glider-001.png" in res
    assert "test-glider-001.graph" in res


@patch("surfer.agents.tools.erddap_dataset_tools._fetch_dataset_variables", return_value={"NC_GLOBAL": {}})
@patch("surfer.agents.tools.erddap_dataset_tools.get_erddap")
def test_plot_map_local_engine_with_color_var_units(mock_get_erddap, mock_metadata, runtime):
    fake_e = make_fake_erddap()
    fake_e.to_pandas = lambda requests_kwargs=None, **kw: pd.DataFrame({
        "latitude (degrees_north)": [39.0], "longitude (degrees_east)": [-70.0],
        "temperature (Celsius)": [12.3],
    })
    mock_get_erddap.return_value = fake_e
    mock_metadata.return_value = {"NC_GLOBAL": {}, "temperature": {"units": "Celsius"}}

    with patch("surfer.agents.tools.erddap_dataset_tools._variable_units", return_value="Celsius"):
        with patch("surfer.agents.tools.erddap_dataset_tools.trajectory_map", return_value="<html>fake map</html>") as mock_traj:
            res = _func(plot_map)(runtime=runtime, dataset_id="whoi_406-2016", color_var="temperature")

    called_kwargs = mock_traj.call_args.kwargs
    assert called_kwargs["cbar_label"] == "temperature (Celsius)"

    artifacts = res.update["artifacts"]
    assert artifacts == [{
        "type": "html", "content": "<html>fake map</html>", "title": "Flight path: whoi_406-2016",
    }]


@patch("surfer.agents.tools.erddap_dataset_tools._fetch_dataset_variables", return_value={"NC_GLOBAL": {}})
@patch("surfer.agents.tools.erddap_dataset_tools.get_erddap")
def test_plot_custom_local_engine(mock_get_erddap, mock_metadata, runtime):
    fake_e = make_fake_erddap()
    fake_e.to_pandas = lambda requests_kwargs=None, **kw: pd.DataFrame({
        "salinity (1)": [35.1], "temperature (Celsius)": [12.3],
    })
    mock_get_erddap.return_value = fake_e

    with patch("surfer.agents.tools.erddap_dataset_tools.scatter_plot", return_value=b"fake-png-bytes") as mock_scatter:
        res = _func(plot_custom)(
            runtime=runtime, dataset_id="whoi_406-2016", x_var="salinity", y_var="temperature",
        )

    called_kwargs = mock_scatter.call_args.kwargs
    assert called_kwargs["x"] == "salinity"
    assert called_kwargs["y"] == "temperature"

    artifacts = res.update["artifacts"]
    assert artifacts == [{"type": "image", "content": b"fake-png-bytes", "title": "salinity vs temperature"}]


def test_load_dataset_error_short_circuits_before_plotting():
    """_load_dataset's ERDDAP | str error path should return the error string
    directly, without ever reaching the plotting step."""
    with patch(
        "surfer.agents.tools.erddap_dataset_tools.get_erddap",
        side_effect=ValueError("Variables ['bogus'] not found. Available: ['time']"),
    ):
        runtime = ToolRuntime(
            state={
                "active_server_url": "https://gliders.ioos.us/erddap",
                "servers": {"https://gliders.ioos.us/erddap": {
                    "url": "https://gliders.ioos.us/erddap", "protocol": "tabledap", "response": "csv",
                }},
            },
            context=None, config={}, stream_writer=None, tool_call_id="fake", store=None,
        )
        res = _func(plot_timeseries)(runtime=runtime, dataset_id="whoi_406-2016", y_var="bogus")
    assert "not found" in res


# get_dataset_download_tool ---------------------------------------------------

@patch("surfer.agents.tools.erddap_dataset_tools._load_dataset")
def test_get_dataset_download_tool_returns_quoted_url(mock_load, runtime):
    fake_e = make_fake_erddap()
    mock_load.return_value = fake_e

    res = _func(get_dataset_download_tool)(runtime=runtime, dataset_id="whoi_406-2016")

    assert "<" not in res
    assert ">" not in res
    assert "test-glider-001" in res


@patch("surfer.agents.tools.erddap_dataset_tools._load_dataset", return_value="Could not load dataset")
def test_get_dataset_download_tool_short_circuits_on_load_error(mock_load, runtime):
    res = _func(get_dataset_download_tool)(runtime=runtime, dataset_id="bogus-id")
    assert res == "Could not load dataset"

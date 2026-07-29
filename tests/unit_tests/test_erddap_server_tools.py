from typing import Any, Callable
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from langchain.tools import ToolRuntime

from surfer.agents.erddap_agent import ERDDAPServer
from surfer.agents.tools.erddap_server_tools import (
    clean_erddap_title,
    filter_categorize_values,
    get_categorize_values,
    get_server_summary,
    get_server_summary_tool,
    search_server,
    search_server_tool,
    validate_categorize_value,
)


def _func(tool: Any) -> Callable[..., str]:
    """@tool-wrapped functions are BaseTool at the type level, which has no `.func`
    attribute -- it's only declared on the concrete StructuredTool subclass that the
    @tool decorator actually returns at runtime. Centralizes the cast in one place."""
    return tool.func


def _runtime(server: ERDDAPServer) -> ToolRuntime:
    return ToolRuntime(
        state={"active_server_url": server.url, "servers": {server.url: server.model_dump()}},
        context=None, config={}, stream_writer=None, tool_call_id="fake", store=None,
    )


# clean_erddap_title --------------------------------------------------------------

def test_clean_erddap_title_underscored_lowercase():
    assert clean_erddap_title("rutgers_university") == "Rutgers University"


def test_clean_erddap_title_strips_leading_trailing_underscores_and_spaces():
    assert clean_erddap_title("_noaa_ ") == "Noaa"


def test_clean_erddap_title_none_returns_empty_string():
    assert clean_erddap_title(None) == ""


# filter_categorize_values ----------------------------------------------------------

def test_filter_categorize_values_no_query_returns_all():
    values = ["rutgers_university", "noaa", "ioos"]
    assert filter_categorize_values(values) == values


def test_filter_categorize_values_case_insensitive_substring_match():
    values = ["Rutgers_University", "NOAA", "ioos"]
    assert filter_categorize_values(values, query="rutgers") == ["Rutgers_University"]


def test_filter_categorize_values_no_match_returns_empty():
    values = ["rutgers_university", "noaa"]
    assert filter_categorize_values(values, query="zzz_nonexistent") == []


# get_categorize_values ---------------------------------------------------------------

@patch("surfer.agents.tools.erddap_server_tools.pd.read_csv")
@patch("surfer.agents.tools.erddap_server_tools.ERDDAP")
def test_get_categorize_values_returns_list(mock_erddap_cls, mock_read_csv):
    mock_erddap_cls.return_value.get_categorize_url.return_value = "https://fake/categorize/institution"
    mock_read_csv.return_value = pd.DataFrame({"Category": ["rutgers_university", "noaa", None]})

    result = get_categorize_values("https://gliders.ioos.us/erddap", "institution")

    assert result == ["rutgers_university", "noaa"]


@patch("surfer.agents.tools.erddap_server_tools.pd.read_csv", side_effect=ValueError("404"))
@patch("surfer.agents.tools.erddap_server_tools.ERDDAP")
def test_get_categorize_values_returns_error_string_on_failure(mock_erddap_cls, mock_read_csv):
    mock_erddap_cls.return_value.get_categorize_url.return_value = "https://fake/categorize/bogus_field"

    result = get_categorize_values("https://gliders.ioos.us/erddap", "bogus_field")

    assert isinstance(result, str)
    assert "Could not retrieve" in result


@patch("surfer.agents.tools.erddap_server_tools.pd.read_csv")
@patch("surfer.agents.tools.erddap_server_tools.ERDDAP")
def test_get_categorize_values_caches_repeat_calls(mock_erddap_cls, mock_read_csv):
    """Direct regression test for the resolve_categorize_values_tool -> search_server_tool
    redundant-fetch latency fix: a second call for the same (server, categorize_by) within
    the TTL must reuse the cached result instead of issuing another live HTTP fetch."""
    import surfer.agents.tools.erddap_server_tools as server_tools
    server_tools._categorize_values_cache.clear()

    mock_erddap_cls.return_value.get_categorize_url.return_value = "https://fake/categorize/institution"
    mock_read_csv.return_value = pd.DataFrame({"Category": ["rutgers_university", "noaa"]})

    first = get_categorize_values("https://gliders.ioos.us/erddap", "institution")
    second = get_categorize_values("https://gliders.ioos.us/erddap", "institution")

    assert first == second == ["rutgers_university", "noaa"]
    mock_read_csv.assert_called_once()


# get_server_summary ------------------------------------------------------------------

@patch("surfer.agents.tools.erddap_server_tools.get_categorize_values")
@patch("surfer.agents.tools.erddap_server_tools.requests.get")
@patch("surfer.agents.tools.erddap_server_tools.pd.read_csv")
@patch("surfer.agents.tools.erddap_server_tools.ERDDAP")
def test_get_server_summary_returns_expected_shape(
    mock_erddap_cls, mock_read_csv, mock_requests_get, mock_get_categorize,
):
    mock_erddap_cls.return_value.get_info_url.return_value = "https://fake/info/index.csv"
    mock_read_csv.return_value = pd.DataFrame({
        "Dataset ID": ["allDatasets", "whoi_406-2016"],
        "tabledap": ["", "https://fake/tabledap/whoi_406-2016"],
        "griddap": ["", None],
        "wms": ["", None],
    })
    mock_requests_get.return_value = Mock(text="View a List of All 42 Datasets")
    mock_get_categorize.side_effect = [
        ["rutgers_university"],  # institution
        ["glider", "qartod_flags"],  # keywords
    ]

    result = get_server_summary("https://gliders.ioos.us/erddap")

    assert result["dataset_count"] == 42
    assert result["protocols"] == ["tabledap"]
    assert result["institutions"] == ["Rutgers University"]
    # qartod_flags should be excluded from keywords via is_qc_keyword
    assert result["keywords"] == ["Glider"]


@patch("surfer.agents.tools.erddap_server_tools.requests.get")
@patch("surfer.agents.tools.erddap_server_tools.pd.read_csv")
@patch("surfer.agents.tools.erddap_server_tools.ERDDAP")
def test_get_server_summary_raises_when_dataset_count_not_found(mock_erddap_cls, mock_read_csv, mock_requests_get):
    mock_erddap_cls.return_value.get_info_url.return_value = "https://fake/info/index.csv"
    mock_read_csv.return_value = pd.DataFrame({
        "Dataset ID": ["allDatasets"], "tabledap": [""], "griddap": [""], "wms": [""],
    })
    mock_requests_get.return_value = Mock(text="no dataset count text here")

    with pytest.raises(ValueError, match="Could not find dataset count"):
        get_server_summary("https://gliders.ioos.us/erddap")


@patch("surfer.agents.tools.erddap_server_tools.get_server_summary")
def test_get_server_summary_tool_formats_result(mock_summary):
    mock_summary.return_value = {
        "dataset_count": 42, "protocols": ["tabledap"],
        "institutions": ["Rutgers University"], "keywords": ["Glider"],
    }
    server = ERDDAPServer(url="https://gliders.ioos.us/erddap")
    result = _func(get_server_summary_tool)(runtime=_runtime(server))

    assert "42" in result
    assert "Rutgers University" in result


@patch("surfer.agents.tools.erddap_server_tools.get_server_summary", side_effect=ValueError("could not parse"))
def test_get_server_summary_tool_returns_error_string_on_failure(mock_summary):
    server = ERDDAPServer(url="https://gliders.ioos.us/erddap")
    result = _func(get_server_summary_tool)(runtime=_runtime(server))

    assert "Could not summarize" in result
    assert "could not parse" in result


# validate_categorize_value -------------------------------------------------------------

@patch("surfer.agents.tools.erddap_server_tools.get_categorize_values")
def test_validate_categorize_value_returns_none_when_valid(mock_get_values):
    mock_get_values.return_value = ["rutgers_university", "noaa"]
    result = validate_categorize_value("https://gliders.ioos.us/erddap", "institution", "rutgers_university")
    assert result is None


@patch("surfer.agents.tools.erddap_server_tools.get_categorize_values")
def test_validate_categorize_value_returns_error_message_when_invalid(mock_get_values):
    mock_get_values.return_value = ["rutgers_university", "noaa"]
    result = validate_categorize_value("https://gliders.ioos.us/erddap", "institution", "bogus_university")
    assert result is not None
    assert "not found" in result
    assert "bogus_university" in result


@patch("surfer.agents.tools.erddap_server_tools.get_categorize_values", return_value="server unreachable")
def test_validate_categorize_value_propagates_error_string(mock_get_values):
    result = validate_categorize_value("https://gliders.ioos.us/erddap", "institution", "rutgers_university")
    assert result == "server unreachable"


# search_server ---------------------------------------------------------------------------

@patch("surfer.agents.tools.erddap_server_tools.pd.read_csv")
@patch("surfer.agents.tools.erddap_server_tools.ERDDAP")
def test_search_server_returns_dataframe(mock_erddap_cls, mock_read_csv):
    mock_erddap_cls.return_value.get_search_url.return_value = "https://fake/search"
    mock_read_csv.return_value = pd.DataFrame({"Dataset ID": ["whoi_406-2016"], "Title": ["WHOI Glider 406"]})

    result = search_server("https://gliders.ioos.us/erddap", "salinity")

    assert isinstance(result, pd.DataFrame)
    assert "whoi_406-2016" in result["Dataset ID"].values


@patch("surfer.agents.tools.erddap_server_tools.pd.read_csv")
@patch("surfer.agents.tools.erddap_server_tools.ERDDAP")
def test_search_server_single_variable_uses_variablename_kwarg(mock_erddap_cls, mock_read_csv):
    mock_erddap_cls.return_value.get_search_url.return_value = "https://fake/search"
    mock_read_csv.return_value = pd.DataFrame({"Dataset ID": [], "Title": []})

    search_server("https://gliders.ioos.us/erddap", "all", variables=["temperature"])

    called_kwargs = mock_erddap_cls.return_value.get_search_url.call_args.kwargs
    assert called_kwargs["variableName"] == "temperature"


@patch("surfer.agents.tools.erddap_server_tools.pd.read_csv")
@patch("surfer.agents.tools.erddap_server_tools.ERDDAP")
def test_search_server_multiple_variables_concatenates_into_query(mock_erddap_cls, mock_read_csv):
    mock_erddap_cls.return_value.get_search_url.return_value = "https://fake/search"
    mock_read_csv.return_value = pd.DataFrame({"Dataset ID": [], "Title": []})

    search_server("https://gliders.ioos.us/erddap", "glider", variables=["temperature", "salinity"])

    called_kwargs = mock_erddap_cls.return_value.get_search_url.call_args.kwargs
    assert called_kwargs["search_for"] == "glider temperature salinity"


@patch("surfer.agents.tools.erddap_server_tools.pd.read_csv", side_effect=ValueError("404"))
@patch("surfer.agents.tools.erddap_server_tools.ERDDAP")
def test_search_server_returns_error_string_on_failure(mock_erddap_cls, mock_read_csv):
    mock_erddap_cls.return_value.get_search_url.return_value = "https://fake/search"

    result = search_server("https://gliders.ioos.us/erddap", "salinity")

    assert isinstance(result, str)
    assert "Search failed" in result


# search_server_tool --------------------------------------------------------------------

@patch("surfer.agents.tools.erddap_server_tools.validate_categorize_value")
def test_search_server_tool_returns_validation_error_before_searching(mock_validate):
    mock_validate.return_value = "'Fake University' not found in institution. Available: ['NOAA']"
    server = ERDDAPServer(url="https://gliders.ioos.us/erddap")

    result = _func(search_server_tool)(runtime=_runtime(server), institution="Fake University")

    assert "not found" in result
    mock_validate.assert_called_once()


@patch("surfer.agents.tools.erddap_server_tools.search_server")
def test_search_server_tool_reports_no_datasets_found(mock_search):
    mock_search.return_value = pd.DataFrame()
    server = ERDDAPServer(url="https://gliders.ioos.us/erddap")

    result = _func(search_server_tool)(runtime=_runtime(server), search_for="zzz_nonexistent")

    assert "No datasets found" in result


@patch("surfer.agents.tools.erddap_server_tools.search_server")
def test_search_server_tool_formats_results(mock_search):
    mock_search.return_value = pd.DataFrame({
        "Dataset ID": ["whoi_406-2016"], "Title": ["WHOI Glider 406"],
    })
    server = ERDDAPServer(url="https://gliders.ioos.us/erddap")

    result = _func(search_server_tool)(runtime=_runtime(server), search_for="salinity")

    assert "whoi_406-2016" in result
    assert "WHOI Glider 406" in result


@patch("surfer.agents.tools.erddap_server_tools.search_server", side_effect=RuntimeError("boom"))
def test_search_server_tool_catches_unexpected_errors(mock_search):
    server = ERDDAPServer(url="https://gliders.ioos.us/erddap")

    result = _func(search_server_tool)(runtime=_runtime(server), search_for="salinity")

    assert "unexpected error" in result.lower()

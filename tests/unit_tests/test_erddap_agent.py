from unittest.mock import Mock, patch

import pytest
import requests
from langchain.tools import ToolRuntime
from langgraph.types import Command

from surfer.agents.erddap_agent import (
    ERDDAPServer,
    check_server_status,
    describe_erddap_error,
    get_active_server,
    get_server,
    is_qc_keyword,
    is_qc_variable,
    register_server_tool,
    set_active_server_tool,
)


# ERDDAPServer.sanitize_url -----------------------------------------------------

def test_sanitize_url_strips_query_string_and_fragment():
    result = ERDDAPServer.sanitize_url("https://gliders.ioos.us/erddap/search?q=temp#frag")
    assert result == "https://gliders.ioos.us/erddap"


def test_sanitize_url_truncates_after_erddap_root():
    result = ERDDAPServer.sanitize_url("https://gliders.ioos.us/erddap/tabledap/whoi_406.html")
    assert result == "https://gliders.ioos.us/erddap"


def test_sanitize_url_strips_trailing_index_html():
    result = ERDDAPServer.sanitize_url("https://gliders.ioos.us/erddap/index.html")
    assert result == "https://gliders.ioos.us/erddap"


def test_sanitize_url_strips_trailing_slash():
    result = ERDDAPServer.sanitize_url("https://gliders.ioos.us/erddap/")
    assert result == "https://gliders.ioos.us/erddap"


def test_sanitize_url_case_insensitive_erddap_root():
    result = ERDDAPServer.sanitize_url("https://gliders.ioos.us/ERDDAP/index.html")
    assert result == "https://gliders.ioos.us/ERDDAP"


# get_server / get_active_server -------------------------------------------------

def _runtime(state):
    return ToolRuntime(state=state, context=None, config={}, stream_writer=None, tool_call_id="fake", store=None)


def test_get_server_returns_registered_server():
    url = "https://gliders.ioos.us/erddap"
    runtime = _runtime({"servers": {url: {"url": url, "protocol": "tabledap", "response": "csv"}}})
    server = get_server(runtime, url)
    assert server.url == url


def test_get_server_raises_when_not_registered():
    runtime = _runtime({"servers": {}})
    with pytest.raises(ValueError, match="not registered"):
        get_server(runtime, "https://gliders.ioos.us/erddap")


def test_get_active_server_returns_active_server():
    url = "https://gliders.ioos.us/erddap"
    runtime = _runtime({
        "active_server_url": url,
        "servers": {url: {"url": url, "protocol": "tabledap", "response": "csv"}},
    })
    server = get_active_server(runtime)
    assert server.url == url


def test_get_active_server_raises_when_none_set():
    runtime = _runtime({"active_server_url": None, "servers": {}})
    with pytest.raises(ValueError, match="No active server set"):
        get_active_server(runtime)


# check_server_status ------------------------------------------------------------

@patch("surfer.agents.erddap_agent.requests.get")
def test_check_server_status_reachable(mock_get):
    mock_get.return_value = Mock(text="ERDDAP_version=2.24\n")
    mock_get.return_value.raise_for_status = Mock()

    result = check_server_status("https://gliders.ioos.us/erddap")

    assert result["reachable"] is True
    assert result["error"] is None
    assert "2.24" in result["version"]


@patch("surfer.agents.erddap_agent.requests.get", side_effect=requests.exceptions.ConnectionError("refused"))
def test_check_server_status_unreachable(mock_get):
    result = check_server_status("https://bogus.example.com/erddap")

    assert result["reachable"] is False
    assert result["version"] is None
    assert "refused" in result["error"]


# register_server_tool / set_active_server_tool ----------------------------------

@patch("surfer.agents.erddap_agent.check_server_status")
def test_register_server_tool_returns_command_with_state_update(mock_status):
    mock_status.return_value = {"reachable": True, "version": "2.24", "error": None}
    runtime = _runtime({})

    result = register_server_tool.func(runtime=runtime, url="https://gliders.ioos.us/erddap/index.html")

    assert isinstance(result, Command)
    assert result.update is not None
    assert result.update["active_server_url"] == "https://gliders.ioos.us/erddap"
    assert "https://gliders.ioos.us/erddap" in result.update["servers"]


@patch("surfer.agents.erddap_agent.check_server_status")
def test_register_server_tool_reports_unreachable_in_message(mock_status):
    mock_status.return_value = {"reachable": False, "version": None, "error": "timed out"}
    runtime = _runtime({})

    result = register_server_tool.func(runtime=runtime, url="https://gliders.ioos.us/erddap")

    assert isinstance(result, Command)
    assert result.update is not None
    message_content = result.update["messages"][0].content
    assert "Reachable: False" in message_content
    assert "timed out" in message_content


def test_set_active_server_tool_switches_active_server():
    url_a = "https://gliders.ioos.us/erddap"
    url_b = "https://hfr.marine.rutgers.edu/erddap"
    runtime = _runtime({
        "active_server_url": url_a,
        "servers": {
            url_a: {"url": url_a, "protocol": "tabledap", "response": "csv"},
            url_b: {"url": url_b, "protocol": "tabledap", "response": "csv"},
        },
    })

    result = set_active_server_tool.func(runtime=runtime, url=url_b)

    assert isinstance(result, Command)
    assert result.update is not None
    assert result.update["active_server_url"] == url_b


def test_set_active_server_tool_raises_for_unregistered_server():
    runtime = _runtime({"active_server_url": None, "servers": {}})
    with pytest.raises(ValueError, match="not registered"):
        set_active_server_tool.func(runtime=runtime, url="https://gliders.ioos.us/erddap")


# is_qc_variable / is_qc_keyword --------------------------------------------------

@pytest.mark.parametrize("name", ["temperature_qc", "salinity_flag", "pressure_qartod_flag", "TEMP_QC"])
def test_is_qc_variable_true_for_qc_patterns(name):
    assert is_qc_variable(name) is True


@pytest.mark.parametrize("name", ["temperature", "salinity", "depth"])
def test_is_qc_variable_false_for_real_variables(name):
    assert is_qc_variable(name) is False


@pytest.mark.parametrize("keyword", ["QARTOD", "qartod_flags"])
def test_is_qc_keyword_true_for_qc_terms(keyword):
    assert is_qc_keyword(keyword) is True


@pytest.mark.parametrize("keyword", ["Red Flag Warning", "Storm Flag", "Temperature"])
def test_is_qc_keyword_false_for_thematic_keywords_with_flag_word(keyword):
    """is_qc_keyword is deliberately narrower than is_qc_variable -- free-form thematic
    keywords like 'Red Flag Warning' shouldn't be treated as QC-related just because
    they contain the word 'flag'."""
    assert is_qc_keyword(keyword) is False


# describe_erddap_error ------------------------------------------------------------

def test_describe_erddap_error_uses_read_when_available():
    class FakeHTTPError(Exception):
        def read(self):
            return b"Error {\n    code=404;\n    message=\"Not Found\";\n}"

    result = describe_erddap_error(FakeHTTPError())
    assert "Not Found" in result


def test_describe_erddap_error_falls_back_to_str():
    result = describe_erddap_error(ValueError("plain error message"))
    assert result == "plain error message"

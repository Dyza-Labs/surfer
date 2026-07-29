from unittest.mock import Mock, patch

import pytest
from langchain.tools import ToolRuntime
from langgraph.types import Command

from surfer.agents.thredds_agent import (
    THREDDSServer,
    check_thredds_status,
    get_active_catalog_url,
    get_active_server,
    get_server,
    register_server_tool,
    set_active_catalog_tool,
    set_active_server_tool,
)


# THREDDSServer.sanitize_url -----------------------------------------------------

def test_sanitize_url_strips_query_string_and_fragment():
    result = THREDDSServer.sanitize_url("https://gliders.ioos.us/thredds/catalog/catalog.xml?dataset=x#frag")
    assert result == "https://gliders.ioos.us/thredds/catalog/catalog.xml"


def test_sanitize_url_rewrites_catalog_html_to_catalog_xml():
    result = THREDDSServer.sanitize_url("https://gliders.ioos.us/thredds/catalog/catalog.html")
    assert result == "https://gliders.ioos.us/thredds/catalog/catalog.xml"


def test_sanitize_url_appends_catalog_xml_to_bare_host():
    result = THREDDSServer.sanitize_url("https://gliders.ioos.us/thredds/catalog")
    assert result == "https://gliders.ioos.us/thredds/catalog/catalog.xml"


def test_sanitize_url_strips_trailing_slash_before_suffixing():
    result = THREDDSServer.sanitize_url("https://gliders.ioos.us/thredds/catalog/")
    assert result == "https://gliders.ioos.us/thredds/catalog/catalog.xml"


def test_sanitize_url_leaves_catalog_xml_untouched():
    result = THREDDSServer.sanitize_url("https://gliders.ioos.us/thredds/catalog/catalog.xml")
    assert result == "https://gliders.ioos.us/thredds/catalog/catalog.xml"


# get_server / get_active_server / get_active_catalog_url ------------------------

def _runtime(state):
    return ToolRuntime(state=state, context=None, config={}, stream_writer=None, tool_call_id="fake", store=None)


def test_get_server_returns_registered_server():
    url = "https://gliders.ioos.us/thredds/catalog/catalog.xml"
    runtime = _runtime({"servers": {url: {"url": url}}})
    server = get_server(runtime, url)
    assert server.url == url


def test_get_server_raises_when_not_registered():
    runtime = _runtime({"servers": {}})
    with pytest.raises(ValueError, match="not registered"):
        get_server(runtime, "https://gliders.ioos.us/thredds/catalog/catalog.xml")


def test_get_active_server_returns_active_server():
    url = "https://gliders.ioos.us/thredds/catalog/catalog.xml"
    runtime = _runtime({"active_server_url": url, "servers": {url: {"url": url}}})
    server = get_active_server(runtime)
    assert server.url == url


def test_get_active_server_raises_when_none_set():
    runtime = _runtime({"active_server_url": None, "servers": {}})
    with pytest.raises(ValueError, match="No active server set"):
        get_active_server(runtime)


def test_get_active_catalog_url_returns_current_position():
    runtime = _runtime({"active_catalog_url": "https://gliders.ioos.us/thredds/catalog/deployments/catalog.xml"})
    assert get_active_catalog_url(runtime) == "https://gliders.ioos.us/thredds/catalog/deployments/catalog.xml"


def test_get_active_catalog_url_raises_when_none_set():
    runtime = _runtime({"active_catalog_url": None})
    with pytest.raises(ValueError, match="No active catalog"):
        get_active_catalog_url(runtime)


# check_thredds_status ------------------------------------------------------------

@patch("surfer.agents.thredds_agent.TDSCatalog")
def test_check_thredds_status_reachable(mock_cls):
    mock_cls.return_value = Mock(catalog_name="IOOS Glider DAC")

    result = check_thredds_status("https://gliders.ioos.us/thredds/catalog/catalog.xml")

    assert result["reachable"] is True
    assert result["error"] is None
    assert result["catalog_name"] == "IOOS Glider DAC"


@patch("surfer.agents.thredds_agent.TDSCatalog", side_effect=Exception("connection refused"))
def test_check_thredds_status_unreachable(mock_cls):
    result = check_thredds_status("https://bogus.example.com/thredds/catalog/catalog.xml")

    assert result["reachable"] is False
    assert result["catalog_name"] is None
    assert "connection refused" in result["error"]


# register_server_tool / set_active_server_tool / set_active_catalog_tool -------

@patch("surfer.agents.thredds_agent.check_thredds_status")
def test_register_server_tool_returns_command_with_state_update(mock_status):
    mock_status.return_value = {"reachable": True, "catalog_name": "IOOS Glider DAC", "error": None}
    runtime = _runtime({})

    result = register_server_tool.func(runtime=runtime, url="https://gliders.ioos.us/thredds/catalog/catalog.html")

    assert isinstance(result, Command)
    assert result.update is not None
    url = "https://gliders.ioos.us/thredds/catalog/catalog.xml"
    assert result.update["active_server_url"] == url
    assert result.update["active_catalog_url"] == url
    assert url in result.update["servers"]


@patch("surfer.agents.thredds_agent.check_thredds_status")
def test_register_server_tool_reports_unreachable_in_message(mock_status):
    mock_status.return_value = {"reachable": False, "catalog_name": None, "error": "timed out"}
    runtime = _runtime({})

    result = register_server_tool.func(runtime=runtime, url="https://gliders.ioos.us/thredds/catalog/catalog.xml")

    assert isinstance(result, Command)
    assert result.update is not None
    message_content = result.update["messages"][0].content
    assert "Reachable: False" in message_content
    assert "timed out" in message_content


def test_set_active_server_tool_switches_active_server_and_resets_catalog():
    url_a = "https://gliders.ioos.us/thredds/catalog/catalog.xml"
    url_b = "https://tds.marine.rutgers.edu/thredds/catalog/catalog.xml"
    runtime = _runtime({
        "active_server_url": url_a,
        "active_catalog_url": "https://gliders.ioos.us/thredds/catalog/deployments/catalog.xml",
        "servers": {url_a: {"url": url_a}, url_b: {"url": url_b}},
    })

    result = set_active_server_tool.func(runtime=runtime, url=url_b)

    assert isinstance(result, Command)
    assert result.update is not None
    assert result.update["active_server_url"] == url_b
    assert result.update["active_catalog_url"] == url_b


def test_set_active_server_tool_raises_for_unregistered_server():
    runtime = _runtime({"active_server_url": None, "servers": {}})
    with pytest.raises(ValueError, match="not registered"):
        set_active_server_tool.func(runtime=runtime, url="https://gliders.ioos.us/thredds/catalog/catalog.xml")


@patch("surfer.agents.thredds_agent.TDSCatalog")
def test_set_active_catalog_tool_updates_position(mock_cls):
    mock_cls.return_value = Mock()
    url = "https://gliders.ioos.us/thredds/catalog/catalog.xml"
    runtime = _runtime({"active_server_url": url, "servers": {url: {"url": url}}})

    sub_catalog = "https://gliders.ioos.us/thredds/catalog/deployments/catalog.xml"
    result = set_active_catalog_tool.func(runtime=runtime, url=sub_catalog)

    assert isinstance(result, Command)
    assert result.update is not None
    assert result.update["active_catalog_url"] == sub_catalog


def test_set_active_catalog_tool_raises_when_no_server_registered():
    runtime = _runtime({"active_server_url": None, "servers": {}})
    with pytest.raises(ValueError, match="No active server set"):
        set_active_catalog_tool.func(runtime=runtime, url="https://gliders.ioos.us/thredds/catalog/deployments/catalog.xml")


@patch("surfer.agents.thredds_agent.TDSCatalog", side_effect=Exception("404 Client Error"))
def test_set_active_catalog_tool_returns_clean_error_on_unfetchable_url(mock_cls):
    """Direct regression test: this tool previously stored any URL string into
    active_catalog_url with zero validation, so a hand-built/stale URL would silently
    poison state and only fail later on an unrelated browse_catalog_tool/get_catalog_summary_tool
    call -- now the fetch is validated here, at the point the bad URL enters state."""
    url = "https://gliders.ioos.us/thredds/catalog/catalog.xml"
    runtime = _runtime({"active_server_url": url, "servers": {url: {"url": url}}})

    result = set_active_catalog_tool.func(
        runtime=runtime, url="https://tds.marine.rutgers.edu/thredds/catalog/cool/glider/allGliders/catalog.xml"
    )

    assert isinstance(result, str)
    assert "Could not fetch catalog" in result

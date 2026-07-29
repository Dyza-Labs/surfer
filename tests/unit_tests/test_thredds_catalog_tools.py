from typing import Any, Callable
from unittest.mock import Mock, patch

from langchain.tools import ToolRuntime
from langgraph.types import Command

from surfer.agents.thredds_agent import THREDDSServer
from surfer.agents.tools.thredds_catalog_tools import (
    CatalogIndexEntry,
    browse_catalog,
    browse_catalog_tool,
    build_catalog_index,
    find_datasets_tool,
    get_catalog_summary_tool,
    search_catalog_index,
)


def _func(tool: Any) -> Callable[..., str]:
    return tool.func


def _server_runtime(server: THREDDSServer, active_catalog_url: str) -> ToolRuntime:
    return ToolRuntime(
        state={
            "active_server_url": server.url,
            "servers": {server.url: server.model_dump()},
            "active_catalog_url": active_catalog_url,
        },
        context=None, config={}, stream_writer=None, tool_call_id="fake", store=None,
    )


def _fake_catalog(catalog_url: str, catalog_name: str, refs: dict[str, Any], datasets: list[str]):
    return Mock(catalog_url=catalog_url, catalog_name=catalog_name, catalog_refs=refs, datasets={d: Mock() for d in datasets})


def _fake_ref(child_catalog):
    ref = Mock()
    ref.follow.return_value = child_catalog
    return ref


# browse_catalog -------------------------------------------------------------------

@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog")
def test_browse_catalog_returns_top_level_listing(mock_cls):
    mock_cls.return_value = _fake_catalog(
        "https://gliders.ioos.us/thredds/catalog/catalog.xml", "IOOS Glider DAC",
        {"deployments": _fake_ref(None)}, [],
    )

    result = browse_catalog("https://gliders.ioos.us/thredds/catalog/catalog.xml")

    assert result["catalog_name"] == "IOOS Glider DAC"
    assert result["sub_catalogs"] == ["deployments"]
    assert result["datasets"] == []


@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog")
def test_browse_catalog_descends_into_matching_sub_catalog(mock_cls):
    child = _fake_catalog(
        "https://gliders.ioos.us/thredds/catalog/deployments/catalog.xml", "deployments",
        {}, ["glider1.nc"],
    )
    mock_cls.return_value = _fake_catalog(
        "https://gliders.ioos.us/thredds/catalog/catalog.xml", "root",
        {"deployments": _fake_ref(child)}, [],
    )

    result = browse_catalog("https://gliders.ioos.us/thredds/catalog/catalog.xml", descend_into="deploy")

    assert result["catalog_url"] == "https://gliders.ioos.us/thredds/catalog/deployments/catalog.xml"
    assert result["datasets"] == ["glider1.nc"]


@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog")
def test_browse_catalog_no_match_returns_error_string(mock_cls):
    mock_cls.return_value = _fake_catalog(
        "https://gliders.ioos.us/thredds/catalog/catalog.xml", "root",
        {"deployments": _fake_ref(None)}, [],
    )

    result = browse_catalog("https://gliders.ioos.us/thredds/catalog/catalog.xml", descend_into="bogus")

    assert isinstance(result, str)
    assert "No sub-catalog matching" in result


@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog", side_effect=Exception("404 Client Error"))
def test_browse_catalog_returns_clean_error_on_unfetchable_url(mock_cls):
    """Direct regression test: a live 404 was seen when set_active_catalog_tool stored a
    hand-built/stale catalog_url and a later browse_catalog_tool call tried to fetch it --
    TDSCatalog() raised an uncaught requests.HTTPError that killed the whole tool node."""
    result = browse_catalog("https://tds.marine.rutgers.edu/thredds/catalog/cool/glider/allGliders/catalog.xml")

    assert isinstance(result, str)
    assert "Could not fetch catalog" in result


@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog")
def test_browse_catalog_returns_clean_error_when_follow_fails(mock_cls):
    bad_ref = Mock()
    bad_ref.follow.side_effect = Exception("404 Client Error")
    mock_cls.return_value = _fake_catalog(
        "https://x/catalog.xml", "root", {"deployments": bad_ref}, [],
    )

    result = browse_catalog("https://x/catalog.xml", descend_into="deployments")

    assert isinstance(result, str)
    assert "Could not follow sub-catalog" in result


@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog")
def test_browse_catalog_ambiguous_match_returns_did_you_mean(mock_cls):
    mock_cls.return_value = _fake_catalog(
        "https://gliders.ioos.us/thredds/catalog/catalog.xml", "root",
        {"OOI-CE": _fake_ref(None), "OOI-CGSN": _fake_ref(None)}, [],
    )

    result = browse_catalog("https://gliders.ioos.us/thredds/catalog/catalog.xml", descend_into="OOI")

    assert isinstance(result, str)
    assert "ambiguous" in result
    assert "OOI-CE" in result and "OOI-CGSN" in result


# browse_catalog_tool ---------------------------------------------------------------

@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog")
def test_browse_catalog_tool_plain_listing_returns_string_not_command(mock_cls):
    mock_cls.return_value = _fake_catalog(
        "https://gliders.ioos.us/thredds/catalog/catalog.xml", "root",
        {"deployments": _fake_ref(None)}, [],
    )
    server = THREDDSServer(url="https://gliders.ioos.us/thredds/catalog/catalog.xml")
    runtime = _server_runtime(server, server.url)

    result = _func(browse_catalog_tool)(runtime=runtime)

    assert isinstance(result, str)
    assert "root" in result


@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog")
def test_browse_catalog_tool_descent_returns_command_updating_active_catalog(mock_cls):
    child = _fake_catalog(
        "https://gliders.ioos.us/thredds/catalog/deployments/catalog.xml", "deployments",
        {}, ["glider1.nc"],
    )
    mock_cls.return_value = _fake_catalog(
        "https://gliders.ioos.us/thredds/catalog/catalog.xml", "root",
        {"deployments": _fake_ref(child)}, [],
    )
    server = THREDDSServer(url="https://gliders.ioos.us/thredds/catalog/catalog.xml")
    runtime = _server_runtime(server, server.url)

    result = _func(browse_catalog_tool)(runtime=runtime, descend_into="deployments")

    assert isinstance(result, Command)
    assert result.update["active_catalog_url"] == "https://gliders.ioos.us/thredds/catalog/deployments/catalog.xml"


@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog")
def test_browse_catalog_tool_no_match_returns_plain_string(mock_cls):
    mock_cls.return_value = _fake_catalog(
        "https://gliders.ioos.us/thredds/catalog/catalog.xml", "root",
        {"deployments": _fake_ref(None)}, [],
    )
    server = THREDDSServer(url="https://gliders.ioos.us/thredds/catalog/catalog.xml")
    runtime = _server_runtime(server, server.url)

    result = _func(browse_catalog_tool)(runtime=runtime, descend_into="bogus")

    assert isinstance(result, str)
    assert "No sub-catalog matching" in result


# build_catalog_index / search_catalog_index ----------------------------------------

@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog")
def test_build_catalog_index_walks_tree_and_collects_leaf_datasets(mock_cls):
    import surfer.agents.tools.thredds_catalog_tools as catalog_tools
    catalog_tools._catalog_index_cache.clear()

    leaf_a = _fake_catalog("https://x/a/catalog.xml", "a", {}, ["a1.nc", "a2.nc"])
    leaf_b = _fake_catalog("https://x/b/catalog.xml", "b", {}, ["b1.nc"])
    root = _fake_catalog("https://x/catalog.xml", "root", {"a": _fake_ref(leaf_a), "b": _fake_ref(leaf_b)}, [])
    mock_cls.return_value = root

    entries = build_catalog_index("https://x/catalog.xml")

    names = {e.name for e in entries}
    assert names == {"a1.nc", "a2.nc", "b1.nc"}
    a1 = next(e for e in entries if e.name == "a1.nc")
    assert a1.parent_path == ["a"]


@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog")
def test_build_catalog_index_does_not_open_any_dataset(mock_cls):
    """Names/paths only -- Dataset objects in the fake tree are bare Mocks with no
    real open() capability, so a passing test here confirms the walk never calls
    anything beyond .datasets/.catalog_refs/.follow()."""
    import surfer.agents.tools.thredds_catalog_tools as catalog_tools
    catalog_tools._catalog_index_cache.clear()

    leaf = _fake_catalog("https://x/a/catalog.xml", "a", {}, ["a1.nc"])
    root = _fake_catalog("https://x/catalog.xml", "root", {"a": _fake_ref(leaf)}, [])
    mock_cls.return_value = root

    build_catalog_index("https://x/catalog.xml")  # would raise if it tried .open() on the Mock dataset


@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog")
def test_build_catalog_index_skips_broken_catalog_ref(mock_cls):
    """Direct regression test for a real 404 hit live on tds.marine.rutgers.edu: a single
    stale/broken catalogRef anywhere in the tree must not sink the whole walk."""
    import surfer.agents.tools.thredds_catalog_tools as catalog_tools
    catalog_tools._catalog_index_cache.clear()

    good_leaf = _fake_catalog("https://x/good/catalog.xml", "good", {}, ["good.nc"])
    broken_ref = Mock()
    broken_ref.follow.side_effect = Exception("404 Client Error")
    root = _fake_catalog(
        "https://x/catalog.xml", "root",
        {"broken": broken_ref, "good": _fake_ref(good_leaf)}, [],
    )
    mock_cls.return_value = root

    entries = build_catalog_index("https://x/catalog.xml")

    assert {e.name for e in entries} == {"good.nc"}


@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog")
def test_build_catalog_index_caches_repeat_calls(mock_cls):
    import surfer.agents.tools.thredds_catalog_tools as catalog_tools
    catalog_tools._catalog_index_cache.clear()

    root = _fake_catalog("https://x/catalog.xml", "root", {}, ["only.nc"])
    mock_cls.return_value = root

    first = build_catalog_index("https://x/catalog.xml")
    second = build_catalog_index("https://x/catalog.xml")

    assert first is second
    mock_cls.assert_called_once()


def test_search_catalog_index_case_insensitive_substring_match():
    entries = [
        CatalogIndexEntry(name="ru28-20221129T1452.nc", catalog_url="https://x", parent_path=[]),
        CatalogIndexEntry(name="whoi_406.nc", catalog_url="https://y", parent_path=[]),
    ]

    result = search_catalog_index(entries, "RU28")

    assert len(result) == 1
    assert result[0].name == "ru28-20221129T1452.nc"


def test_search_catalog_index_no_match_returns_empty():
    entries = [CatalogIndexEntry(name="whoi_406.nc", catalog_url="https://y", parent_path=[])]
    assert search_catalog_index(entries, "zzz_nonexistent") == []


# find_datasets_tool -----------------------------------------------------------------

@patch("surfer.agents.tools.thredds_catalog_tools.build_catalog_index")
def test_find_datasets_tool_reports_matches(mock_build_index):
    mock_build_index.return_value = [
        CatalogIndexEntry(name="ru28-20221129T1452.nc", catalog_url="https://x/ru28/catalog.xml", parent_path=["a"]),
    ]
    server = THREDDSServer(url="https://gliders.ioos.us/thredds/catalog/catalog.xml")
    runtime = _server_runtime(server, server.url)

    result = _func(find_datasets_tool)(runtime=runtime, query="ru28")

    assert "ru28-20221129T1452.nc" in result
    assert "https://x/ru28/catalog.xml" in result


@patch("surfer.agents.tools.thredds_catalog_tools.build_catalog_index")
def test_find_datasets_tool_reports_no_match(mock_build_index):
    mock_build_index.return_value = []
    server = THREDDSServer(url="https://gliders.ioos.us/thredds/catalog/catalog.xml")
    runtime = _server_runtime(server, server.url)

    result = _func(find_datasets_tool)(runtime=runtime, query="zzz_nonexistent")

    assert "No datasets matching" in result


# get_catalog_summary_tool ------------------------------------------------------------

@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog")
def test_get_catalog_summary_tool_reports_current_level_only(mock_cls):
    mock_cls.return_value = Mock(
        catalog_url="https://gliders.ioos.us/thredds/catalog/deployments/catalog.xml",
        catalog_name="deployments",
        catalog_refs={"a": Mock(), "b": Mock()},
        datasets={"d1.nc": Mock()},
        services=[Mock(name="all")],
    )
    server = THREDDSServer(url="https://gliders.ioos.us/thredds/catalog/catalog.xml")
    runtime = _server_runtime(server, "https://gliders.ioos.us/thredds/catalog/deployments/catalog.xml")

    result = _func(get_catalog_summary_tool)(runtime=runtime)

    assert "deployments" in result
    assert "Sub-catalogs: 2" in result
    assert "Datasets at this level: 1" in result


@patch("surfer.agents.tools.thredds_catalog_tools.TDSCatalog", side_effect=Exception("404 Client Error"))
def test_get_catalog_summary_tool_returns_clean_error_on_unfetchable_url(mock_cls):
    server = THREDDSServer(url="https://gliders.ioos.us/thredds/catalog/catalog.xml")
    runtime = _server_runtime(server, "https://tds.marine.rutgers.edu/thredds/catalog/cool/glider/allGliders/catalog.xml")

    result = _func(get_catalog_summary_tool)(runtime=runtime)

    assert "Could not fetch catalog" in result

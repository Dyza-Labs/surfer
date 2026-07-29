"""
Real-network integration tests against the two THREDDS servers used during development.
Like test_multi_server.py, these do NOT invoke an LLM -- they call the tool-layer plain
functions directly against real servers, so they need no OPENROUTER_API_KEY.

Each server is skipped individually (not failed) if unreachable, since live third-party
infrastructure can flake independently of anything in this codebase.
"""
import pytest

from surfer.agents.thredds_agent import THREDDSServer, check_thredds_status
from surfer.agents.tools.thredds_catalog_tools import browse_catalog, build_catalog_index, search_catalog_index
from surfer.agents.tools.thredds_dataset_tools import describe_thredds_dataset, open_leaf_dataset, resolve_leaf_dataset

SERVERS = [
    pytest.param("https://tds.marine.rutgers.edu/thredds/catalog/catalog.html", id="rutgers-tds"),
    pytest.param("https://gliders.ioos.us/thredds/catalog/catalog.html", id="ioos-gliders"),
]

# Sub-catalog root to index/search from, and a name substring expected to match within it.
# Both servers' full top-level trees are impractically slow to index in a test (Rutgers
# hosts far more than glider data -- ROMS model runs, meteorology, CODAR, ...; even
# gliders.ioos.us has 50+ sub-catalogs under 'deployments', each a live HTTP fetch). This
# scopes the index build to a small, known sub-catalog on each server, the same way a real
# agent would first browse_catalog_tool down before calling find_datasets_tool.
INDEX_ROOTS = [
    pytest.param("https://tds.marine.rutgers.edu/thredds/catalog/cool/glider/catalog.xml", "CE05MOAS", id="rutgers-tds"),
    pytest.param("https://gliders.ioos.us/thredds/catalog/deployments/OOI-CE/catalog.xml", "ce_1012", id="ioos-gliders"),
]


def _sanitized(url: str) -> str:
    return THREDDSServer.sanitize_url(url)


def _skip_if_unreachable(url: str) -> None:
    status = check_thredds_status(url)
    if not status["reachable"]:
        pytest.skip(f"{url} is currently unreachable: {status['error']}")


@pytest.mark.parametrize("url", SERVERS)
def test_server_is_reachable(url):
    sanitized = _sanitized(url)
    status = check_thredds_status(sanitized)
    if not status["reachable"]:
        pytest.skip(f"{sanitized} is currently unreachable: {status['error']}")
    assert status["catalog_name"]


@pytest.mark.parametrize("url", SERVERS)
def test_browse_catalog_top_level_has_sub_catalogs(url):
    sanitized = _sanitized(url)
    _skip_if_unreachable(sanitized)

    result = browse_catalog(sanitized)

    assert not isinstance(result, str), f"browse_catalog failed: {result}"
    assert result["sub_catalogs"], f"Expected at least one sub-catalog at {sanitized}"


@pytest.mark.parametrize("index_root, name_query", INDEX_ROOTS)
def test_find_datasets_tool_locates_a_real_deployment(index_root, name_query):
    """Exercises the real build_catalog_index tree walk against a live server -- the one
    test that should actually pay that cost, per the plan's verification section."""
    _skip_if_unreachable(index_root)

    index = build_catalog_index(index_root)
    assert index, f"No datasets found anywhere under {index_root}"

    matches = search_catalog_index(index, name_query)
    assert matches, f"No dataset matching '{name_query}' found under {index_root}"


@pytest.mark.parametrize("index_root, name_query", INDEX_ROOTS)
def test_resolve_and_open_real_trajectory_dataset(index_root, name_query):
    _skip_if_unreachable(index_root)

    index = build_catalog_index(index_root)
    matches = search_catalog_index(index, name_query)
    if not matches:
        pytest.skip(f"No dataset matching '{name_query}' found under {index_root} to resolve")

    entry = matches[0]
    dataset = resolve_leaf_dataset(entry.catalog_url, entry.name)
    assert not isinstance(dataset, str), f"resolve_leaf_dataset failed: {dataset}"

    ds = open_leaf_dataset(dataset)
    try:
        assert ds.attrs.get("cdm_data_type") == "TrajectoryProfile"
        assert "trajectory" in ds.sizes
        assert "profile" in ds.sizes
        assert "obs" in ds.sizes
    finally:
        ds.close()


@pytest.mark.parametrize("index_root, name_query", INDEX_ROOTS)
def test_describe_thredds_dataset_returns_expected_shape(index_root, name_query):
    _skip_if_unreachable(index_root)

    index = build_catalog_index(index_root)
    matches = search_catalog_index(index, name_query)
    if not matches:
        pytest.skip(f"No dataset matching '{name_query}' found under {index_root} to describe")

    entry = matches[0]
    result = describe_thredds_dataset(entry.catalog_url, entry.name)

    assert not isinstance(result, str), f"describe_thredds_dataset failed for {entry.name}: {result}"
    assert "bounds" in result
    assert "variables" in result
    assert len(result["variables"]) > 0

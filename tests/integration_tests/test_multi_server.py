"""
Real-network integration tests across a range of ERDDAP server sizes (29 to ~8,900
datasets). Unlike test_graph.py / test_erddap_agent_mocked.py, these do NOT invoke an
LLM (no graph.ainvoke()) -- they call the tool-layer plain functions directly against
real servers, so they need no OPENROUTER_API_KEY. A single mid-size fixture server (as
used everywhere else in the suite) can't surface scale-dependent issues like pagination
limits or slow/huge categorize responses -- this file exists to catch those specifically.

Each server is skipped individually (not failed) if unreachable, since live third-party
infrastructure can flake independently of anything in this codebase.
"""
import pytest

from surfer.agents.erddap_agent import ERDDAPServer, check_server_status
from surfer.agents.tools.erddap_dataset_tools import describe_dataset
from surfer.agents.tools.erddap_server_tools import get_categorize_values, get_server_summary, search_server

SERVERS = [
    pytest.param("https://hfr.marine.rutgers.edu/erddap/index.html", 29, id="rutgers-hfr-29"),
    pytest.param("https://hfradar.ioos.us/erddap/info/index.html", 200, id="ioos-hfradar-200"),
    pytest.param("https://gliders.ioos.us/erddap/index.html", 2459, id="ioos-gliders-2459"),
    pytest.param("https://slocum-data.marine.rutgers.edu/erddap/info/index.html", 944, id="rutgers-slocum-944"),
    pytest.param("https://upwell.pfeg.noaa.gov/erddap/index.html", 8893, id="noaa-upwell-8893"),
]


def _sanitized(url: str) -> str:
    return ERDDAPServer.sanitize_url(url)


def _skip_if_unreachable(url: str) -> None:
    status = check_server_status(url)
    if not status["reachable"]:
        pytest.skip(f"{url} is currently unreachable: {status['error']}")


@pytest.mark.parametrize("url, expected_order_of_magnitude", SERVERS)
def test_server_is_reachable(url, expected_order_of_magnitude):
    sanitized = _sanitized(url)
    status = check_server_status(sanitized)
    if not status["reachable"]:
        pytest.skip(f"{sanitized} is currently unreachable: {status['error']}")
    assert status["version"]


@pytest.mark.parametrize("url, expected_order_of_magnitude", SERVERS)
def test_get_server_summary_returns_plausible_dataset_count(url, expected_order_of_magnitude):
    sanitized = _sanitized(url)
    _skip_if_unreachable(sanitized)

    result = get_server_summary(sanitized)

    assert result["dataset_count"] > 0
    # Real counts drift over time as datasets are added/retired -- check the right order
    # of magnitude rather than an exact match.
    assert result["dataset_count"] >= expected_order_of_magnitude * 0.5
    assert isinstance(result["protocols"], list) and result["protocols"]
    assert isinstance(result["institutions"], list)
    assert isinstance(result["keywords"], list)


@pytest.mark.parametrize("url, expected_order_of_magnitude", SERVERS)
def test_search_server_returns_results_with_dataset_id_column(url, expected_order_of_magnitude):
    sanitized = _sanitized(url)
    _skip_if_unreachable(sanitized)

    result = search_server(sanitized, "temperature")

    assert not isinstance(result, str), f"search_server failed: {result}"
    assert not result.empty
    assert "Dataset ID" in result.columns


@pytest.mark.parametrize("url, expected_order_of_magnitude", SERVERS)
def test_resolve_standard_name_categorize_values_succeeds(url, expected_order_of_magnitude):
    """The categorize/standard_name endpoint is the one most likely to expose a
    slow/huge-response issue on a large server -- confirm it still returns promptly
    and without error even at the ~8,900-dataset scale."""
    sanitized = _sanitized(url)
    _skip_if_unreachable(sanitized)

    result = get_categorize_values(sanitized, "standard_name")

    assert not isinstance(result, str), f"get_categorize_values failed: {result}"
    assert len(result) > 0


@pytest.mark.parametrize("url, expected_order_of_magnitude", SERVERS)
def test_describe_dataset_succeeds_on_a_real_search_result(url, expected_order_of_magnitude):
    """Pulls a real dataset ID from that server's own search results instead of
    hardcoding one, so this doesn't rot when a specific dataset gets retired."""
    sanitized = _sanitized(url)
    _skip_if_unreachable(sanitized)

    search_result = search_server(sanitized, "temperature")
    if isinstance(search_result, str) or search_result.empty or "Dataset ID" not in search_result.columns:
        pytest.skip(f"No searchable dataset found on {sanitized} to describe")

    dataset_id = search_result["Dataset ID"].iloc[0]
    result = describe_dataset(sanitized, dataset_id)

    assert not isinstance(result, str), f"describe_dataset failed for {dataset_id}: {result}"
    assert "bounds" in result
    assert "variables" in result

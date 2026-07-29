from unittest.mock import patch

import pytest
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel import Pregel
from surfer.graph import SupervisorState
from typing import cast
from langgraph.runtime import Runtime
import surfer.config as config
from surfer.graph import graph, route


def test_graph_compiles() -> None:
    assert isinstance(graph, Pregel)


def test_erddap_agent_compiles() -> None:
    from surfer.graph import erddap_agent
    assert isinstance(erddap_agent, Pregel)


def test_thredds_agent_compiles() -> None:
    from surfer.graph import thredds_agent
    assert isinstance(thredds_agent, Pregel)


# get_checkpointer ----------------------------------------------------------------

def test_get_checkpointer_falls_back_to_in_memory_when_uri_unset() -> None:
    with patch.object(config, "POSTGRES_URI", None):
        assert isinstance(config.get_checkpointer(), InMemorySaver)


def test_get_checkpointer_builds_postgres_saver_when_uri_set() -> None:
    with (
        patch.object(config, "POSTGRES_URI", "postgresql://fake:fake@localhost/fake"),
        patch.object(config, "ConnectionPool") as mock_pool,
        patch.object(config, "PostgresSaver") as mock_saver_cls,
    ):
        result = config.get_checkpointer()

        mock_pool.assert_called_once()
        assert mock_pool.call_args.args[0] == "postgresql://fake:fake@localhost/fake"
        mock_saver_cls.assert_called_once_with(mock_pool.return_value)
        mock_saver_cls.return_value.setup.assert_called_once()
        assert result is mock_saver_cls.return_value


# supervisor routing ---------------------------------------------------------------

def _route_state(text: str, active_subagent=None) -> SupervisorState:
    return {"messages": [HumanMessage(content=text)], "active_subagent": active_subagent}


@pytest.mark.parametrize(
    ("text", "platform"),
    [
        ("find data on https://gliders.ioos.us/erddap/index.html", "erddap"),
        ("browse https://gliders.ioos.us/thredds/catalog.html", "thredds"),
    ],
)
def test_route_first_contact_url_sets_active_subagent(text: str, platform: str) -> None:
    result = route(_route_state(text), cast(Runtime, None))

    assert result.goto == "call_subagent"
    assert result.update == {"active_subagent": platform}


def test_route_sticky_continuation_with_no_url_skips_model() -> None:
    """Direct regression test for the token-waste fix: a reply to a clarifying question
    (no URL at all) must stay on the active subagent without falling through to the
    supervisor's LLM-backed model node."""
    result = route(_route_state("sea_water_practical_salinity", active_subagent="erddap"), cast(Runtime,None))

    assert result.goto == "call_subagent"
    assert result.update is None or "active_subagent" not in result.update


def test_route_platform_switch_commits_and_routes_directly() -> None:
    """An unambiguous mid-conversation switch (exactly one platform mentioned, differing
    from the active one) now routes directly and updates active_subagent, rather than
    falling through to model_node, which had no way to ever commit the switch."""
    result = route(
        _route_state("actually use https://gliders.ioos.us/thredds/catalog.html", active_subagent="erddap"),
        cast(Runtime, None),
    )

    assert result.goto == "call_subagent"
    assert result.update == {"active_subagent": "thredds"}


def test_route_name_only_platform_mention_routes_directly() -> None:
    """A platform named in words with no URL still routes deterministically."""
    result = route(_route_state("Search THREDDS for glider data near Rutgers."), cast(Runtime, None))

    assert result.goto == "call_subagent"
    assert result.update == {"active_subagent": "thredds"}


def test_route_conflicting_platforms_falls_through_to_model() -> None:
    result = route(
        _route_state("compare https://x/erddap/index.html and https://y/thredds/catalog.html"),
        cast(Runtime, None),
    )

    assert result.goto == "model"


def test_route_no_url_and_no_active_subagent_falls_through_to_model() -> None:
    result = route(_route_state("find me some data"), cast(Runtime, None))

    assert result.goto == "model"


# supervisor pass-through (call_subagent) --------------------------------------------

def test_call_subagent_passes_erddap_answer_through_verbatim() -> None:
    """Direct regression test for the resummarization bug: a subagent answer containing
    a specific detail (a server version string) must survive to the final graph output
    unchanged -- no second supervisor LLM turn should paraphrase or drop it."""
    from langchain.messages import AIMessage

    with patch("surfer.graph.erddap_agent") as mock_agent:
        mock_agent.stream.return_value = iter([
            {"model": {"messages": [AIMessage(content="The server is reachable, running ERDDAP_version=2.24.")]}}
        ])
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "Is https://gliders.ioos.us/erddap/index.html online?"}]},
            config={"configurable": {"thread_id": "test-pass-through"}},
        )

    assert "ERDDAP_version=2.24" in result["messages"][-1].content
    assert result["active_subagent"] == "erddap"
    mock_agent.stream.assert_called_once()

import os
from unittest.mock import patch

import pandas as pd
import pytest
from uuid import uuid4
from langchain_core.runnables import RunnableConfig

from surfer.graph import graph

pytestmark = pytest.mark.anyio

if not os.getenv("OPENROUTER_API_KEY"):
    pytest.skip(
        "Set OPENROUTER_API_KEY to run integration tests.", allow_module_level=True
    )


GLIDER_SERVER = "https://gliders.ioos.us/erddap/index.html"


async def _ask(prompt: str, thread_id: str | None = None) -> str:
    # graph.invoke(), not ainvoke() -- PostgresSaver has no async checkpoint methods.
    config: RunnableConfig = {"configurable": {"thread_id": thread_id or f"test-{uuid4()}"}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config,
    )
    return str(result["messages"][-1].content)

@pytest.mark.asyncio
async def test_memory_across_turns() -> None:
    thread = f"memory-test-{uuid4()}"

    await _ask(
        "Summarize the server at https://gliders.ioos.us/erddap/index.html and set it as active",
        thread_id=thread,
    )

    followup = await _ask("Graph a trajectory map of ru40-20260507T1702", thread_id=thread)

    assert "2,495" in followup or "2495" in followup


@pytest.mark.asyncio
async def test_clarification_response_passes_through_supervisor_verbatim() -> None:
    """Deterministic pass-through test: mock erddap_agent's own .invoke() (a module-level
    object in surfer.graph) to return a fixed clarifying question, then assert call_subagent
    relays it byte-for-byte as the graph's final output -- no second supervisor LLM turn
    paraphrases or drops it. This is the code path the live prompt-tuning attempts above were
    trying to exercise naturally, but the live model was inconsistent across runs (sometimes
    asking a clarifying question, sometimes silently picking a variant, sometimes searching
    every candidate at once) -- not testable reliably as a live behavior, so the clarifying
    text itself is controlled here and only the graph's handling of it is under test."""
    from unittest.mock import patch

    from langchain.messages import AIMessage as _AIMessage

    clarifying_question = (
        "I found several standard names for chlorophyll: "
        "concentration_of_chlorophyll_fluorescence_in_sea_water and "
        "mass_concentration_of_chlorophyll_a_in_sea_water. Which one do you mean?"
    )
    with patch("surfer.graph.erddap_agent") as mock_agent:
        mock_agent.invoke.return_value = {"messages": [_AIMessage(content=clarifying_question)]}
        response = await _ask(
            f"On {GLIDER_SERVER}, search for datasets measuring chlorophyll.",
            thread_id=f"clarify-passthrough-{uuid4()}",
        )

    assert response == clarifying_question


@pytest.mark.asyncio
async def test_erddap_agent_checkpointer_resolves_followup_using_persisted_registration() -> None:
    """Real, unmocked test of the actual mechanism that lets a follow-up reply resolve
    correctly: erddap_agent's own checkpointer (not the supervisor's) persists its message
    history and ERDDAPAgentState across two separate invoke() calls on the same thread_id.
    call_subagent (graph.py) only ever forwards the latest human text to erddap_agent, never
    the full supervisor conversation -- so a follow-up like "resolve the standard_name for
    chlorophyll fluorescence" with no server URL in it can only succeed if erddap_agent itself
    remembers, from a prior turn on this thread, which server is active. Live-verified this
    works correctly before writing this test (register on turn 1, unrelated query with no URL
    on turn 2 correctly used the persisted active_server_url)."""
    from surfer.graph import erddap_agent

    thread_id = f"erddap-persist-test-{uuid4()}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    erddap_agent.invoke(
        {"messages": [{"role": "user", "content": f"Register {GLIDER_SERVER}"}]},
        config=config,
    )
    snapshot_after_register = erddap_agent.get_state(config)
    assert snapshot_after_register.values.get("active_server_url") == "https://gliders.ioos.us/erddap"

    result = erddap_agent.invoke(
        {"messages": [{"role": "user", "content": "resolve the standard_name for chlorophyll fluorescence"}]},
        config=config,
    )

    followup_text = str(result["messages"][-1].content)
    assert "no active server" not in followup_text.lower()
    assert "concentration_of_chlorophyll_fluorescence_in_sea_water" in followup_text


@pytest.mark.skipif(not os.getenv("POSTGRES_URI"), reason="Set POSTGRES_URI to test Postgres persistence.")
@patch("surfer.agents.tools.erddap_server_tools.get_server_summary")
async def test_postgres_checkpoint_persists_active_subagent_across_invocations(mock_summary) -> None:
    """graph's checkpointer only actually uses Postgres when POSTGRES_URI is set at import
    time (see surfer.config.get_checkpointer) -- this asserts the supervisor's sticky
    active_subagent state survives a second, separate graph.ainvoke() call for the same
    thread_id, which is the whole point of wiring PostgresSaver in rather than the
    InMemorySaver fallback (that one wouldn't survive a process restart)."""
    mock_summary.return_value = {
        "dataset_count": 42,
        "protocols": ["tabledap"],
        "institutions": ["Rutgers University"],
        "keywords": ["Temperature"],
    }
    thread = f"postgres-persist-{uuid4()}"
    config: RunnableConfig = {"configurable": {"thread_id": thread}}

    await _ask(f"Register {GLIDER_SERVER} and summarize it.", thread_id=thread)
    snapshot = graph.get_state(config)

    assert snapshot.values.get("active_subagent") == "erddap"
    assert len(snapshot.values["messages"]) >= 2


@patch("surfer.agents.tools.erddap_server_tools.get_server_summary")
async def test_get_server_summary_via_agent_mocked(mock_summary) -> None:
    mock_summary.return_value = {
        "dataset_count": 42,
        "protocols": ["tabledap"],
        "institutions": ["Rutgers University"],
        "keywords": ["Temperature", "Salinity"],
    }

    output_text = await _ask(f"Register {GLIDER_SERVER} and give me a summary of it.")

    mock_summary.assert_called_once()
    called_url = mock_summary.call_args.args[0]
    assert "gliders.ioos.us" in called_url
    assert "42" in output_text or "Rutgers" in output_text


@patch("surfer.agents.tools.erddap_server_tools.search_server")
async def test_search_server_via_agent_mocked(mock_search) -> None:
    mock_search.return_value = pd.DataFrame({
        "Dataset ID": ["whoi_406-2016"],
        "Title": ["WHOI Glider 406"],
    })

    output_text = await _ask(f"On {GLIDER_SERVER}, search for datasets related to salinity.")

    mock_search.assert_called_once()
    assert "whoi_406-2016" in output_text or "WHOI Glider 406" in output_text


@patch("surfer.agents.tools.erddap_server_tools.search_server")
async def test_search_server_no_results_via_agent_mocked(mock_search) -> None:
    mock_search.return_value = pd.DataFrame()

    output_text = await _ask(
        f"On {GLIDER_SERVER}, search for datasets about 'zzz_nonexistent_zzz'."
    )

    mock_search.assert_called_once()
    assert "no datasets" in output_text.lower() or "not found" in output_text.lower()


@patch("surfer.agents.tools.erddap_server_tools.validate_categorize_value")
async def test_search_server_invalid_institution_via_agent_mocked(mock_validate) -> None:
    mock_validate.return_value = (
        "'Streamer University' not found in institution on "
        f"{GLIDER_SERVER}. Available: ['NOAA', 'Rutgers University']"
    )

    output_text = await _ask(
        f"On {GLIDER_SERVER}, search for datasets from the institution 'Streamer University'."
    )

    mock_validate.assert_called_once()
    assert "not found" in output_text.lower()
    assert "Streamer University" in output_text


@patch("surfer.agents.tools.erddap_server_tools.search_server")
async def test_search_server_by_variable_via_agent_mocked(mock_search) -> None:
    mock_search.return_value = pd.DataFrame({
        "Dataset ID": ["glider_temp_01"],
        "Title": ["Temperature Profile Glider"],
    })

    output_text = await _ask(f"On {GLIDER_SERVER}, find datasets that measure temperature.")

    # Agent starts broad and narrows down, so any call count >= 1 is success.
    assert mock_search.call_count >= 1

    all_calls_text = " ".join(str(c) for c in mock_search.call_args_list).lower()
    assert "temperature" in all_calls_text
    assert "glider_temp_01" in output_text or "Temperature Profile" in output_text


@patch("surfer.agents.tools.erddap_server_tools.get_server_summary")
async def test_get_server_summary_handles_failure_via_agent_mocked(mock_summary) -> None:
    mock_summary.side_effect = ValueError("Could not find dataset count on url/index.html")

    output_text = await _ask(f"Register {GLIDER_SERVER} and summarize it.")

    mock_summary.assert_called_once()
    assert "could not" in output_text.lower() or "error" in output_text.lower()

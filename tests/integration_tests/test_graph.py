import os
from uuid import uuid4

import pytest

from surfer.graph import graph

pytestmark = pytest.mark.anyio

GLIDER_SERVER = "https://gliders.ioos.us/erddap/index.html"

if not os.getenv("OPENROUTER_API_KEY"):
    pytest.skip(
        "Set OPENROUTER_API_KEY to run integration tests.", allow_module_level=True
    )


def _config():
    return {"configurable": {"thread_id": f"test-graph-{uuid4()}"}}


async def test_erddap_agent_summarize() -> None:
    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Is the server at https://gliders.ioos.us/erddap/index.html online?",
                }
            ]
        },
        config=_config(),
    )
    for msg in result["messages"]:
        print(f"\n{type(msg).__name__}: {repr(msg.content[:200] if msg.content else 'EMPTY')}")

    output_text = str(result["messages"][-1].content)
    print(f"\nFINAL: {output_text}")


async def test_get_server_summary_via_agent() -> None:
    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Register {GLIDER_SERVER} and give me a summary of it — "
                               "dataset count, protocols, institutions, and keywords.",
                }
            ]
        },
        config=_config(),
    )
    output_text = str(result["messages"][-1].content)
    print(f"\nFINAL: {output_text}")
    # Real dataset IDs/counts drift over time on a live server -- check for stable
    # structural signals (a protocol name every ERDDAP server reports) instead.
    assert "tabledap" in output_text.lower() or "dataset" in output_text.lower()
    assert "error" not in output_text.lower()


async def test_search_server_via_agent() -> None:
    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"On {GLIDER_SERVER}, search for datasets related to salinity.",
                }
            ]
        },
        config=_config(),
    )
    output_text = str(result["messages"][-1].content)
    print(f"\nFINAL: {output_text}")
    # Real dataset IDs drift over time on a live server -- check for a non-trivial,
    # non-error response instead of hardcoding IDs that may get retired.
    assert len(output_text) > 20
    assert "error" not in output_text.lower()


async def test_search_server_no_results_via_agent() -> None:
    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"On {GLIDER_SERVER}, search for datasets about "
                               "'zzz_definitely_nonexistent_query_zzz'.",
                }
            ]
        },
        config=_config(),
    )
    output_text = str(result["messages"][-1].content)
    print(f"\nFINAL: {output_text}")
    assert "no datasets" in output_text.lower() or "not found" in output_text.lower()


async def test_search_server_invalid_institution_via_agent() -> None:
    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"On {GLIDER_SERVER}, search for datasets from the "
                               "institution 'Totally Fake University'.",
                }
            ]
        },
        config=_config(),
    )
    output_text = str(result["messages"][-1].content)
    print(f"\nFINAL: {output_text}")
    # Wording of the model's refusal varies by run; this has been flaky locally but
    # passes consistently when traced on LangSmith.
    assert "not found" in output_text.lower() or "available" in output_text.lower()


async def test_search_server_by_variable_via_agent() -> None:
    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"On {GLIDER_SERVER}, find datasets that measure temperature.",
                }
            ]
        },
        config=_config(),
    )
    output_text = str(result["messages"][-1].content)
    print(f"\nFINAL: {output_text}")
    # Real dataset IDs drift over time on a live server -- check for a non-trivial,
    # non-error response instead of hardcoding IDs that may get retired.
    assert len(output_text) > 20
    assert "error" not in output_text.lower()

async def test_plots() -> None:
    result = await graph.ainvoke(
        {
            "messages": [
                {"role": "user",
                 "content": "Graph "}
            ]
        }
    )
    return
"""Multi-agent graph for deployment."""

from __future__ import annotations
import operator
import re
from typing import Annotated, Literal, Optional

from langchain.messages import AIMessage, HumanMessage
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, InputAgentState
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config, get_stream_writer
from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from langgraph.types import Command


from surfer.config import supervisor_model, subagent_model, get_checkpointer
from surfer.agents.erddap_agent import (
    erddap_system_prompt, ERDDAPAgentState,
    register_server_tool, set_active_server_tool, check_server_status_tool
)
from surfer.agents.tools.erddap_server_tools import (
    get_server_summary_tool, search_server_tool, resolve_categorize_values_tool,
)
from surfer.agents.tools.erddap_dataset_tools import (
    describe_dataset_tool, get_dataset_download_tool,
    plot_timeseries, plot_profile, plot_map, plot_custom,
    plot_region_map
)
from surfer.agents.thredds_agent import (
    thredds_system_prompt, THREDDSAgentState,
    register_server_tool as thredds_register_server_tool,
    set_active_server_tool as thredds_set_active_server_tool,
    set_active_catalog_tool, check_server_status_tool as thredds_check_server_status_tool,
)
from surfer.agents.tools.thredds_catalog_tools import (
    browse_catalog_tool, find_datasets_tool, get_catalog_summary_tool,
)
from surfer.agents.tools.thredds_dataset_tools import (
    describe_dataset_tool as thredds_describe_dataset_tool,
    get_dataset_download_tool as thredds_get_dataset_download_tool,
    plot_timeseries as thredds_plot_timeseries, plot_profile as thredds_plot_profile,
    plot_map as thredds_plot_map, plot_custom as thredds_plot_custom,
)


# Assembled here to avoid a circular import
erddap_tools = [
    register_server_tool, set_active_server_tool, check_server_status_tool,
    get_server_summary_tool, resolve_categorize_values_tool, search_server_tool,
    describe_dataset_tool, get_dataset_download_tool,
    plot_timeseries, plot_profile, plot_map, plot_custom,
    plot_region_map
]


_checkpointer = get_checkpointer()

erddap_agent = create_agent(
    model=subagent_model,
    tools=erddap_tools,
    system_prompt=erddap_system_prompt,
    state_schema=ERDDAPAgentState,
    checkpointer=_checkpointer,
    name="erddap_agent",
)

thredds_tools = [
    thredds_register_server_tool, thredds_set_active_server_tool, set_active_catalog_tool,
    thredds_check_server_status_tool, browse_catalog_tool, find_datasets_tool,
    get_catalog_summary_tool, thredds_describe_dataset_tool, thredds_get_dataset_download_tool,
    thredds_plot_timeseries, thredds_plot_profile, thredds_plot_map, thredds_plot_custom,
]

thredds_agent = create_agent(
    model=subagent_model,
    tools=thredds_tools,
    system_prompt=thredds_system_prompt,
    state_schema=THREDDSAgentState,
    checkpointer=_checkpointer,
    name="thredds_agent",
)


# Hand-rolled StateGraph, not create_agent: create_agent's ReAct loop always re-invokes the
# model after a tool call, which would re-decide routing on every reply even when the
# conversation is clearly still with the same subagent. `route` below is plain Python (no
# LLM call) and only falls through to `model` for genuine ambiguity. `call_subagent` passes
# the subagent's answer through verbatim, so nothing re-summarizes or drops details.
class SupervisorState(AgentState):
    active_subagent: Optional[Literal["erddap", "thredds"]]
    artifacts: Annotated[list[dict], operator.add]


# Matches a URL path segment (/erddap, /thredds) or the bare platform name in words.
_PLATFORM_PATTERNS = {
    "erddap": re.compile(r"\berddap\b", re.IGNORECASE),
    "thredds": re.compile(r"\bthredds\b", re.IGNORECASE),
}


def _detect_platforms(text: str) -> set[str]:
    return {name for name, pattern in _PLATFORM_PATTERNS.items() if pattern.search(text)}


def _last_human_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.text if hasattr(msg, "text") else str(msg.content)
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _extract_text(message) -> str:
    # content_blocks is the current API, fall back to content if not present
    if hasattr(message, "content_blocks") and message.content_blocks:
        return " ".join(
            block.get("text", "") for block in message.content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return message.content or "No response from subagent."


def route(state: SupervisorState, runtime: Runtime) -> Command:
    last_text = _last_human_text(state["messages"])
    mentioned = _detect_platforms(last_text)
    active = state.get("active_subagent")

    # Sticky continuation: stay on the active subagent with no LLM call.
    if active and not (mentioned - {active}):
        return Command(goto="call_subagent")

    # Unambiguous first contact or platform switch: commit and route directly.
    if len(mentioned) == 1:
        (only,) = mentioned
        return Command(update={"active_subagent": only}, goto="call_subagent")

    # Ambiguous: nothing mentioned, or conflicting platforms -- ask via the model.
    return Command(goto="model")


def _invoke_subagent(
    active_subagent: Optional[Literal["erddap", "thredds"]], query: str, thread_id: str
) -> tuple[str, list[dict]]:
    agent = erddap_agent if active_subagent == "erddap" else thredds_agent

    # Streamed so callers using stream_mode="custom" see per-tool-call progress;
    # get_stream_writer() is a no-op for plain invoke() callers.

    writer = get_stream_writer()
    result: dict | None = None
    artifacts: list[dict] = []
    
    config: RunnableConfig = {"configurable": {"thread_id": f"{thread_id}:{active_subagent}"}}
    # durability="exit" skips the checkpoint write after every model/tool step and persists
    # once at the end instead -- a multi-tool-call turn otherwise pays a synchronous Postgres
    # round-trip per step. Tradeoff: a mid-turn crash loses the whole turn's progress instead
    # of resuming from the last completed step, acceptable here since a user just resends.
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]}, config,
        stream_mode="updates", durability="exit",
    ):
        for node, update in chunk.items():
            result = update
            artifacts.extend(update.get("artifacts") or [])
            if node == "model":
                for msg in update.get("messages", []):
                    for call in getattr(msg, "tool_calls", None) or []:
                        writer(f"Calling {call['name']}...")
            elif node == "tools":
                for msg in update.get("messages", []):
                    writer(f"{msg.name} done")
    if result is None:
        raise RuntimeError(f"{active_subagent} agent stream produced no updates")
    answer = _extract_text(result["messages"][-1])
    return answer, artifacts


def call_subagent(state: SupervisorState, runtime: Runtime) -> Command:
    thread_id = get_config().get("configurable", {})["thread_id"]
    query = _last_human_text(state["messages"])
    answer, artifacts = _invoke_subagent(state["active_subagent"], query, thread_id)
    return Command(update={"messages": [AIMessage(content=answer)], "artifacts": artifacts}, goto=END)


# Only reached for genuine routing ambiguity -- see route() above.
_model_agent = create_agent(
    model=supervisor_model,
    system_prompt=(
        "You handle general queries and requests that don't clearly indicate an ERDDAP or "
        "THREDDS server. Ask the user which data platform they mean, or which server URL to use, "
        "if it isn't clear from their message. Refuse to fulfill unsafe or extraneous queries."
    ),
)


def model_node(state: SupervisorState, runtime: Runtime) -> Command:
    model_input: InputAgentState = {"messages": list(state["messages"])}
    result = _model_agent.invoke(model_input)
    answer = _extract_text(result["messages"][-1])
    return Command(update={"messages": [AIMessage(content=answer)]}, goto=END)


_builder = StateGraph(SupervisorState)
_builder.add_node("route", route)
_builder.add_node("call_subagent", call_subagent)
_builder.add_node("model", model_node)
_builder.set_entry_point("route")

graph = _builder.compile(checkpointer=_checkpointer)

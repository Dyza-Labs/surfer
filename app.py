from typing import cast
from uuid import uuid4

import streamlit as st
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from surfer.graph import SupervisorState, graph

st.set_page_config(
    page_title = "Surfer",
    page_icon = "🏄‍♂️"
)

st.title("SURFER: Quick, Surface-Level Access To Ocean Data 🏄‍♂️")
AVATARS = {
    "user": "🦈",
    }

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"ui-{uuid4()}"

# Each entry is a {"role", "text", "artifacts", "error"} dict
if "history" not in st.session_state:
    st.session_state.history = []


def render_artifacts(artifacts: list[dict], turn_idx: int) -> None:
    """Renders plots and maps to the chat window. Artifacts do not persist across
    browser tab refreshes.

    turn_idx keys widgets by conversation position: history redraws pass the turn's
    index, the live turn passes len(history) (its future index), so keys stay stable
    across reruns and can't collide when two artifacts share a title."""
    for i, artifact in enumerate(artifacts):
        title = artifact.get("title", "artifact")
        safe_name = title.replace(" ", "_").replace(":", "")
        key = f"dl-{turn_idx}-{i}-{safe_name}"

        if artifact["type"] == "image":
            st.image(artifact["content"])
            st.download_button(
                f"⬇ Download {title} (PNG)",
                data=artifact["content"],
                file_name=f"{safe_name}.png",
                mime="image/png",
                key=key,
            )
        elif artifact["type"] == "html":
            st.iframe(artifact["content"], height=720)
            st.download_button(
                f"⬇ Download {title} (HTML)",
                data=artifact["content"],
                file_name=f"{safe_name}.html",
                mime="text/html",
                key=key,
            )


def render_turn(turn: dict, turn_idx: int) -> None:
    """Renders one chat turn as an error box (with expandable detail) or
    markdown, followed by any artifacts."""
    if turn.get("error"):
        st.error(turn["text"])
        with st.expander("Details"):
            st.code(turn["error"])
    else:
        st.markdown(turn["text"])
    render_artifacts(turn["artifacts"], turn_idx)


for turn_idx, turn in enumerate(st.session_state.history):
    with st.chat_message(turn["role"], avatar=AVATARS.get(turn["role"])):
        render_turn(turn, turn_idx)

EXAMPLE_PROMPTS = [
    "Search for datasets with chlorophyll A data on https://gliders.ioos.us/erddap/index.html",
    "Graph the trajectory of glider maracoos_05-20250404T1319 from https://slocum-data.marine.rutgers.edu/erddap",
    "What data is available on the THREDDS server at https://tds.marine.rutgers.edu/thredds/catalog/catalog.html?",
]

picked = None
if not st.session_state.history:
    picked = st.pills(
        "Try an example prompt:", EXAMPLE_PROMPTS,
        selection_mode="single", label_visibility="visible",
        key="example_pill",
    )

if prompt := (st.chat_input("Ask about ERDDAP or THREDDS servers...") or picked):
    # Appending to history hides the pills widget
    st.session_state.history.append({"role": "user", "text": prompt, "artifacts": []})
    with st.chat_message("user", avatar=AVATARS.get("user")):
        st.markdown(prompt)

    config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant", avatar=AVATARS.get("assistant")):
        with st.status("Working...", expanded=True) as status:
            # active_subagent/artifacts must not be set here -- active_subagent has no
            # reducer, so resupplying it would clobber the persisted checkpoint value.
            input_state = cast(SupervisorState, {"messages": [HumanMessage(content=prompt)]})
            update: dict | None = None
            error_detail = None
            try:
                for mode, chunk in graph.stream(
                    input_state, config, stream_mode=["updates", "custom"], durability="exit",
                ):
                    if mode == "custom":
                        status.write(chunk)
                    elif mode == "updates" and isinstance(chunk, dict):
                        update = chunk
                if update is None:
                    raise RuntimeError("graph.stream produced no updates")
            except Exception as exc:
                answer, artifacts = "Something went wrong while processing your request.", []
                error_detail = f"{type(exc).__name__}: {exc}"
            else:
                final = next(iter(update.values()))
                answer = str(final["messages"][-1].content)
                artifacts = final.get("artifacts", [])

            if error_detail:
                status.update(label="Failed", state="error", expanded=False)
            else:
                status.update(label="Done", state="complete", expanded=False)

        turn = {"role": "assistant", "text": answer, "artifacts": artifacts, "error": error_detail}
        render_turn(turn, len(st.session_state.history))

    st.session_state.history.append(turn)

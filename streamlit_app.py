from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from src.simulator import SupportResponse, SupportSimulator

st.set_page_config(
    page_title="Smart Customer Support Simulator",
    page_icon=":speech_balloon:",
    layout="wide",
)


@dataclass(frozen=True)
class UiModeOption:
    key: str
    label: str
    simulator_mode: str
    description: str


UI_MODE_OPTIONS = (
    UiModeOption(
        key="fast_private",
        label="Fast & Private Mode",
        simulator_mode="raw_slm",
        description="Small local model only. No retrieval, tools, or order database access.",
    ),
    UiModeOption(
        key="full_intelligence",
        label="Full Intelligence Mode",
        simulator_mode="raw_llm",
        description="Full local model for general reasoning. No private retrieval, tools, or order database access.",
    ),
    UiModeOption(
        key="agent",
        label="Agent",
        simulator_mode="agent",
        description="Direct agent testing with retrieval, order lookup, ticketing, and escalation.",
    ),
    UiModeOption(
        key="rag",
        label="RAG",
        simulator_mode="rag",
        description="Retrieval-only testing over the local knowledge base.",
    ),
)
UI_MODE_BY_KEY = {option.key: option for option in UI_MODE_OPTIONS}


def ensure_session_state() -> None:
    if "simulator" not in st.session_state:
        st.session_state.simulator = SupportSimulator()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_mode_key" not in st.session_state:
        st.session_state.active_mode_key = "full_intelligence"


def reset_conversation() -> None:
    st.session_state.simulator = SupportSimulator()
    st.session_state.messages = []


def get_active_mode() -> UiModeOption:
    return UI_MODE_BY_KEY.get(
        st.session_state.get("active_mode_key"),
        UI_MODE_BY_KEY["full_intelligence"],
    )


def response_to_record(
    response: SupportResponse, mode_option: UiModeOption
) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": response.answer,
        "mode_label": mode_option.label,
        "simulator_mode": mode_option.simulator_mode,
        "route": response.route,
        "intent": response.intent,
        "confidence": response.confidence,
        "sources": response.sources,
        "actions": response.actions,
        "ticket_id": response.ticket_id,
    }


def render_assistant_metadata(message: dict[str, object]) -> None:
    confidence = float(message["confidence"])
    st.caption(
        f"Selection: {message['mode_label']} | Mode: {message['simulator_mode']} | Route: {message['route']} | Intent: {message['intent']} | Confidence: {confidence:.0%}"
    )

    if message["ticket_id"]:
        st.info(f"Ticket created: {message['ticket_id']}")

    if message["sources"]:
        with st.expander("Sources"):
            for source in message["sources"]:
                st.markdown(f"- {source}")

    if message["actions"]:
        with st.expander("Actions"):
            for action in message["actions"]:
                st.markdown(f"- {action}")


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_assistant_metadata(message)


def render_mode_buttons() -> None:
    columns = st.columns(len(UI_MODE_OPTIONS))

    for column, mode_option in zip(columns, UI_MODE_OPTIONS):
        is_active = mode_option.key == st.session_state.active_mode_key
        if column.button(
            mode_option.label,
            type="primary" if is_active else "secondary",
            use_container_width=True,
            key=f"mode_{mode_option.key}",
        ):
            st.session_state.active_mode_key = mode_option.key
            st.rerun()
        column.caption(mode_option.description)


ensure_session_state()

st.title("Smart Customer Support Simulator")
render_mode_buttons()
active_mode = get_active_mode()

header_left, header_right = st.columns([4, 1])
header_left.markdown(
    f"**Current mode:** {active_mode.label} (`{active_mode.simulator_mode}`)"
)
if header_right.button("New Conversation", use_container_width=True):
    reset_conversation()
    st.rerun()

if (
    active_mode.simulator_mode == "agent"
    and st.session_state.simulator.agent_runtime is None
):
    st.warning(
        "The LangChain agent is unavailable in this session. Agent requests will use the grounded fallback until the agent runtime is restored."
    )

render_chat_history()

prompt = st.chat_input(f"Send a message in {active_mode.label}")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.simulator.handle_message(
                prompt,
                mode=active_mode.simulator_mode,
            )
        st.markdown(response.answer)
        assistant_message = response_to_record(response, active_mode)
        render_assistant_metadata(assistant_message)

    st.session_state.messages.append(assistant_message)

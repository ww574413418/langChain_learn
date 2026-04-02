import time

import streamlit as st

from agent.react_agent import agent
from memory.chat_session_store import (
    DEFAULT_SESSION_TITLE,
    cleanup_inactive_empty_sessions,
    create_chat_session,
    delete_chat_session,
    get_chat_session,
    upsert_chat_session,
)


def refresh_chat_sessions() -> list[dict]:
    active_thread_id = st.session_state.get("thread_id")
    sessions = cleanup_inactive_empty_sessions(
        st.session_state["user_id"],
        active_thread_id,
    )
    st.session_state["chat_sessions"] = sessions
    return sessions


def ensure_active_thread() -> dict | None:
    sessions = refresh_chat_sessions()
    active_thread_id = st.session_state.get("thread_id")

    if not sessions:
        st.session_state["thread_id"] = None
        return None

    if not active_thread_id:
        st.session_state["thread_id"] = sessions[0]["thread_id"]
        return sessions[0]

    active_session = get_chat_session(st.session_state["user_id"], active_thread_id)
    if active_session:
        return active_session

    st.session_state["thread_id"] = sessions[0]["thread_id"]
    return sessions[0]


def save_active_messages(messages: list[dict]) -> dict:
    active_thread_id = st.session_state["thread_id"]
    updated_session = upsert_chat_session(
        st.session_state["user_id"],
        active_thread_id,
        messages,
        fallback_title=DEFAULT_SESSION_TITLE,
    )
    refresh_chat_sessions()
    return updated_session


def create_and_switch_session() -> None:
    new_session = create_chat_session(st.session_state["user_id"])
    st.session_state["thread_id"] = new_session["thread_id"]
    refresh_chat_sessions()


def delete_and_switch_session(thread_id: str) -> None:
    sessions = delete_chat_session(st.session_state["user_id"], thread_id)
    if thread_id == st.session_state.get("thread_id"):
        st.session_state["thread_id"] = sessions[0]["thread_id"] if sessions else None

    refresh_chat_sessions()


def get_final_response(chunks: list[str]) -> str:
    for chunk in reversed(chunks):
        content = chunk.strip()
        if content:
            return content
    return ""


def render_sidebar(active_thread_id: str) -> None:
    with st.sidebar:
        st.subheader("会话列表")
        if st.button("新建会话", use_container_width=True):
            create_and_switch_session()
            st.rerun()

        for session in st.session_state["chat_sessions"]:
            session_thread_id = session["thread_id"]
            session_title = session.get("title") or DEFAULT_SESSION_TITLE
            is_active = session_thread_id == active_thread_id

            session_col, delete_col = st.columns([5, 1])

            if session_col.button(
                session_title,
                key=f"switch_{session_thread_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["thread_id"] = session_thread_id
                st.rerun()

            if delete_col.button(
                "×",
                key=f"delete_{session_thread_id}",
                use_container_width=True,
                help="删除会话",
            ):
                delete_and_switch_session(session_thread_id)
                st.rerun()


# title
st.set_page_config(
    page_title="我的智能客服",
    page_icon="🤖",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] button {
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "user_id" not in st.session_state:
    st.session_state["user_id"] = "0001"

if "agent" not in st.session_state:
    st.session_state["agent"] = agent

if "chat_sessions" not in st.session_state:
    st.session_state["chat_sessions"] = []

active_session = ensure_active_thread()
active_thread_id = st.session_state["thread_id"]

render_sidebar(active_thread_id)

st.title("智能客服")
if active_thread_id:
    st.caption(f"当前会话 ID: {active_thread_id}")
st.divider()

history_messages = active_session.get("messages", []) if active_session else []
for message in history_messages:
    st.chat_message(message["role"]).write(message["content"])

if not active_session:
    st.info("还没有会话，点击左侧“新建会话”开始，或直接在下方输入消息。")

prompt = st.chat_input("请输入你的问题")

if prompt:
    if not active_thread_id:
        active_session = create_chat_session(st.session_state["user_id"])
        active_thread_id = active_session["thread_id"]
        st.session_state["thread_id"] = active_thread_id
        refresh_chat_sessions()

    st.chat_message("user").write(prompt)

    current_messages = list(active_session.get("messages", [])) if active_session else []
    current_messages.append({"role": "user", "content": prompt})
    save_active_messages(current_messages)

    response_chunks = []
    with st.spinner(text="thinking..."):
        result = st.session_state["agent"].execute_stream(
            prompt,
            active_thread_id,
            st.session_state["user_id"],
        )

        def stream_result(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(stream_result(result, response_chunks))

    assistant_message = get_final_response(response_chunks)
    if assistant_message:
        current_messages.append({"role": "assistant", "content": assistant_message})
        save_active_messages(current_messages)

    st.rerun()

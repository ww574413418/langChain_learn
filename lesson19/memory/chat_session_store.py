import json
import os
import uuid
from datetime import datetime
from memory.memory_note_store import delete_session_notes
from memory.thread_summary_store import delete_thread_summary
from utils.path_tool import get_abs_path

DEFAULT_SESSION_TITLE = "新建会话"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_chat_sessions_path(user_id: str | int) -> str:
    return get_abs_path(f"memory/chat_sessions/{user_id}.json")


def load_chat_sessions(user_id: str | int) -> list[dict]:
    path = get_chat_sessions_path(user_id)
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            sessions = json.load(f)
    except Exception:
        return []

    if not isinstance(sessions, list):
        return []
    return sort_chat_sessions(sessions)


def save_chat_sessions(user_id: str | int, sessions: list[dict]) -> None:
    path = get_chat_sessions_path(user_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sort_chat_sessions(sessions), f, ensure_ascii=False, indent=2)


def sort_chat_sessions(sessions: list[dict]) -> list[dict]:
    return sorted(
        sessions,
        key=lambda item: (
            item.get("updated_at", ""),
            item.get("created_at", ""),
            item.get("thread_id", ""),
        ),
        reverse=True,
    )


def build_session_title(messages: list[dict], fallback: str = DEFAULT_SESSION_TITLE) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue

        content = str(message.get("content", "")).strip()
        if not content:
            continue

        if len(content) <= 18:
            return content
        return f"{content[:18]}..."

    return fallback


def is_empty_temporary_session(session: dict) -> bool:
    messages = session.get("messages", [])
    return session.get("is_temporary", False) and not messages


def new_chat_session(
    title: str = DEFAULT_SESSION_TITLE,
    thread_id: str | None = None,
    is_temporary: bool = True,
) -> dict:
    now = now_text()
    return {
        "thread_id": thread_id or str(uuid.uuid4()),
        "title": title,
        "messages": [],
        "is_temporary": is_temporary,
        "created_at": now,
        "updated_at": now,
    }


def ensure_chat_sessions(user_id: str | int) -> list[dict]:
    sessions = load_chat_sessions(user_id)
    if sessions:
        return sessions

    first_session = new_chat_session()
    save_chat_sessions(user_id, [first_session])
    return [first_session]


def create_chat_session(
    user_id: str | int,
    title: str = DEFAULT_SESSION_TITLE,
) -> dict:
    sessions = load_chat_sessions(user_id)
    new_session = new_chat_session(title=title)
    sessions.append(new_session)
    save_chat_sessions(user_id, sessions)
    return new_session


def get_chat_session(user_id: str | int, thread_id: str) -> dict | None:
    sessions = load_chat_sessions(user_id)
    for session in sessions:
        if session.get("thread_id") == thread_id:
            return session
    return None


def upsert_chat_session(
    user_id: str | int,
    thread_id: str,
    messages: list[dict],
    fallback_title: str = DEFAULT_SESSION_TITLE,
) -> dict:
    sessions = load_chat_sessions(user_id)
    now = now_text()

    for session in sessions:
        if session.get("thread_id") != thread_id:
            continue

        session["messages"] = messages
        session["title"] = build_session_title(messages, session.get("title") or fallback_title)
        session["is_temporary"] = False if messages else session.get("is_temporary", True)
        session["updated_at"] = now
        save_chat_sessions(user_id, sessions)
        return session

    new_session = new_chat_session(
        title=build_session_title(messages, fallback_title),
        thread_id=thread_id,
        is_temporary=not bool(messages),
    )
    new_session["messages"] = messages
    new_session["updated_at"] = now
    sessions.append(new_session)
    save_chat_sessions(user_id, sessions)
    return new_session


def cleanup_inactive_empty_sessions(
    user_id: str | int,
    active_thread_id: str | None,
) -> list[dict]:
    sessions = load_chat_sessions(user_id)
    remaining_sessions = [
        session for session in sessions
        if session.get("thread_id") == active_thread_id or not is_empty_temporary_session(session)
    ]

    if len(remaining_sessions) == len(sessions):
        return sort_chat_sessions(sessions)

    removed_thread_ids = [
        session.get("thread_id")
        for session in sessions
        if session.get("thread_id") != active_thread_id and is_empty_temporary_session(session)
    ]

    save_chat_sessions(user_id, remaining_sessions)
    for thread_id in removed_thread_ids:
        delete_thread_summary(thread_id)
        delete_session_notes(thread_id)

    return sort_chat_sessions(remaining_sessions)


def delete_chat_session(user_id: str | int, thread_id: str) -> list[dict]:
    sessions = load_chat_sessions(user_id)
    remaining_sessions = [
        session for session in sessions
        if session.get("thread_id") != thread_id
    ]
    save_chat_sessions(user_id, remaining_sessions)
    delete_thread_summary(thread_id)
    delete_session_notes(thread_id)
    return sort_chat_sessions(remaining_sessions)

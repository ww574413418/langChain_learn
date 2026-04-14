from memory.memory_selector import (
    select_memory_plan,
    pick_profile_fields,
    format_profile_subset
)
from memory.memory_retriever import (
    retrieve_relevant_notes,
    format_retrieved_notes
)
from memory.profile_store import get_user_profile
from memory.thread_summary_store import load_thread_summary

'''
这层负责把你现有 memory 能力组装成 QA 可消费的上下文。
不要把这些逻辑散在 node 里。
'''

def build_memory_read_artifacts(
        query:str,
        user_id:str,
        thread_id:str,
)->dict:
    plan = select_memory_plan(query)

    profile_context = ""

    if plan.need_profile:
        user_profile = get_user_profile(user_id)
        profile_subset = pick_profile_fields(user_profile, plan.profile_fields)
        profile_context = format_profile_subset(profile_subset)

    notes_context = ""

    if plan.need_notes and plan.note_query:
        notes = retrieve_relevant_notes(
            user_id,
            thread_id,
            plan.note_query,
            top_n=max(plan.top_k * 2, 8),
            top_k=plan.top_k
        )
        notes_context = format_retrieved_notes(notes)

    summary_context = ""
    thread_summary = load_thread_summary(thread_id)
    if thread_summary and thread_summary.get("summary"):
        summary_context = f"当前thread的历史摘要:{thread_summary["summary"]}"

    context_parts = [
        profile_context,
        notes_context,
        summary_context,
    ]

    memory_context = "\n".join(part for part in context_parts if part)

    return {
        "memory_plan": plan.model_dump(),
        "profile_context": profile_context,
        "notes_context": notes_context,
        "summary_context": summary_context,
        "memory_context": memory_context,
    }

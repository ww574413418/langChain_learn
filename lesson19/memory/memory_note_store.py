import re

from pydantic import BaseModel,Field
import json
import os
from utils.path_tool import get_abs_path
from utils.logger_handler import logger as log
from datetime import datetime

class MemoryNote(BaseModel):
    id:str = Field(description="note id")
    text:str = Field(description="记忆内容")
    category: str = Field(description="记忆类别")
    keywords: list[str] = Field(default_factory=list, description="便于检索的关键词")
    scope:str = Field(description="session or global")
    source_thread_id:str | None = Field(default=None)
    confidence:float = Field(default=0.8)
    created_at:str = Field(description="创建者时间")
    updated_at:str = Field(description="更新时间")
    last_seen_at:str
    status: str = Field(default="pending") # pending 表示还在 staging area,promoted 表示已经沉淀进 global,discarded 表示已判断不进 global，但保留痕迹
    consolidation_action: str | None = Field(default=None)
    consolidated_at: str | None = Field(default=None)
    target_global_id: str | None = Field(default=None)

def get_global_notes_path(user_id:str|int)->str:
    return get_abs_path(f"memory/global_notes/{user_id}.json")

def get_session_notes_path(thread_id:str|int)->str:
    return get_abs_path(f"memory/session_notes/{thread_id}.json")


def load_notes(path:str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_notes(path:str,notes:list[dict])->None:
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(notes,f,ensure_ascii=False,indent=2)


def load_session_notes(thread_id: str | int, include_history: bool = False):
    notes = load_notes(get_session_notes_path(thread_id))
    if include_history:
        return notes

    return [
        note for note in notes
        if note.get("status") == "pending"
    ]


def load_global_notes(user_id:str|int) ->list[dict]:
    return load_notes(get_global_notes_path(user_id))

def update_session_notes_status(
    thread_id: str | int,
    decisions: list[dict],
) -> list[dict]:
    notes = load_notes(get_session_notes_path(thread_id))
    if not notes:
        return []

    updated_notes = []
    used_ids = set()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for decision in decisions:
        session_note_text = normalize_note_text(decision.get("session_note_text", ""))
        action = decision.get("action")
        target_global_id = decision.get("target_global_id")

        for note in notes:
            if note.get("id") in used_ids:
                continue

            if note.get("status", "pending") not in {"pending"}:
                continue

            if normalize_note_text(note.get("text", "")) != session_note_text:
                continue

            note["updated_at"] = now
            note["consolidated_at"] = now
            note["consolidation_action"] = action
            note["target_global_id"] = target_global_id

            if action == "discard":
                note["status"] = "discarded"
            else:
                note["status"] = "promoted"

            updated_notes.append(note)
            used_ids.add(note.get("id"))
            break

    save_notes(get_session_notes_path(thread_id), notes)
    return updated_notes


def append_session_note(
        user_id:str|int,
        source_thread_id:str,
        text:str,
        keywords:list[str] | None,
        category:str,
) ->dict:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = get_session_notes_path(source_thread_id)
    notes = load_notes(path)
    confidence = estimate_note_confidence(text,category,keywords)

    for existing_note in notes:
        if existing_note.get("category") != category:
            continue

        if existing_note.get("status") != "pending":
            continue

        old_text = existing_note.get("text","")

        if is_same_note(old_text,text) or is_contained_note(old_text,text):
            merge = merge_note(existing_note,text,keywords,confidence)
            save_notes(path,notes)
            return merge

    note = MemoryNote(
        id=f"session_note_{len(notes) + 1}",
        user_id=user_id,
        text=text,
        keywords=keywords or [],
        category=category,
        confidence=confidence,
        source_thread_id=source_thread_id,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        last_seen_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        scope="session"
    )

    notes.append(note.model_dump())
    save_notes(path,notes)
    return note.model_dump()


def estimate_note_confidence(text:str,category:str,keywords:list[str]|None = None)->float:
    score = {
        "home": 0.85,
        "device": 0.85,
        "preference": 0.65,
        "demand": 0.8,
        "other": 0.65,
    }.get(category, 0.60)

    keywords = keywords or []

    if any(word in text for word in ["可能", "说明", "应该", "推测"]):
        score -= 0.20

    if len(text) > 40:
        score -= 0.05

    if len(keywords) >= 2:
        score += 0.03

    return max(0.0, min(score, 0.95))

def normalize_note_text(text:str)->str:
    text = text.strip()
    text = re.sub(r"[，。！？；：、“”‘’（）()、\s,.!?;:]", "", text)
    return text

def is_same_note(text1:str,text2:str)->bool:
    '''
    2个文本完全想到
    :param text1:
    :param text2:
    :return:
    '''
    return normalize_note_text(text1) == normalize_note_text(text2)

def is_contained_note(text1:str,text2:str)->bool:
    '''
    判断2个文本是否包含
    :param text1:
    :param text2:
    :return:
    '''
    t1 = normalize_note_text(text1)
    t2 = normalize_note_text(text2)
    if not t1 or not t2:
        return False
    return t1 in t2 or t2 in t1


def merge_note(existing_note:dict,text:str,keywords:list[str],confidence:float)->dict:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    old_text = existing_note.get("text","")
    if len(normalize_note_text(text)) > len(normalize_note_text(old_text)):
        existing_note["text"] = text

    old_keywords = existing_note.get("keywords", [])
    existing_note["keywords"] = list(dict.fromkeys(old_keywords + (keywords or [])))
    existing_note["updated_at"] = now
    existing_note["last_seen_at"] = now
    existing_note["confidence"] = max(existing_note.get("confidence",0.0), confidence)

    return existing_note

def keyword_overlap_count(note1:dict,note2:dict)->int:
    '''
    返回更长的keywords的
    :param note1:
    :param note2:
    :return:
    '''
    k1 = set(note1.get("keywords",[]))
    k2 = set(note2.get("keywords",[]))
    return len(k1 & k2)

def should_merge_notes(note1:dict,note2:dict)->bool:
    '''
    判断是否需要合并
    :param note1:
    :param note2:
    :return:
    '''
    if note1.get("category") != note2.get("category"):
        return False
    if is_same_note(note1.get("text",""),note2.get("text","")):
        return True
    if is_contained_note(note1.get("text",""),note2.get("text","")):
        return True
    if keyword_overlap_count(note1,note2) >=2:
        return True

    return False

def merge_notes_records(note1:dict,note2:dict)->dict:
    '''
    将session notes合并成global session
    :param note1:
    :param note2:
    :return:
    '''
    text1 = note1.get("text","")
    text2 = note2.get("text","")

    batter_text = text1
    if len(normalize_note_text(text2)) > len(normalize_note_text(text1)):
        batter_text = text2

    merged_keywords = list(dict.fromkeys(note1.get("keywords", []) + note2.get("keywords", [])))

    return {
        **note1,
        "text": batter_text,
        "keywords": merged_keywords,
        "confidence":max(note1.get("confidence",0),note2.get("confidence",0)),
        "updated_at":max(note1.get("updated_at",""),note2.get("updated_at","")),
        "last_seen_at":max(note1.get("last_seen_at",""),note2.get("last_seen_at","")),
        "scope":"global"
    }

def clear_session_notes(thread_id: str | int) -> None:
    save_notes(get_session_notes_path(thread_id), [])



if __name__ == "__main__":
    print(append_session_note("0001", "123123","房东自带老扫地机器，不能拖地", ["扫地机", "不能拖地"], "device"))
    print(append_session_note("0001", "123123","房东自带老扫地机器，不能拖地", ["扫地机", "不能拖地"], "device"))
    # print(append_session_note("0001", "家里有一个大阳台，容易落灰", ["阳台", "落灰"]))

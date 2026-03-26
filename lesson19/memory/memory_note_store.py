from pydantic import BaseModel,Field
import json
import os
from utils.path_tool import get_abs_path
from utils.logger_handler import logger as log
from datetime import datetime

class MemoryNote(BaseModel):
    id:str = Field(description="note id")
    text:str = Field(description="记忆内容")
    keywords:list[str] = Field(default=list,description="便于检索的关键词")
    created_at:str = Field(description="创建者时间")
    updated_at:str = Field(description="更新时间")
    category: str = Field(description="记忆类别")


def get_memory_notes_path(user_id:str|int)->str:
    return get_abs_path(f"memory/memory_notes/{user_id}.json")


def load_memory_notes(user_id:str|int) ->list[dict]:
    memory_notes_path = get_memory_notes_path(user_id)
    if not os.path.exists(memory_notes_path):
        return []

    try:
        with open(memory_notes_path,"r",encoding="utf-8") as f:
            memory_notes = json.load(f)
            return memory_notes
    except Exception as e:
        print(f"load memory notes error:{e}")
        log.error(f"load memory notes error:{e}")
        return []

def save_memory_notes(user_id:str|int,notes:list[dict]) -> None:
    '''
    将notes保存到本地
    :param user_id:
    :param notes:
    :return:
    '''
    memory_notes_path = get_memory_notes_path(user_id)
    os.makedirs(os.path.dirname(memory_notes_path),exist_ok=True)

    with open(memory_notes_path,"w",encoding="utf-8") as f:
        json.dump(notes,f,ensure_ascii=False,indent=2)

def normalize_note_text(text: str) -> str:
    return text.strip()


def find_existing_note(notes: list[dict], text: str) -> dict | None:
    normalized = normalize_note_text(text)
    for note in notes:
        if normalize_note_text(note.get("text", "")) == normalized:
            return note
    return None


def add_memory_note(user_id:str|int,text:str,keywords:list[str],category:str="other") -> dict:
    notes:list = load_memory_notes(user_id)
    note = MemoryNote(
        id=f"note_{len(notes) + 1}",
        text=text,
        keywords=keywords,
        category=category,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    existing_note = find_existing_note(notes,text)
    if existing_note:
        old_keywords = existing_note.get("keywords", [])
        merged_keywords = list(dict.fromkeys(old_keywords + (keywords or [])))
        existing_note["keywords"] = merged_keywords
        existing_note["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not existing_note.get("category"):
            existing_note["category"] = category
        save_memory_notes(user_id, notes)
        return existing_note
    log.info(f"add_memory_note:{note}")
    notes.append(note.model_dump())
    save_memory_notes(user_id,notes)
    return note.model_dump()

if __name__ == "__main__":
    print(add_memory_note("0001", "房东自带老扫地机器，不能拖地", ["扫地机", "不能拖地"]))
    print(add_memory_note("0001", "家里有一个大阳台，容易落灰", ["阳台", "落灰"]))
    print(load_memory_notes("0001"))

from typing import Optional,Literal
from pydantic import BaseModel,Field
from model.model_factory import chat_model
from utils.prompts_loader import memory_note_filter_prompt
from utils.logger_handler import logger as log


class NoteKeepDecision(BaseModel):
    '''
    keep = false 后面的字段都可以为空
           ture 后面的字段最好补全
    '''
    keep:bool = Field(description="这条note是否值得进入长期记忆链路")
    normalized_text:Optional[str] = Field(default=None,description="保留时的规范化文本")
    normalized_keywords:list[str] = Field(default_factory=[],description="保留时的规范化keywords")
    category:Optional[Literal["home","device","preference","demand","other"]] = Field(
        default=None,
        description="保留时的分类"
    )
    reason:Optional[str] = Field(default=None,description="做出该判断的原因")

def validate_keep_decision(note: dict, decision: NoteKeepDecision) -> NoteKeepDecision:
    original_text = note.get("text", "").strip()
    original_keywords = note.get("keywords", [])
    original_category = note.get("category", None)

    if not decision.keep:
        decision.normalized_text = None
        decision.normalized_keywords = []
        decision.category = None
        return decision

    if not decision.normalized_text:
        decision.normalized_text = original_text

    if not decision.normalized_keywords:
        decision.normalized_keywords = original_keywords

    if not decision.category:
        decision.category = original_category

    return decision

def filter_one_candidate_note(note:dict)->NoteKeepDecision:
    if hasattr(note, "model_dump"):
        note = note.model_dump()

    structured_llm = chat_model.with_structured_output(NoteKeepDecision)
    simplified_note = {
        "text":note.get("text",""),
        "keywords":note.get("keywords",[]),
        "category":note.get("category","")
    }
    prompt = f"{memory_note_filter_prompt()} \n\n 候选记忆:{simplified_note}"
    result = structured_llm.invoke(prompt)
    log.info(f"filter_one_candidate_note: {result}")
    return validate_keep_decision(note,result)

def filter_candidate_notes(notes: list[dict]) -> list[NoteKeepDecision]:
    results = []

    for note in notes:
        decision = filter_one_candidate_note(note)
        results.append(decision)

    return results

if __name__ == "__main__":
    test_notes = [
        {
            "text": "房东自带的老扫地机不能拖地",
            "category": "device",
            "keywords": ["扫地机", "拖地"]
        },
        {
            "text": "扫地机器人和洗地机有什么区别",
            "category": "demand",
            "keywords": ["扫地机器人", "洗地机", "区别"]
        },
        {
            "text": "想买个安静点的扫地机",
            "category": "demand",
            "keywords": ["扫地机", "安静"]
        }
    ]

    for note in test_notes:
        print("====")
        print(filter_one_candidate_note(note))

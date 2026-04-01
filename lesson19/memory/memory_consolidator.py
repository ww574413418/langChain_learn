'''
基于llm 从session note中提取信息存入到global note中
'''
from pydantic import BaseModel,Field
from typing import Literal,Optional
from utils.prompts_loader import memory_consolidator_prompt
from utils.logger_handler import logger as log
from model.model_factory import chat_model
from datetime import datetime

# 先产出显式决策，再执行决策。
class ConsolidationDecision(BaseModel):
    action:Literal["add","merge","discard"] = Field(description="对 session note 的处理动作")
    session_note_text:str = Field(description="当前需要处理的 session note 的内容")
    target_global_id:Optional[str] = Field(default=None,description="当 action=merge 时,目标 global note 的 id")
    merged_text:Optional[str] = Field(default=None,description="当 action=add或merge时,最终保留的文本")
    merged_keywords:list[str] = Field(default_factory=list,description="最终要保留的关键字")
    category:Optional[str] = Field(default=None,description="最终要保留的类别")
    reason:Optional[str] = Field(default=None,description="做出决策的原因")
    relation_type: Optional[
        Literal["same_fact", "richer_version", "related_but_distinct", "no_value"]
    ] = Field(default=None, description="session note 与目标 global note 的关系类型")


class ConsolidationPlan(BaseModel):
    decisions: list[ConsolidationDecision] = Field(default_factory=list,description="本轮 consolidation 的决策列表")


def simplify_session_notes(session_notes:list[dict])-> list[dict]:
    result = []
    for note in session_notes:
        result.append({
            "text":note.get("text",""),
            "category":note.get("category",""),
            "keywords":note.get("keywords",[])
        })
    return result

def simplify_global_notes(global_notes:list[dict])-> list[dict]:
    result = []
    for note in global_notes:
        result.append({
            "id":note.get("id",""),
            "text":note.get("text",""),
            "category":note.get("category",""),
            "keywords":note.get("keywords",[])
        })
    return result

def normalize_keywords(keywords: list[str]) -> list[str]:
    result = []
    seen = set()

    for kw in keywords or []:
        if not isinstance(kw, str):
            continue
        kw = kw.strip()
        if not kw:
            continue
        if kw in seen:
            continue
        seen.add(kw)
        result.append(kw)

    return result


def validate_decision(
    decision: ConsolidationDecision,
    session_note: dict,
) -> ConsolidationDecision:
    valid_categories = {"home", "device", "preference", "demand", "other"}
    session_text = session_note.get("text", "").strip()
    session_category = session_note.get("category", "other")
    session_keywords = session_note.get("keywords", [])

    if decision.relation_type == "same_fact":
        decision.action = "merge"

    if decision.relation_type == "richer_version":
        decision.action = "merge"

    if decision.relation_type == "related_but_distinct":
        decision.action = "add"
        decision.target_global_id = None

    if decision.relation_type == "no_value":
        decision.action = "discard"

    if decision.category and decision.category not in valid_categories:
        decision.category = "other"

    if decision.action == "add" and decision.target_global_id:
        decision.action = "merge"

    if decision.action == "merge" and not decision.target_global_id:
        decision.action = "discard"
        decision.reason = "merge action missing target_global_id"

    if decision.action == "discard":
        decision.target_global_id = None
        decision.merged_text = None
        decision.merged_keywords = []
        decision.category = None
        return decision

    if decision.action in {"add", "merge"} and not decision.merged_text:
        decision.merged_text = session_text

    if decision.action in {"add", "merge"} and not decision.category:
        decision.category = session_category

    if decision.category not in valid_categories:
        decision.category = "other"

    if not decision.merged_keywords:
        decision.merged_keywords = session_keywords

    decision.merged_keywords = normalize_keywords(decision.merged_keywords)[:3]

    if decision.action == "add":
        decision.target_global_id = None

    return decision


def build_decision_for_one_session_note(
    session_note: dict,
    global_notes: list[dict]
) -> ConsolidationDecision:
    '''
    它的职责非常明确：
        输入：1 条 session note + 全部 global notes
        输出：1 条 ConsolidationDecision
    不是让模型去“总结 memory”，而是让它做 3 选 1 决策：
    add
        这条 session note 是新事实，global 里没有，应该新增
    merge
        这条 session note 和某条 global note 是同一事实或更完整版本，应该合并到已有 global
    discard
        这条 session note 不值得进入 global，比如：
        临时性
        无价值
        重复
        推断/噪音
    :param session_note:
    :param global_notes:
    :return:
    '''
    structured_llm = chat_model.with_structured_output(ConsolidationDecision)
    simplified_session_note = {
        "text": session_note.get("text", ""),
        "category": session_note.get("category", ""),
        "keywords": session_note.get("keywords", []),
    }
    simplified_global_notes = simplify_global_notes(global_notes)

    prompt = (
        f"{memory_consolidator_prompt}\n\n"
        f"current session note:\n{simplified_session_note}\n\n"
        f"global notes:\n{simplified_global_notes}"
    )

    result = structured_llm.invoke(prompt)
    log.info(f"build_decision_for_one_session_note: {result}")
    return validate_decision(result,session_note)

def build_consolidation_plan(
        session_notes:list[dict],
        global_notes:list[dict]
) -> ConsolidationPlan:
    '''
    整合所有的决策
    :param session_notes:
    :param global_notes:
    :return:
    '''
    decisions = []

    for session_note in session_notes:
        decision = build_decision_for_one_session_note(session_note, global_notes)
        decisions.append(decision)

    return ConsolidationPlan(decisions=decisions)

def build_next_global_id(global_notes: list[dict]) -> str:
    return f"global_{len(global_notes)}"

def merge_unique_keywords(old_keywords: list[str], new_keywords: list[str]) -> list[str]:
    '''
    合并关键词
    :param old_keywords:
    :param new_keywords:
    :return:
    '''
    return list(dict.fromkeys((old_keywords or []) + (new_keywords or [])))

def apply_consolidation_plan(global_notes:list[dict],plan:ConsolidationPlan) -> list[dict]:
    '''
    将ConsolidationPlan 应用到 global note,返回更新后的global note
    :param global_notes:
    :param plan:
    :return:
    '''
    working_notes = [dict(note) for note in global_notes]

    for decision in plan.decisions:
        if decision.action == "discard":
            continue

        if decision.action == "add":
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 新建一个note
            new_note = {
                "id": build_next_global_id(working_notes),
                "text": decision.merged_text,
                "category": decision.category or "other",
                "keywords": decision.merged_keywords,
                "scope": "global",
                "source_thread_id": None,
                "confidence": 0.8,
                "created_at": now,
                "updated_at": now,
                "last_seen_at": now,
                "status": "active",
            }
            working_notes.append(new_note)
            continue

        if decision.action == "merge":
            target_idx = None
            # 找到需要合并的note
            for idx,note in enumerate(working_notes):
                if note["id"] == decision.target_global_id:
                    target_idx = idx
                    break
            if target_idx is None:
                continue


            target_note = dict(working_notes[target_idx])
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            target_note["text"] = decision.merged_text or target_note.get("text","")
            target_note["category"] = decision.category or target_note.get(
                "category", "other"
            )
            target_note["keywords"] = merge_unique_keywords(
                target_note.get("keywords", []),
                decision.merged_keywords,
            )
            target_note["updated_at"] = now
            target_note["last_seen_at"] = now
            target_note["scope"] = "global"

            working_notes[target_idx] = target_note

    return working_notes


def consolidate_with_plan(
        session_notes:list[dict],
        global_notes:list[dict]) -> tuple[ConsolidationPlan,list[dict]]:
    plan = build_consolidation_plan(session_notes, global_notes)
    return plan, apply_consolidation_plan(global_notes, plan)




if __name__ == "__main__":
    session_notes = [
        {
            "text": "秋冬天阳台那边会有点冷",
            "category": "home",
            "keywords": ["秋冬", "阳台", "冷"],
        }
    ]

    global_notes = [
        {
            "id": "global_0",
            "text": "家里有阳台，没封窗，容易落灰",
            "category": "home",
            "keywords": ["阳台", "封窗", "落灰"],
            "scope": "global",
            "source_thread_id": "123123",
            "confidence": 0.85,
            "created_at": "2026-03-31 10:00:00",
            "updated_at": "2026-03-31 10:00:00",
            "last_seen_at": "2026-03-31 10:00:00",
            "status": "active",
        }
    ]

    plan, updated_global_notes = consolidate_with_plan(session_notes, global_notes)

    print("=== PLAN ===")
    print(plan)

    print("\n=== UPDATED GLOBAL NOTES ===")
    for note in updated_global_notes:
        print(note)

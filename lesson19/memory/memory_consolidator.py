'''
基于llm 从session note中提取信息存入到global note中
'''
from pydantic import BaseModel,Field
from typing import Literal,Optional
from utils.prompts_loader import memory_consolidator_prompt
from utils.logger_handler import logger as log
from model.model_factory import chat_model

# 先产出显式决策，再执行决策。
class ConsolidationDecision(BaseModel):
    action:Literal["add","merge","discard"] = Field(description="对 session note 的处理动作")
    session_note_text:str = Field(description="当前需要处理的 session note 的内容")
    target_global_id:Optional[str] = Field(default=None,description="当 action=merge 时,目标 global note 的 id")
    merged_text:Optional[str] = Field(default=None,description="当 action=add或merge时,最终保留的文本")
    merged_keywords:list[str] = Field(default_factory=list,description="最终要保留的关键字")
    category:Optional[str] = Field(default=None,description="最终要保留的类别")
    reason:Optional[str] = Field(default=None,description="做出决策的原因")

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

def validate_decision(decision: ConsolidationDecision) -> ConsolidationDecision:
    '''
    对提取的global note的结构做最后的检查
    :param decision:
    :return:
    '''
    valid_categories = {"home", "device", "preference", "demand", "other"}

    if decision.category and decision.category not in valid_categories:
        decision.category = "other"

    if decision.action == "merge" and not decision.target_global_id:
        decision.action = "discard"
        decision.reason = "merge action missing target_global_id"

    if decision.action in {"add", "merge"} and not decision.merged_text:
        decision.action = "discard"
        decision.reason = "missing merged_text"

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
    return validate_decision(result)

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

def apply_consolidation_plan(global_notes:list[dict],plan:ConsolidationPlan) -> list[dict]:
    '''
    将ConsolidationPlan 应用到 global note,返回更新后的global note
    :param global_notes:
    :param plan:
    :return:
    '''
    pass


if __name__ == "__main__":

    session_notes = [{
    "text": "房东自带的老扫地机不能拖地",
    "category": "device",
    "keywords": ["扫地机", "拖地"]
    }]


    global_note = [
    {
        "id": "global_0",
        "text": "房东自带老扫地机器，不能拖地",
        "category": "device",
        "keywords": ["扫地机", "不能拖地"]
    }
]

    plan = build_consolidation_plan(session_notes, global_note)
    print(plan)

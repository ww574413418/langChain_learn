from typing import Optional
from pydantic import BaseModel,Field
from model.model_factory import chat_model
from utils.prompts_loader import select_memory_plan_prompt

class MemoryPlan(BaseModel):
    need_profile:bool = Field(default=False,description="当前问题是否需要读取结构化用户信息")
    profile_fields:list[str] = Field(default=[],description="需要读取的profile 字段名,例如budge投,pets等")
    need_notes:bool = Field(default=False,description="当前问题是否需要检索长期记忆片段")
    note_query:Optional[str] = Field(default=None,description="用于检索 memory notes 的查询语句，不需要检索时为 null")
    top_k:int = Field(default=5,description="检索长期记忆片段的top_k")


def select_memory_plan(query:str) -> MemoryPlan:
    structure_llm = chat_model.with_structured_output(MemoryPlan)
    prompt = f"{select_memory_plan_prompt()} \n\n 用户问题:{query}"
    return structure_llm.invoke(prompt)

def pick_profile_fields(user_profile:dict,profile_fields:list[str]) -> dict:
    '''
    通过需要的用户画像字段,从用户画像中拿取字段信息
    :param user_profile: 用户画像
    :param profile_fields: 需要的用户画像字段
    :return:
    '''
    result = {}

    for field in profile_fields:
        if field not in user_profile:
            continue

        value = user_profile.get(field)

        if value in (None,"",[],{}):
            continue

        result[field] = value

    return result

def format_profile_subset(profile_subset: dict) -> str:
    if not profile_subset:
        return ""

    lines = ["当前问题相关的用户画像信息："]

    pets = profile_subset.get("pets", {})
    pet_parts = []
    if pets.get("cat") is not None:
        pet_parts.append(f"{pets['cat']}只猫")
    if pets.get("dog") is not None:
        pet_parts.append(f"{pets['dog']}条狗")
    if pet_parts:
        lines.append(f"- 宠物：{'，'.join(pet_parts)}")

    budget = profile_subset.get("budget")
    if budget is not None:
        lines.append(f"- 预算：{budget}")

    location = profile_subset.get("location")
    if location:
        lines.append(f"- 城市：{location}")

    preferences = profile_subset.get("preferences", {})
    pref_parts = []
    if preferences.get("low_noise"):
        pref_parts.append("低噪音")
    if preferences.get("easy_maintenance"):
        pref_parts.append("易维护")
    if pref_parts:
        lines.append(f"- 偏好：{'、'.join(pref_parts)}")

    home_features = profile_subset.get("home_features", {})
    env_parts = []
    if home_features.get("carpet"):
        env_parts.append("有地毯")
    if home_features.get("balcony"):
        env_parts.append("有阳台")
    if env_parts:
        lines.append(f"- 家庭环境：{'、'.join(env_parts)}")

    other_facts = profile_subset.get("other_facts", [])
    if other_facts:
        lines.append(f"- 其他长期信息：{'；'.join(other_facts)}")

    return "\n".join(lines)


if __name__ == "__main__":
    tests = [
        "推荐一款适合我家的扫拖一体机",
        "扫地机器人和洗地机有什么区别",
        "我家有猫，预算 2000，想买个安静点的",
    ]

    for q in tests:
        print("query:", q)
        print(select_memory_plan(q))
        print("-" * 30)


    from memory.profile_store import get_user_profile

    user_profile = get_user_profile("0001")
    plan = select_memory_plan("推荐一款适合我家的扫拖一体机")
    picked = pick_profile_fields(user_profile, plan.profile_fields)

    print("plan:", plan)
    print("picked_profile:", picked)

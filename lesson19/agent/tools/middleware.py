from typing import Callable
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger as log
from utils.prompts_loader import load_system_prompt,load_report_prompt
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from memory.profile_store import get_user_profile

@wrap_tool_call
def monitor_tool(
        # 请求数据的封装
        request: ToolCallRequest,
        # 执行函数本身
        handler:Callable[[ToolCallRequest],ToolMessage|Command]
) -> ToolMessage|Command:
    log.info(f"[tool_monitor]executed tool name:{request.tool_call["name"]}")
    log.info(f"[tool_monitor]parameters:{request.tool_call["args"]}")

    try:
        result = handler(request)
        if request.tool_call["name"] == "fill_context4report":
            request.runtime.context["report"] = True

        log.info(f"[tool_monitor]tool{request.tool_call["name"]} invocation success")
        return result
    except Exception as e:
        log.error(f"[tool_monitor]tool{request.tool_call["name"]}invocation error:{e}")
        raise e

@before_model
def log_before_model(
        state: AgentState,#整个agent的状态记录
        runtime: Runtime # 整个agent执行过程中的上下文信息
) -> None:
    log.info(f"[log_before_model] invoke model soon,there are {len(state["messages"])} info")
    log.debug(f"[log_before_model] {type(state["messages"][-1])} | {state["messages"][-1].content.strip()}")

    for i, msg in enumerate(state["messages"]):
        log.info(
            f"[state_message] index={i}, "
            f"type={type(msg).__name__}, "
            f"source={msg.additional_kwargs.get('lc_source') if hasattr(msg, 'additional_kwargs') else None}, "
            f"content={str(msg.content)[:80]}"
        )
    return None

def format_user_profile(user_profile: dict) -> str:

    lines = ["当前用户画像："]

    # 宠物
    pets = user_profile.get("pets", {})
    pet_parts = []
    cat_count = pets.get("cat")
    dog_count = pets.get("dog")

    if cat_count:
        pet_parts.append(f"{cat_count}只猫")
    if dog_count:
        pet_parts.append(f"{dog_count}条狗")

    if pet_parts:
        lines.append(f"- 宠物：{'，'.join(pet_parts)}")
    else:
        lines.append("- 宠物：暂无")

    # 预算
    budget = user_profile.get("budget")
    lines.append(f"- 预算：{budget}" if budget is not None else "- 预算：暂无")

    # 城市
    location = user_profile.get("location")
    lines.append(f"- 城市：{location}" if location else "- 城市：暂无")

    # 偏好
    preferences = user_profile.get("preferences", {})
    pref_parts = []
    if preferences.get("low_noise"):
        pref_parts.append("低噪音")
    if preferences.get("easy_maintenance"):
        pref_parts.append("易维护")

    if pref_parts:
        lines.append(f"- 偏好：{'、'.join(pref_parts)}")
    else:
        lines.append("- 偏好：暂无")

    # 家庭环境
    home_features = user_profile.get("home_features", {})
    env_parts = []
    if home_features.get("carpet"):
        env_parts.append("有地毯")

    if env_parts:
        lines.append(f"- 家庭环境：{'、'.join(env_parts)}")
    else:
        lines.append("- 家庭环境：暂无")

    return "\n".join(lines)


@dynamic_prompt # 每一次在生成提示词之前调用
def report_prompt_switch(request:ModelRequest):
    '''
    动态提示词切换
    :return:
    '''
    is_report= request.runtime.context.get("report",False)

    is_user_id = request.runtime.context.get("user_id",None)

    # 需要report提示模板
    if is_report:
        return load_report_prompt()

    base_prompt = load_system_prompt()

    if not is_user_id:
        return base_prompt

    user_profile = get_user_profile(is_user_id)
    profile_text = format_user_profile(user_profile)
    log.info(f"[report_prompt_switch] user profile:{profile_text}")

    return f"{base_prompt} \n\n{profile_text}"


@before_model
def trim_history(state: AgentState,runtime: Runtime):
    messages = state["messages"]

    log.info(f"[trim_history] trim history,current history length:{len(messages)}")

    # 历史消息小于6条 不处理
    if len(messages) <= 6:
        return None

    # 如果历史消息太长,只保留6条
    recent_message = messages[-6:]

    # RemoveMessage 告诉langgraph 旧的state里面的消息都删掉,在塞入剪裁后的消息
    return {
        ''' 
            RemoveMessage:表示一条“删除消息的更新指令”，不是普通聊天消息。它用于修改 state["messages"]。
            
            REMOVE_ALL_MESSAGES:不是一个消息 id 列表，也不是若干 id 的集合。它是一个特殊常量，意思接近于：
            “不是删某一条，而是删当前 state 里的全部消息。”
            
            *recent_message 列表里的每条消息逐个展开到当前列表里。
            这段代码的意思为:先清空当前 state 里的全部消息，再把我挑出来的最近 6 条消息重新写回去
        '''
        "messages":[
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *recent_message
        ]
    }
from typing import Callable
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger as log
from utils.prompts_loader import load_system_prompt,load_report_prompt

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

    return None

@dynamic_prompt # 每一次在生成提示词之前调用
def report_prompt_switch(request:ModelRequest):
    '''
    动态提示词切换
    :return:
    '''
    is_report= request.runtime.context.get("report",False)

    # 需要report提示模板
    if is_report:
        return load_report_prompt()

    return load_system_prompt()
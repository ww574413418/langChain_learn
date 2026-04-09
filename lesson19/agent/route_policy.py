from typing import Literal
from pydantic import BaseModel,Field
from utils.logger_handler import logger as log


class RoutePolicy(BaseModel):
    route: Literal["qa","rag_qa","report","tool_qa"]
    prompt_mode: Literal["main","report"]
    tool_mode: Literal["default","report","rag","tool"]
    memory_mode: Literal["full","light","none"]
    output_mode: Literal["answer","report_markdown","tool_answer"]

    description:str = Field(default="这条 route 的职责说明")


ROUTE_POLICY_MATRIX = {
    "qa": RoutePolicy(
        route="qa",
        prompt_mode="main",
        tool_mode="default",
        memory_mode="full",
        output_mode="answer",
        description="普通咨询、推荐、对比解释、基于 memory 的个性化回"
    ),
    "rag_qa": RoutePolicy(
        route="rag_qa",
        prompt_mode="main",
        tool_mode="rag"
        , memory_mode="light",
        output_mode="answer",
        description="知识库事实问答、功能说明、故障排查、使用说明"
    ),
    "report": RoutePolicy(
        route="report",
        prompt_mode="report",
        tool_mode="report",
        memory_mode="light",
        output_mode="report_markdown",
        description="月报、使用报告、结构化分析与保养建议",
    ),
    "tool_qa": RoutePolicy(
        route="tool_qa",
        prompt_mode="main",
        tool_mode="tool",
        memory_mode="none",
        output_mode="tool_answer",
        description="单步工具查询类问题，如天气",
    )
}

def get_route_policy(route:str) -> RoutePolicy:
    try:
        return ROUTE_POLICY_MATRIX[route]
    except KeyError as e:
        log.error(f"get_route_policy can not find route:{route}")
        raise ValueError(f"unsupported route: {route}") from e



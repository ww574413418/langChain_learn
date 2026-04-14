from typing import Literal,Optional,TypedDict

RouteLiteral = Literal["qa", "rag_qa", "report","tool_qa"]

class WorkflowState(TypedDict,total=False):
    query:str
    user_id:str
    thread_id:str

    route:RouteLiteral
    route_reason:str

    policy_tool_mode:str
    policy_prompt_mode:str
    policy_memory_mode:str
    policy_output_mode:str

    memory_write_done:bool
    memory_context:str

    rag_context:str
    report_month: str
    resolved_report_month: str
    report_month_resolution: str
    report_month_source: str
    report_data: str
    report_context: str

    tool_name: str
    tool_provided_args: dict
    tool_missing_required_args: list[str]
    tool_resolved_args: dict
    tool_resolution: dict
    tool_result: str

    final_answer:str

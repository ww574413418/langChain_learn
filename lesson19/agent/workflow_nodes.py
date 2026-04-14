from agent.request_parse.report_request_parser import extract_report_request_params
from agent.request_parse.tool_request_parser import ToolRequestParams, extract_tool_request_params
from agent.service.report_service import (
    fetch_external_report_data,
    format_report_data,
    resolve_report_month_for_user,
)
from agent.service.tool_service import (
    build_tool_execution_plan,
    fetch_weather_data,
    format_weather_result,
    render_weather_answer,
)
from agent.workflow_state import WorkflowState
from agent.route_policy import get_route_policy
from agent.query_router import route_query
from memory.memory_note_extractor import extract_candidate_notes
from memory.memory_note_store import append_session_note
from model.model_factory import chat_model
from utils.logger_handler import logger as log
from memory.profile_store import get_user_profile,update_user_profile
from memory.profile_extractor import extractor_userProfile_patch
from memory.memory_note_filter import filter_candidate_notes
from rag.mixedRecall.rag_service import rag_summary_service
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.prompts_loader import load_rag_answer_prompt,load_report_writer_prompt
from memory.memory_consolidator import consolidate_session_notes_with_plan


rag_answer_chain = (PromptTemplate.from_template(load_rag_answer_prompt())
                    | chat_model
                    | StrOutputParser())


report_writer_chain = (
    PromptTemplate.from_template(load_report_writer_prompt())
    | chat_model
    | StrOutputParser()
)

'''
router_node 不负责执行 workflow。
它只负责把“后面该怎么执行”写进 state。
'''
def router_node(state:WorkflowState) ->dict:
    query = state["query"]

    decision = route_query(query)
    policy = get_route_policy(decision.route)

    log.info(
        f"[router_node] route={decision.route}, "
        f"reason={decision.reason}, "
        f"tool_mode={policy.tool_mode}"
    )

    return {
        "route": decision.route,
        "route_reason": decision.reason,
        "policy_tool_mode": policy.tool_mode,
        "policy_prompt_mode": policy.prompt_mode,
        "policy_memory_mode": policy.memory_mode,
        "policy_output_mode": policy.output_mode,
    }


'''
这个节点是“写侧 memory”。
它不是给模型读上下文，它是在更新你的长期记忆材料。
'''
def memory_write_node(state:WorkflowState) ->dict:
    query = state["query"]
    user_id = state["user_id"]
    thread_id = state["thread_id"]

    # 获取当前用户的profile
    current_profile = get_user_profile(user_id)
    # 提取当前thread,可以保存到user prfile 的信息
    user_profile_patch = extractor_userProfile_patch(query, current_profile)

    if user_profile_patch:
        update_user_profile(user_id, user_profile_patch)

    # 提取当前thread,可以保存到session note的信息
    note_result = extract_candidate_notes(query)

    # 更新session note
    if note_result.notes:
        filtered_notes = filter_candidate_notes(
            [note.model_dump() for note in note_result.notes]
        )

        for decison in filtered_notes:
            if not decison.keep:
                continue
            if not decison.normalized_text:
                continue
            if not decison.category:
                continue

            append_session_note(
                user_id,
                thread_id,
                decison.normalized_text,
                decison.normalized_keywords,
                decison.category,
            )

    log.info("[memory_write_node] completed")

    return {
        "memory_write_done": True,
    }


'''
给 graph 做条件分支。
'''
def route_condition(state:WorkflowState) ->str:
    route = state["route"]

    if route == "qa":
        return "qa"
    elif route == "rag_qa":
        return "rag_qa"
    elif route == "report":
        return "report"
    elif route == "tool_qa":
        return "tool_qa"
    else:
        raise Exception(f"[route_condition] unknown route: {route}")


def rag_retrieve_node(state:WorkflowState) -> dict:
    query = state["query"]
    rag_context = rag_summary_service.rag_summary(query)
    log.info(
        f"[rag_retrieve_node] retrieved context length={len(rag_context) if rag_context else 0}"
    )

    # if rag_context:
    #     state["rag_context"] = rag_context

    return {
        "rag_context": rag_context or "",
    }


def rag_answer_node(state:WorkflowState)->dict:

    query = state["query"]
    rag_context = state.get("rag_context","")
    memory_context = state.get("memory_context","")

    final_answer = rag_answer_chain.invoke({
        "query":query,
        "rag_context":rag_context,
        "memory_context":memory_context
    })

    log.info(f"[rag_answer_node] final_answer={final_answer}")

    return {
        "final_answer":final_answer.strip() if final_answer else ""
    }


def consolidate_node(state: WorkflowState) -> dict:
    user_id = state["user_id"]
    thread_id = state["thread_id"]
    consolidate_session_notes_with_plan(user_id, thread_id)
    log.info("[consolidate_node] completed")

    return {}

def report_fetch_node(state:WorkflowState) -> dict:
    '''
    职责只做“数据准备”，不要直接写报告。
    :param user_id:来自state
    :param month: 如果user query中没有设计month,调用函数获取month
    :param external_data:直接调用函数获取
    :return:
    '''
    user_id = state["user_id"]
    query = state["query"]

    report_request_params = extract_report_request_params(query)
    requested_month = report_request_params.target_month
    month_source = report_request_params.month_source

    resolved_month, month_resolution = resolve_report_month_for_user(
        user_id=user_id,
        requested_month=requested_month,
        month_source=month_source,
    )

    report_data = fetch_external_report_data(user_id, resolved_month)
    report_data_text = format_report_data(report_data)

    # report_context = rag_summary_service.rag_summary("扫地机器人月度报告解读、保养建议、耗材维护建议")

    log.info(
        f"[report_fetch_node] month={requested_month}, "
        f"month_source={month_source}, "
        f"report_data_len={len(report_data_text)}, "
        # f"report_context_len={len(report_context) if report_context else 0}"
    )

    return {
        "report_month": requested_month,
        "report_month_source": month_source,
        "resolved_report_month": resolved_month,
        "report_month_resolution": month_resolution,
        "report_data": report_data_text,
        "report_context": "",
    }


def report_writer_node(state:WorkflowState):
    '''
    职责只做“根据已准备好的材料写报告”。
    '''
    query = state["query"]
    requested_report_month = state.get("report_month", "")
    resolved_report_month = state.get("resolved_report_month", "")
    report_month_resolution = state.get("report_month_resolution", "")
    report_data = state.get("report_data", "")
    report_context = state.get("report_context", "")

    final_answer = report_writer_chain.invoke(
        {
            "query": query,
            "requested_report_month": requested_report_month,
            "resolved_report_month": resolved_report_month,
            "report_month_resolution": report_month_resolution,
            "report_data": report_data,
            "report_context": report_context,
        }
    )

    log.info("[report_writer_node] completed")

    return {
        "final_answer": final_answer.strip() if final_answer else "",
    }


def tool_request_parse_node(state: WorkflowState) -> dict:
    query = state["query"]
    request = extract_tool_request_params(query)

    log.info(
        f"[tool_request_parse_node] tool_name={request.tool_name}, "
        f"provided_args={request.provided_args}, "
        f"missing_required_args={request.missing_required_args}"
    )

    return {
        "tool_name": request.tool_name,
        "tool_provided_args": request.provided_args,
        "tool_missing_required_args": request.missing_required_args,
    }


def tool_execute_node(state: WorkflowState) -> dict:
    user_id = state["user_id"]

    request = ToolRequestParams(
        tool_name=state["tool_name"],
        provided_args=state.get("tool_provided_args", {}),
        missing_required_args=state.get("tool_missing_required_args", []),
    )

    plan = build_tool_execution_plan(user_id=user_id, request=request)

    if plan.missing_required_args:
        return {
            "tool_resolved_args": plan.resolved_args,
            "tool_resolution": plan.resolution,
            "final_answer": "请告诉我你想查询哪个城市的天气。",
        }

    city = plan.resolved_args["city"]
    weather_data = fetch_weather_data(city)
    city_resolution = plan.resolution["city"]

    tool_result = format_weather_result(weather_data)
    final_answer = render_weather_answer(weather_data, city_resolution)

    log.info(
        f"[tool_execute_node] tool_name={plan.tool_name}, "
        f"resolved_args={plan.resolved_args}, "
        f"resolution={plan.resolution}"
    )

    return {
        "tool_resolved_args": plan.resolved_args,
        "tool_resolution": plan.resolution,
        "tool_result": tool_result,
        "final_answer": final_answer,
    }

if __name__ == "__main__":
    state = {
        "query": "帮我生成这个月的用户使用报告",
        "user_id": "1007",
        "thread_id": "workflow_report_fetch_test_001",
    }

    result = report_fetch_node(state)
    print(result["report_month"])
    print(result["report_month_source"])
    print(result["report_data"])
    print(result["report_context"][:300])

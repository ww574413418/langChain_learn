from model.model_factory import  chat_model
from utils.prompts_loader import load_system_prompt
from agent.tools.agent_tools import (get_weather,get_user_id,get_user_location,
                                     get_current_month,fetch_external_data,rag_summarize_rrf)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch,persist_thread_summary
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage,ToolMessage
from langchain.agents.middleware import SummarizationMiddleware
from memory.profile_extractor import extractor_userProfile_patch
from memory.profile_store import update_user_profile, get_user_profile
from memory.memory_note_store import append_session_note
from memory.memory_note_extractor import extract_candidate_notes
from utils.logger_handler import logger as log
from memory.memory_consolidator import consolidate_session_notes_with_plan
from memory.memory_note_filter import filter_candidate_notes
import uuid
from agent.query_router import route_query
from agent.route_policy import get_route_policy

class ReactAgent:

    def __init__(self):

        self.default_tools = [
            rag_summarize_rrf
        ]

        self.default_agent = create_agent(
            model = chat_model,
            system_prompt = load_system_prompt(),
            tools = self.default_tools,
            middleware = [monitor_tool,
                          report_prompt_switch,
                          # 先总结旧消息 再保留最近几条
                          SummarizationMiddleware(
                              model = chat_model,
                              trigger=("tokens",1000), # 历史消息大约到 4000 tokens 时，开始总结。
                              keep=("messages",5) # 保留最近 20 条消息原文，较早部分压缩成 summary。
                          ),
                          log_before_model,
                          persist_thread_summary],
            checkpointer= InMemorySaver() # 引入临时会话保存点
        )

        self.report_tools = [
            get_weather,
            get_user_location,
            get_user_id,
            get_current_month,
            fetch_external_data,
            rag_summarize_rrf,
        ]

        self.report_agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=self.report_tools,
            middleware=[
                monitor_tool,
                report_prompt_switch,
                # 先总结旧消息 再保留最近几条
                SummarizationMiddleware(
                    model=chat_model,
                    trigger=(
                        "tokens",
                        1000,
                    ),  # 历史消息大约到 4000 tokens 时，开始总结。
                    keep=(
                        "messages",
                        5,
                    ),  # 保留最近 20 条消息原文，较早部分压缩成 summary。
                ),
                log_before_model,
                persist_thread_summary,
            ],
            checkpointer=InMemorySaver(),
        )

    def execute_stream(self, query: str, thread_id: str, user_id: str):
        # 1.router
        route_decision = route_query(query)
        route = route_decision.route
        policy = get_route_policy(route)

        log.info(f"[route_policy] {policy}")

        # 2.shared pre-write memory
        current_profile = get_user_profile(user_id)
        user_profile_patch = extractor_userProfile_patch(query,current_profile)

        if user_profile_patch:
            update_user_profile(user_id,user_profile_patch)

        # 获取用户输入中需要长期记忆的内容
        note_result = extract_candidate_notes(query)
        if note_result.notes:
            # 对note进行过滤
            filtered_notes = filter_candidate_notes(
                [note.model_dump() for note in note_result.notes]
            )

            for decision in filtered_notes:
                if not decision.keep:
                    continue
                if not decision.normalized_text:
                    continue
                if not decision.category:
                    continue

                append_session_note(user_id,
                                    thread_id,
                                    decision.normalized_text,
                                    decision.normalized_keywords,
                                    decision.category)
        # 3.disaptch
        if policy.tool_mode == "report":
            yield from self._run_report_path(query,thread_id,user_id,route)
        elif policy.tool_mode == "rag_qa":
            yield from self._run_rag_qa_path(query,thread_id,user_id,route)
        else:
            yield from self._run_default_path(query, thread_id, user_id, route)

        # 4. consolidate 每次只一次之后,将session note整理成global note
        consolidate_session_notes_with_plan(user_id,thread_id)

    def _run_report_path(self, query, thread_id, user_id, route):
        log.info(f"_run_report_path query:{query}")
        input_dic = {"messages": [{"role": "user", "content": query}]}
        # 用于保存历史信息
        config = {
            "run_name": "lesson19_customer_agent",
            "tags": ["lesson19", "agent", "stream", "customer-service"],
            "metadata": {
                "app": "lesson19",
                "entrypoint": "streamlit",
                "user_id": str(user_id),
                "thread_id": str(thread_id),
            },
            "configurable": {"thread_id": thread_id},
        }
        res = self.report_agent.stream(
            input_dic,
            stream_mode="values",
            context={
                "user_id": user_id,
                "thread_id": thread_id,
                "route": route,
            },
            config=config,
        )

        for chunk in res:
            latest_message = chunk["messages"][-1]
            print(type(latest_message), latest_message)
            # 检测是否是调用方法的信息
            if isinstance(latest_message, ToolMessage):
                continue

            if isinstance(latest_message, AIMessage):
                # 是ai message 但是是ai在决定如何使用tool的信息,跳过
                if latest_message.tool_calls:
                    continue
                if latest_message.content:
                    yield latest_message.content.strip() + "\n"

    def _run_default_path(self, query, thread_id, user_id, route):
        log.info(f"_run_default_path query:{query}")
        input_dic = {"messages": [{"role": "user", "content": query}]}
        # 用于保存历史信息
        config = {
            "run_name": "lesson19_customer_agent",
            "tags": ["lesson19", "agent", "stream", "customer-service"],
            "metadata": {
                "app": "lesson19",
                "entrypoint": "streamlit",
                "user_id": str(user_id),
                "thread_id": str(thread_id),
            },
            "configurable": {"thread_id": thread_id},
        }
        # context={"report":False} 就是切换prompt的标志
        res = self.default_agent.stream(
            input_dic,
            stream_mode="values",
            context={
                "user_id": user_id,
                "thread_id": thread_id,
                "route": route,
            },
            config=config,
        )

        for chunk in res:
            latest_message = chunk["messages"][-1]
            print(type(latest_message), latest_message)
            # 检测是否是调用方法的信息
            if isinstance(latest_message, ToolMessage):
                continue

            if isinstance(latest_message, AIMessage):
                # 是ai message 但是是ai在决定如何使用tool的信息,跳过
                if latest_message.tool_calls:
                    continue
                if latest_message.content:
                    yield latest_message.content.strip() + "\n"


agent = ReactAgent()

if __name__ == '__main__':
    thread_id = f"sum_test_{uuid.uuid4().hex[:8]}"
    print("thread_id:", thread_id)

    # print("---" * 20)
    # res = agent.execute_stream("帮我写一份用户报告。", "sum_test_4264af7c", "0001")
    # for chunk in res:tool
    #     print(chunk, end="", flush=True)

    # print("---" * 20)
    # res = agent.execute_stream("预算2000，想买个安静点的，帮我推荐", "sum_test_4264af7c", "0001")
    # for chunk in res:
    #     print(chunk, end="", flush=True)
    #
    # print("---" * 20)
    res = agent.execute_stream(
        "扫拖一体机器人可以只扫地不拖地吗？", "sum_test_4264af7c", "0001"
    )
    for chunk in res:
        print(chunk, end="", flush=True)

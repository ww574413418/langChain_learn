from model.model_factory import  chat_model
from utils.prompts_loader import load_system_prompt
from agent.tools.agent_tools import (rag_summarize,get_weather,get_user_id,get_user_location,
                                     get_current_month,fetch_external_data,fill_context4report,
                                     rag_summarize_mixRecall,rag_summarize_rrf)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch,trim_history
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

class ReactAgent:

    def __init__(self):
        self.agent = create_agent(
            model = chat_model,
            system_prompt = load_system_prompt(),
            tools = [get_weather,get_user_id,get_user_location,get_current_month,
                          fetch_external_data,fill_context4report,rag_summarize_rrf],
            middleware = [monitor_tool,
                          report_prompt_switch,
                          # 先总结旧消息 再保留最近几条
                          SummarizationMiddleware(
                              model = chat_model,
                              trigger=("tokens",1000), # 历史消息大约到 4000 tokens 时，开始总结。
                              keep=("messages",5) # 保留最近 20 条消息原文，较早部分压缩成 summary。
                          ),
                          log_before_model],
            checkpointer= InMemorySaver() # 引入临时会话保存点
        )

    def execute_stream(self,query:str,thread_id:str,user_id:str):
        input_dic = {
            "messages":
                [{"role": "user", "content": query}]
        }
        # 用于保存历史信息
        config = {
            "configurable":{
                "thread_id":thread_id
            }
        }

        current_profile = get_user_profile(user_id)
        userProfile_patch = extractor_userProfile_patch(query,current_profile)

        if userProfile_patch:
            update_user_profile(user_id,userProfile_patch)

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

        # context={"report":False} 就是切换prompt的标志
        res = self.agent.stream(input_dic,stream_mode="values",context={"report":False,"user_id":user_id,"thread_id":thread_id},config=config)

        for chunk in res:
            latest_message = chunk["messages"][-1]
            print(type(latest_message), latest_message)
            # 检测是否是调用方法的信息
            if isinstance(latest_message,ToolMessage):
                continue

            if isinstance(latest_message,AIMessage):
                # 是ai message 但是是ai在决定如何使用tool的信息,跳过
                if latest_message.tool_calls:
                    continue
                if latest_message.content:
                    yield latest_message.content.strip() + "\n"
        # 每次只一次之后,将session note整理成global note
        consolidate_session_notes_with_plan(user_id,thread_id)
agent = ReactAgent()

if __name__ == '__main__':
    # res = agent.execute_stream("扫地机器人和洗地机有什么区别","123123","0001")
    # for chunk in res:
    #     print(chunk,end="",flush=True)

    print("---"*20)
    res = agent.execute_stream("想买个安静点的扫地机", "123123", "0001")
    for chunk in res:
        print(chunk, end="", flush=True)

    print("---" * 20)
    res = agent.execute_stream("房东自带的老扫地机不能拖地", "123123", "0001")
    for chunk in res:
        print(chunk, end="", flush=True)

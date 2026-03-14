from model.model_factory import  chat_model
from utils.prompts_loader import load_system_prompt
from agent.tools.agent_tools import (rag_summarize,get_weather,get_user_id,get_user_location,
                                     get_current_month,fetch_external_data,fill_context4report)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch
from langchain.agents import create_agent

class ReactAgent:

    def __init__(self):
        self.agent:ReactAgent = create_agent(
            model = chat_model,
            system_prompt = load_system_prompt,
            tools = [rag_summarize,get_weather,get_user_id,get_user_location,get_current_month,
                          fetch_external_data,fill_context4report],
            middleware = [monitor_tool,log_before_model,report_prompt_switch]
        )

    def execute_stream(self,query:str):
        input_dic = {
            "messages":[
                {"role":"user","content":query}
            ]
        }
        # context={"report":False} 就是切换prompt的标志
        res = self.agent.stream(input_dic,stream_mode="values",context={"report":False})

        for chunk in res:
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


ReactAgent = ReactAgent()

if __name__ == '__main__':
    res = ReactAgent.execute_stream("APP无法连接机器人怎么办？")
    for chunk in res:
        print(chunk,end="",flush=True)



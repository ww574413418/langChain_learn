import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key=os.getenv("SILICON_FLOW")
base_url=os.getenv("SILICON_URL")

llm = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model="Qwen/Qwen3-VL-32B-Thinking",
)

@tool(description="获取体重,返回值是整数,单位kg")
def get_weight()->int:
    return 72

@tool(description="获取身高,返回值是整数,单位cm")
def get_height()->int:
    return 173

agent = create_agent(
    model=llm,
    tools=[get_weight,get_height],
    system_prompt='''你是一个严格遵守ReAct框架的智能体,必须按照[thought->action->observation]的流程来解决问题,
    切每一轮仅能思考并调用一个工具,禁止单词思考调用多个工具.并且告诉我思考过程,工具调用的原因,按照[thought->action->observation]
    的结构告诉我
    '''
)

res = agent.stream(
    {
        "messages":[
            {"role":"user","content":"请计算我体重的BMI"}
        ]
    },
    stream_mode="values"
)

for chunk in res:
    latest_message = chunk["messages"][-1]
    print(type(latest_message).__name__,latest_message.content)

    try:
        if latest_message.tool_calls:
            print(f"工具调用:{[ tc["name"] for tc in latest_message.tool_calls]}")
    except AttributeError as e:
        pass


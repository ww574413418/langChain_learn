import os
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key=os.getenv("SILICON_FLOW")
base_url=os.getenv("SILICON_URL")

llm = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model="Qwen/Qwen3-VL-32B-Thinking",
)

@tool(description="查询股票价格")
def query_price(name:str):
    return f"{name}股票的价格是:20元"

@tool(description="查询股票信息")
def get_info(name:str):
    return f"{name}股票的信息如下:股票代码:000001,股票名称:上证指数."


agent = create_agent(
    model=llm,
    tools=[query_price,get_info],
    system_prompt="你是一个股票查询助手,请告知我思考过程,展示你为什么要调用这个工具"
)

res = agent.stream(
    {
    "messages":[
        {"role":"user","content":"微软的股价是多少?并介绍一下"}
    ],
    },
    stream_mode="values"
)

for chunk in res:
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        print(type(latest_message).__name__,":",latest_message.content)
    try:
        if latest_message.tool_calls:
            print(f"工具调用:{[ tc["name"] for tc in latest_message.tool_calls]}")
    except AttributeError as e:
        pass
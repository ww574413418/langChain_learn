import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key=os.getenv("SILICON_FLOW")
base_url=os.getenv("SILICON_URL")

chat = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model="Qwen/Qwen3-VL-32B-Thinking",
)

@tool(description="查询天气")
def query_weather(location: str) -> str:
    '''
    查询天气
    :param location:
    :return:
    '''
    return "晴天"


agent = create_agent(
    model=chat,
    tools=[query_weather],
    system_prompt="你是一个天气查询助手"
)

res = agent.invoke({
    "messages":[
        {"role":"user","content":"明天深圳天气如何?"},
    ]
})

for msg in res["messages"]:
    print(type(msg).__name__,msg.content)


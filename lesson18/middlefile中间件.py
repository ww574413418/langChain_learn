import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent,AgentState
from langchain.agents.middleware import before_agent, after_agent, before_model, after_model, wrap_model_call, \
    wrap_tool_call
from langgraph.runtime import Runtime


load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key=os.getenv("SILICON_FLOW")
base_url=os.getenv("SILICON_URL")

llm = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model="Qwen/Qwen3-VL-32B-Thinking",
)

@tool(description="传入位置信息,返回天气")
def query_weather(location: str) -> str:
    '''
    查询天气
    :param location:
    :return:
    '''
    return f"{location}是晴天"

'''
1.agent执行前
2.agent执行后
3.model执行前
4.model执行后
5.工具执行中
6.模型执行中
'''

@before_agent
def log_before_agnet(state: AgentState, runtime: Runtime) -> None:
    print(f"agent执行前,附带有{len(state["messages"])}条信息")

@after_agent
def log_after_agent(state:  AgentState, runtime: Runtime) -> None:
    print(f"agent执行后,附带有{len(state['messages'])}条信息")

@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    print(f"模型执行前,附带有{len(state['messages'])}条信息")

@after_model
def log_after_model(state:  AgentState, runtime: Runtime) -> None:
    print(f"模型执行后,附带有{len(state['messages'])}条信息")

@wrap_model_call
def wrap_model_call(request,handler):
    '''
    模型执行中
    :param request:
    :param handler:
    :return:
    '''
    print("模型调用了")
    return handler(request)

@wrap_tool_call
def wrap_tool_call(request,handler):
    '''
    工具执行中
    :param request:
    :param handler:
    :return:
    '''
    print(f"调用了{request.tool_call["name"]}工具")
    print(f"传入参数{request.tool_call["args"]}")
    return handler(request)


agent = create_agent(
    model=llm,
    tools=[query_weather],
    system_prompt="你是一个天气查询助手",
    middleware=[
        log_before_agnet,
        log_after_agent,
        log_before_model,
        log_after_model,
        wrap_model_call,
        wrap_tool_call
    ]
)

res = agent.invoke({
    "messages":[
        {"role":"user","content":"明天深圳天气如何?"},
    ]
})

for msg in res["messages"]:
    print(type(msg).__name__,msg.content)


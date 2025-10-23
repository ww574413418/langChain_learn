# 使用langchain 运行 chat
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

llm = ChatOpenAI(
    api_key = api_key,
    base_url = base_url,
    model= "gemini-2.5-flash",
    temperature = 0.5,#0稳重,1创意
    max_tokens = 100
)
messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请介绍一下caterpillar这个公司"),
]
response = llm.invoke(messages)
print(response)

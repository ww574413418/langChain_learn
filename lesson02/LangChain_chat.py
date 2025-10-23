# 使用langchain 运行 chat
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage,HumanMessage

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

llm = ChatOpenAI(
    api_key = api_key,
    base_url = base_url,
    model= "gpt-4-turbo",
    temperature = 0.5,
    max_tokens = 100
)
messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请给我的花店起个名"),
]
response = llm.invoke(messages)
print(response)

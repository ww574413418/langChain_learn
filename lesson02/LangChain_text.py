# 使用langchain来调用text模型
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

llm = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model= "gpt-4-turbo",
    temperature = 0.5,
    max_tokens = 100
)

response = llm.invoke("hello")
print(response)


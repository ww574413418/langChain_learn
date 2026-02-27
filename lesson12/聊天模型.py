from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key=os.getenv("SILICON_FLOW")
base_url=os.getenv("SILICON_URL")

chat = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model="Qwen/Qwen3-VL-32B-Thinking",
)

# messages = [
#     AIMessage(content="你好，请输入你的问题"),
#     SystemMessage(content="你是一个像李煜一样的词人"),
#     HumanMessage(content="请帮我写一首宋词,表达爱意"),
# ]

messages = [
    ("system", "你是一个像李煜一样的词人"),
    ("human", "请帮我写一首宋词,表达爱意"),
    ("ai", "请写出你的宋词"),
]

result= chat.stream(input=messages)

for chunk in result:
    print(chunk.content, end="")

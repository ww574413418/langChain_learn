from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key=os.getenv("SILICON_FLOW")
base_url=os.getenv("SILICON_URL")

model = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model="Qwen/Qwen3-VL-32B-Thinking",
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个夸夸大师，专门夸赞别人"),
    MessagesPlaceholder(variable_name="history"),
    ("ai", "请你输出一段夸夸"),
    ("human", "我是个傻逼"),
])

history_content = [
    ("human","逗比一点的夸夸"),
    ("ai","简短,真实一些"),
]

print(prompt.invoke({"history": history_content}).to_string())

chain = prompt | model
res = chain.stream({"history": history_content})

for chunk in res:
    print(chunk.content, end="", flush=True)

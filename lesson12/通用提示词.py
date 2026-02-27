from langchain_core.prompts import PromptTemplate
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

template = PromptTemplate.from_template("我的邻居姓{lastname},生了个{gender},请你帮我取一个名字,简单回答")


chain = template | model

res = chain.invoke({"lastname":"张","gender":"男"})
print(res.content)


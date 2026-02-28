# 将模型的输出结果从默认的AIMessage编程str
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
import os
from dotenv import load_dotenv

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key=os.getenv("SILICON_FLOW")
base_url=os.getenv("SILICON_URL")

model = ChatOpenAI(
    model="tencent/Hunyuan-MT-7B",
    api_key=api_key,
    base_url=base_url,
)

prompt = PromptTemplate.from_template('''
    我的名字叫:{fullname},我有一个{gender}孩,请你帮我取一个名字
''')

parser = StrOutputParser()

chain = prompt | model | parser | model

res:AIMessage = chain.stream({"fullname":"张三","gender":"男"})

for chunk in res:
    print(chunk.content, end="", flush=True)
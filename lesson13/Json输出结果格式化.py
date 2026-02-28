#将模型输出的结果转成json
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

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


prompt1 = PromptTemplate.from_template('''
    我的邻居姓:{lastname},刚生了一个{gender},请起名字,并封装成json格式返回给我,
    要求key是name,value就是你起的名字.请严格遵守格式要求.
''')

prompt2 = PromptTemplate.from_template('''
    姓名:{name},请你帮我分析他的含义
''')

# 接受AImessage格式数据,返回python字典
jsonParser = JsonOutputParser()

chain = prompt1 | model | jsonParser | prompt2 | model

res = chain.stream({"lastname":"张","gender":"男"})

for chunk in res:
    print(chunk.content, end="", flush=True)


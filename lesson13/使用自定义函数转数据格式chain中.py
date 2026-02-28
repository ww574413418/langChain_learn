from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
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

my_func1 = RunnableLambda(lambda x: {"name":x.content})
strParse = StrOutputParser()

# RunnableLambda也支持普通函数
def extra_name(x):
    return {"name":x.content}

my_func2 = RunnableLambda(extra_name)

# RunnableLambda也支持传入现了 __call__ 的类实例
class NameExtrator:
    def __call__(self,x):
        return {"name":x.content}

my_func3 = RunnableLambda(NameExtrator())


prompt1 = PromptTemplate.from_template("我的邻居姓{lastname},生了个{gender}孩,请你帮我取一个名字,简单回答")
prompt2 = PromptTemplate.from_template("姓名{name},请帮我解析含义")

chain = prompt1 | model | my_func2 | prompt2 | model
# chain = prompt1 | model | (lambda x:{"name":x.content}) | prompt2 | model
res = chain.stream(input={"lastname":"张","gender":"男"})
for chunk in res:
    print(chunk.content, end="")
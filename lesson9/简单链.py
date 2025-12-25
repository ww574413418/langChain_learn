from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

template = "{flower}的花语是?"

llm = ChatOpenAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=api,
    base_url=base_url,
)

# 创建链（LCEL / RunnableSequence）
prompt = PromptTemplate.from_template(template)
chain = prompt | llm

# 传参必须是 dict，键名要和模板变量一致
result = chain.invoke({"flower": "rose"})
print(result)
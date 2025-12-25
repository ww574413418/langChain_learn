import os
from dotenv import load_dotenv

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")

api = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# 第一个chain
template = """
你是一个植物学家。给定花的名称和类型，你需要为这种花写一个200字左右的介绍。
花名: {name}
颜色: {color}
植物学家: 这是关于上述花的介绍:
"""

llm = ChatOpenAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key=api,
    base_url=base_url,
)

prompt = PromptTemplate.from_template(template=template)

chain1 = prompt | llm | StrOutputParser()

# 第二个chain

template = """
你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇200字左右的评论。

鲜花介绍:
{introduction}
花评人对上述花的评论:
"""

prompt = PromptTemplate.from_template(template=template)

chain2 = prompt | llm |StrOutputParser()

# 第三个chain
template = """
你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300字左右。

鲜花介绍:
{introduction}
花评人对上述花的评论:
{review}

社交媒体帖子:
"""

prompt = PromptTemplate.from_template(template=template)

chain3 = prompt | llm | StrOutputParser()


# 将3个链串起来
chain = (
    {"introduction":chain1}
    # RunnablePassthrough RunnablePassthrough.assign(...)：在不丢原数据的情况下，新增字段
    # 输入是 dict，输出还是 dict，并且在原 dict 上“追加/计算”新键值。
    #你这里的目标是：
	# •	保留 introduction
	# •	计算出 review
	# •	最终把两者一起传给 chain3（因为 chain3 模板里同时用 {introduction} 和 {review}）
    #  itemgetter("introduction")：从字典里取某个字段
    | RunnablePassthrough.assign(review = (itemgetter ("introduction") | chain2))
    | RunnablePassthrough.assign(social_post_text = chain3)
)

result = chain.invoke({"name":"rose","color":"black"})
print(result)
print(result["introduction"])
print(result["review"])
print(result["social_post_text"])

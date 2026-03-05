from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
'''
用户提问 + 向量库中检索的参考资料
'''

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key=os.getenv("SILICON_FLOW")
base_url=os.getenv("SILICON_URL")

model = ChatOpenAI(
    model="tencent/Hunyuan-MT-7B",
    api_key=api_key,
    base_url=base_url,
)

prompt = ChatPromptTemplate.from_messages([
    ("system","以我提供的已知资料为主,简洁专业的回答用户的问题.参考资料{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human","用户提问:{input}")
])

vector_store = InMemoryVectorStore(
    embedding=OllamaEmbeddings(model="qwen3-embedding:0.6b"),
)

vector_store.add_texts(
    texts=["减肥就是要少吃多练", "在减脂期间吃东西很重要", "清淡少油控制卡路里摄入并运动起来", "跑步是很好的运动哦"],
    ids=["1", "2", "3", "4"]
)

input_text = "要如何减肥"

result = vector_store.similarity_search(
    query=input_text,
    k=2
)

refernce_text = "[" + ",".join([doc.page_content for doc in result]) + "]"

def print_prompt(prompt:AIMessage):
    print(prompt.to_string())
    print("="*20)
    return prompt

chain = prompt | print_prompt | model

res = chain.stream({"input":input_text, "context":refernce_text,"history":[]})

for chunk in res:
    print(chunk.content, end="",flush=True)




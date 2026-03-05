'''
将向量检索的结果加入到chain中
'''
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_community.document_loaders import CSVLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
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

loader = CSVLoader(
    file_path="info.csv",
    encoding="utf-8",
    source_column="source"
)
documents = loader.load()

vect_store = InMemoryVectorStore(
    embedding=OllamaEmbeddings(model="qwen3-embedding:0.6b"),
)

vect_store.add_texts(
    texts=["减肥就是要少吃多练", "在减脂期间吃东西很重要", "清淡少油控制卡路里摄入并运动起来", "跑步是很好的运动哦"],
    ids=["1","2","3","4"]
)


prompt = ChatPromptTemplate.from_messages([
    ("system","以我提供的已知资料为主,简洁专业的回答用户的问题.参考资料{context}"),
    MessagesPlaceholder("history"),
    ("human","用户输入:{input}")
])

input_text = "我该如何减肥"

retriever = vect_store.as_retriever(search_kwargs={"k":2})


def print_prompt(prompt:AIMessage) -> AIMessage:
    print(prompt.to_string())
    print("="*20)
    return prompt

def format_fun(docs:list[Document]) -> str:
    if not docs:
        return "无相关参考资料"
    formatted_str = "[" + ",".join(doc.page_content for doc in docs) + "]"
    return formatted_str

# 将参考信息加入chain中
chain = (
    {"input":RunnablePassthrough(),
     "context":retriever | format_fun,
     "history":RunnableLambda(lambda _:[])
     } | prompt | print_prompt | model
)

res = chain.stream(input_text)
for chunk in res:
    print(chunk.content, end="", flush=True)


from dotenv import load_dotenv
import os

# load api key
load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key = os.getenv("SILICON_FLOW")
base_url = os.getenv("SILLICON_URL")
DB_PATH = "./local_qdrant"

# 1.Load 导入Document Loaders
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader

# 加载Documents
base_dir = os.path.join(os.path.dirname(__file__), "oneFlowers")

documents = []
for file in os.listdir(base_dir):
    # 构建完整的文件路径
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())


# 2.Split 将Documents切分成块以便后续进行嵌入和向量存储
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
chunked_documents = text_splitter.split_documents(documents)


# 3.Store 将分割嵌入并存储在矢量数据库Qdrant中
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional

embedding = OpenAIEmbeddings(
    api_key = api_key,
    base_url = base_url,
    model="BAAI/bge-large-zh-v1.5"
)

# vectorstore 向量数据库
vectorstore = Qdrant.from_documents(
    documents=chunked_documents, # 以分块的文档
    embedding=embedding, # 用OpenAI的Embedding Model做嵌入
    location=":memory:",  # in-memory 存储
    # location="./local_qdrant",  # ✅ 本地持久化
    collection_name="my_documents",# 指定collection_name
    batch_size = 16
)

# 4. Retrieval 准备模型和Retrieval链
import logging # 导入Logging工具
from langchain_openai import ChatOpenAI # ChatOpenAI模型
#MultiQueryRetriever 的内部机制是：它会让 LLM（例如 Gemini）自动改写用户问题，
# 生成多种语义等价的提问（比如“公司假期政策是什么？”、“放假规定是怎样的？”），
# 从而从向量数据库里找到更全面的文档。
from langchain_classic.retrievers import MultiQueryRetriever
# RetrievalQA链能让模型在“查资料 + 生成回答”两步之间自动衔接。
from langchain_classic.chains import RetrievalQA


# 设置Logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

deepbricks_api = os.getenv("API_KEY")
deepbricks_url = os.getenv("BASE_URL")

# 实例化一个大模型工具
llm = ChatOpenAI(
    api_key = deepbricks_api,
    base_url = deepbricks_url,
    model_name="gemini-2.5-flash",
    temperature=0.5
)

# 实例化一个MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# 实例化一个RetrievalQA链,将大模型和retriever_from_llm链接在一块
qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever_from_llm)


# 5. Output 问答系统的UI实现
from flask import Flask, request, render_template

app = Flask(__name__)  # Flask APP


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 接收用户输入作为问题
        question = request.form.get('question')

        # RetrievalQA链 - 读入问题，生成答案
        result = qa_chain({"query": question})

        # 把大模型的回答结果返回网页进行渲染
        return render_template('index.html', result=result)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8000)

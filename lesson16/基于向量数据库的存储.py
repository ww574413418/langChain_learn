from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma

embed = OllamaEmbeddings(model="qwen3-embedding:0.6b")

vec_store = Chroma(
    collection_name="test",#g给当前的向量存储起名字
    embedding_function=embed,
    persist_directory="./chroma_db"#指定数据存放的文件夹
)

loader = CSVLoader(
    file_path="info.csv",
    encoding="utf-8",
    source_column="source"
)

documents = loader.load()

vec_store.add_documents(
    documents=documents,
    ids=[str(i) for i in range(len(documents))]
)
result = vec_store.similarity_search(
    query="python是不是简单易学",
    k=2,
    filter={"source":"传智教育"}#过滤数据来源
)

print(result)

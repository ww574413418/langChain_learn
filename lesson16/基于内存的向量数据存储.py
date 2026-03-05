from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import CSVLoader

embed = OllamaEmbeddings(model="qwen3-embedding:0.6b")

vector_store = InMemoryVectorStore(
    embedding=embed,
)

loader = CSVLoader(
    file_path="info.csv",
    encoding="utf-8",
    source_column="source"
)
 
documents = loader.load()

# 向量化文档
vector_store.add_documents(
    documents=documents,
    ids=[str(i) for i in range(len(documents))] # 给添加的文件添加id
)

# 删除id为0 和 1 的文档
vector_store.delete(ids=["0", "1"])

result = vector_store.similarity_search(
     query="python是不是简单易学",
     k=2#要几个结果
)

print(result)
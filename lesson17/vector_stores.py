from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
import config_data as config
class VectorStoreService(object):
    def __init__(self,embedding):
        '''
        :param embedding: 嵌入模型的传入
        :return:
        '''

        self.embedding = embedding
        self.vector_store = Chroma(
            embedding_function=embedding,
            collection_name=config.collection_name,
            persist_directory=config.persist_directory
        )

    def get_retrieve(self) -> VectorStoreRetriever:
        '''
        获取向量库中的retriver(检索信息)
        :return:
        '''
        return self.vector_store.as_retriever(search_kwargs={"k":config.similarity_threshold})

if __name__ == '__main__':
    service = VectorStoreService(OllamaEmbeddings(model="qwen3-embedding:0.6b"))
    retrieve = service.get_retrieve()
    res = retrieve.invoke("如何减肥")
    print(res)
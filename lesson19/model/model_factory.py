'''
model factory
'''
from abc import ABC,abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
import os
from utils.config_handler import  rag_config
from model.rerank_client import SiliconRerankModel
from model.rerank_adapter import build_reranker_adapter
from model.rerank_runnable import RerankerRunnable
from utils.runtime_env import load_runtime_env, setup_langsmith_defaults

load_runtime_env()
setup_langsmith_defaults()
api_key = os.getenv("SILICON_FLOW")
base_url = os.getenv("SILICON_URL")

class BaseModelFactory(ABC):

    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseModel ]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=rag_config["chat_model_name"],
            api_key=api_key,
            base_url=base_url
        )

class EmbeddingFactory(BaseModelFactory):
    def generator(self) -> Embeddings:
        return OllamaEmbeddings(model=rag_config["embedding_model_name"])

class EmbeddingFactorySilicon(BaseModelFactory):
    def generator(self) -> Embeddings:
        return OpenAIEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=rag_config["embedding_model_name_silicon"]
        )

class RerankingFactorySilicon(BaseModelFactory) :
    def generator(self) -> Optional[Embeddings | BaseModel]:
        return SiliconRerankModel(
            api_key=api_key,
            base_url=f"{base_url}/rerank",
            model=rag_config["reranking_model_name_silicon"],
        )

class RerankerAdapter(BaseModelFactory) :
    '''
    将reranker_client的普通python客户端封装成runnable类,使用runnable lambda
    支持langchian的功能
    '''
    def generator(self) -> Optional[Embeddings | BaseModel]:
        reranker_client = RerankingFactorySilicon().generator()
        return build_reranker_adapter(reranker_client)

class RerankerRunnableFactory(BaseModelFactory) :
    '''
    将reranker_client的普通python客户端封装成runnable类
    支持langchian的功能
    '''
    def generator(self) -> Optional[Embeddings | BaseModel]:
        reranker_client = RerankingFactorySilicon().generator()
        return RerankerRunnable(reranker=reranker_client)


chat_model:ChatOpenAI = ChatModelFactory().generator()
embedding_model = EmbeddingFactory().generator()
embedding_model_silicon = EmbeddingFactorySilicon().generator()
reranking_model_silicon = RerankingFactorySilicon().generator()
reranker_adapter = RerankerAdapter().generator()
reranker_runnable = RerankerRunnableFactory().generator()

if __name__ == '__main__':
    print(api_key)
    print(base_url)
    print(chat_model)
    print(embedding_model)

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
from dotenv import load_dotenv
from utils.config_handler import  rag_config
from model.rerank_client import SiliconRerankModel

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key = os.getenv("SILICON_FLOW")
base_url = os.getenv("SILICON_URL")

class BaseModelFactory(ABC):

    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseModel ]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseModel ]:
        return ChatOpenAI(
            model=rag_config["chat_model_name"],
            api_key=api_key,
            base_url=base_url
        )

class EmbeddingFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseModel ]:
        return OllamaEmbeddings(model=rag_config["embedding_model_name"])

class EmbeddingFactorySilicon(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseModel ]:
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

chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingFactory().generator()
embedding_model_silicon = EmbeddingFactorySilicon().generator()
reranking_model_silicon = RerankingFactorySilicon().generator()

if __name__ == '__main__':
    print(api_key)
    print(base_url)
    print(chat_model)
    print(embedding_model)
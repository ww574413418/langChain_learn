from typing import Any
from langchain_core.documents import Document
from model.rerank_client import SiliconRerankModel
from langchain_core.runnables import RunnableSerializable,RunnableConfig
from utils.logger_handler import logger as log
from pydantic import ConfigDict


class RerankerRunnable(RunnableSerializable[dict,dict]):
    # 因为 SiliconRerankModel 是你自定义普通类，Pydantic 有时不认识。
    # 这时候要再加一个配置，允许任意类型。
    model_config = ConfigDict(arbitrary_types_allowed=True)
    reranker:SiliconRerankModel

    def invoke(self,input_data:dict,config:RunnableConfig|None=None,**kwargs:Any)->dict:
        '''
        :param input_data: 
        :param config: langchian运行配置入口
        :param kwargs: 让runnable更加兼容框架的调用方式
        :return: 
        '''
        query:str = input_data["query"]
        docs:list[Document] = input_data["docs"]

        log.info(f"reranker invoke,query:{query},docs:{docs}")
        if not docs:
            return {
                "query":query,
                "docs":[]
            }

        reranked_doc = self.reranker.rerank_documents(query,docs)

        return {
            "query":query,
            "docs":reranked_doc
        }

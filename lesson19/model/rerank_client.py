import json
import requests
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from utils.config_handler import chroma_config
from utils.logger_handler import logger as log

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api = os.getenv("SILICON_FLOW")


class SiliconRerankModel:
    def __init__(self,api_key:str,base_url:str,model:str):
        self.api_key =  api_key
        self.base_url = base_url
        self.model = model

    def rerank(self ,query:str,documents:list[str],top_n:int = 5) -> list[dict]:
        payload={
            "model":self.model,
            "query":query,
            "documents":documents,
            "top_n":top_n
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            # post到 v1/rerank 拿到相应
            response = requests.post(url=f"{self.base_url}",headers=headers,json=payload)

            # 转成json
            result = response.json()
            return result.get("results")
        except Exception as e:
            log.error(f"rerank error,error={e}")
            print(f"rerank error,error={e}")

    def rerank_documents(self,query:str, docs:list[Document]):
        documents = [doc.page_content for doc in docs]

        rerank_docs = []
        rerank_result = self.rerank(query,documents,10)

        for item in rerank_result:
            index = item.get("index")
            score = item.get("relevance_score")

            doc = docs[index]
            doc.metadata["rerank_score"] = score
            rerank_docs.append(docs[index])

        return rerank_docs



if __name__ == '__main__':
    model = SiliconRerankModel(api_key=api,base_url="https://api.siliconflow.cn/v1",model="Qwen/Qwen3-Reranker-8B")
    res = model.rerank("APP无法连接机器人怎么办？",["APP无法连接机器人怎么办？","APP无法连接机器人怎么办？","APP无法连接机器人怎么办？"])
    print(res)
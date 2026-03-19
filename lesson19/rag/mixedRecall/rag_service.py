'''
get user input and search relative info from chroma db,
and then summary the search result and user input
'''
from langchain_core.documents import Document
from utils.logger_handler import logger as log
from rag.mixedRecall.vector_store import vector_store
from langchain_core.prompts import PromptTemplate
from model import model_factory
from utils.prompts_loader import load_rag_prompt,load_refine_query_prompt
from langchain_core.output_parsers import StrOutputParser
from utils.config_handler import chroma_config
from langchain_community.retrievers import BM25Retriever
import jieba


class rag_summary_service:
    def __init__(self):
        self.vect_store = vector_store
        self.prompt_text = load_rag_prompt()
        self.refine_query = load_refine_query_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.refine_query_template = PromptTemplate.from_template(self.refine_query)
        self.model = model_factory.chat_model
        self.retriever = vector_store.get_retriever()
        self.chain = self._init_chain()
        self.cross_encoder = model_factory.reranker_runnable #注入重排序模型



    def _init_chain(self):
        log.info("build llm chain")
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain


    def retriever_document(self,user_input:str) -> list[Document]:
        '''
        search relative info by user input
        :param user_input: user input
        :return:
        '''
        dense_docs = self.dense_retriever(user_input,top_k=chroma_config["k"])
        sparse_docs = self.sparse_retriever(user_input,top_k=chroma_config["k"])
        fuse_docs = self.rrf_fuse(dense_docs,sparse_docs)
        rerank_docs = self.cross_encoder_rerank(user_input,fuse_docs)
        log.info(f"get relative info:{rerank_docs}")

        return rerank_docs[:5]

    def rag_summary(self,user_input:str) -> str:
        context_doc = self.retriever_document(user_input)

        context = ""
        counter = 0

        for doc in context_doc:
            counter += 1
            context += f"[citation {counter}]:reference doc:{doc.page_content} | reference meta :{doc.metadata} \n"

        log.info(f"context info:{context}")

        if len(context.strip()) == 0:
            # not found relative info
            refine_query = self.rewrite_query(user_input)
            log.info(f"refine query:{refine_query}")
            context_doc = self.retriever_document(refine_query)
            log.info(f"refine query :{context_doc}")
            for doc in context_doc:
                context += f"[citation {counter}]:reference doc:{doc.page_content} | reference meta :{doc.metadata} \n"

        return self.chain.invoke({"input":user_input,"context":context})

    def rewrite_query(self, user_input):
        chain = self.refine_query_template | self.model | StrOutputParser()
        res = chain.invoke(user_input)
        return res

    def dense_retriever(self, user_input, top_k) -> list[Document]:
        '''
        向量匹配
        :param user_input:
        :param top_k:
        :return:
        '''
        docs = self.retriever.invoke(user_input)
        return docs[:top_k]

    def Chinese_tokenize(self, text) -> list[str]:
        '''
        中文分词器,帮助bm25更好分词
        :param text:
        :return:
        '''
        words = jieba.lcut(text)
        return [w for w in words if w.strip()]

    def sparse_retriever(self, user_input, top_k) ->list[Document]:
        '''
        关键词匹配
        :param user_input:
        :param top_k:
        :return:
        '''
        try:
            doc_chunks = self.vect_store.get_all_split_documents()
            if not doc_chunks:
                log.error(f"there are no chunk data,chunks:{len(doc_chunks)}")
                return []

            bm25 = BM25Retriever.from_documents(doc_chunks,preprocess_func=self.Chinese_tokenize)
            bm25.k = top_k
            docs = bm25.invoke(user_input)
            return docs
        except Exception as e:
            log.error(f"sparse retriever error,error={e}")
            return []

    def get_doc_key(self,chunk:Document) ->str:
        """
        获取文档的key
        :return:
        """
        doc_id = chunk.metadata.get("doc_id")
        if not doc_id:
            log.error(f"there are no doc_id in chunk,chunk={chunk}")
            doc_id=chunk.page_content[:100]

        chunk_id = chunk.metadata.get("chunk_id")
        if not chunk_id:
            log.error(f"there are no chunk_id in chunk,chunk={chunk}")
            chunk_id = chunk.metadata["source"]

        key= f"{doc_id}_{chunk_id}"
        return key

    '''
    用来记录每个文档的累积分数
    
    score = 1 / (rrf_k + rank)
    rrf_k = 60 设置为默认值
    
     fused_map = {
        "doc_key":{
            "score":0.0, #累积PRF分
            "doc":Document #要返回的document对象
        }
    }
    '''
    def rrf_fuse(self, dense_docs, sparse_docs,rrf_k = 60) -> list[Document]:
        '''
            1.拿到排名 rank
            2.拿到当前 doc
            3.生成 doc_key
            4.累加它的 RRF 分数
        :param dense_docs:  向量索引结果
        :param sparse_docs: 关键字索引结果
        :param rrf_k:
        :return:
        '''

        fused_map = {}

        for rank,doc in enumerate(dense_docs,start=1):
            doc_key = self.get_doc_key(doc)
            score = 1 / (rrf_k + rank)

            if doc_key not in fused_map:
                fused_map[doc_key] = {
                    "score":score,
                    "doc":doc
                }
            else:
                fused_map[doc_key]["score"] += score

        for rank,doc in enumerate(sparse_docs,start=1):
            doc_key = self.get_doc_key(doc)
            score = 1 / (rrf_k + rank)

            if doc_key not in fused_map:
                fused_map[doc_key] = {
                    "score":score,
                    "doc":doc
                }
            else:
                fused_map[doc_key]["score"] += score
        # 把字典转成列表
        ranked = list(fused_map.values())
        # 按 score 降序排序
        ranked.sort(key=lambda x: x["score"], reverse=True)


        for idx, (doc_key, doc_info) in enumerate(fused_map.items(), start=1):
            log.info(f"RRF fusion result #{idx} , score: {doc_info['score']:.6f} , doc_content: {doc_info['doc'].page_content[:100]}...")


        #把排序结果还原成 Document 列表
        return [doc["doc"] for doc in ranked]

    def cross_encoder_rerank(self, user_input:str, fuse_docs:list[Document]) ->list[Document]:
        '''
        使用cross_encoder对rrf召回的信息和用户的输入,进行相关性评价
        :param user_input: 用户输入
        :param fuse_docs: rrf召回的相关知识库数据
        :return:
        '''
        result = self.cross_encoder.invoke(input_data={"query":user_input,"docs":fuse_docs})
        return result["docs"]


rag_summary_service = rag_summary_service()

if __name__ == '__main__':

    print(rag_summary_service.retriever_document("APP无法连接机器人怎么办？"))
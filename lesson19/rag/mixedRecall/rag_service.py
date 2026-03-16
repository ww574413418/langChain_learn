'''
get user input and search relative info from chroma db,
and then summary the search result and user input
'''
from langchain_core.documents import Document
from utils.logger_handler import logger as log
from rag.vector_store import vector_store
from langchain_core.prompts import PromptTemplate
from model import model_factory
from utils.prompts_loader import load_rag_prompt,load_refine_query_prompt
from langchain_core.output_parsers import StrOutputParser
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
        docs = self.retriever.invoke(user_input)
        log.info("get relative info")

        ranked_docs = self.rerank_documents(user_input, docs)
        return ranked_docs[:5]

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


    def extract_keywords(self, query: str) -> list[str]:
        '''
        extract keywords of user query
        :param query:
        :return:
        '''
        # 定义jieba那些词不需要处理
        stop_words = {
            "怎么", "怎么办", "如何", "为什么", "一下", "一个", "一些",
            "处理", "方法", "问题", "原因", "的", "了", "吗", "呢"
        }

        words = jieba.lcut(query)
        keywords = []

        for word in words:
            word = word.strip()
            if not word:
                continue
            if word in stop_words:
                continue
            if len(word) < 2:
                 continue
            keywords.append(word)

        return  keywords

    def keyword_score(self, query: str, doc: Document) -> int:
        '''
        set score for each keyword
        :param query:
        :param doc:
        :return:
        '''
        keywords = self.extract_keywords(query)
        content = doc.page_content
        score = 0

        for kw in keywords:
            if kw in content:
                score += 1

        if query.replace("？", "").replace("?", "") in content:
            score += 10

        if "怎么办" in content or "怎么处理" in content:
            score += 1

        return score

    def rerank_documents(self, query: str, docs: list[Document]) -> list[Document]:
        '''
        rerank documents by keyword score
        :param query:
        :param docs:
        :return:
        '''
        ranked = []
        total = len(docs)

        for idx,doc in enumerate(docs):
            vector_store = total - idx
            kw_score = self.keyword_score(query, doc)
            final_score = vector_store + kw_score*3

            log.info(
                f"rank | idx={idx} vector_score={vector_store}"
                f"kw_score={kw_score} final_score={final_score}"
                f"source={doc.metadata.get("source","")}"
            )
            ranked.append((final_score, doc))

        ranked.sort(key=lambda x: x[0], reverse=True)
        '''
        result = []
        for item in ranked:
            score, doc = item
            result.append(doc)
        return result
        '''
        return [doc for _, doc in ranked]



rag_summary_service = rag_summary_service()

if __name__ == '__main__':

    print(rag_summary_service.rag_summary("APP无法连接机器人怎么办？"))
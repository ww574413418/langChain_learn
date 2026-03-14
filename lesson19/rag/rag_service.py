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
        res = self.retriever.invoke(user_input)
        log.info("get relative info")
        return res

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

rag_summary_service = rag_summary_service()

if __name__ == '__main__':

    print(rag_summary_service.rag_summary("APP无法连接机器人怎么办？"))
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from vector_stores import VectorStoreService
from langchain_ollama import OllamaEmbeddings
import config_data
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableSequence,RunnableWithMessageHistory
from file_history_store import get_history


load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key=os.getenv("SILICON_FLOW")
base_url=os.getenv("SILICON_URL")

class RagService(object):
    def __init__(self):
        self.vector_store_service = VectorStoreService(
            embedding=OllamaEmbeddings(model=config_data.embedding_model),
        )
        self.prompt_tempate = ChatPromptTemplate.from_messages([
            (
                "system",
                "以我提供的资料为主,简洁专业地回答问题。"
                "若会话历史里出现过明确数值(如170cm),必须优先原样复述该数值,"
                "不要擅自改写成区间。仅当用户明确要求区间时才给区间。"
                "参考资料:{context}",
            ),
            ("system","会话历史消息如下:"),
            MessagesPlaceholder("chat_history"),
            ("human","用户输入:{input}")
        ])
        self.chat_model = ChatOpenAI(
            model = config_data.chat_model,
            api_key = api_key,
            base_url = base_url
        )
        self.chain = self.get_chain()

    def print_prompt(self,prompt:AIMessage)->AIMessage:
        print("="*20)
        print(prompt.to_string())
        print("="*20)
        return prompt

    def get_chain(self) -> RunnableSequence:
        '''
        获取最终执行链
        :return:
        '''

        retriever = self.vector_store_service.get_retrieve()

        def format_document(documents:list[Document]) -> str:
            '''
            将向量库中的文档格式化成字符串
            :param documents:
            :return:
            '''
            if not documents:
                return "无相关资料"

            formate_str ="\n".join([doc.page_content for doc in documents])
            return formate_str



        # 由于memory_chain要求传入一个dict,而retriever要接收一个str,所以使用extra_input将input取出来给而retriever要接收一个str
        extra_input = RunnableLambda(lambda x:x["input"])

        chain = (
            {
                "input": extra_input,
                "context": extra_input | retriever | format_document,
                "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
            }
            | self.prompt_tempate
            | self.chat_model
        )

        memory_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        return memory_chain

if __name__ == '__main__':
    # session id配置
    session_config = {
        "configurable":{
            "session_id":"user_001"
        }
    }
    service = RagService()
    res = service.chain.invoke({"input":"hello"},session_config)
    for chunk in res:
        print(chunk.content,end="",flush=True)

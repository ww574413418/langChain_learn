import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder


load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api = os.getenv("SILICON_FLOW")
base_url = os.getenv("SILICON_URL")

llm = ChatOpenAI(
    model="tencent/Hunyuan-MT-7B",
    temperature=0.5,
    api_key=api,
    base_url=base_url,
)

prompt = ChatPromptTemplate.from_messages([
    ("system","你是一个花点导购,回答要简洁,你会收到历史对话history,必须优先依据 history，不能说没收到"),
    MessagesPlaceholder("history"),
    ("human","{input}")
])

chain = prompt | llm | StrOutputParser()

# 简单的内存存储,按seesion_id保存会话历史
_store: dict[str,InMemoryChatMessageHistory] = {}
# 设置窗口大小
WINDOW = 8

def get_history(session_id:str) -> InMemoryChatMessageHistory:
    # setdefault(k,v)如果_store中没有k,则将v放进去并返回v
    hist = _store.setdefault(session_id,InMemoryChatMessageHistory())

    if len(hist.messages) > WINDOW:
        hist.messages = hist.messages[-WINDOW:]
    return hist

conservation = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

if __name__ == '__main__':
    session_id = "user1"

    resp1 = conservation.invoke(
        {"input":"我姐姐要过生日,我需要一束鲜花"},
        config={"configurable":{"session_id":session_id}}
    )

    print("回复1:",resp1)

    resp2 = conservation.invoke(
        {"input": "预算300,给我推荐"},
        config={"configurable": {"session_id": session_id}}
    )

    print("回复2:", resp2)

    resp3 = conservation.invoke(
        {"input": "我为什么要买花?"},
        config={"configurable": {"session_id": session_id}}
    )

    print("回复3:", resp3)

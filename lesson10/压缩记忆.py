import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api = os.getenv("SILICON_FLOW")
base_url = os.getenv("SILLICON_URL")

llm = ChatOpenAI(
    model="tencent/Hunyuan-MT-7B",
    temperature=0.5,
    api_key=api,
    base_url=base_url,
)


# 简单的内存存储,按seesion_id保存会话历史
_store: dict[str,InMemoryChatMessageHistory] = {}

# 压缩更早的历史记录(总结)
_summary_store = {}


# 当消息超过该阈值的时候,把更早的会话进行压缩成summary(避免上下文无限增长)
SUMMARY_TRIGGER = 4

summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是对话压缩器,将更早的对话压缩成简要的摘要,保留关键事实(任务,预算,偏好,约束,已做决定,为完成事项).输出中文,尽量不要超多150字."
    ),
    (
        "human",
        "以后摘要:{existing_summary},需要压缩的历史,{history_text}请输出新的摘要"
    )
])
# 使用llm将就消息变成摘要
summary_chain = summary_prompt | llm | StrOutputParser()

def update_summary(seesion_id:str,message_to_summary) -> None:
    '''把更早的消息 压缩进_summary_store[session_id]'''
    if not message_to_summary:
        return

    existing = _summary_store.get(seesion_id,"")
    summary = summary_chain.invoke({
        "existing_summary":existing,"history_text":message_to_summary
    })
    _summary_store[seesion_id] = summary;

def get_summary(session_id: str) -> str:
    return _summary_store.get(session_id,"")

# 设置窗口大小
WINDOW = 8

def get_history(session_id:str) -> InMemoryChatMessageHistory:
    # setdefault(k,v)如果_store中没有k,则将v放进去并返回v
    hist = _store.setdefault(session_id,InMemoryChatMessageHistory())

    # 历史过长,把更早的内容压缩到summary,在保留最近的WINDOW条
    if len(hist.messages) > SUMMARY_TRIGGER:
        older = hist.messages[-WINDOW:]
        update_summary(session_id,older)
        hist.messages = hist.messages[-WINDOW:]

    if len(hist.messages) > WINDOW:
        hist.messages = hist.messages[-WINDOW:]
    return hist

# 将history 和 summary都放进去
prompt = ChatPromptTemplate.from_messages([
    ("system","你是一个花点导购,回答要简洁,你会收到历史对话history,必须优先依据 history，不能说没收到"),
    ("system", "已知摘要（summary）：{summary}"),
    MessagesPlaceholder("history"),
    ("human","{input}")
])

# RunnableWithMessageHistory 只负责把 history 塞进来，
# 但不会自动帮你把 _summary_store 的 summary 塞进 prompt。
def inject_summary(inputs: dict, config) -> dict:
    session_id = (config.get("configurable", {}) or {}).get("session_id")
    summary = _summary_store.get(session_id, "") if session_id else ""
    return {**inputs, "summary": summary}

# RunnableLambda 的作用：把你写的一个普通 Python 函数，包装成 LangChain 的“可组合步骤（Runnable）”，
# 这样它就能像 prompt | llm | parser 一样用 | 串进链里。
chain = RunnableLambda(inject_summary) | prompt |  llm | StrOutputParser()

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

    resp4 = conservation.invoke(
        {"input": "我为什么要买花?"},
        config={"configurable": {"session_id": session_id}}
    )

    print("回复4:", resp4)

    resp5 = conservation.invoke(
        {"input": "给我一个之前对话的摘要"},
        config={"configurable": {"session_id": session_id}}
    )

    print("回复5:", resp5)
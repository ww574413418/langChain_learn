import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key=os.getenv("SILICON_FLOW")
base_url=os.getenv("SILICON_URL")

model = ChatOpenAI(
    model="tencent/Hunyuan-MT-7B",
    api_key=api_key,
    base_url=base_url,
)

def print_prompt(full_prompt:AIMessage):
    print("="*20,full_prompt.to_string(),"="*20)
    return full_prompt

# prompt = PromptTemplate.from_template("""
#     你需要根据历史回应用户的问题.对话历史{chat_history}.当前用户的输入{input},请给出回应:
# """)

prompt = ChatPromptTemplate.from_messages([
    ("system","根据历史会话,回应用户的问题,对话历史:"),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
])
base_chain = prompt | print_prompt | model | StrOutputParser()

# 保存历史信息
chat_history = {}

# 获取历史信息
def getHistory(session_id  ) -> InMemoryChatMessageHistory:
    if session_id not in chat_history:
        chat_history[session_id] = InMemoryChatMessageHistory()
    return chat_history[session_id]

conversation_chain = RunnableWithMessageHistory(
    base_chain, #被附加历史消息的runnable
    getHistory, #获取历史会话
    input_messages_key="input", #声明用户输入信息占位
    history_messages_key="chat_history" #声明历史消占位符
)

if __name__ == '__main__':
    # 固定格式,添加langchian配置,为当前程序配置seesin id
    session_config = {
        "configurable":{"session_id":"user_001"}
    }

    print(conversation_chain.invoke({"input":"小明有一只猫"},config=session_config))
    print(conversation_chain.invoke({"input":"小刚有2条狗"},config=session_config))
    print(conversation_chain.invoke({"input":"请问他们一共有几只宠物"},config=session_config))


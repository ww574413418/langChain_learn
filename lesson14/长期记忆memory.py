import os
import json
from typing import Sequence
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import message_to_dict, messages_to_dict, AIMessage, BaseMessage,messages_from_dict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import  ChatPromptTemplate,MessagesPlaceholder

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id:str,store_path:str):
        self.session_id = session_id #会话id
        self.store_path = store_path #不同会话的存储文件,所在文件路径
        # 完整文件路径
        self.file_path = os.path.join(self.store_path,self.session_id)
        # 确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)


    # 重写add_messages方法
    def add_messages(self,message:Sequence[BaseMessage]) -> None:
        # Sequence序列 list
        all_messages = list(self.messages) # 已有的消息 BaseMessage 里面又一个messages的list
        all_messages.extend(message) # 新的和已有的合成一个list

        # 将数据写入到文件中
        # 类对象写入文件 会变成二进制,将对象转成字典,转成json
        # new_messages = []
        # for message in all_messages:
        #     d = message_to_dict(message)
        #     new_messages.append(d)

        new_messages = [message_to_dict(message) for message in all_messages]

        # 将数据写入文件
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump(new_messages,f,ensure_ascii=False)

    # 获取历史会话
    @property # @property装饰器,messages方法当作成员属性用
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                messages = json.load(f)
                return messages_from_dict(messages) # 将list[json]文件转成list[BaseMessage]
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f)

def get_prompt(full_prompt:AIMessage):
    print("="*20,full_prompt.to_string(),"="*20)
    return full_prompt

def get_history(session_id:str):
    return FileChatMessageHistory(session_id,"/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/lesson14/memory")


load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key=os.getenv("SILICON_FLOW")
base_url=os.getenv("SILICON_URL")

model = ChatOpenAI(
    model="tencent/Hunyuan-MT-7B",
    api_key=api_key,
    base_url=base_url,
)

prompt = ChatPromptTemplate.from_messages([
    ("system","请根据历史记录信息,回答用户的问题(简单回答)"),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
])

base_chain = prompt  |  model | StrOutputParser()

conversation_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

if __name__ == '__main__':
    session_config = {
        "configurable":{"session_id":"user_001"}
    }

    # print(conversation_chain.invoke({"input":"小明有一只猫"},config=session_config))
    # print(conversation_chain.invoke({"input":"小王有2只狗"},config=session_config))
    print(conversation_chain.invoke({"input":"他们一共有几只宠物"},config=session_config))
from typing import Sequence
from langchain_core.chat_history import  BaseChatMessageHistory
from langchain_core.messages import messages_to_dict, messages_from_dict, message_to_dict, BaseMessage
import json
import os

def get_history(session_id:str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return FileChatMessageHistory(session_id, os.path.join(current_dir, "history"))
class FileChatMessageHistory(BaseChatMessageHistory):
        def __init__(self,session_id:str,store_path:str):
            self.store_path = store_path
            self.session_id = session_id
            # 完整保存路径
            self.file_path = os.path.join(self.store_path, f"{self.session_id}.json")
            # 确保文件存在
            os.makedirs(os.path.dirname(self.file_path),exist_ok=True)


        def add_messages(self, messages: Sequence[BaseMessage]) -> None:
            '''
            存储历史会话
            :param messages: 历史会话
            :return:
            '''
            all_messages = list(self.messages)
            all_messages.extend(messages)

            new_message = [message_to_dict(message) for message in all_messages]
            with open(self.file_path,"w",encoding="utf-8") as f:
                json.dump(new_message,f,ensure_ascii=False)


        @property
        def messages(self)->list[BaseMessage]:
            '''
            获得历史会话
            :return:
            '''
            try:
                with open(self.file_path,"r",encoding="utf-8") as f:
                    messages = json.load(f)
                    return messages_from_dict(messages)
            except FileNotFoundError:
                return []

        def clear(self) -> None:
            '''
            清楚历史会话
            :return:
            '''
            with open(self.file_path,"w",encoding="utf-8") as f:
                json.dump([],f)




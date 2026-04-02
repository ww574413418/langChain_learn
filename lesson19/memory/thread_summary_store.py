'''
实现summary单独落盘
'''
from datetime import datetime
from utils.path_tool import get_abs_path
import os
import json

def get_thread_summary_path(thread_id:str|int):
    summary_path = get_abs_path(f"memory/thread_summaries/{thread_id}.json")
    return summary_path

def load_thread_summary(thread_id:str|int) -> dict:
    summary_path = get_thread_summary_path(thread_id)
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_thread_summary(thread_id:str|int,summary:str):
    new_summary = {
        "thread_id":thread_id,
        "summary":summary,
        "updated_at":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    summary_path = get_thread_summary_path(thread_id)
    # 先创建,防止目录不存在
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path,"w",encoding="utf-8") as f:
        json.dump(new_summary,f,ensure_ascii=False,indent=2)


def delete_thread_summary(thread_id: str | int) -> None:
    summary_path = get_thread_summary_path(thread_id)
    if os.path.exists(summary_path):
        os.remove(summary_path)

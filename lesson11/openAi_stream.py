from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")

api_key = os.getenv("ollama_key")
base_url = os.getenv("ollama_url")

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

response= client.chat.completions.create(
    model="gemma3:1b",
    messages= [
        {"role":"system","content":"你是个python编程专家,并且很耐心解释"},
        {"role":"assistant","content":"好的,我是编程专家"},
        {"role":"user","content":"用python输出hello world"}
    ],
    stream=True
)

# 处理流式结果
for chunk in response:
    print(chunk.choices[0].delta.content,
          end="", # 每一个chunk以空格结尾
          flush=True # 立即刷新缓冲区
          )

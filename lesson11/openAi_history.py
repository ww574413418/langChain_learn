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
        {"role":"system","content":"你是一个ai助理"},
        {"role":"user","content":"小明又2条宠物狗"},
        {"role":"assistant","content":"好的"},
        {"role": "user", "content": "小红有3只猫咪"},
        {"role":"assistant","content":"好的"},
        {"role": "user", "content": "一共有多少个宠物呢?"},
    ]
)

result = response.choices[0].message.content
print(result)

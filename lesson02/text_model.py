from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

client = OpenAI(api_key=api_key,base_url=base_url);
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    temperature=0.5,
    max_tokens=100,
    messages=[
        {
            "role": "system",
            "content": "You are a creative AI.",
        },
        {
            "role": "user",
            "content": "请教我一下,什么是广义相对论,用小白能懂的语言",
        }
    ]
)
# content 只显示回答内容,不显示英文的思维链条和其他信息
print(response.choices[0].message.content)


from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

client = OpenAI(api_key=api_key,base_url=base_url)

response = client.chat.completions.create(
    model = "gpt-4-turbo",
    messages= [
        {
            "role" : "user",
            "content" : "hello"
        }
    ]
)
print(response.choices[0].message)
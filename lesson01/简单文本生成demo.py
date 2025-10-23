import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

llm = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL,model="gemini-2.5-flash")

text = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
print(text.content)


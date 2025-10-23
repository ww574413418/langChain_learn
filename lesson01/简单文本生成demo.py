from langchain_openai import ChatOpenAI

API_KEY = "sk-Q4qtQDCC3axW4QmzI4VLAdmxpniZcsJ0nioOOKD24D5t2dlg"
BASE_URL = "https://api.deepbricks.ai/v1/"

llm = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL,model="GPT-4o-mini")


text = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
print(text.content)


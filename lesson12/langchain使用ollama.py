from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma3:1b")
result = llm.stream(input="请问你是谁?")

for chunk in result: 
    print(chunk.content,end="",flush=True)




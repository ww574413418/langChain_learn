from langchain_openai import ChatOpenAI

client = ChatOpenAI(
    model="gemma3:1b",
    temperature=0.5,
    base_url="http://localhost:11434/v1",  # Added /v1 here
    api_key="ollama"                       # Ollama doesn't check keys, but needs a placeholder
)

result = client.invoke("介绍一下你自己")
print(result)
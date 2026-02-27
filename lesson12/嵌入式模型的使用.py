from langchain_ollama import OllamaEmbeddings
import asyncio

embed = OllamaEmbeddings(model="qwen3-embedding:0.6b")

print(embed.embed_query("你好"))
async def main():
    vecs = await embed.aembed_documents(["我喜欢你", "hello", "晚上吃啥"])
    print(len(vecs), len(vecs[0]))

asyncio.run(main())


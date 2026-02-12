import os
from dotenv import load_dotenv
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import  create_async_playwright_browser
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent


async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=async_browser)
tools = toolkit.get_tools()

print(tools)


load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api = os.getenv("SILICON_FLOW")
serpapi_api_key=os.getenv("SERPAPI_API_KEY")
base_url = os.getenv("SILICON_URL")

llm = ChatOpenAI(
    model="Qwen/Qwen3-30B-A3B",
    temperature=0.5,
    api_key=api,
    base_url=base_url,
)

agent = create_agent(
    tools = tools,
    llm = llm,
)

async def main():
    response = await agent.invoke("What are the headers on python.langchain.com?")
    print(response)

import asyncio
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
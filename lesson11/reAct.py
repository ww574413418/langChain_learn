import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool # 把一个普通函数包装成langchain的工具,这样agent才可以看到它
from langchain_community.utilities import SerpAPIWrapper # LangChain 社区提供的 SerpAPI 工具封装
# LangGraph 的预置 ReAct Agent 构建器 把 llm + tools 组装成一个“会思考并调用工具”的 Agent
from langchain.agents import create_agent



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

# engine : google 用google进行搜索
# num : 5 返回5条结果
serp = SerpAPIWrapper(params={"engine":"google","num":5}, serpapi_api_key=serpapi_api_key)

# 定义工具 搜索
@tool
def serp_search(query:str) -> str:
    '''
    通过serp 去搜索google然后返回搜索的结果
    :param query: 搜索内容
    :return: 搜索结果
    '''

    data = serp.results(query)
    results = []
    # 从organic_results拿取搜索结果,如果没有就返回[],取前5条数据
    for r in data.get("organic_results",[])[:5] :
        results.append({
            "title":r.get("title",""), # 字段不存在则返回“”
            "url":r.get("link",""),
            "snippet":r.get("snippet","")
        })
    # 把结果组装成结构化对象：{"query":..., "results":[...]} json格式
    return json.dumps({"query":query,"results":results},ensure_ascii=False)

# 定义工具2
@tool
def add_markup(price:float,pct:float = 0.15) -> float:
    '''
    :param price: 鲜花价格
    :param pct: 赔率
    :return: 返回新的鲜花价格
    '''
    return round(price * (1 + pct),2) # 保留2为小数

# 创建ReAct agent
agent = create_agent(model=llm,tools=[serp_search,add_markup])

'''
明确告诉它：
	•	先搜索（用哪个工具）
	•	再计算（用哪个工具）
	•	输出格式必须是 JSON（非常关键：减少胡说八道）
'''

prompt = '''
    请完成：
1) 用 serp_search 搜索“玫瑰 平均价格 元/枝（或元/束）”，选择一个可解释的平均价，并给出来源URL；
2) 用 add_markup 计算加价15%的价格；
最终只输出 JSON：
{
  "market_price": <number>,
  "unit": "<string>",
  "source_urls": ["<url>", "..."],
  "marked_up_price_15pct": <number>,
  "notes": "<string>"
}
'''

out = agent.invoke({"messages": [("user", prompt)]})
#取 out["messages"] 最后一条（通常是最终 AI 输出）
# print(out["messages"][-1].content)
for i, m in enumerate(out["messages"], 1):
    print(f"\n--- step {i} ---")
    print("type:", type(m).__name__)
    # 普通内容
    if getattr(m, "content", None):
        print("content:", m.content)
    # 工具调用（如果有）
    if getattr(m, "tool_calls", None):
        print("tool_calls:", m.tool_calls)
from utils.prompts_loader import router_query_prompt
from model.model_factory import chat_model
from typing import Literal
from pydantic import BaseModel,Field
from utils.logger_handler import logger as log

class RouteDecision(BaseModel):
    route:Literal["qa","rag_qa","report","tool_qa"] = Field(description="当前请求应该进入的处理路径")
    reason:str = Field(description="简洁说明做出该判断的原因")


def route_query(query:str) -> RouteDecision:
    structured_llm = chat_model.with_structured_output(RouteDecision)
    prompt = f"{router_query_prompt()} \n\n用户问题:{query}"
    result:RouteDecision = structured_llm.invoke(prompt)
    log.info(f"RouteDecision route={result.route}")
    return result

if __name__ == "__main__":
    tests = [
        "帮我生成这个月的用户使用报告",
        "APP无法连接机器人怎么办？",
        "今天天气怎么样？",
        "预算2000，想买个安静点的，帮我推荐",
    ]

    for q in tests:
        print("query:", q)
        print(route_query(q))
        print("-" * 40)
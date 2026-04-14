from typing import Literal
from pydantic import BaseModel,Field

ToolName = Literal["weather"]

class ToolRequestParams(BaseModel):
    tool_name:str
    provided_args: dict[str,str] = Field(default_factory=dict)
    missing_required_args: list[str] = Field(default_factory=list)

SUPPORTED_CITIES = ["北京", "上海", "深圳", "广州", "杭州", "成都"]
WEATHER_KEYWORDS = ["天气", "气温", "下雨", "湿度", "风力", "开窗"]


def detect_tool_name(query: str) -> ToolName:
    if any(keyword in query for keyword in WEATHER_KEYWORDS):
        return "weather"
    raise ValueError(f"unsupported tool query: {query}")

def parse_city_from_query(query: str) -> str | None:
    for city in SUPPORTED_CITIES:
        if city in query:
            return city
    return None


def extract_tool_request_params(query: str) -> ToolRequestParams:
    tool_name = detect_tool_name(query)

    if tool_name == "weather":
        city = parse_city_from_query(query)
        provided_args = {}
        missing_required_args = []

        if city:
            provided_args["city"] = city
        else:
            missing_required_args.append("city")

        return ToolRequestParams(
            tool_name="weather",
            provided_args=provided_args,
            missing_required_args=missing_required_args,
        )

    raise ValueError(f"unsupported tool_name for query: {query}")


if __name__ == "__main__":
    tests = [
        "今天天气怎么样？",
        "北京天气怎么样？",
        "帮我看看上海今天适不适合开窗",
    ]

    for query in tests:
        print(query, "->", extract_tool_request_params(query).model_dump())
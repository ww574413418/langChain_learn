from typing import Literal
from pydantic import BaseModel,Field
from agent.request_parse.tool_request_parser import ToolRequestParams
from memory.profile_store import get_user_profile
'''
参数补全
参数来源记录
工具执行
'''
CityResolution = Literal["explicit", "profile", "default"]


class ToolExecutionPlan(BaseModel):
    tool_name: Literal["weather"]
    resolved_args: dict[str, str] = Field(default_factory=dict)
    resolution: dict[str, str] = Field(default_factory=dict)
    missing_required_args: list[str] = Field(default_factory=list)


def build_tool_execution_plan(
    user_id: str | int,
    request: ToolRequestParams,
) -> ToolExecutionPlan:
    if request.tool_name != "weather":
        raise ValueError(f"unsupported tool_name: {request.tool_name}")

    city = request.provided_args.get("city")
    resolution = {}
    missing_required_args = []

    if city:
        resolution["city"] = "explicit"
    else:
        user_profile = get_user_profile(user_id)
        profile_city = user_profile.get("location")

        if profile_city:
            city = profile_city
            resolution["city"] = "profile"
        else:
            city = "北京"
            resolution["city"] = "default"

    if not city:
        missing_required_args.append("city")

    return ToolExecutionPlan(
        tool_name="weather",
        resolved_args={"city": city} if city else {},
        resolution=resolution,
        missing_required_args=missing_required_args,
    )


def fetch_weather_data(city: str) -> dict:
    return {
        "city": city,
        "weather": "晴",
        "humidity": "55%",
        "wind_force": "3级",
        "wind_direction": "东南风",
        "precipitation": "近期无降水",
    }


def format_weather_result(weather_data: dict) -> str:
    return (
        f"城市：{weather_data['city']}\n"
        f"天气：{weather_data['weather']}\n"
        f"空气湿度：{weather_data['humidity']}\n"
        f"风力：{weather_data['wind_force']}\n"
        f"风向：{weather_data['wind_direction']}\n"
        f"降水：{weather_data['precipitation']}"
    )


def render_weather_answer(weather_data: dict, city_resolution: CityResolution) -> str:
    city = weather_data["city"]

    source_text = {
        "explicit": f"基于你指定的城市 {city}",
        "profile": f"基于用户资料中的城市 {city}",
        "default": f"当前未提供城市，先按默认城市 {city}",
    }[city_resolution]

    return (
        f"{source_text}，当前天气如下：\n\n"
        f"- 天气：{weather_data['weather']}\n"
        f"- 空气湿度：{weather_data['humidity']}\n"
        f"- 风力：{weather_data['wind_force']}\n"
        f"- 风向：{weather_data['wind_direction']}\n"
        f"- 降水：{weather_data['precipitation']}"
    )


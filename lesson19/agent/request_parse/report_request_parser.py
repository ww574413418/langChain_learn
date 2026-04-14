import re
from datetime import datetime
from typing import Literal
from pydantic import BaseModel,Field

'''
这个文件负责：

从 query 中解析 report 请求参数
当前先专注 target_month
返回结构化结果
不负责取数据，不负责写报告
'''
class ReportRquestParam(BaseModel):
    target_month:str = Field(description="标准化后的目标月份，格式 YYYY-MM")
    month_source: Literal["explicit", "relative", "defaulted"] = Field(
        description="月份来源：用户明确给出 / 相对时间解析 / 系统默认"
    )


def normalize_year_month(year:int,month:int) ->str:
    return f"{year}-{month:02d}"

def parse_explicit_month(query: str) -> str | None:
    query = query.strip()

    match = re.search(r"(20\d{2})[-年](\d{1,2})月?", query)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        if 1 <= month <= 12:
            return normalize_year_month(year, month)

    match = re.search(r"(?<!\d)(\d{1,2})月", query)
    if match:
        month = int(match.group(1))
        if 1 <= month <= 12:
            year = datetime.now().year
            return normalize_year_month(year, month)

    return None

def parse_relative_month(query: str) -> str | None:
    """
    当前先支持：
    - 这个月 / 本月
    - 上个月
    """
    now = datetime.now()

    if "这个月" in query or "本月" in query:
        return normalize_year_month(now.year, now.month)

    if "上个月" in query:
        year = now.year
        month = now.month - 1
        if month == 0:
            year -= 1
            month = 12
        return normalize_year_month(year, month)

    return None

def default_month() -> str:
    now = datetime.now()
    return normalize_year_month(now.year, now.month)

def extract_report_request_params(query: str) -> ReportRquestParam:
    explicit_month = parse_explicit_month(query)
    if explicit_month:
        return ReportRquestParam(
            target_month=explicit_month,
            month_source="explicit"
        )

    relative_month = parse_relative_month(query)

    if relative_month:
        return ReportRquestParam(
            target_month=relative_month,
            month_source="relative"
        )

    return ReportRquestParam(
        target_month=default_month(),
        month_source="defaulted"
    )

if __name__ == "__main__":
    tests = [
        "帮我生成这个月的用户使用报告",
        "帮我生成上个月的用户使用报告",
        "帮我生成2025年12月的用户使用报告",
        "帮我生成12月的用户使用报告",
        "帮我生成用户使用报告",
    ]

    for query in tests:
        result = extract_report_request_params(query)
        print(query, "->", result.model_dump())
import json
from datetime import datetime
from agent.tools.agent_tools import generate_external_data,external_data
from utils.logger_handler import logger as log
from typing import Literal
'''
这个文件负责：

report 数据获取
对外暴露 plain service
不做 tool wrapper
workflow 直接调用它
'''


def get_current_month()->str:
    now = datetime.now()
    return f"{now.year}-{now.month:02d}"


def fetch_external_report_data(user_id:str|int,month:str) -> dict:
    """
    直接给 workflow / service 层调用。
    不走 tool wrapper。
    """
    user_id = str(user_id)
    month = str(month)

    try:
        generate_external_data()
        return external_data[user_id][month]
    except KeyError as e:
        log.error(f"fetch_external_data can not find user:{user_id},month={month}")
        raise ValueError(f"can not find user:{user_id},month={month}") from e



def format_report_data(report_data:dict) ->str:
    if not report_data:
        return ""
    return json.dumps(report_data, ensure_ascii=False, indent=2)


def list_available_report_months(user_id: str | int) -> list[str]:
    '''
    查询可以用的分月信息
    :param user_id:
    :return:
    '''
    user_id = str(user_id)
    generate_external_data()

    user_records = external_data.get(user_id, {})
    months = list(user_records.keys())
    months.sort()
    return months



MonthResolution = Literal["exact", "fallback_to_latest_available"]


def resolve_report_month_for_user(
    user_id: str | int,
    requested_month: str,
    month_source: str,
) -> tuple[str, MonthResolution]:

    available_months = list_available_report_months(user_id)

    if not available_months:
        raise ValueError(f"no report data found for user_id={user_id}")

    if requested_month in available_months:
        return requested_month, "exact"

    if month_source in {"relative", "defaulted"}:
        return available_months[-1], "fallback_to_latest_available"

    raise ValueError(
        f"report data not found for user_id={user_id}, requested_month={requested_month}"
    )


if __name__ == "__main__":
    result = fetch_external_report_data("1007", "2025-12")
    print(result)
    print(format_report_data(result))

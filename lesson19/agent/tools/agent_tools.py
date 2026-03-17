import random

from langchain.tools import tool
from rag.rag_service import rag_summary_service
from rag.simpleMixedRecall.rag_service import rag_summary_service as rag_summary_service_mixRecall
from rag.mixedRecall.rag_service import rag_summary_service as rag_summary_service_rrf
from utils.config_handler import agents_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger as log
import os

user_ids = ["1001","1002","1003","1004","1005","1006","1007","1008"]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06","2025-07"
        , "2025-08", "2025-09", "2025-10", "2025-11","2025-12"]

external_data = {}

@tool(description="search info from vector store")
def rag_summarize(query:str) ->str:
    return rag_summary_service.rag_summary(query)

@tool(description="search info from vector store by mix recall")
def rag_summarize_mixRecall(query:str) ->str:
    return rag_summary_service_mixRecall.rag_summary(query)

@tool(description="search info from vector store by rrf")
def rag_summarize_rrf(query:str) ->str:
    return rag_summary_service_rrf.rag_summary(query)

@tool(description="get city`s weather ")
def get_weather(city:str)->str:
    return (f"city:{city},weather is sun,air humidity is 55%,wind force is 3bft and direction is 144ºSE, "
            f"There has been no precipitation recently  ")

@tool(description="get user`s city")
def get_user_location() ->str:
    return random.choice(["shenzhen","beijing","shanghai"])

@tool(description="get user`s id")
def get_user_id():
    return random.choice(user_ids)


@tool(description="get current month")
def get_current_month()->str:
    return random.choice(month_arr)

def generate_external_data():
    if not external_data:
        external_data_path = get_abs_path(agents_config["external_data_path"])
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"external file is not exist,path{external_data_path}")

        with open(external_data_path,"r",encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr:list[str] = line.strip().split(",")

                user_id:str = arr[0].replace("\"","")
                feature:str = arr[1].replace("\"","")
                efficiency:str = arr[2].replace("\"","")
                consumables:str = arr[3].replace("\"","")
                comparison:str = arr[4].replace("\"","")
                time:str = arr[5].replace("\"","")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "feature":feature,
                    "consumables":consumables,
                    "efficiency":efficiency,
                    "comparison":comparison
                }


@tool(description="get user`s daily active record")
def fetch_external_data(user_id:str|int,month:str)->str:
    user_id = str(user_id)
    month = str(month)
    generate_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        log.error(f"fetch_external_data can not find user:{user_id}")


@tool(description="automatically calls its middleware to populate the report context. "
                  "This function is self-contained (no parameters/return value); "
                  "the context is injected dynamically upon execution.")
def fill_context4report():
    '''
    改方法被调用的时候,说明需要更改提示词
    :return:
    '''
    return "fill_context4report has been invoked"


if __name__ == '__main__':
    print(fetch_external_data("1001","2025-01"))
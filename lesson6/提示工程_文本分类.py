text = '''
1.'2023-02-15, 寓意吉祥的节日, 股票佰笃［BD］美股开盘价10美元, 虽然经历了波动, 但最终以13美元收盘, 成交量
微幅增加至460, 000, 投资者情绪较为平稳。', 
2.'2023-04-05, 市场迎来轻松氛围, 股票盘古 (0021) 开盘价23元, 尽管经历了波动, 但最终以26美元收盘, 成交量
缩小至310, 000, 投资者保持观望态度。', 
'''

schema = '''
# 定义不同实体下的具备属性
schema =｛
'金融'：［'日期', '股票名称', '开盘价', '收盘价', '成交量'］, 
｝
'''

examples_data = [
    {
        "content": "2023-01-10, 股市震荡。股票强大科技A股今日开盘价100人民币, 一度飙升至105人民币, 随后回落至98人民币, 最终以102人民币收盘, 成交量达到520000。",
        "answers": {
            "日期": "2023-01-10",
            "股票名称": "强大科技A股",
            "开盘价": "100人民币",
            "收盘价": "102人民币",
            "成交量": "520000"
        }
    },
    {
        "content": "2024-05-16, 股市利好。股票英伟达美股今日开盘价105美元, 一度飙升至109美元, 随后回落至100美元, 最终以116美元收盘, 成交量达到3560000。",
        "answers": {
            "日期": "2024-05-16",
            "股票名称": "英伟达美股",
            "开盘价": "105美元",
            "收盘价": "116美元",
            "成交量": "3560000"
        }
    }
]

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")

api_key = os.getenv("ollama_key")
base_url = os.getenv("ollama_url")

client =  OpenAI(
    base_url=base_url,
    api_key=api_key,
)

response = client.chat.completions.create(
    model="gemma3:1b",
    messages=[
        {"role":"system","content":"你是一个分类专家"},
        {"role":"assistant","content":f"请按照我的要求数据整理成json格式,要求:{schema}"},
        {"role":"assistant","content":f"参照我给你的示例:{examples_data}"},
        {"role":"user","content":text},
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")





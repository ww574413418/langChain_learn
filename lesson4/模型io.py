import os
import pandas as pd
from dotenv import load_dotenv
# 导入LangChain中的提示模板
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


# 创建原始模板
template = """您是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
{format_instructions}
"""


from langchain_openai import OpenAI
load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key = os.getenv("SILICON_FLOW")
base_url = os.getenv("SILLICON_URL")


model = OpenAI(
    api_key=api_key,
    base_url=base_url,
    model_name="tencent/Hunyuan-MT-7B",
)


# 定义我们想要接收的响应模式
from langchain_core.output_parsers import PydanticOutputParser

class MyOutput(BaseModel):
    flower:str = Field(description="花名")
    price:float = Field(description="价格")
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="问什么要这样写这个文案")

output_parser = PydanticOutputParser(pydantic_object=MyOutput)
format_instructions = output_parser.get_format_instructions()

# 根据原始模板创建LangChain提示模板
prompt = PromptTemplate(
    template = template,
    input_variables=["flower_name", "price"],
    partial_variables={"format_instructions": format_instructions},
)
# 打印LangChain提示模板的内容
# print(prompt)

# 用于存储返回的格式化数据
df = pd.DataFrame(columns=["flower", "price", "description", "reason"])

# 多种花的列表
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

for flower,price in zip(flowers,prices):

    input = prompt.format(flower_name=flower,price=price)

    output = model.invoke(input)

    #解析输出
    parsed_output = output_parser.parse(output)

    # ✅ 转换为字典后再加字段
    row = parsed_output.model_dump()
    row["flower"] = flower
    row["price"] = price

    df.loc[len(df)] = row


df.to_csv("flower document.csv",index=False)
print(df)






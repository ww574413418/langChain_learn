import os
from dotenv import load_dotenv

# ----part1 定义模型文件和模型
load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")


api_key = os.getenv("SILICON_FLOW")
base_url = os.getenv("SILLICON_URL")

from langchain_openai import OpenAI
llm = OpenAI(
    model="tencent/Hunyuan-MT-7B",
    api_key=api_key,
    base_url=base_url,
    temperature=0.1,
    max_tokens=128
    )


# -----part2 定义模型输出格式
import pandas as pd
# 用于保存结果
df = pd.DataFrame(columns=["flower_type","price","description","reason"])

# 准备一些模拟数据
flowers = ["玫瑰","百合","康乃馨"]
prices = ["1","2","3"]

# 定义我们需要的格式
from pydantic import BaseModel,Field


class FlowerDescription(BaseModel):
    flower_type: str = Field(description="鲜花种类")
    price: float = Field(description="鲜花价格")
    description: str = Field(description="鲜花描述")
    reason: str = Field(description="推荐理由")



# -------part3 创建输出解析器
from langchain_core.output_parsers import PydanticOutputParser
# 实例化一个解析器
parse = PydanticOutputParser(pydantic_object=FlowerDescription)

# 获取输出格式指示
format_instruction = parse.get_format_instructions()


# ---------part4 创建提示模版
from langchain_core.prompts import PromptTemplate

template = """您是一位专业的鲜花店文案撰写员。对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？{format_instructions}"""



prompt= PromptTemplate.from_template(
    template = template,
    # 给占位字段赋值
    partial_variables={"format_instructions":format_instruction}
)



for flower,price in zip(flowers,prices):
    # 根据提示准备模型的输入
    input = prompt.format(flower=flower,price=price)

    # 获取模型的输出
    output = llm.invoke(input)

    # 解析输出结果
    parsed_out = parse.parse(output)
    parsed_out_dict = parsed_out.model_dump()

    df.loc[len(df)] = parsed_out_dict

print(df.to_dict(orient="records"))


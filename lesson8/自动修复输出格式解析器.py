import os
from dotenv import load_dotenv

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")

api_key = os.getenv("SILICON_FLOW")
base_url = os.getenv("SILLICON_URL")

from langchain_core.output_parsers import pydantic
from pydantic import Field,BaseModel
from typing import List
class Flower(BaseModel):
    name:str = Field(description="the name of flower")
    colors:List[str] = Field(description="the colors of this flower")

flower_quer = "Gernerate the characters for a random flower"

# 定义一个不正确的输出格式,json要求字段名和字符串必须用双引号
misformatted = "{'name': '康乃馨', 'colors': ['粉红色','白色','红色','紫色','黄色']}"

# 定义一个用于解析输出pydantic解析器
parser = pydantic.PydanticOutputParser(pydantic_object=Flower)
# 使用解析器解析不正确的输出格式
# parser.parse(misformatted)

from langchain_openai import ChatOpenAI
from langchain_classic.output_parsers import OutputFixingParser


llm = ChatOpenAI(
    model="tencent/Hunyuan-MT-7B",
    api_key=api_key,
    base_url=base_url,
    max_tokens=2080,
    temperature=0
)
# 使用OutputFixingParser创建一个新的解析器,能够纠正不正确的格式
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

result = fixing_parser.parse(misformatted)
print(result)




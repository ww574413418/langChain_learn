import os
from dotenv import load_dotenv

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=api_key,
    base_url=base_url,
    temperature=0,
    max_tokens=512
)

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

class Action(BaseModel):
    action:str = Field(description="action to take")
    action_input:str = Field(description="input to the action")

# 定义字符串模板
template =  '''Base on the user question, provide an Action and Action Input for what step should be taken.
{format_instrutions}
Question:{query}
Response:
'''

parser = PydanticOutputParser(pydantic_object = Action)

# 定义一个提示模版
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template = template,
    input_variables = ["query"],
    partial_variables = {"format_instrutions":parser.get_format_instructions()}
)


prompt_value = prompt.format_prompt(query="what are the colors of orchid")

# 定义一个错误格式的字符串 只提供了action字段,没有提供action_input字段
bad_response = '''{"action":"search"}'''


# 先定义OutputFixingParser 尝试在检测到错误的时候,让大模型对修复输出格式
from langchain_classic.output_parsers import OutputFixingParser

fix_parse = OutputFixingParser.from_llm(parser=parser,llm=llm)
parse_result = fix_parse.parse(bad_response)

# action='search' action_input='search query'
# 我们可以看到OutputFixingParser只是修复了action_input没有输入的问题,它并没有根据我们的prompt
# 给一个正确的action_input
print("the result of outfix parser:",parse_result)
# parser.parse(bad_response)

# 尝试使用RetryWitchErrorOutputParse解析器
from langchain_classic.output_parsers import RetryOutputParser
retry_parser = RetryOutputParser.from_llm(parser=parser,llm=llm)
parser_result = retry_parser.parse_with_prompt(bad_response,prompt_value)
# action='search' action_input='colors of orchid'
# 重新请求模型,不仅仅修复了action_input为空的问题,输入的问题也符合prompt的要求,更符合预期ßß
print("the result of retry_parser:",parser_result)

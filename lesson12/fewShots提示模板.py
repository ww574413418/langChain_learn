from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate,PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key=os.getenv("SILICON_FLOW")
base_url=os.getenv("SILICON_URL")

llm = ChatOpenAI(
    model="tencent/Hunyuan-MT-7B",
    api_key=api_key,
    base_url=base_url,
)

template = PromptTemplate.from_template("单词:{word},反义词:{antonym}")
example_data = [
    {"word":"good", "antonym":"bad"},
    {"word":"happy", "antonym":"sad"},
]

few_shot_prompt = FewShotPromptTemplate(
    examples=example_data,
    example_prompt=template,
    prefix="请给出单词的相反词,请按example_prompt输出结果",
    suffix="单词:{input_word},反义词",
    input_variables=["input_word"],
)

chain = few_shot_prompt | llm

res = chain.invoke(input={"input_word":"东"})
print(res.content)
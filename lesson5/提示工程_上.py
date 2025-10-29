from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptWithTemplates
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key = os.getenv("SILICON_FLOW")
base_url = os.getenv("SILLICON_URL")

# 构建模型
model= OpenAI(
    api_key = api_key,
    base_url = base_url,
    model_name = "tencent/Hunyuan-MT-7B"
)

# 创建模板
template = """你是业务咨询顾问。对于一个面向{market}市场的,你给一个销售{product}的电商公司，起一个好的名字？"""
prompt_template = PromptTemplate(
    template = template,
    input_variables = ["market","product"]
)
#
# prompt_template = PromptTemplate.from_template(template)
print(prompt_template.format(market=123,product="666"))


# 导入聊天消息类模板
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

template = "你是一位专业顾问，负责为专注于{product}的公司起名。"

sysMsg = SystemMessagePromptTemplate.from_template(template)

human_template = "公司主打产品是{product_detail}。"
humanMsg = HumanMessagePromptTemplate.from_template(human_template)

prompt_template = ChatPromptTemplate.from_messages([sysMsg,humanMsg])

prompt = prompt_template.format(product=123,product_detail=123)


# 1. 创建一些示例
samples = [
  {
    "flower_type": "玫瑰",
    "occasion": "爱情",
    "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
  },
  {
    "flower_type": "康乃馨",
    "occasion": "母亲节",
    "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
  },
  {
    "flower_type": "百合",
    "occasion": "庆祝",
    "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
  },
  {
    "flower_type": "向日葵",
    "occasion": "鼓励",
    "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
  }
]

# 2. 创建一个提示模板
template="鲜花类型: {flower_type}\n场合: {occasion}\n文案: {ad_copy}"
prompt_sample = PromptTemplate(
    template=template,
    input_variables=["flower_type", "occasion", "ad_copy"]
    )

# **把字典内容解包为关键字参数,**samples[0]拿出第一个数据,并解包为关键字参数
# print(prompt_sample.format(**samples[0]))

# 3. 创建一个FewShotPromptTemplate对象
from langchain_core.prompts.few_shot import FewShotPromptTemplate
prompt = FewShotPromptTemplate(
    examples = samples,
    example_prompt = prompt_sample,
    # 要在示例的后面，接着放一段模板（suffix)
    # suffix 就是 few-shot 示例后面给模型的“待填空问题”，
    suffix = "鲜花类型: {flower_type}\n场合: {occasion}\n",
    input_variables = ["flower_type","occasion"] # 并指定模板(suffix)里有哪些变量
)

# print(prompt.format(flower_type="玫瑰",occasion="情人节"))

ouput = model.invoke(prompt.format(flower_type="玫瑰",occasion="情人节"))

# print(ouput)


# 5. 使用示例选择器
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

# 初始化示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    samples,
    OpenAIEmbeddings(api_key=api_key,
                     base_url=base_url,
                     model="netease-youdao/bce-embedding-base_v1"
                     ),
    Chroma,
    k=1
)

# 创建一个使用示例选择器的FewShotPromptTemplate对象
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=prompt_sample,
    suffix="鲜花类型: {flower_type}\n场合: {occasion}",
    input_variables=["flower_type", "occasion"]
)

# 鲜花类型: 玫瑰
# 场合: 爱情
# 文案: 玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。
#
# 鲜花类型: 红玫瑰
# 场合: 爱情
print(prompt.format(flower_type="红玫瑰", occasion="爱情"))


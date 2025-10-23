
import os
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# ---- Part I 初始化图像字幕生成模型
# 指定要使用的工具模型（HuggingFace中的image-caption模型）
hf_model = "Salesforce/blip-image-captioning-large"
print("正在初始化图像字幕生成模型...")

# 初始化处理器和工具模型
# 预处理器将准备图像供模型使用
processor = BlipProcessor.from_pretrained(hf_model)
# 然后我们初始化工具模型本身
model = BlipForConditionalGeneration.from_pretrained(hf_model)
print("初始化图像字幕生成模型成功")


# ---- Part II 定义图像字幕生成工具类
class ImageCapTool(BaseTool):
    name = "Image captioner"
    description = "使用该工具可以生成图片的文字描述，需要传入图片的URL."

    def _run(self, url: str):
        # 下载图像并将其转换为PIL对象
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        # 预处理图像
        inputs = processor(image, return_tensors="pt")
        # 生成字幕
        out = model.generate(**inputs, max_new_tokens=20)
        # 获取字幕
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


API_KEY = "sk-Q4qtQDCC3axW4QmzI4VLAdmxpniZcsJ0nioOOKD24D5t2dlg"
BASE_URL = "https://api.deepbricks.ai/v1/"

llm = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL,model="GPT-4o-mini")

print("初始化大语言模型成功")
# 使用工具初始化智能体并运行
tools = [ImageCapTool()]
agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True,
)

img_url = "https://lf3-static.bytednsdoc.com/obj/eden-cn/lkpkbvsj/ljhwZthlaukjlkulzlp/eec79e20058499.563190744f903.jpg"
# agent.run(input=f"{img_url}\n请创作合适的中文推广文案")
agent.invoke(input=f"图片链接如下：{img_url}\n 请为这张图创作合适的中文推广文案")

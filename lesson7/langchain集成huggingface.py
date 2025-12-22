import os
import torch
'''
AutoTokenizer : 根据模型名称加载对应的tokenizer
AutoModelForCausalLM : 自动加载“因果语言模型(casual LM)”,用于文本生成
pipeline : transforms提供的高层封装,可以方便做text-generation这种任务
'''
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline
# 定义一个带占位符的“提示模板”
from langchain_core.prompts import PromptTemplate
# 将HuggingFacePipeline模型转成langchain的模型
from langchain_huggingface import HuggingFacePipeline

# 镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_id = "meta-llama/Llama-3.2-3B-Instruct"
# 检查当前环境是否支持 Apple 的 MPS（Metal Performance Shaders）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("使用设备:",device)

# 1. 加载tokenizer&model
# 根据 model_id 去 Hugging Face Hub 下载该模型对应的 tokenizer 配置和 vocab 文件。
# 返回的 tokenizer 用于：
# 	•把字符串 prompt 转成 input_ids（数字序列）。
# 	•把模型输出的 input_ids 再转回人类可读的文本。
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 某些 Llama 类模型原生没有 pad_token（填充符）,如果pad_token没有定义,则使用eos_token(end-of-sequence，序列结束符)
# 这样在做 padding 或 text-generation 时，如果需要用 pad_token_id，就不会报错。
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 加载模型(模型的参数根绝设备来选择)
# 如果不是 CPU（比如 MPS / CUDA），用 float16（半精度），节省显存/内存、加快推理。
# 如果是 CPU，则保持 float32（单精度），避免某些运算在 CPU 上 half 精度不稳定/不支持
dtype = torch.float16 if device.type != "cpu" else torch.float32


model = AutoModelForCausalLM.from_pretrained(
    # 模型id
    model_id,
    # 根据上面的精度类型来加载权重
    torch_dtype=dtype
)

# 将模型部署到GPU中
model.to(device)
# 将模型切换到推理模式evaluation mode
# 关闭 dropout、batchnorm 的训练行为等。
# 对于生成任务是标准操作。
model.eval()

# 2.构造transforms的text-generation pipline
gen_pipe = pipeline(
    task = "text-generation",
    model = model,
    tokenizer=tokenizer,
    device=0 if device.type == "cuda" else -1,
    max_new_tokens=1024,
    do_sample=False,# 不采样 -> 确定性
    #temperature=None, # 覆盖掉默认 0.6
    #top_p=None,  # 覆盖掉默认 0.9
    pad_token_id=tokenizer.pad_token_id,
)

# 3.封装为 langchain的llm
llm = HuggingFacePipeline(pipeline=gen_pipe)

# 4. PromptTemplate + 链
template = '''你是一个只能进行精确数学运算的计算器。

请务必严格遵守以下规则：
1. 不要简化步骤，不要猜测，不要使用语言模型补全。
2. 所有计算必须逐步列出，逐项相加，不得跳步。
3. 在最终答案前先展示：
   - 元素总数 N
   - 求和过程（每10个为一组）
   - 总和 SUM
   - 均值 = SUM / N
   - 方差 = 所有 (xi - 均值)^2 的平均值（逐项列出）
   - 最大值 = 从列表中逐一比较得出
   - 最小值 = 从列表中逐一比较得出
4. 最终给出：
   - 最大值
   - 最小值
   （全部保留两位小数）

输出格式示例：
思考过程：
...

最终答案：
最大值：xx.xx
最小值：yy.yy

请对下列数据执行上述步骤：
{input}
'''
prompt = PromptTemplate.from_template(template)

chain = prompt | llm

if __name__ == '__main__':
    res = chain.invoke(
        {
            "input":'''[56, 94, 40, 3, 32, 76,1,1,1,1]'''
        }
    )
    print(res)
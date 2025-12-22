import os
from mlx_lm import load, generate

# 镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 下载使用mlx的模型,适配apple silicon
model_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"  # ✅ 官方示例用的默认模型
prompt = "简单介绍一下你自己,最好是阐述一下你和Llama原生的区别"


model,tokenizer = load(model_name)


print("开始生成")

response = generate(
    model,
    tokenizer,
    prompt,
    max_tokens=512,
    verbose=True#看到生成速度
)

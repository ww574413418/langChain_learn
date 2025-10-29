import os
from dotenv import load_dotenv



load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")
api_key = os.getenv("SILICON_FLOW")
base_url = os.getenv("SILLICON_URL")



from tree_of_thoughts import TotAgent,ToTDFSAgent
from langchain_openai import ChatOpenAI

os.makedirs("agent_workspace", exist_ok=True)

# 定义模型的输出格式
json_schema = {
    "title": "ThoughtEvaluation",
    "description": "Evaluate a single reasoning step with score",
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": "The reasoning step proposed by the model"
        },
        "evaluation": {
            "type": "number",
            "description": "A score between 0 and 1 indicating how good the thought is"
        }
    },
    "required": ["thought", "evaluation"]
}

class OpenAIWapper:
    def __init__(self,model_name="tencent/Hunyuan-MT-7B"):
        self.model_name = model_name
        self.llm = ChatOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    model=model_name,
                    )
        self.llm = self.llm.with_structured_output(
            json_schema,
            method="json_schema"
        )
    def run(self,task:str=None):
        # # 添加严格的格式要求到提示词中
        # formatted_task = f"""
        #         请严格按照以下 JSON 格式返回结果，不要包含任何其他文本：
        #         {{
        #             "thought": "你的推理步骤",
        #             "evaluation": 0.95
        #         }}
        #
        #         原始任务：{task}
        #         """
        response = self.llm.invoke(task)
        return response


agent = TotAgent(
    use_openai_caller=False,
    model =OpenAIWapper("tencent/Hunyuan-MT-7B"),
)



dfs_agent = ToTDFSAgent(
    agent=agent,
    #继续深入/作为高质量候选的通过分（0~1）。评分 ≥ 0.8 的思路会被保留并继续扩展；越高越严格、分支更少但更容易错过解。
    threshold=0.8,
    #一次搜索里“生成→评估→剪枝→下探”的层数/轮数上限
    max_loops=1,
    #prune_threshold=0.5：早停剪枝线（01）。评分 < 0.5 的分支直接丢弃不再扩展。
    prune_threshold=0.5,
    #每一层并行生成/评估的候选思路数（也常等价于并发工作线程数）
    number_of_agents=4
)
agent.workspace_dir = "/tmp/agent_workspace"

# Define the initial state for the DFS algorithm
initial_state="""
Your task: is to use 4 numbers and basic arithmetic operations (+-*/) 
to obtain 24 in 1 equation, return only the math
"""

# Run the DFS algorithm to solve the problem and obtain the final thought
final_thought = dfs_agent.run(initial_state)

print(final_thought)


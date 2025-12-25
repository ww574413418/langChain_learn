import os
from dotenv import load_dotenv

load_dotenv("/Users/grubby/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/langChain/langChain_learn/env")

api = os.getenv("SILICON_FLOW")
base_url = os.getenv("SILLICON_URL")

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel,Field
# TypedDict state字典结构,Literal限定字段只能取某些值,Optional字段可以空
from typing import TypedDict,Literal,Optional
# langgraph StateGraph状态图构建器  START/END 图的起点终点
from langgraph.graph import StateGraph,START,END



#  1) 定义 State（图里流转的数据）每个图的节点的数据结构
class GraphState(TypedDict):
    input:str
    destination:Optional[str]
    output:Optional[str]


#  2) 路由器：让模型只输出 destination
# 定义路由目标集合,把3个路由拼成一段文本,让模型知道他可以选择的目的地
# 把路由选项写成一段自然语言，让 LLM 自由判断
destination_text = "\n".join([
    "鲜花养护:保持花的健康、如何浇水、施肥等",
    "鲜花装饰:如何搭配花、如何装饰场地等",
    "default:其他问题",
])

# 定义路由器的输出格式,必须是一个对象,且destination只能是这三个值之一,Field(...)表示必填
class Route(BaseModel):
    destination:Literal["鲜花养护","鲜花装饰","default"] = Field(...)


llm = ChatOpenAI(
    model="tencent/Hunyuan-MT-7B",
    temperature=0.7,
    api_key=api,
    base_url=base_url,
)

#
router_prompt = ChatPromptTemplate.from_template(
    """
        你是路由器.根据用户的输入选择最合适的目标
        可选目标:{destination}
        
        只输出一个 destination(鲜花养护/鲜花装饰/default).
        用户输入{input}
    """
).partial(destination=destination_text)# 提前固化destination,以后不需要在传入

# with_structured_output 让模型必须解析成pydantic结构
router_llm = llm.with_structured_output(Route)


def rounter_node(state: GraphState) -> GraphState:
    r: Route = (router_prompt | router_llm).invoke({"input":state["input"]});
    return {"destination":r.destination}


# 决定conditonal edge往那一条边走
# 把destination专程吓一跳的分支的key
def pick_next(state: GraphState) -> Literal["鲜花养护","鲜花装饰","default"]:
    return (state["destination"] or "default"); # 兜底


# 3) 各目的地节点（下游链）
botanyCare_chain = (
        ChatPromptTemplate.from_template('''你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
                        下面是需要你来回答的问题:
                        {input}''') | llm | StrOutputParser()
    )

botanyDress_chain = (
    ChatPromptTemplate.from_template('''你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
                        下面是需要你来回答的问题:
                        {input}''') | llm | StrOutputParser()
)

default_chain = (
        ChatPromptTemplate.from_template("你是通用助手，简洁回答：{input}")| llm | StrOutputParser()
)


# 定义三个图节点
def botanyCare_node(state: GraphState) -> GraphState:
    return {"output":botanyCare_chain.invoke({"input":state["input"]})}

def botanyDress_node(state: GraphState) -> GraphState:
    return {"output":botanyDress_chain.invoke({"input":state["input"]})}

def default_node(state: GraphState) -> GraphState:
    return {"output":default_chain.invoke({"input":state["input"]})}


# 4) 组图：START -> route -> (conditional) -> leaf -> END
g = StateGraph(GraphState)
# 将图节点加入图
g.add_node("route",rounter_node)
g.add_node("botanyCare",botanyCare_node)
g.add_node("botanyDress",botanyDress_node)
g.add_node("default",default_node)
# 定义开始节点
g.add_edge(START,"route")
# 定义route节点走完之后,调用pick_next(state)得到一个字符串,用字符串在下列映射列表中查找吓一跳的节点名
g.add_conditional_edges("route",pick_next,{
    "鲜花养护":"botanyCare",
    "鲜花装饰":"botanyDress",
    "default":"default",
})

# 三个分支节点执行完便结束
g.add_edge("botanyCare",END)
g.add_edge("botanyDress",END)
g.add_edge("default",END)

# 编译成可执行图
app = g.compile()

# 5) 调用

res = app.invoke({"input":"明天天气如何"})
print(res["output"])

trace = ["START"]

for event in app.stream({"input": "明天天气如何"},
                        stream_mode="updates"):
    # event 形如：{"route": {"destination": "鲜花养护"}} 或 {"botanyCare": {"output": "..."}}
    node = next(iter(event.keys()))
    trace.append(node)

trace.append("END")
print(" -> ".join(trace))

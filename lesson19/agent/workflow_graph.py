from langgraph.graph import END,START,StateGraph
from agent.workflow_state import WorkflowState
from agent.workflow_nodes import (
    router_node,
    memory_write_node,
    route_condition,
    rag_retrieve_node,
    rag_answer_node,
    report_fetch_node,
    report_writer_node,
    consolidate_node
)

'''
qa_node \\ report_node \\ tool_node 
占位置,只是为了先把 graph 编译、跑通。
'''
def qa_node(state: WorkflowState) -> dict:
    return {
        "final_answer": "qa_node not implemented yet"
    }


def report_node(state: WorkflowState) -> dict:
    return {
        "final_answer": "report_node not implemented yet"
    }


def tool_node(state: WorkflowState) -> dict:
    return {
        "final_answer": "tool_node not implemented yet"
    }


def build_workflow():
    graph = StateGraph(WorkflowState)

    graph.add_node("router_node", router_node)
    graph.add_node("memory_write_node", memory_write_node)
    graph.add_node("rag_retrieve_node",rag_retrieve_node)
    graph.add_node("rag_answer_node",rag_answer_node)
    graph.add_node("qa_node",qa_node)
    graph.add_node("report_node",report_node)
    graph.add_node("tool_node",tool_node)
    graph.add_node("consolidate_node",consolidate_node)
    graph.add_node("report_fetch_node", report_fetch_node)
    graph.add_node("report_writer_node", report_writer_node)

    graph.add_edge(START, "router_node")
    graph.add_edge("router_node", "memory_write_node")

    graph.add_conditional_edges(
        "memory_write_node",
        route_condition,
        {
            "qa": "qa_node",
            "rag_qa": "rag_retrieve_node",
            "tool_qa": "tool_node",
            "report": "report_fetch_node",
        },
    )

    graph.add_edge("rag_retrieve_node", "rag_answer_node")
    graph.add_edge("qa_node","consolidate_node")
    graph.add_edge("rag_answer_node","consolidate_node")
    graph.add_edge("report_node","consolidate_node")
    graph.add_edge("tool_node","consolidate_node")
    graph.add_edge("report_fetch_node", "report_writer_node")
    graph.add_edge("report_writer_node", "consolidate_node")

    graph.add_edge("consolidate_node", END)

    return graph.compile()

workflow = build_workflow()

if __name__ == "__main__":
    state = {
        "query": "帮我生成这个月的用户使用报告",
        "user_id": "1007",
        "thread_id": "workflow_report_test_002",
    }

    result = workflow.invoke(state)
    print(result)

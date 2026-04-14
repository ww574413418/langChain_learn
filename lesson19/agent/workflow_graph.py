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
    tool_request_parse_node,
    tool_execute_node,
    memory_read_node,
    qa_node,
    consolidate_node
)


def build_workflow():
    graph = StateGraph(WorkflowState)

    graph.add_node("router_node", router_node)
    graph.add_node("memory_write_node", memory_write_node)
    graph.add_node("rag_retrieve_node",rag_retrieve_node)
    graph.add_node("rag_answer_node",rag_answer_node)
    graph.add_node("memory_read_node", memory_read_node)
    graph.add_node("qa_node",qa_node)
    graph.add_node("tool_request_parse_node", tool_request_parse_node)
    graph.add_node("tool_execute_node", tool_execute_node)
    graph.add_node("consolidate_node",consolidate_node)
    graph.add_node("report_fetch_node", report_fetch_node)
    graph.add_node("report_writer_node", report_writer_node)

    graph.add_edge(START, "router_node")
    graph.add_edge("router_node", "memory_write_node")

    graph.add_conditional_edges(
        "memory_write_node",
        route_condition,
        {
            "qa": "memory_read_node",
            "rag_qa": "rag_retrieve_node",
            "tool_qa": "tool_request_parse_node",
            "report": "report_fetch_node",
        },
    )

    graph.add_edge("rag_retrieve_node", "rag_answer_node")
    graph.add_edge("rag_answer_node","consolidate_node")
    graph.add_edge("report_fetch_node", "report_writer_node")
    graph.add_edge("report_writer_node", "consolidate_node")
    graph.add_edge("tool_request_parse_node", "tool_execute_node")
    graph.add_edge("tool_execute_node", "consolidate_node")
    graph.add_edge("memory_read_node", "qa_node")
    graph.add_edge("qa_node", "consolidate_node")
    graph.add_edge("consolidate_node", END)

    return graph.compile()

workflow = build_workflow()

if __name__ == "__main__":
    state = {
        "query": "结合我之前说过的需求，再给我推荐一下",
        "user_id": "0001",
        "thread_id": "workflow_qa_test_002",
    }

    result = workflow.invoke(state)
    print(result)

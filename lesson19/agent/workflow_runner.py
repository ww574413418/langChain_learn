from typing import Any,Generator
from pydantic import BaseModel,Field
from agent.workflow_graph import workflow
from agent.workflow_state import WorkflowState
from utils.logger_handler import logger as log


class WorkflowResult(BaseModel):
    '''
    工作流结果
    '''
    query:str
    user_id:str
    thread_id:str
    route:str
    final_answer:str = ""
    raw_state:dict[str,Any] = Field(default_factory=dict)

class WorkflowRunner:
    def __init__(self,complied_workflow=None):
        self.workflow = complied_workflow or workflow

    def build_initial_state(
            self,
            query:str,
            user_id:str,
            thread_id:str,
        ) -> WorkflowState:

        return {
            "query": query,
            "user_id": str(user_id),
            "thread_id": str(thread_id),
        }

    def run(
            self,
            query:str,
            user_id:str,
            thread_id:str,
    ) -> WorkflowResult:

        initial_state = self.build_initial_state(query, user_id, thread_id)

        log.info(
            f"[workflow_runner] start query={query}, "
            f"thread_id={thread_id}, user_id={user_id}"
        )

        result_state = self.workflow.invoke(initial_state)
        final_answer = self.extract_final_answer(result_state)

        route = result_state.get("route","")

        if not final_answer:
            raise ValueError("workflow finished without final_answer")

        log.info(
            f"[workflow_runner] completed route={route}, "
            f"final_answer_len={len(final_answer)}"
        )

        return WorkflowResult(
            query=query,
            user_id=str(user_id),
            thread_id=str(thread_id),
            route=route,
            final_answer=final_answer,
            raw_state=result_state,
        )

    def execute_stream(self,
                       query:str,
                       thread_id:str,
                       user_id:str) ->Generator[str,None,None]:
        """
        先保持和旧 ReactAgent 兼容的接口。
        当前 graph 还不是 token streaming，所以这里先一次性产出最终答案。
        后续如果 graph 支持增量流式，再只改这里，不改 app 层。
        """
        result = self.run(
            query=query,
            thread_id=thread_id,
            user_id=user_id,
        )
        yield result.final_answer

    @staticmethod
    def extract_final_answer(state: dict[str, Any]) -> str:
        final_answer = state.get("final_answer", "")
        if not isinstance(final_answer, str):
            return ""
        return final_answer.strip()

runner = WorkflowRunner()

if __name__ == "__main__":
    result = runner.run(
        query="预算 2000，想买个安静点的，帮我推荐",
        thread_id="workflow_runner_test_001",
        user_id="0001",
    )

    print("route:", result.route)
    print("final_answer:", result.final_answer)
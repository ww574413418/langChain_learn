from agent.react_agent import agent


def run_lesson19_example(inputs: dict) -> dict:
    question = inputs["question"]
    user_id = inputs["user_id"]
    thread_id = inputs["thread_id"]

    answer = "".join(agent.execute_stream(question, thread_id, user_id)).strip()

    return {
        "answer": answer
    }

if __name__ == '__main__':
    print(run_lesson19_example({
        "question": "推荐一款适合我家的扫拖一体机",
        "user_id": "0001",
        "thread_id": "0001"
    }))
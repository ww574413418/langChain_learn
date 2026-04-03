from langsmith import Client
from evals.run_lesson19_eval import run_lesson19_example
from evals.evaluators import expected_points_coverage

DATASET_NAME = "lesson19_v1_core_eval"


def main():
    client = Client()

    results = client.evaluate(
        run_lesson19_example,
        data=DATASET_NAME,
        evaluators=[expected_points_coverage],
        experiment_prefix="lesson19_v1_baseline",
        description="First offline eval for lesson19 baseline agent",
        max_concurrency=1,
    )

    print(results)


if __name__ == "__main__":
    main()

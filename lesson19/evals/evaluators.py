def expected_points_coverage(outputs: dict, reference_outputs: dict) -> dict:
    answer = outputs.get("answer", "")
    expected_points = reference_outputs.get("expected_points", [])

    hit_count = 0
    missed = []

    for point in expected_points:
        if point in answer:
            hit_count += 1
        else:
            missed.append(point)

    score = hit_count / len(expected_points) if expected_points else 0.0

    return {
        "key": "expected_points_coverage",
        "score": score,
        "comment": f"missed={missed}"
    }


if __name__ == "__main__":
    outputs = {
        "answer": "这款机器比较安静，适合你之前提到的扫地机需求。"
    }

    reference_outputs = {
        "expected_points": ["安静", "扫地机", "需求"]
    }

    result = expected_points_coverage(outputs, reference_outputs)
    print(result)

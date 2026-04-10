import json
import requests
import time

API_URL = "http://127.0.0.1:8000/ask"


def evaluate():
    with open("evaluation/test_cases.json") as f:
        test_cases = json.load(f)

    results = []

    for case in test_cases:
        query = case["query"]

        start = time.time()
        response = requests.post(API_URL, json={
            "query": query,
            "llm_model": "gpt-4.1-mini"
        }).json()
        latency = time.time() - start

        results.append({
            "query": query,
            "response": response["response"],
            "latency": round(latency, 2),
            "category": case["category"]
        })

    with open("evaluation/results/results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    evaluate()
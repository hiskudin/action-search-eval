"""
Evaluation client. Fetches daily batches from the server and submits predictions.

This shows candidates the expected workflow:
  1. Fetch the day's queries from the server
  2. Run your model on each query
  3. Submit predictions back to the server
  4. Get your score

Usage:
    python evaluate.py --day 1
"""
import argparse
import json
import requests

SERVER_URL = "http://localhost:5117"


def fetch_queries(day: int) -> list[dict]:
    r = requests.get(f"{SERVER_URL}/day/{day}")
    r.raise_for_status()
    return r.json()["queries"]


def submit_predictions(day: int, predictions: list[dict]) -> dict:
    r = requests.post(
        f"{SERVER_URL}/day/{day}/submit",
        json={"predictions": predictions},
    )
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser(description="Evaluate your model against a daily batch")
    parser.add_argument("--day", type=int, required=True, help="Day number (1-10)")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    from baseline import load_actions, build_action_index, predict, MODEL_NAME

    model = SentenceTransformer(MODEL_NAME)
    actions = load_actions()
    action_embs, action_ids = build_action_index(model, actions)

    queries = fetch_queries(args.day)
    print(f"Day {args.day}: {len(queries)} queries")

    predictions = []
    for q in queries:
        predicted_action = predict(model, action_embs, action_ids, q["query"])
        predictions.append({"id": q["id"], "action_id": predicted_action})

    result = submit_predictions(args.day, predictions)
    print(f"Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
    print("Per category:")
    for cat, stats in result["per_category"].items():
        print(f"  {cat}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    main()

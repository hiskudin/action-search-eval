"""
Evaluation client. Fetches daily batches from the server and submits predictions.

This shows candidates the expected workflow:
  1. Fetch the day's queries from the server
  2. Run your model on each query
  3. Submit predictions back to the server
  4. Get your score

Usage:
    python evaluate.py            # run all available days
    python evaluate.py --day 3    # run a single day
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
    parser = argparse.ArgumentParser(description="Evaluate your model against daily batches")
    parser.add_argument("--day", type=int, default=None, help="Single day to evaluate (default: all available)")
    args = parser.parse_args()

    # ── Load the baseline model ──
    # Replace this section with your own model.
    from sentence_transformers import SentenceTransformer
    from baseline import load_actions, build_action_index, predict, MODEL_NAME

    model = SentenceTransformer(MODEL_NAME)
    actions = load_actions()
    action_embs, action_ids = build_action_index(model, actions)

    # Determine which days to run
    if args.day is not None:
        days = [args.day]
    else:
        days = list(range(1, 11))

    total_correct = 0
    total_queries = 0
    results = {}

    for day in days:
        try:
            queries = fetch_queries(day)
        except requests.HTTPError:
            continue

        predictions = []
        for q in queries:
            predicted_action = predict(model, action_embs, action_ids, q["query"])
            predictions.append({"id": q["id"], "action_id": predicted_action})

        result = submit_predictions(day, predictions)
        results[day] = result
        total_correct += result["correct"]
        total_queries += result["total"]

        print(f"Day {day:2d}: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
        for cat, stats in result["per_category"].items():
            print(f"       {cat}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    if len(results) > 1:
        overall = total_correct / total_queries if total_queries else 0
        print(f"\nOverall: {overall:.1%} ({total_correct}/{total_queries}) across {len(results)} days")


if __name__ == "__main__":
    main()

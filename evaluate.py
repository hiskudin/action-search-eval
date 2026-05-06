"""Evaluation client. Fetches daily batches from the server and submits predictions.

Workflow:
  1. Fetch the day's queries from the server
  2. Run the model on each query
  3. Submit predictions back to the server
  4. Get the score

Usage:
    python evaluate.py            # run all available days
    python evaluate.py --day 3    # run a single day

Model lives in `models/predictor.py`. See RESULTS.md for the full iteration log.
"""
import argparse

import requests

from models.predictor import Predictor
from models.v8_ensemble import EnsemblePredictor

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
    parser.add_argument("--day", type=int, default=None,
                        help="Single day to evaluate (default: all available)")
    parser.add_argument("--online", action="store_true",
                        help="After each day's submission, fold its labels into the training pool")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use the V8 ensemble (off-the-shelf + fine-tuned encoders, weight_ft=0.25)")
    args = parser.parse_args()

    if args.ensemble:
        predictor = EnsemblePredictor(weight_ft=0.25)
        print("Using V8 ensemble (off-the-shelf + fine-tuned, w_ft=0.25)")
    else:
        predictor = Predictor()
        print(f"Using encoder: {predictor.config.encoder}")
    days = [args.day] if args.day is not None else list(range(1, 11))

    total_correct = 0
    total_queries = 0
    results = {}

    for day in days:
        try:
            queries = fetch_queries(day)
        except requests.HTTPError:
            continue

        preds = predictor.predict_batch([q["query"] for q in queries])
        predictions = [{"id": q["id"], "action_id": p} for q, p in zip(queries, preds)]

        result = submit_predictions(day, predictions)
        results[day] = result
        total_correct += result["correct"]
        total_queries += result["total"]

        print(f"Day {day:2d}: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
        for cat, stats in result["per_category"].items():
            print(f"       {cat}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

        if args.online:
            wrong_ids = {m["id"] for m in result.get("mistakes", [])}
            wrong_expected = {m["id"]: m["expected"] for m in result.get("mistakes", [])}
            update_q, update_l = [], []
            for q, p in zip(queries, predictions):
                qid = q["id"]
                label = wrong_expected[qid] if qid in wrong_ids else p["action_id"]
                update_q.append(q["query"])
                update_l.append(label)
            predictor.update(update_q, update_l)

    if len(results) > 1:
        overall = total_correct / total_queries if total_queries else 0
        print(f"\nOverall: {overall:.1%} ({total_correct}/{total_queries}) across {len(results)} days")


if __name__ == "__main__":
    main()

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

    # ── Model: kNN over labeled train queries + log-prior from train frequency ──
    # See RESULTS.md for the full iteration log. The baseline scored 33.3% on
    # days 1-10; this pipeline scores 66.7% (+33.4pt absolute).
    import json
    from collections import Counter
    from pathlib import Path

    import numpy as np
    from sentence_transformers import SentenceTransformer

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    K = 10
    PRIOR_LAMBDA = 0.1
    DATA_DIR = Path(__file__).parent / "data"

    with open(DATA_DIR / "actions.json") as f:
        actions = json.load(f)
    with open(DATA_DIR / "train.jsonl") as f:
        train = [json.loads(l) for l in f]

    action_ids = [a["id"] for a in actions]
    counts = Counter(s["action_id"] for s in train)
    total = sum(counts.values()) + len(action_ids)
    log_prior = {aid: np.log((counts.get(aid, 0) + 1) / total) for aid in action_ids}

    model = SentenceTransformer(MODEL_NAME)
    train_queries = [s["query"] for s in train]
    train_labels = [s["action_id"] for s in train]
    train_embs = model.encode(train_queries, normalize_embeddings=True, show_progress_bar=False)

    def predict_query(query: str) -> str:
        q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        sims = (q @ train_embs.T)[0]
        top_idx = np.argpartition(-sims, K)[:K]
        scores = {aid: PRIOR_LAMBDA * log_prior[aid] for aid in action_ids}
        for i in top_idx:
            scores[train_labels[i]] += float(sims[i])
        return max(scores, key=scores.get)

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
            predictions.append({"id": q["id"], "action_id": predict_query(q["query"])})

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

"""Shared eval harness — load splits, score a predictor, print per-day + dev/val/all."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def load_splits():
    splits = {"train": load_jsonl(DATA / "train.jsonl")}
    for d in range(1, 11):
        splits[f"day{d:02d}"] = load_jsonl(DATA / "days" / f"day_{d:02d}.jsonl")
    return splits


def load_actions():
    with open(DATA / "actions.json") as f:
        return json.load(f)


def evaluate(predict_batch, name="model"):
    """predict_batch(queries: list[str]) -> list[action_id]."""
    splits = load_splits()
    print(f"\n=== {name} ===")
    print(f"{'split':<10} {'acc':>7} {'n':>5}")

    per = {}
    for sp_name, samples in splits.items():
        preds = predict_batch([s["query"] for s in samples])
        correct = sum(p == s["action_id"] for p, s in zip(preds, samples))
        per[sp_name] = (correct, len(samples))
        print(f"{sp_name:<10} {correct/len(samples):>7.1%} {len(samples):>5}")

    def agg(keys):
        c = sum(per[k][0] for k in keys)
        n = sum(per[k][1] for k in keys)
        return c, n, c / n if n else 0.0

    dev_keys = [f"day{d:02d}" for d in range(1, 9)]
    val_keys = ["day09", "day10"]
    all_keys = dev_keys + val_keys

    for label, keys in [("dev (1-8)", dev_keys), ("val (9-10)", val_keys), ("all days", all_keys)]:
        c, n, a = agg(keys)
        print(f"{label:<12} {a:>6.1%}  ({c}/{n})")

    return per

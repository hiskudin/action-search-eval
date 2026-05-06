"""Diagnose model failure modes on train + days 1-10.

Defaults to the V0 baseline (action-text cosine). Pass --model v2 to diagnose
the canonical predictor in models/predictor.py.

    python scripts/diagnose.py            # baseline (V0)
    python scripts/diagnose.py --model v2 # canonical V2 predictor
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def load_actions():
    with open(DATA / "actions.json") as f:
        return json.load(f)


def baseline_predictor():
    from sentence_transformers import SentenceTransformer
    from baseline import build_action_index, MODEL_NAME

    model = SentenceTransformer(MODEL_NAME)
    actions = load_actions()
    embs, ids = build_action_index(model, actions)

    def predict_batch(queries):
        q = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
        sims = q @ embs.T
        return [ids[i] for i in sims.argmax(axis=1)]

    return predict_batch


def v2_predictor():
    from models.predictor import Predictor

    return Predictor().predict_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "v2"], default="baseline")
    args = ap.parse_args()

    predict_batch = baseline_predictor() if args.model == "baseline" else v2_predictor()

    actions = load_actions()
    action_by_id = {a["id"]: a for a in actions}

    splits = {"train": load_jsonl(DATA / "train.jsonl")}
    for d in range(1, 11):
        splits[f"day{d:02d}"] = load_jsonl(DATA / "days" / f"day_{d:02d}.jsonl")

    all_errors = []
    print(f"\n=== {args.model} ===")
    print(f"{'split':<10} {'acc':>7} {'n':>5}")
    for name, samples in splits.items():
        preds = predict_batch([s["query"] for s in samples])
        correct = sum(p == s["action_id"] for p, s in zip(preds, samples))
        print(f"{name:<10} {correct/len(samples):>7.1%} {len(samples):>5}")
        for s, p in zip(samples, preds):
            if p != s["action_id"]:
                all_errors.append({"split": name, "query": s["query"],
                                   "true": s["action_id"], "pred": p})

    print(f"\nTotal errors: {len(all_errors)}")

    by_cat = defaultdict(lambda: [0, 0])
    by_conn = defaultdict(lambda: [0, 0])
    same_connector_err = 0
    cross_connector_err = 0

    for name, samples in splits.items():
        if name == "train":
            continue
        for s in samples:
            cat = action_by_id[s["action_id"]]["category"]
            conn = action_by_id[s["action_id"]]["connector"]
            by_cat[cat][1] += 1
            by_conn[conn][1] += 1

    day_errors = [e for e in all_errors if e["split"] != "train"]
    for e in day_errors:
        cat = action_by_id[e["true"]]["category"]
        conn_t = action_by_id[e["true"]]["connector"]
        conn_p = action_by_id[e["pred"]]["connector"]
        by_cat[cat][0] += 1
        by_conn[conn_t][0] += 1
        if conn_t == conn_p:
            same_connector_err += 1
        else:
            cross_connector_err += 1

    print(f"\nError breakdown — days 1-10:")
    print(f"  cross-connector (right verb, wrong tool): {cross_connector_err} ({cross_connector_err/max(len(day_errors),1):.0%})")
    print(f"  same-connector  (wrong verb on right tool): {same_connector_err} ({same_connector_err/max(len(day_errors),1):.0%})")

    print("\nErrors by category (errs/total) — days 1-10:")
    for c, (e, t) in sorted(by_cat.items(), key=lambda x: -x[1][0]):
        print(f"  {c:<20} {e:>3}/{t:<3}  ({e/t:.0%} err)")

    print("\nErrors by connector (top 10) — days 1-10:")
    for c, (e, t) in sorted(by_conn.items(), key=lambda x: -x[1][0])[:10]:
        print(f"  {c:<20} {e:>3}/{t:<3}  ({e/t:.0%} err)")

    print("\nMost-confused (true -> pred), days 1-10:")
    confusions = Counter((e["true"], e["pred"]) for e in day_errors)
    for (t, p), n in confusions.most_common(15):
        print(f"  {n}x  {t}  ->  {p}")


if __name__ == "__main__":
    main()

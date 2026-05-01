"""Diagnose baseline failure modes on train + days 1-10."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from baseline import build_action_index, load_actions, MODEL_NAME

DATA = Path(__file__).parent / "data"


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def main():
    model = SentenceTransformer(MODEL_NAME)
    actions = load_actions()
    action_by_id = {a["id"]: a for a in actions}
    action_embs, action_ids = build_action_index(model, actions)

    splits = {"train": load_jsonl(DATA / "train.jsonl")}
    for d in range(1, 11):
        splits[f"day{d:02d}"] = load_jsonl(DATA / "days" / f"day_{d:02d}.jsonl")

    all_errors = []
    print(f"{'split':<10} {'acc':>7} {'n':>5}")
    for name, samples in splits.items():
        queries = [s["query"] for s in samples]
        q_emb = model.encode(queries, normalize_embeddings=True)
        sims = q_emb @ action_embs.T
        preds = [action_ids[i] for i in sims.argmax(axis=1)]
        correct = sum(p == s["action_id"] for p, s in zip(preds, samples))
        print(f"{name:<10} {correct/len(samples):>7.1%} {len(samples):>5}")
        for s, p in zip(samples, preds):
            if p != s["action_id"]:
                all_errors.append({"split": name, "query": s["query"],
                                   "true": s["action_id"], "pred": p})

    print(f"\nTotal errors: {len(all_errors)}")

    by_cat = defaultdict(lambda: [0, 0])
    by_conn = defaultdict(lambda: [0, 0])
    for name, samples in splits.items():
        if name == "train":
            continue
        for s in samples:
            cat = action_by_id[s["action_id"]]["category"]
            conn = action_by_id[s["action_id"]]["connector"]
            by_cat[cat][1] += 1
            by_conn[conn][1] += 1
        for e in all_errors:
            if e["split"] == name:
                cat = action_by_id[e["true"]]["category"]
                conn = action_by_id[e["true"]]["connector"]
                by_cat[cat][0] += 1
                by_conn[conn][0] += 1

    print("\nErrors by category (errs/total) — days 1-10:")
    for c, (e, t) in sorted(by_cat.items(), key=lambda x: -x[1][0]):
        print(f"  {c:<20} {e:>3}/{t:<3}  ({e/t:.0%} err)")

    print("\nErrors by connector — days 1-10:")
    for c, (e, t) in sorted(by_conn.items(), key=lambda x: -x[1][0])[:10]:
        print(f"  {c:<20} {e:>3}/{t:<3}  ({e/t:.0%} err)")

    print("\nMost-confused (true -> pred), days 1-10:")
    confusions = Counter((e["true"], e["pred"]) for e in all_errors if e["split"] != "train")
    for (t, p), n in confusions.most_common(15):
        print(f"  {n}x  {t}  ->  {p}")

    print("\nSample error queries (15):")
    day_errs = [e for e in all_errors if e["split"] != "train"]
    for e in day_errs[:15]:
        print(f"  [{e['split']}] '{e['query']}'")
        print(f"      true={e['true']}  pred={e['pred']}")


if __name__ == "__main__":
    main()

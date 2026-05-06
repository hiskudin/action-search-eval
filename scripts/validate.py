"""Validation pass — confirms the numbers in RESULTS.md are honest.

Three things:
  1. Leave-one-out accuracy on train (the raw-train number is inflated by
     self-matches in the index).
  2. Bootstrap 95% CI on V2-cold across days 1-10.
  3. Bootstrap 95% CI on V5-online across days 1-10 (sequential simulation).

CI is per-query: resample the 498 boolean correct/wrong outcomes with
replacement. Per-day bootstrap on 10 days is too unstable to be worth
reporting.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.predictor import Predictor

DATA = ROOT / "data"
RNG = np.random.default_rng(0)
N_BOOT = 5000


def load_day(d):
    with open(DATA / "days" / f"day_{d:02d}.jsonl") as f:
        return [json.loads(l) for l in f]


def bootstrap_ci(correct: np.ndarray, n=N_BOOT) -> tuple[float, float, float]:
    means = []
    n_obs = len(correct)
    for _ in range(n):
        idx = RNG.integers(0, n_obs, n_obs)
        means.append(correct[idx].mean())
    means = np.array(means)
    return float(means.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def loo_train_accuracy() -> float:
    """Predict each train query with itself excluded from the index."""
    p = Predictor()
    train_embs = p.train_embs
    train_labels = p.train_labels
    n = len(train_labels)
    # Reuse one model encode pass (already cached in train_embs).
    sims = train_embs @ train_embs.T
    np.fill_diagonal(sims, -np.inf)  # exclude self
    correct = 0
    for i in range(n):
        row = sims[i]
        top_idx = np.argpartition(-row, p.config.k)[: p.config.k]
        scores = {aid: p.config.prior_lambda * p.log_prior[aid] for aid in p.action_ids}
        for j in top_idx:
            scores[train_labels[j]] += float(row[j])
        pred = max(scores, key=scores.get)
        correct += pred == train_labels[i]
    return correct / n


def v2_cold_outcomes() -> np.ndarray:
    p = Predictor()
    out = []
    for d in range(1, 11):
        samples = load_day(d)
        preds = p.predict_batch([s["query"] for s in samples])
        for pred, s in zip(preds, samples):
            out.append(pred == s["action_id"])
    return np.array(out, dtype=bool)


def v5_online_outcomes() -> np.ndarray:
    p = Predictor()
    out = []
    for d in range(1, 11):
        samples = load_day(d)
        queries = [s["query"] for s in samples]
        preds = p.predict_batch(queries)
        for pred, s in zip(preds, samples):
            out.append(pred == s["action_id"])
        # Online update: fold true labels into the index.
        p.update(queries, [s["action_id"] for s in samples])
    return np.array(out, dtype=bool)


def main():
    print("=" * 60)
    print("Validation report")
    print("=" * 60)

    print("\n[1] Leave-one-out train accuracy (vs reported 85.0%)")
    loo = loo_train_accuracy()
    print(f"    LOO train accuracy: {loo:.1%}")

    print("\n[2] V2 cold — bootstrap 95% CI on days 1-10")
    v2 = v2_cold_outcomes()
    mean, lo, hi = bootstrap_ci(v2)
    print(f"    point  : {v2.mean():.1%} ({v2.sum()}/{len(v2)})")
    print(f"    bootstrap mean: {mean:.1%}")
    print(f"    95% CI : [{lo:.1%}, {hi:.1%}]")

    print("\n[3] V5 online — bootstrap 95% CI on days 1-10")
    v5 = v5_online_outcomes()
    mean5, lo5, hi5 = bootstrap_ci(v5)
    print(f"    point  : {v5.mean():.1%} ({v5.sum()}/{len(v5)})")
    print(f"    bootstrap mean: {mean5:.1%}")
    print(f"    95% CI : [{lo5:.1%}, {hi5:.1%}]")

    print("\n[4] V5 vs V2 paired delta")
    diff = v5.astype(int) - v2.astype(int)
    diffs_boot = []
    n_obs = len(diff)
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n_obs, n_obs)
        diffs_boot.append(diff[idx].mean())
    diffs_boot = np.array(diffs_boot)
    print(f"    mean Δ: {diff.mean():+.1%}")
    print(f"    95% CI of Δ: [{np.percentile(diffs_boot, 2.5):+.1%}, {np.percentile(diffs_boot, 97.5):+.1%}]")
    print(f"    fraction of bootstraps with Δ>0: {(diffs_boot > 0).mean():.1%}")


if __name__ == "__main__":
    main()

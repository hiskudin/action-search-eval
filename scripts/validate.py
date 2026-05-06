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


BASE = "sentence-transformers/all-MiniLM-L6-v2"


def base_predictor() -> Predictor:
    from models.predictor import Config
    return Predictor(Config(encoder=BASE))


def loo_train_accuracy() -> float:
    """Predict each train query with itself excluded from the index."""
    p = base_predictor()
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
    return _cold_outcomes(base_predictor())


def v5_online_outcomes() -> np.ndarray:
    return _online_outcomes(base_predictor())


def v6_outcomes(online: bool) -> np.ndarray:
    from models.predictor import Config
    ft_path = ROOT / "models" / "ft_minilm"
    p = Predictor(Config(encoder=str(ft_path)))
    return _online_outcomes(p) if online else _cold_outcomes(p)


def _cold_outcomes(p: Predictor) -> np.ndarray:
    out = []
    for d in range(1, 11):
        samples = load_day(d)
        preds = p.predict_batch([s["query"] for s in samples])
        for pred, s in zip(preds, samples):
            out.append(pred == s["action_id"])
    return np.array(out, dtype=bool)


def _online_outcomes(p: Predictor) -> np.ndarray:
    out = []
    for d in range(1, 11):
        samples = load_day(d)
        queries = [s["query"] for s in samples]
        preds = p.predict_batch(queries)
        for pred, s in zip(preds, samples):
            out.append(pred == s["action_id"])
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

    def paired_delta(a, b, label):
        diff = b.astype(int) - a.astype(int)
        diffs_boot = []
        n_obs = len(diff)
        for _ in range(N_BOOT):
            idx = RNG.integers(0, n_obs, n_obs)
            diffs_boot.append(diff[idx].mean())
        diffs_boot = np.array(diffs_boot)
        print(f"    {label}: mean Δ {diff.mean():+.1%}, 95% CI "
              f"[{np.percentile(diffs_boot, 2.5):+.1%}, "
              f"{np.percentile(diffs_boot, 97.5):+.1%}], "
              f"P(Δ>0)={(diffs_boot > 0).mean():.1%}")

    print("\n[4] Paired deltas")
    paired_delta(v2, v5, "V5 - V2     ")

    print("\n[5] V6 cold (fine-tuned encoder, no online)")
    v6c = v6_outcomes(online=False)
    mean6c, lo6c, hi6c = bootstrap_ci(v6c)
    print(f"    point  : {v6c.mean():.1%} ({v6c.sum()}/{len(v6c)})")
    print(f"    95% CI : [{lo6c:.1%}, {hi6c:.1%}]")

    print("\n[6] V6 online (fine-tuned encoder + online learning)")
    v6o = v6_outcomes(online=True)
    mean6o, lo6o, hi6o = bootstrap_ci(v6o)
    print(f"    point  : {v6o.mean():.1%} ({v6o.sum()}/{len(v6o)})")
    print(f"    95% CI : [{lo6o:.1%}, {hi6o:.1%}]")

    print("\n[7] More paired deltas")
    paired_delta(v2, v6c, "V6c - V2    ")
    paired_delta(v5, v6o, "V6o - V5    ")
    paired_delta(v2, v6o, "V6o - V2    ")

    print("\n[8] V8 ensemble (online, weight_ft=0.25)")
    from models.v8_ensemble import evaluate as v8_eval
    v8, _ = v8_eval(weight_ft=0.25, online=True)
    mean8, lo8, hi8 = bootstrap_ci(v8)
    print(f"    point  : {v8.mean():.1%} ({v8.sum()}/{len(v8)})")
    print(f"    95% CI : [{lo8:.1%}, {hi8:.1%}]")

    print("\n[9] Paired deltas vs V8")
    paired_delta(v6o, v8, "V8 - V6o    ")
    paired_delta(v5, v8, "V8 - V5     ")
    paired_delta(v2, v8, "V8 - V2     ")


if __name__ == "__main__":
    main()

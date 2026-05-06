"""V8: ensemble of off-the-shelf and fine-tuned encoders.

V6 day 10 was 76% but V5 day 10 was 86% — V6 regressed on some days. The
ensemble averages per-action scores from both encoders so each picks up the
other's slack.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.predictor import Config, Predictor

BASE = "sentence-transformers/all-MiniLM-L6-v2"
FT_PATH = ROOT / "models" / "ft_minilm"
DATA = ROOT / "data"


class EnsemblePredictor:
    def __init__(self, weight_ft: float = 0.5):
        if not FT_PATH.exists():
            raise FileNotFoundError(
                f"{FT_PATH} not found — run scripts/finetune.py first."
            )
        self.base = Predictor(Config(encoder=BASE))
        self.ft = Predictor(Config(encoder=str(FT_PATH)))
        self.weight_ft = weight_ft
        self.action_ids = self.base.action_ids

    def predict_batch(self, queries: list[str]) -> list[str]:
        s_base = self.base.score_batch(queries)
        s_ft = self.ft.score_batch(queries)
        out = []
        w_ft = self.weight_ft
        w_base = 1.0 - w_ft
        for sb, sf in zip(s_base, s_ft):
            combined = {aid: w_base * sb[aid] + w_ft * sf[aid] for aid in self.action_ids}
            out.append(max(combined, key=combined.get))
        return out

    def update(self, queries: list[str], labels: list[str]) -> None:
        self.base.update(queries, labels)
        self.ft.update(queries, labels)


def load_day(d):
    with open(DATA / "days" / f"day_{d:02d}.jsonl") as f:
        return [json.loads(l) for l in f]


def evaluate(weight_ft: float, online: bool) -> tuple[np.ndarray, list[float]]:
    p = EnsemblePredictor(weight_ft=weight_ft)
    out = []
    per_day = []
    for d in range(1, 11):
        samples = load_day(d)
        queries = [s["query"] for s in samples]
        preds = p.predict_batch(queries)
        day_correct = []
        for pred, s in zip(preds, samples):
            ok = pred == s["action_id"]
            out.append(ok); day_correct.append(ok)
        per_day.append(sum(day_correct) / len(day_correct))
        if online:
            p.update(queries, [s["action_id"] for s in samples])
    return np.array(out, dtype=bool), per_day


if __name__ == "__main__":
    print("Sweep weight_ft (online=True):")
    for w in [0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 1.0]:
        outcomes, _ = evaluate(weight_ft=w, online=True)
        print(f"  weight_ft={w:>4}  acc={outcomes.mean():.1%}")

"""V6: V2/V5 pipeline on top of a fine-tuned MiniLM encoder.

Run scripts/finetune.py first to produce models/ft_minilm/.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.predictor import Config, Predictor

FT_PATH = Path(__file__).resolve().parent / "ft_minilm"
DATA = Path(__file__).resolve().parent.parent / "data"


def predictor_v6() -> Predictor:
    if not FT_PATH.exists():
        raise FileNotFoundError(
            f"{FT_PATH} not found — run `python scripts/finetune.py` first."
        )
    return Predictor(Config(encoder=str(FT_PATH)))


def load_day(d):
    with open(DATA / "days" / f"day_{d:02d}.jsonl") as f:
        return [json.loads(l) for l in f]


def evaluate(online: bool = False) -> tuple[np.ndarray, list[float]]:
    p = predictor_v6()
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
    for online in [False, True]:
        outcomes, per_day = evaluate(online=online)
        label = "V6 cold (fine-tuned, no online)" if not online else "V6 online (fine-tuned + V5)"
        print(f"\n=== {label} ===")
        for d, acc in enumerate(per_day, 1):
            print(f"Day {d:2d}: {acc:.1%}")
        print(f"Overall: {outcomes.mean():.1%} ({outcomes.sum()}/{len(outcomes)})")

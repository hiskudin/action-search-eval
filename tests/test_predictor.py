"""Regression test on the canonical predictor.

Pins V2's accuracy on days 1-10 above a floor so refactors that silently break
the model fail loudly. The current measured floor is 66.7% (332/498); we leave
some headroom for hyperparameter changes and pin at 60%.
"""
import json
from pathlib import Path

import pytest

from models.predictor import Predictor

DATA = Path(__file__).resolve().parent.parent / "data"
FLOOR = 0.60


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def _load_day(d: int):
    with open(DATA / "days" / f"day_{d:02d}.jsonl") as f:
        return [json.loads(l) for l in f]


def test_predictor_returns_known_action_ids(predictor):
    sample = _load_day(1)[:5]
    preds = predictor.predict_batch([s["query"] for s in sample])
    assert all(p in predictor.action_ids for p in preds)
    assert len(preds) == 5


def test_predictor_meets_accuracy_floor(predictor):
    correct = total = 0
    for d in range(1, 11):
        samples = _load_day(d)
        preds = predictor.predict_batch([s["query"] for s in samples])
        correct += sum(p == s["action_id"] for p, s in zip(preds, samples))
        total += len(samples)
    acc = correct / total
    assert acc >= FLOOR, f"Predictor accuracy {acc:.1%} below floor {FLOOR:.0%}"

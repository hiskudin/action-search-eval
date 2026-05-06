"""Canonical predictor (V2): kNN over labeled train queries + log-prior.

Single source of truth for the production pipeline. Both `evaluate.py` and the
in-process eval harness (`models/_eval.py`) import from here.

Iteration history and ablations live alongside in `models/v1_knn.py`,
`models/v2_prior.py`, `models/v3_rerank.py`, `models/v3b_rerank_train.py`,
`models/v4_hybrid.py`. RESULTS.md documents the full iteration log.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FT_PATH = Path(__file__).resolve().parent / "ft_minilm"


def _default_encoder() -> str:
    """Use the fine-tuned MiniLM if `scripts/finetune.py` has been run; else off-the-shelf."""
    if FT_PATH.exists() and (FT_PATH / "config.json").exists():
        return str(FT_PATH)
    return "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Config:
    encoder: str = ""  # populated from _default_encoder() if empty
    k: int = 10
    prior_lambda: float = 0.1
    smoothing: float = 1.0


CONFIG = Config()


class Predictor:
    def __init__(self, config: Config = CONFIG):
        if not config.encoder:
            config = Config(encoder=_default_encoder(), k=config.k,
                            prior_lambda=config.prior_lambda, smoothing=config.smoothing)
        self.config = config
        with open(DATA_DIR / "actions.json") as f:
            actions = json.load(f)
        with open(DATA_DIR / "train.jsonl") as f:
            train = [json.loads(l) for l in f]

        self.action_ids = [a["id"] for a in actions]
        self.train_labels = [s["action_id"] for s in train]
        train_queries = [s["query"] for s in train]

        counts = Counter(self.train_labels)
        total = sum(counts.values()) + config.smoothing * len(self.action_ids)
        self.log_prior = {
            aid: float(np.log((counts.get(aid, 0) + config.smoothing) / total))
            for aid in self.action_ids
        }

        self.model = SentenceTransformer(config.encoder)
        self.train_embs = self.model.encode(
            train_queries, normalize_embeddings=True, show_progress_bar=False
        )
        self._counts = counts

    def update(self, queries: list[str], labels: list[str]) -> None:
        """Append (query, label) pairs to the kNN index and refresh the prior.

        Used for online learning: after each day's predictions are graded,
        fold the day's true labels into the training pool for future days.
        """
        if not queries:
            return
        new_embs = self.model.encode(
            queries, normalize_embeddings=True, show_progress_bar=False
        )
        self.train_embs = np.vstack([self.train_embs, new_embs])
        self.train_labels.extend(labels)
        for label in labels:
            self._counts[label] += 1
        total = sum(self._counts.values()) + self.config.smoothing * len(self.action_ids)
        self.log_prior = {
            aid: float(np.log((self._counts.get(aid, 0) + self.config.smoothing) / total))
            for aid in self.action_ids
        }

    def score_batch(self, queries: list[str]) -> list[dict[str, float]]:
        """Return the per-action score dict for each query (no argmax)."""
        if not queries:
            return []
        q = self.model.encode(
            queries, normalize_embeddings=True, show_progress_bar=False
        )
        sims = q @ self.train_embs.T
        out = []
        for row in sims:
            top_idx = np.argpartition(-row, self.config.k)[: self.config.k]
            scores = {
                aid: self.config.prior_lambda * self.log_prior[aid]
                for aid in self.action_ids
            }
            for i in top_idx:
                scores[self.train_labels[i]] += float(row[i])
            out.append(scores)
        return out

    def predict_batch(self, queries: list[str]) -> list[str]:
        return [max(s, key=s.get) for s in self.score_batch(queries)]

    def predict(self, query: str) -> str:
        return self.predict_batch([query])[0]

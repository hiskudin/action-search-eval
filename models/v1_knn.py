"""V1: kNN over training queries + action descriptions, weighted vote."""
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from _eval import evaluate, load_actions, load_splits

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K = 10


def build_predictor():
    model = SentenceTransformer(MODEL)
    actions = load_actions()
    train = load_splits()["train"]
    action_ids = [a["id"] for a in actions]

    texts = [s["query"] for s in train]
    labels = [s["action_id"] for s in train]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def predict_batch(queries):
        q = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
        sims = q @ embs.T
        out = []
        for row in sims:
            top_idx = np.argpartition(-row, K)[:K]
            scores = {aid: 0.0 for aid in action_ids}
            for i in top_idx:
                scores[labels[i]] += float(row[i])
            out.append(max(scores, key=scores.get))
        return out

    return predict_batch


if __name__ == "__main__":
    evaluate(build_predictor(), name=f"V1 kNN (k={K})")

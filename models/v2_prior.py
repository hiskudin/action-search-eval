"""V2: V1 kNN + log-prior from train action frequency."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from _eval import evaluate, load_actions, load_splits

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K = 10
PRIOR_LAMBDA = 0.1  # tuned on dev (see RESULTS.md)
SMOOTHING = 1.0


def build_predictor(prior_lambda=PRIOR_LAMBDA):
    model = SentenceTransformer(MODEL)
    actions = load_actions()
    action_ids = [a["id"] for a in actions]
    train = load_splits()["train"]

    counts = Counter(s["action_id"] for s in train)
    total = sum(counts.values()) + SMOOTHING * len(action_ids)
    log_prior = {
        aid: np.log((counts.get(aid, 0) + SMOOTHING) / total) for aid in action_ids
    }

    texts = [s["query"] for s in train]
    labels = [s["action_id"] for s in train]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def predict_batch(queries):
        q = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
        sims = q @ embs.T
        out = []
        for row in sims:
            top_idx = np.argpartition(-row, K)[:K]
            scores = {aid: prior_lambda * log_prior[aid] for aid in action_ids}
            for i in top_idx:
                scores[labels[i]] += float(row[i])
            out.append(max(scores, key=scores.get))
        return out

    return predict_batch


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        # Sweep prior_lambda on dev
        for lam in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]:
            pred = build_predictor(prior_lambda=lam)
            splits = load_splits()
            dev_c = dev_n = val_c = val_n = 0
            for d in range(1, 11):
                samples = splits[f"day{d:02d}"]
                preds = pred([s["query"] for s in samples])
                for p, s in zip(preds, samples):
                    if d <= 8:
                        dev_n += 1
                        dev_c += p == s["action_id"]
                    else:
                        val_n += 1
                        val_c += p == s["action_id"]
            print(f"  lambda={lam:>4}  dev={dev_c/dev_n:.1%}  val={val_c/val_n:.1%}")
    else:
        evaluate(build_predictor(), name=f"V2 kNN+prior (k={K}, lambda={PRIOR_LAMBDA})")

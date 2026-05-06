"""V3b: bi-encoder kNN over train -> cross-encoder reranks (query, train_query) pairs."""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from _eval import evaluate, load_actions, load_splits

BIENC = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "BAAI/bge-reranker-base"
CAND_K = 20
PRIOR_LAMBDA = 0.1


def build_predictor(cand_k=CAND_K, alpha=1.0):
    bi = SentenceTransformer(BIENC)
    ce = CrossEncoder(RERANKER)
    actions = load_actions()
    action_ids = [a["id"] for a in actions]
    train = load_splits()["train"]

    counts = Counter(s["action_id"] for s in train)
    total = sum(counts.values()) + len(action_ids)
    log_prior = {aid: np.log((counts.get(aid, 0) + 1) / total) for aid in action_ids}

    train_queries = [s["query"] for s in train]
    train_labels = [s["action_id"] for s in train]
    embs = bi.encode(train_queries, normalize_embeddings=True, show_progress_bar=False)

    def predict_batch(queries):
        q = bi.encode(queries, normalize_embeddings=True, show_progress_bar=False)
        sims = q @ embs.T

        # Build pairs: (query, train_query) for top-cand_k train neighbors
        pairs, owners = [], []
        cand_idx_per_q = []
        for qi, row in enumerate(sims):
            top_idx = np.argpartition(-row, cand_k)[:cand_k]
            cand_idx_per_q.append(top_idx)
            for ti in top_idx:
                pairs.append((queries[qi], train_queries[ti]))
                owners.append((qi, ti))

        ce_scores = ce.predict(pairs, show_progress_bar=False)
        # Aggregate per (query, action_id) using rerank scores (sigmoid)
        rerank_by_q_t = {(qi, ti): float(s) for (qi, ti), s in zip(owners, ce_scores)}

        out = []
        for qi, top_idx in enumerate(cand_idx_per_q):
            scores = {aid: PRIOR_LAMBDA * log_prior[aid] for aid in action_ids}
            for ti in top_idx:
                # combine: bi-enc cosine + alpha * sigmoid(rerank logit)
                bi_s = float(sims[qi][ti])
                r = rerank_by_q_t[(qi, ti)]
                r_sig = 1.0 / (1.0 + np.exp(-r))
                scores[train_labels[ti]] += bi_s + alpha * r_sig
            out.append(max(scores, key=scores.get))
        return out

    return predict_batch


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        for cand_k, alpha in [(10, 0.5), (10, 1.0), (20, 0.5), (20, 1.0), (20, 2.0), (30, 1.0)]:
            pred = build_predictor(cand_k=cand_k, alpha=alpha)
            splits = load_splits()
            dev_c = dev_n = val_c = val_n = 0
            for d in range(1, 11):
                samples = splits[f"day{d:02d}"]
                preds = pred([s["query"] for s in samples])
                for p, s in zip(preds, samples):
                    if d <= 8:
                        dev_n += 1; dev_c += p == s["action_id"]
                    else:
                        val_n += 1; val_c += p == s["action_id"]
            print(f"  cand_k={cand_k} alpha={alpha}  dev={dev_c/dev_n:.1%}  val={val_c/val_n:.1%}")
    else:
        evaluate(build_predictor(), name=f"V3b rerank-train (cand_k={CAND_K})")

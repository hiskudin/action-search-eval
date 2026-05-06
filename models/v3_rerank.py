"""V3: V2 candidate set + cross-encoder reranker over (query, action_text)."""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from _eval import evaluate, load_actions, load_splits

BIENC = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "BAAI/bge-reranker-base"
K = 10
TOP_N = 5
PRIOR_LAMBDA = 0.1


def build_action_text(actions, train, with_examples=True, max_examples=3):
    """For each action, build text the reranker will score against."""
    by_action = defaultdict(list)
    for s in train:
        by_action[s["action_id"]].append(s["query"])
    out = {}
    for a in actions:
        base = f"{a['label']}: {a['description']}"
        if with_examples and by_action[a["id"]]:
            ex = by_action[a["id"]][:max_examples]
            base += " Examples: " + "; ".join(ex)
        out[a["id"]] = base
    return out


def build_predictor(top_n=TOP_N, with_examples=True, alpha=0.0):
    bi = SentenceTransformer(BIENC)
    ce = CrossEncoder(RERANKER)
    actions = load_actions()
    action_ids = [a["id"] for a in actions]
    train = load_splits()["train"]

    counts = Counter(s["action_id"] for s in train)
    total = sum(counts.values()) + len(action_ids)
    log_prior = {aid: np.log((counts.get(aid, 0) + 1) / total) for aid in action_ids}

    texts = [s["query"] for s in train]
    labels = [s["action_id"] for s in train]
    embs = bi.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    action_text = build_action_text(actions, train, with_examples=with_examples)

    def predict_batch(queries):
        q = bi.encode(queries, normalize_embeddings=True, show_progress_bar=False)
        sims = q @ embs.T
        # Stage 1: V2 scoring -> top_n candidates per query, keep V2 scores
        v2_scores_per_q = []
        candidates_per_q = []
        for row in sims:
            top_idx = np.argpartition(-row, K)[:K]
            scores = {aid: PRIOR_LAMBDA * log_prior[aid] for aid in action_ids}
            for i in top_idx:
                scores[labels[i]] += float(row[i])
            ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
            v2_scores_per_q.append({aid: s for aid, s in ranked})
            candidates_per_q.append([aid for aid, _ in ranked])

        # Stage 2: rerank
        pairs, owners = [], []
        for qi, (query, cands) in enumerate(zip(queries, candidates_per_q)):
            for c in cands:
                pairs.append((query, action_text[c]))
                owners.append((qi, c))
        ce_scores_raw = ce.predict(pairs, show_progress_bar=False)

        rerank_per_q = defaultdict(dict)
        for (qi, c), s in zip(owners, ce_scores_raw):
            rerank_per_q[qi][c] = float(s)

        out = []
        for qi in range(len(queries)):
            r = rerank_per_q[qi]
            v = v2_scores_per_q[qi]
            # softmax-normalize rerank within candidates so alpha is interpretable
            rs = np.array(list(r.values()))
            rs = np.exp(rs - rs.max()); rs = rs / rs.sum()
            r_norm = dict(zip(r.keys(), rs))
            combined = {c: v[c] + alpha * r_norm[c] for c in r}
            out.append(max(combined, key=combined.get))
        return out

    return predict_batch


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        for top_n, with_ex, alpha in [
            (5, True, 0.0), (5, True, 0.05), (5, True, 0.1), (5, True, 0.2),
            (5, True, 0.5), (5, True, 1.0), (5, False, 0.1), (3, True, 0.1),
        ]:
            pred = build_predictor(top_n=top_n, with_examples=with_ex, alpha=alpha)
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
            print(f"  top_n={top_n} ex={with_ex} alpha={alpha:>4}  dev={dev_c/dev_n:.1%}  val={val_c/val_n:.1%}")
    else:
        evaluate(build_predictor(), name=f"V3 rerank (top_n={TOP_N})")

"""V4: V2 dense + TF-IDF lexical, fused via reciprocal rank fusion.

Also includes an encoder-swap experiment (MiniLM -> bge-small-en-v1.5).
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).parent))
from _eval import evaluate, load_actions, load_splits

K = 10
PRIOR_LAMBDA = 0.1


def build_v2_scores(model, action_ids, log_prior, train_embs, train_labels, queries):
    q = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    sims = q @ train_embs.T
    out = []
    for row in sims:
        top_idx = np.argpartition(-row, K)[:K]
        scores = {aid: PRIOR_LAMBDA * log_prior[aid] for aid in action_ids}
        for i in top_idx:
            scores[train_labels[i]] += float(row[i])
        out.append(scores)
    return out


def build_predictor(encoder_name, use_lexical=True, rrf_k=60):
    model = SentenceTransformer(encoder_name)
    actions = load_actions()
    action_ids = [a["id"] for a in actions]
    train = load_splits()["train"]

    counts = Counter(s["action_id"] for s in train)
    total = sum(counts.values()) + len(action_ids)
    log_prior = {aid: np.log((counts.get(aid, 0) + 1) / total) for aid in action_ids}

    train_queries = [s["query"] for s in train]
    train_labels = [s["action_id"] for s in train]
    train_embs = model.encode(train_queries, normalize_embeddings=True, show_progress_bar=False)

    # Lexical index: train queries + action descriptions
    lex_texts = list(train_queries)
    lex_labels = list(train_labels)
    for a in actions:
        lex_texts.append(f"{a['connector']} {a['label']} {a['description']}")
        lex_labels.append(a["id"])
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    lex_mat = vec.fit_transform(lex_texts)

    def predict_batch(queries):
        dense_scores = build_v2_scores(
            model, action_ids, log_prior, train_embs, train_labels, queries
        )

        if not use_lexical:
            return [max(s, key=s.get) for s in dense_scores]

        q_lex = vec.transform(queries)
        lex_sims = (q_lex @ lex_mat.T).toarray()  # (Q, N_lex)
        # Aggregate lexical per action: max similarity over rows owned by that action.
        out = []
        for qi, row in enumerate(lex_sims):
            lex_per_action = defaultdict(float)
            for j, score in enumerate(row):
                if score > lex_per_action[lex_labels[j]]:
                    lex_per_action[lex_labels[j]] = score

            # RRF: rank both score vectors over all action_ids
            d_rank = {aid: r for r, aid in enumerate(
                sorted(action_ids, key=lambda a: -dense_scores[qi][a]))}
            l_rank = {aid: r for r, aid in enumerate(
                sorted(action_ids, key=lambda a: -lex_per_action.get(a, 0.0)))}
            fused = {aid: 1.0 / (rrf_k + d_rank[aid]) + 1.0 / (rrf_k + l_rank[aid])
                     for aid in action_ids}
            out.append(max(fused, key=fused.get))
        return out

    return predict_batch


if __name__ == "__main__":
    splits = load_splits()
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        configs = [
            ("sentence-transformers/all-MiniLM-L6-v2", False, 60, "MiniLM dense-only (=V2)"),
            ("sentence-transformers/all-MiniLM-L6-v2", True, 60, "MiniLM + TFIDF RRF k=60"),
            ("sentence-transformers/all-MiniLM-L6-v2", True, 30, "MiniLM + TFIDF RRF k=30"),
            ("BAAI/bge-small-en-v1.5", False, 60, "bge-small dense-only"),
            ("BAAI/bge-small-en-v1.5", True, 60, "bge-small + TFIDF RRF k=60"),
        ]
        for enc, use_lex, rrfk, label in configs:
            pred = build_predictor(enc, use_lexical=use_lex, rrf_k=rrfk)
            dev_c = dev_n = val_c = val_n = 0
            for d in range(1, 11):
                samples = splits[f"day{d:02d}"]
                preds = pred([s["query"] for s in samples])
                for p, s in zip(preds, samples):
                    if d <= 8:
                        dev_n += 1; dev_c += p == s["action_id"]
                    else:
                        val_n += 1; val_c += p == s["action_id"]
            print(f"  {label:<40} dev={dev_c/dev_n:.1%}  val={val_c/val_n:.1%}")
    else:
        evaluate(
            build_predictor("sentence-transformers/all-MiniLM-L6-v2", use_lexical=True),
            name="V4 hybrid",
        )

"""
Minimal baseline: embed actions and queries with MiniLM, match by cosine similarity.
"""
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_actions():
    with open(DATA_DIR / "actions.json") as f:
        return json.load(f)


def build_action_index(model, actions):
    """Embed all action descriptions, return (embeddings, action_ids)."""
    texts = [f"{a['label']}: {a['description']}" for a in actions]
    embeddings = model.encode(texts, normalize_embeddings=True)
    action_ids = [a["id"] for a in actions]
    return embeddings, action_ids


def predict(model, action_embeddings, action_ids, query):
    """Return the best-matching action_id for a query."""
    q_emb = model.encode([query], normalize_embeddings=True)
    sims = (q_emb @ action_embeddings.T)[0]
    best = int(np.argmax(sims))
    return action_ids[best]


def evaluate_file(jsonl_path):
    """Run baseline on a JSONL file, print accuracy."""
    model = SentenceTransformer(MODEL_NAME)
    actions = load_actions()
    action_embs, action_ids = build_action_index(model, actions)

    correct = 0
    total = 0
    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            pred = predict(model, action_embs, action_ids, sample["query"])
            if pred == sample["action_id"]:
                correct += 1
            total += 1

    print(f"Accuracy: {correct}/{total} = {correct/total:.3f}")


if __name__ == "__main__":
    evaluate_file(DATA_DIR / "train.jsonl")

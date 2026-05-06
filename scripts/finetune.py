"""Fine-tune MiniLM with MultipleNegativesRankingLoss on query-query pairs.

For each pair of train queries that share an action_id, create (anchor, positive).
MNRL treats every OTHER positive in the batch as a negative, so in-batch hard
negatives arise naturally — and we ensure same-verb-different-connector pairs
land in the same batch by ordering.

This formulation matches the inference pattern: at test time we compute
query-to-train-query cosine similarity for kNN, so we train for exactly that.
"""
import json
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

DATA = ROOT / "data"
OUT_DIR = ROOT / "models" / "ft_minilm"
SEED = 0
EPOCHS = 5
BATCH = 32
RANDOM = random.Random(SEED)


def verb_from_id(aid: str) -> str:
    parts = aid.split("_", 1)
    return parts[1] if len(parts) > 1 else aid


def main():
    with open(DATA / "actions.json") as f:
        actions = json.load(f)
    with open(DATA / "train.jsonl") as f:
        train = [json.loads(l) for l in f]

    by_action: dict[str, list[str]] = defaultdict(list)
    for s in train:
        by_action[s["action_id"]].append(s["query"])

    pairs = []
    for aid, queries in by_action.items():
        for q1, q2 in combinations(queries, 2):
            pairs.append((q1, q2, aid))
    RANDOM.shuffle(pairs)

    # Order pairs so same-verb actions are adjacent — increases the chance
    # that a batch contains both `slack_send_message` and `teams_send_message`
    # pairs, which are the hard negatives we want.
    by_verb: dict[str, list] = defaultdict(list)
    for p in pairs:
        by_verb[verb_from_id(p[2])].append(p)
    interleaved = []
    for verb_pairs in by_verb.values():
        interleaved.extend(verb_pairs)
    pairs = interleaved

    examples = [InputExample(texts=[a, b]) for a, b, _ in pairs]
    print(f"Pairs: {len(examples)}, batch={BATCH}, epochs={EPOCHS}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    loader = DataLoader(examples, shuffle=False, batch_size=BATCH)
    loss = losses.MultipleNegativesRankingLoss(model=model)

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=EPOCHS,
        warmup_steps=int(0.1 * len(loader) * EPOCHS),
        output_path=str(OUT_DIR),
        show_progress_bar=False,
    )
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

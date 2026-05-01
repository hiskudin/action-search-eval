"""
Evaluation server. Serves daily batches of queries for candidates to predict against.

Usage:
    python server.py

Endpoints:
    GET  /actions          - full action catalog
    GET  /data/train       - initial training data
    GET  /day/<n>          - day n evaluation queries (no labels)
    POST /day/<n>/submit   - submit predictions for day n, returns metrics
"""
import json
from pathlib import Path
from flask import Flask, jsonify, request

app = Flask(__name__)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
EXPOSED_DAYS_DIR = DATA_DIR / "days"       # days 1-10 (candidate-visible)
HIDDEN_DAYS_DIR = ROOT / "_internal" / "days"  # days 11-30 (held-out)

# Load data at startup
with open(DATA_DIR / "actions.json") as f:
    ACTIONS = json.load(f)

TRAIN_DATA = []
with open(DATA_DIR / "train.jsonl") as f:
    for line in f:
        TRAIN_DATA.append(json.loads(line))

DAY_DATA = {}
for days_dir in [EXPOSED_DAYS_DIR, HIDDEN_DAYS_DIR]:
    if not days_dir.exists():
        continue
    for p in sorted(days_dir.glob("day_*.jsonl")):
        day_num = int(p.stem.split("_")[1])
        samples = []
        with open(p) as f:
            for line in f:
                samples.append(json.loads(line))
        DAY_DATA[day_num] = samples


@app.route("/actions")
def get_actions():
    return jsonify(ACTIONS)


@app.route("/data/train")
def get_train():
    return jsonify(TRAIN_DATA)


@app.route("/day/<int:day_num>")
def get_day(day_num):
    """Return queries for a given day (no labels)."""
    if day_num not in DAY_DATA:
        return jsonify({"error": f"Day {day_num} not available. Days 1-{max(DAY_DATA.keys())} exist."}), 404
    queries = [{"id": i, "query": s["query"]} for i, s in enumerate(DAY_DATA[day_num])]
    return jsonify({"day": day_num, "queries": queries})


@app.route("/day/<int:day_num>/submit", methods=["POST"])
def submit_day(day_num):
    """
    Submit predictions for a day. Expects JSON:
      {"predictions": [{"id": 0, "action_id": "slack_send_message"}, ...]}

    For ranked predictions (top-k), use action_ids instead of action_id:
      {"predictions": [{"id": 0, "action_ids": ["slack_send_message", "teams_send_message"]}, ...]}

    Returns accuracy, per-category breakdown, mistakes, and (if ranked) top-3 accuracy and MRR.
    """
    if day_num not in DAY_DATA:
        return jsonify({"error": f"Day {day_num} not found"}), 404

    body = request.get_json()
    if not body or "predictions" not in body:
        return jsonify({"error": "Expected {predictions: [{id, action_id}, ...]}"}), 400

    # Parse predictions - support both action_id (single) and action_ids (ranked)
    preds_single = {}
    preds_ranked = {}
    has_ranked = False
    for p in body["predictions"]:
        pid = p["id"]
        if "action_ids" in p:
            has_ranked = True
            preds_ranked[pid] = p["action_ids"]
            preds_single[pid] = p["action_ids"][0] if p["action_ids"] else None
        else:
            preds_single[pid] = p.get("action_id")

    ground_truth = DAY_DATA[day_num]
    action_cat = {a["id"]: a["category"] for a in ACTIONS}

    correct = 0
    total = len(ground_truth)
    cat_correct = {}
    cat_total = {}
    mistakes = []
    top3_correct = 0
    rr_sum = 0.0

    for i, sample in enumerate(ground_truth):
        cat = action_cat.get(sample["action_id"], "unknown")
        cat_total[cat] = cat_total.get(cat, 0) + 1
        expected = sample["action_id"]
        predicted = preds_single.get(i)

        if predicted == expected:
            correct += 1
            cat_correct[cat] = cat_correct.get(cat, 0) + 1
        else:
            mistakes.append({
                "id": i,
                "query": sample["query"],
                "predicted": predicted,
                "expected": expected,
            })

        if has_ranked and i in preds_ranked:
            ranked = preds_ranked[i]
            if expected in ranked[:3]:
                top3_correct += 1
            if expected in ranked:
                rr_sum += 1.0 / (ranked.index(expected) + 1)

    accuracy = correct / total if total > 0 else 0
    per_category = {}
    for cat in cat_total:
        per_category[cat] = {
            "correct": cat_correct.get(cat, 0),
            "total": cat_total[cat],
            "accuracy": cat_correct.get(cat, 0) / cat_total[cat],
        }

    result = {
        "day": day_num,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_category": per_category,
        "mistakes": mistakes,
    }

    if has_ranked:
        result["top_3_accuracy"] = round(top3_correct / total, 4) if total > 0 else 0
        result["mrr"] = round(rr_sum / total, 4) if total > 0 else 0

    return jsonify(result)


if __name__ == "__main__":
    print("Action Search Eval Server")
    print(f"  {len(ACTIONS)} actions loaded")
    print(f"  {len(TRAIN_DATA)} training samples")
    print(f"  {len(DAY_DATA)} daily batches")
    app.run(port=5117, debug=False)

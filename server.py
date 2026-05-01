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
DAYS_DIR = ROOT / "_internal" / "days"

# Load data at startup
with open(DATA_DIR / "actions.json") as f:
    ACTIONS = json.load(f)

TRAIN_DATA = []
with open(DATA_DIR / "train.jsonl") as f:
    for line in f:
        TRAIN_DATA.append(json.loads(line))

DAY_DATA = {}
for p in sorted(DAYS_DIR.glob("day_*.jsonl")):
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
    Returns accuracy and per-category breakdown.
    """
    if day_num not in DAY_DATA:
        return jsonify({"error": f"Day {day_num} not found"}), 404

    body = request.get_json()
    if not body or "predictions" not in body:
        return jsonify({"error": "Expected {predictions: [{id, action_id}, ...]}"}), 400

    preds = {p["id"]: p["action_id"] for p in body["predictions"]}
    ground_truth = DAY_DATA[day_num]

    correct = 0
    total = len(ground_truth)
    cat_correct = {}
    cat_total = {}

    action_cat = {a["id"]: a["category"] for a in ACTIONS}

    for i, sample in enumerate(ground_truth):
        cat = action_cat.get(sample["action_id"], "unknown")
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if preds.get(i) == sample["action_id"]:
            correct += 1
            cat_correct[cat] = cat_correct.get(cat, 0) + 1

    accuracy = correct / total if total > 0 else 0
    per_category = {}
    for cat in cat_total:
        per_category[cat] = {
            "correct": cat_correct.get(cat, 0),
            "total": cat_total[cat],
            "accuracy": cat_correct.get(cat, 0) / cat_total[cat],
        }

    return jsonify({
        "day": day_num,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_category": per_category,
    })


if __name__ == "__main__":
    print(f"Action Search Eval Server")
    print(f"  {len(ACTIONS)} actions loaded")
    print(f"  {len(TRAIN_DATA)} training samples")
    print(f"  {len(DAY_DATA)} daily batches")
    app.run(port=5117, debug=False)

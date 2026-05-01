"""
Tests for action-search-eval.

Covers: data loading, server endpoints, baseline predictions, scoring logic, edge cases.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture()
def actions():
    with open(ROOT / "data" / "actions.json") as f:
        return json.load(f)


@pytest.fixture()
def train_data():
    samples = []
    with open(ROOT / "data" / "train.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


@pytest.fixture()
def client():
    os.environ["FLASK_ENV"] = "testing"
    from server import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ── Data loading ──

def test_actions_count(actions):
    assert len(actions) == 31


def test_action_schema(actions):
    required = {"id", "connector", "label", "description", "category"}
    for a in actions:
        assert required.issubset(a.keys()), f"Missing keys in {a['id']}"


def test_actions_unique_ids(actions):
    ids = [a["id"] for a in actions]
    assert len(ids) == len(set(ids)), "Duplicate action IDs"


def test_train_count(train_data):
    assert len(train_data) >= 150


def test_train_schema(train_data):
    for sample in train_data:
        assert "query" in sample
        assert "action_id" in sample


def test_train_action_ids_exist(train_data, actions):
    valid_ids = {a["id"] for a in actions}
    for sample in train_data:
        assert sample["action_id"] in valid_ids


# ── Server endpoints ──

def test_get_actions(client, actions):
    r = client.get("/actions")
    assert r.status_code == 200
    data = r.get_json()
    assert len(data) == 31


def test_get_train(client, train_data):
    r = client.get("/data/train")
    assert r.status_code == 200
    data = r.get_json()
    assert len(data) >= 150


def test_get_day_1(client):
    r = client.get("/day/1")
    assert r.status_code == 200
    data = r.get_json()
    assert "day" in data
    assert "queries" in data
    assert len(data["queries"]) > 0


def test_get_day_404(client):
    r = client.get("/day/999")
    assert r.status_code == 404


def test_submit_day_1(client):
    r = client.get("/day/1")
    queries = r.get_json()["queries"]

    preds = [{"id": q["id"], "action_id": "slack_send_message"} for q in queries]
    r = client.post(
        "/day/1/submit",
        json={"predictions": preds},
    )
    assert r.status_code == 200
    data = r.get_json()
    assert "accuracy" in data
    assert "per_category" in data
    assert data["correct"] <= data["total"]


def test_submit_bad_request(client):
    r = client.post("/day/1/submit", json={})
    assert r.status_code == 400


# ── Scoring ──

def test_submit_scores_correct_predictions(client):
    r = client.get("/day/1")
    queries = r.get_json()["queries"]

    # Get ground truth for day 1
    day_file = ROOT / "data" / "days" / "day_01.jsonl"
    with open(day_file) as f:
        ground_truth = [json.loads(line) for line in f]

    # Submit all-correct predictions
    preds = [{"id": i, "action_id": gt["action_id"]} for i, gt in enumerate(ground_truth)]
    r = client.post("/day/1/submit", json={"predictions": preds})
    data = r.get_json()
    assert data["accuracy"] == 1.0
    assert data["correct"] == data["total"]
    assert data["mistakes"] == []


def test_submit_category_breakdown(client):
    """Submit uniform predictions with an action_id absent from day 1; verify 0% accuracy."""
    r = client.get("/day/1")
    queries = r.get_json()["queries"]

    # outlook_get_schedule does not appear in day 1 ground truth
    preds = [{"id": q["id"], "action_id": "outlook_get_schedule"} for q in queries]
    r = client.post("/day/1/submit", json={"predictions": preds})
    data = r.get_json()
    assert data["accuracy"] == 0.0
    assert data["correct"] == 0
    assert len(data["per_category"]) > 0
    # Every category should have 0 correct
    for cat, stats in data["per_category"].items():
        assert stats["correct"] == 0


def test_submit_partial_predictions(client):
    """Submitting predictions for only some queries should score missing ones as wrong."""
    r = client.get("/day/1")
    queries = r.get_json()["queries"]
    total = len(queries)

    # Submit only the first prediction
    preds = [{"id": 0, "action_id": "slack_send_message"}]
    r = client.post("/day/1/submit", json={"predictions": preds})
    data = r.get_json()
    assert data["total"] == total
    # At most 1 can be correct
    assert data["correct"] <= 1


def test_submit_duplicate_ids(client):
    """Submitting duplicate prediction IDs should use the last one (dict behavior)."""
    r = client.get("/day/1")
    queries = r.get_json()["queries"]

    preds = [
        {"id": 0, "action_id": "slack_send_message"},
        {"id": 0, "action_id": "jira_create_issue"},  # overwrites first
    ]
    r = client.post("/day/1/submit", json={"predictions": preds})
    assert r.status_code == 200
    data = r.get_json()
    assert data["total"] == len(queries)


def test_submit_extra_predictions(client):
    """Submitting predictions for IDs beyond the query set should not crash."""
    r = client.get("/day/1")
    queries = r.get_json()["queries"]

    preds = [{"id": q["id"], "action_id": "slack_send_message"} for q in queries]
    preds.append({"id": 9999, "action_id": "slack_send_message"})
    r = client.post("/day/1/submit", json={"predictions": preds})
    assert r.status_code == 200
    data = r.get_json()
    assert data["total"] == len(queries)


def test_submit_returns_mistakes(client):
    """Incorrect predictions should appear in the mistakes list."""
    preds = [{"id": 0, "action_id": "outlook_get_schedule"}]
    r = client.post("/day/1/submit", json={"predictions": preds})
    data = r.get_json()
    assert "mistakes" in data
    # At least query 0 should be a mistake (outlook_get_schedule is unlikely to be correct)
    id_0_mistakes = [m for m in data["mistakes"] if m["id"] == 0]
    assert len(id_0_mistakes) == 1
    assert id_0_mistakes[0]["predicted"] == "outlook_get_schedule"
    assert "expected" in id_0_mistakes[0]
    assert "query" in id_0_mistakes[0]


def test_submit_ranked_predictions(client):
    """Submitting action_ids (ranked list) should return top_3_accuracy and mrr."""
    day_file = ROOT / "data" / "days" / "day_01.jsonl"
    with open(day_file) as f:
        ground_truth = [json.loads(line) for line in f]

    # Put the correct answer at rank 2 (index 1) for each query
    preds = [
        {"id": i, "action_ids": ["outlook_get_schedule", gt["action_id"], "jira_list_issues"]}
        for i, gt in enumerate(ground_truth)
    ]
    r = client.post("/day/1/submit", json={"predictions": preds})
    data = r.get_json()

    # Top-1 should be 0% (first prediction is always wrong)
    assert data["accuracy"] == 0.0
    # Top-3 should be 100% (correct answer is always in top 3)
    assert data["top_3_accuracy"] == 1.0
    # MRR should be 0.5 (correct answer always at rank 2 → 1/2)
    assert data["mrr"] == 0.5


# ── Baseline ──

def test_baseline_predict_returns_valid_action_id(actions):
    from baseline import build_action_index, predict
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))
    embeddings, ids = build_action_index(model, actions)

    pred = predict(model, embeddings, ids, "send a message to the team")
    assert pred in ids


def test_baseline_accuracy_threshold(actions, train_data):
    """Baseline should achieve >30% accuracy on a sample of training data."""
    from baseline import build_action_index, predict
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))
    embeddings, ids = build_action_index(model, actions)

    # Test on first 50 training samples for speed
    sample = train_data[:50]
    correct = sum(
        1 for s in sample
        if predict(model, embeddings, ids, s["query"]) == s["action_id"]
    )
    accuracy = correct / len(sample)
    assert accuracy > 0.3, f"Baseline accuracy {accuracy:.1%} is below 30% threshold"


def test_baseline_empty_query(actions):
    """Baseline should handle an empty query without crashing."""
    from baseline import build_action_index, predict
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))
    embeddings, ids = build_action_index(model, actions)

    pred = predict(model, embeddings, ids, "")
    assert pred in ids


def test_baseline_long_query(actions):
    """Baseline should handle an unusually long query without crashing."""
    from baseline import build_action_index, predict
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))
    embeddings, ids = build_action_index(model, actions)

    long_query = "send a message to the team " * 100
    pred = predict(model, embeddings, ids, long_query)
    assert pred in ids


def test_evaluate_full_flow(client, actions):
    """End-to-end: fetch queries, predict with baseline, submit, verify score."""
    from baseline import build_action_index, predict
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))
    embeddings, ids = build_action_index(model, actions)

    r = client.get("/day/1")
    queries = r.get_json()["queries"]

    predictions = [
        {"id": q["id"], "action_id": predict(model, embeddings, ids, q["query"])}
        for q in queries
    ]

    r = client.post("/day/1/submit", json={"predictions": predictions})
    data = r.get_json()
    assert data["accuracy"] > 0, "Baseline should get at least one prediction right"
    assert data["total"] == len(queries)
    assert data["correct"] + len(data["mistakes"]) == data["total"]

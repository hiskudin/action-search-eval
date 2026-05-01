# Action Search Eval

Build a system that matches natural language queries to the correct connector action.

## The Task

You are given:
- A **catalog of 31 actions** across various connectors (Slack, BambooHR, Salesforce, etc.) in `data/actions.json`
- **Labeled training data** of (query, action_id) pairs in `data/train.jsonl`
- **Labeled evaluation data** for days 1-10 in `data/days/` (you're encouraged to examine these)
- A **server** (`server.py`) that serves daily evaluation batches

Your job: build a model/pipeline that takes a natural language query (e.g. *"send a message to the team channel"*) and returns the correct `action_id` (e.g. `slack_send_message`).

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Workflow

1. **Start the server:**
   ```bash
   uv run python server.py
   ```

2. **Explore the data:**
   - `GET /actions` - the full action catalog
   - `GET /data/train` - the training data
   - `GET /day/1` - evaluation queries for day 1 (no labels)

3. **Build your model.** A bare-bones baseline is in `baseline.py` for reference. You should improve on it.

4. **Submit predictions:**
   ```bash
   uv run python evaluate.py --day 1
   ```
   Or POST directly:
   ```bash
   curl -X POST http://localhost:5117/day/1/submit \
     -H "Content-Type: application/json" \
     -d '{"predictions": [{"id": 0, "action_id": "slack_send_message"}, ...]}'
   ```
   The server returns accuracy, per-category breakdown, and a list of mistakes.
   For ranked predictions (top-k), submit `action_ids` instead of `action_id` to get top-3 accuracy and MRR.

5. **Evaluate across all 10 days.** Each day is a separate batch. Your aggregate performance matters, plus a held-out set you won't see.

## Data Format

**Action:**
```json
{"id": "slack_send_message", "connector": "slack", "label": "Send Message", "description": "Send a message to a Slack channel or user", "category": "messaging"}
```

**Training sample:**
```json
{"query": "post a message on slack", "action_id": "slack_send_message"}
```

**Day query (from server):**
```json
{"id": 0, "query": "post a message on slack"}
```

## What We Evaluate

- Accuracy across all 10 days
- Accuracy on a held-out test set you won't have access to
- Quality and clarity of your approach
- How you handle the pipeline end to end (Observability/Monitoring)


## Guidelines

- Keep it simple.
- Use whatever libraries or approaches you like.
- The baseline uses `all-MiniLM-L6-v2` but you're free to use any model.
- Spend around ~2 hours.
- LLM use encouraged but you are be responsible for all output.

Have fun.

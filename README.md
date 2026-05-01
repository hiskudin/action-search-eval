# Action Search Eval

Build a system that matches natural language queries to the correct connector action.

## The Task

You are given:
- A **catalog of 31 actions** across various connectors (Slack, BambooHR, Salesforce, etc.) in `data/actions.json`
- **Labeled training data** of (query, action_id) pairs in `data/train.jsonl`
- A **server** (`server.py`) that serves daily evaluation batches

Your job: build a model pipeline that takes a natural language query (e.g. *"send a message to the team channel"*) and returns the correct `action_id` (e.g. `slack_send_message`).

We are going to run your pipeline on a simulated month but we are only giving you access to the first ten days for training. We will use your version of the `evaluate.py` script, feel free to make any modification to `main` for your work but the days.jsonl will be the same and we want this to be evaluated iteratively to simulate a real-world environment

## Setup

```bash
uv sync
```

## Workflow

1. **Start the server:**
   ```bash
   uv run server.py
   ```

2. **Explore the data:**
   - `GET /actions` - the full action catalog
   - `GET /data/train` - the training data
   - `GET /day/1` - evaluation queries for day 1 (no labels)

3. **Build your model.** A bare-bones baseline is in `baseline.py` for reference. You should improve on it.

4. **Submit predictions:**
   ```bash
   uv run evaluate.py --day 1
   ```
   Or POST directly:
   ```bash
   curl -X POST http://localhost:5117/day/1/submit \
     -H "Content-Type: application/json" \
     -d '{"predictions": [{"id": 0, "action_id": "slack_send_message"}, ...]}'
   ```
   The server returns accuracy and per-category breakdown.

5. **Evaluate across all 10 days.** Each day is a separate batch. Your aggregate performance across all days is what matters.

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

- Accuracy across all ten days
- Quality and clarity of your approach
- How you handle the pipeline end to end.

## Guidelines

- The baseline uses `all-MiniLM-L6-v2` but you're free to use anything open-source.
- You have ~2 hours.
- Feel free to use LLMs but you are responsible for everything in the submission.

Have fun.

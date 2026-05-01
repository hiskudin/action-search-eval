# Results

Iterative log of model improvements. Validation split: days 9–10 (held out from tuning). Dev split: days 1–8.

## V0 — Baseline (MiniLM, action text only)

`baseline.py` as shipped. Embeds `f"{label}: {description}"` for each of 31 actions with `all-MiniLM-L6-v2`, returns argmax cosine over the action index. Encoder normalized.

### Numbers

| split | acc | n |
|---|---|---|
| train | 41.5% | 193 |
| day01 | 40.0% | 50 |
| day02 | 34.0% | 50 |
| day03 | 38.0% | 50 |
| day04 | 32.0% | 50 |
| day05 | 24.5% | 49 |
| day06 | 32.0% | 50 |
| day07 | 38.0% | 50 |
| day08 | 28.6% | 49 |
| day09 | 26.0% | 50 |
| day10 | 40.0% | 50 |
| **dev (1–8)** | **33.4%** | 398 |
| **val (9–10)** | **33.0%** | 100 |
| **all days** | **33.3%** | 498 |

### Failure modes (from `diagnose.py`)

By category (days 1–10, errors/total):
- storage 83%, project 78%, ats 73%, calendar 72%, crm 65%, messaging 63%, hris 52%

By connector (top offenders): google_drive 92%, jira 89%, hubspot 87%, gcal 81%, dropbox 75%, slack 73%.

Three error families:

1. **Right verb, wrong connector (~50% of errors).** Queries don't name the platform. Top confusions: `bamboo_create_employee → lever_create_candidate` (10×), `slack_send_message → teams_*` (15× combined), `gdrive_upload ↔ dropbox_upload` (7×), `hubspot_get_contact → salesforce_create_lead` (8×), `workday_get_worker → bamboo_get_employee` (7×).
2. **Within-connector verb confusion.** `slack_send_message ↔ slack_list_channels` (12× both directions), `jira_create_issue ↔ jira_list_issues` (13×). Action descriptions aren't discriminative enough.
3. **Heavily paraphrased, task-oriented queries.** "do I have anything at 3pm" → `gcal_list_events`; "where should this conversation go" → `slack_list_channels`. These don't lexically resemble action `label + description`, but probably resemble training queries.

### Implications for next iterations

- (1) → connector prior from train frequency, or richer action text.
- (2) → cross-encoder reranker on top-k.
- (3) → kNN over training queries (predicted to be the single biggest lift).

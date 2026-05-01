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

## V1 — kNN over training queries

`models/v1_knn.py`. Same MiniLM encoder. Index = the 193 train queries (no action-description rows). For each query, take top-k=10 nearest train queries by cosine, sum similarities per action_id, predict the argmax.

Hyperparameter sweep on dev (days 1–8):

| variant | dev | val |
|---|---|---|
| k=1, train-only | 53.0% | 56.0% |
| k=3, train-only | 57.5% | 54.0% |
| k=5, train-only | 65.1% | 63.0% |
| **k=10, train-only** | **65.3%** | **68.0%** |
| k=5, train + actions | 64.1% | 66.0% |
| k=10, train + actions | 65.3% | 66.0% |

Adding action descriptions as weight-0.5 rows didn't help — train queries dominate the signal already. Picked k=10, train-only.

### Numbers

| split | acc | n |
|---|---|---|
| train | 85.5% | 193 |
| day01 | 72.0% | 50 |
| day02 | 70.0% | 50 |
| day03 | 76.0% | 50 |
| day04 | 64.0% | 50 |
| day05 | 59.2% | 49 |
| day06 | 62.0% | 50 |
| day07 | 60.0% | 50 |
| day08 | 59.2% | 49 |
| day09 | 62.0% | 50 |
| day10 | 74.0% | 50 |
| **dev (1–8)** | **65.3%** | 398 |
| **val (9–10)** | **68.0%** | 100 |
| **all days** | **65.9%** | 498 |

**Δ vs V0: +32.6pt on all days, +35.0pt on val.** Confirms diagnosis #3 — paraphrased queries cluster around train queries far better than around action descriptions. Train accuracy dropped from 94.8% (k=5 incl. self-match) to 85.5% (k=10 spreads vote across more neighbors), which is fine — train accuracy isn't the goal.

## V2 — kNN + log-prior from train action frequency

`models/v2_prior.py`. Hypothesis from V0 diagnosis: ~50% of errors were "right verb, wrong connector". Idea: nudge ambiguous predictions toward more-trained actions via `score(a) += λ · log P(a)` where `P(a)` is Laplace-smoothed train frequency.

**Sanity check before tuning**: train action counts range 5–11 (low spread, σ=1.86), days 1–10 counts range 4–30 (much wider). Pearson corr between train and day frequencies = 0.69 — directionally aligned but train is too uniform to encode the day skew well. So the prior is mathematically sound but expected to be weak.

Sweep on dev (days 1–8):

| λ | dev | val |
|---|---|---|
| 0.00 | 65.3% | 68.0% |
| 0.02 | 65.6% | 67.0% |
| 0.05 | 65.6% | 67.0% |
| **0.10** | **66.3%** | **68.0%** |
| 0.20 | 64.8% | 66.0% |
| 0.50 | 60.6% | 61.0% |

Picked λ=0.1.

### Numbers

| split | acc | n |
|---|---|---|
| train | 85.0% | 193 |
| day01 | 72.0% | 50 |
| day02 | 70.0% | 50 |
| day03 | 76.0% | 50 |
| day04 | 66.0% | 50 |
| day05 | 59.2% | 49 |
| day06 | 66.0% | 50 |
| day07 | 62.0% | 50 |
| day08 | 59.2% | 49 |
| day09 | 62.0% | 50 |
| day10 | 74.0% | 50 |
| **dev (1–8)** | **66.3%** | 398 |
| **val (9–10)** | **68.0%** | 100 |
| **all days** | **66.7%** | 498 |

**Δ vs V1: +1.0pt dev, +0.0pt val.** Tiny but real on dev. As predicted, the train distribution is too flat to act as a strong prior — the day distribution has actions appearing 4–30 times that train has 5–11 times, so the prior is a coarse approximation. Keeping it (cheap, doesn't hurt val), but the connector-ambiguity error family is still mostly intact and is the main thing for V3 to attack.

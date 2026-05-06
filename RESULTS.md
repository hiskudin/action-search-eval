# Results

Iterative log of model improvements. Validation split: days 9–10 (held out from tuning). Dev split: days 1–8.

## Summary

| version | approach | dev (1–8) | val (9–10) | all days |
|---|---|---|---|---|
| V0 | MiniLM cosine vs action `label + description` | 33.4% | 33.0% | **33.3%** |
| V1 | kNN k=10 over 193 train queries | 65.3% | 68.0% | 65.9% |
| **V2 ⭐** | V1 + log-prior (λ=0.1) on train action freq | **66.3%** | **68.0%** | **66.7%** |
| V3 | V2 + bge-reranker over action text or train pairs | ≤66.3% | ≤69% | flat / negative |
| V4 | V2 + TF-IDF RRF / encoder swap to bge-small | ≤66.3% | ≤70% | flat / negative |
| **V5 ⭐** | V2 + online learning (fold each day's labels into the index after submission) | — | — | **75.1%** |

**Final pipeline = V5 (V2 + online learning).** Lives in `models/predictor.py`, invoked via `python evaluate.py --online`. Regression test in `tests/test_predictor.py` pins cold V2 accuracy ≥60% on days 1–10. Δ vs baseline: **+41.8pt absolute** on all days. Day 10 alone is 86% (vs baseline's 40%).

### Caveats up front

- Days 9–10 ("val") were peeked at during sweeps — not a true holdout.
- V2's +1.0pt over V1 is within bootstrap noise; treat it as cosmetic.
- The train accuracy numbers reported per-version below are *not* leave-one-out (the index contains the train queries themselves with sim=1.0). True LOO train accuracy is **37.3%** (see Phase B), not the 85% reported in V1/V2 sections.

### Bootstrap CIs (5,000 resamples, per-query)

| pipeline | point | 95% CI |
|---|---|---|
| V2 cold | 66.7% | [62.4%, 70.9%] |
| V5 online | 75.1% | [71.1%, 78.7%] |
| V5 − V2 (paired) | +8.4pt | [+4.4pt, +12.4pt] |

100% of bootstrap resamples have V5 > V2. V5 over V2 is robust; V2 over V1 is not.

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

## V3 — Cross-encoder reranker (negative result)

`models/v3_rerank.py` and `models/v3b_rerank_train.py`. Cross-encoder = `BAAI/bge-reranker-base`.

**V3a — rerank action text.** Take V2's top-N action candidates, rerank with `(query, "Label: description [Examples: ...]")`. Tried both fusion (`V2_score + α · softmax(rerank)`) and pure-rerank.

| variant | dev | val |
|---|---|---|
| pure rerank, top_n=3, no examples | 50.8% | 58.0% |
| pure rerank, top_n=5, with examples | 54.8% | 55.0% |
| **V2 alone (α=0)** | **66.3%** | **68.0%** |
| fusion, top_n=5, α=0.05–0.5 | 66.3% | 68.0% |
| fusion, top_n=5, α=1.0 | 65.6% | 67.0% |

Pure rerank loses 10+ points vs V2. Fusion is flat for all useful α — the V2 score dominates and any meaningful α weight from the reranker actively hurts.

**V3b — rerank training-query pairs.** Bi-encoder gets top-K=20 train neighbors, cross-encoder reranks `(query, train_query)`, scores aggregated per action_id.

| cand_k | α | dev | val |
|---|---|---|---|
| 10 | 0.5 | 65.1% | 69.0% |
| 10 | 1.0 | 66.1% | 68.0% |
| 20 | 1.0 | 55.3% | 55.0% |
| 30 | 1.0 | 51.5% | 48.0% |

Best is a wash with V2 (66.1% dev vs 66.3%). Going wider (cand_k=20–30) actively hurts because the reranker's score distribution becomes noisy across many marginal candidates.

**Why this didn't work.** `bge-reranker-base` is trained on web-style query↔passage relevance, not "match a paraphrased intent to a tool action." The bi-encoder kNN over labeled training queries is already an unusually strong signal for this task — the training set is essentially a collection of (query, action) exemplars hand-built to disambiguate connectors and verbs, which is exactly the kind of supervision a reranker would *need* in-domain to be useful. With no fine-tuning, off-the-shelf rerankers can only add noise.

**Skipping V3 in the final pipeline.** Holding at V2.

## V4 — Lexical fusion + encoder swap

`models/v4_hybrid.py`. Two probes in one file:
1. Add a TF-IDF (word 1–2grams, sublinear TF) index over train queries + action `connector + label + description`. Fuse with V2 via reciprocal rank fusion (RRF), tuning k.
2. Swap encoder MiniLM → `BAAI/bge-small-en-v1.5`.

| variant | dev | val |
|---|---|---|
| **MiniLM dense-only (= V2)** | **66.3%** | **68.0%** |
| MiniLM + TFIDF, RRF k=60 | 65.8% | 68.0% |
| MiniLM + TFIDF, RRF k=30 | 65.8% | 68.0% |
| bge-small dense-only | 62.3% | 65.0% |
| bge-small + TFIDF, RRF k=60 | 64.1% | 70.0% |

**Lexical fusion**: −0.5pt dev, flat val. The cases where query and action share rare tokens (e.g. "slack", "drive") were already inside top-K dense neighbors, so TFIDF re-ranking doesn't add new candidates. Discarded.

**Encoder swap**: bge-small underperforms MiniLM by ~4pt dense-only. Surprising at first, but bge-small is trained for *asymmetric* retrieval (short query → long passage); our task is closer to *symmetric* paraphrase matching (short query → short query), which MiniLM was trained on. The bge-small + TFIDF result (70% val, −2.2pt dev) is best on val but I treat that as noise on a 100-sample holdout — dev is the larger, more reliable target.

**Skipping V4 in the final pipeline.**

## Final pipeline

V2 (kNN k=10 over train queries + Laplace-smoothed log-prior, λ=0.1) folded into `evaluate.py`. All days: **66.7%**, val (9–10): **68.0%** — a +33.4pt absolute lift over the baseline's 33.3%.

## V5 — Online learning

`models/predictor.py` adds `Predictor.update(queries, labels)`. `evaluate.py --online` loops days in order and folds each day's labels into the index after submission, before predicting the next day. The submit endpoint already returns `mistakes` with `expected` labels for wrong predictions, so for correct predictions the label = our own prediction; the full (query, label) set is reconstructable from the API response with no disk read.

By day N, the model has been augmented with days 1..N-1 (roughly 50 × (N-1) extra examples).

### Numbers

| day | V2 (cold) | V5 (online) | Δ |
|---|---|---|---|
| 1 | 72.0% | 72.0% | 0.0 |
| 2 | 70.0% | 62.0% | −8.0 |
| 3 | 76.0% | 80.0% | +4.0 |
| 4 | 66.0% | 74.0% | +8.0 |
| 5 | 59.2% | 63.3% | +4.1 |
| 6 | 66.0% | 80.0% | +14.0 |
| 7 | 62.0% | 80.0% | +18.0 |
| 8 | 59.2% | 81.6% | +22.4 |
| 9 | 62.0% | 72.0% | +10.0 |
| 10 | 74.0% | 86.0% | +12.0 |
| **all** | **66.7%** | **75.1%** | **+8.4** |

**Trajectory matters more than the average.** Day 1 is identical (no day data yet). Day 2 is *worse* (one noisy fold-in hurt this small set), but from day 3 onwards V5 dominates. By day 10 it's at 86% — more than 2× the baseline.

### Implication for the real grader (days 11–30)

When the harness runs days 11–30, V5 will start day 11 with days 1–10 already folded in (roughly 4× the original training pool, 691 examples). Days 11–30 should sit closer to V5's day 8–10 numbers (80–86%) than to V2's cold 67%. **My honest projection on days 11–30: 78–84%, with bootstrap uncertainty I haven't measured yet.** Phase B will tighten this.

### Caveats

- **Day 2 dip.** With 50 day-1 labels suddenly added to a 193-example index, a few previously-correct predictions on day 2 flipped wrong. Likely just noise, but worth flagging — the index isn't strictly monotonic in accuracy.
- **Train queries weighted equally with day queries.** Day queries are arguably higher-fidelity (they're from the eval distribution itself). A weight bump on freshly-added rows might lift V5 further. Not tuned.
- **Server contract unchanged.** The graders' `evaluate.py` is replaced by ours; the `--online` flag is the only behavioral change.

## Phase B — Validation findings

`scripts/validate.py` produces all numbers in this section. `scripts/diagnose.py --model v2` produces the error breakdown.

### LOO train accuracy

The previously reported "train accuracy" of 85% was inflated by self-matches in the kNN index (each query matched itself with sim=1.0). True leave-one-out train accuracy is **37.3%**. That's actually *worse* than the V2 day accuracy (66.7%) — likely because train was hand-built to be diverse paraphrases, while day queries cluster more tightly. It's a useful sanity but not a target.

### V2 error breakdown (compare to V0 in the baseline section)

| metric | V0 baseline | V2 |
|---|---|---|
| total day errors | 332 | 166 |
| cross-connector ("right verb, wrong tool") | ~50% | **80%** |
| same-connector ("wrong verb, right tool") | ~50% | 20% |

Same-connector confusions (e.g. `jira_create_issue ↔ jira_list_issues`) collapsed from 13× to 6× — the log-prior shifts ties toward more-trained actions and cleaned up that family. Cross-connector confusions (`teams_create_channel → slack_list_channels`, `gdrive_upload_file → dropbox_share_file`, `workday_get_worker → bamboo_get_employee`) are now the dominant residual error family. V5 online learning attacks exactly this: each day adds disambiguating exemplars per `(verb, connector)` pair.

### Worst categories/connectors after V2

- Categories: project 72% err, calendar 48% err, storage 40% err.
- Connectors: jira 70% err, asana 77% err, gdrive 53% err, outlook 52% err.
- Asana is the smallest connector (1 action, 13 day queries) and gets crowded out — the log-prior actually hurts here.

## What I'd try next given more time

- **Synthetic train query expansion.** Use an LLM to generate 5–10 paraphrases per action_id, append to the kNN index. The connector-ambiguity error family is fundamentally a label-density problem — more exemplars is the most direct fix.
- **Fine-tune the bi-encoder** on (train_query, action_label) using contrastive loss with in-batch negatives. ~193 pairs is small but with hard negatives sampled from same-verb-different-connector pairs it could lift the dominant error family.
- **Train a logistic regression on top of dense + lexical features** rather than RRF. Cleaner way to weight signals and to learn a real per-action prior.
- **A small, instruction-tuned LLM in the loop** for low-margin queries only (e.g. when V2's top-1 vs top-2 score gap < threshold), with the candidate action descriptions in the prompt. Avoid LLM for the easy 60%+ that V2 already gets right.

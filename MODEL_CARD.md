# Model card — French client-satisfaction estimator (v1)

| | |
|---|---|
| **Version** | v1 · `local-tfidf-v1` |
| **Task** | French binary sentiment with calibrated abstention, plus population-level satisfaction estimation |
| **Backbone** | TF-IDF (word 1–2 grams + `char_wb` 3–5 grams) → LogisticRegression |
| **Training data** | `tblard/allocine` — 40,000 stratified rows (seed 42) of 160,000 |
| **Artifact size** | 8.5 MB |
| **Evaluated** | 2026-10-05, test split, read once |

---

## What it does

Takes French review text and returns one of **positif / négatif / neutre**, with a calibrated probability. Over a body of feedback it also returns a **bias-corrected satisfaction rate with a confidence interval**, and a per-topic breakdown.

The two outputs have very different reliability, and the distinction matters more than any single number here:

> **The aggregate estimate is reliable. Individual predictions are not.**

---

## Intended use

- Estimating what share of a body of French feedback is positive.
- Ranking topics by how much dissatisfaction they attract.
- Triage: surfacing which comments a human should read first.

## Not suitable for

- **Decisions about an individual customer.** Two human annotators typically agree only ~0.7 of the time on 3-class sentiment, so no model on this task supports per-person action at any accuracy it reaches.
- **Automated moderation** or any action taken without human review.
- **Non-French text.** The model emits confident nonsense on English; the demo warns but does not block.
- **Domains far from reviews.** See the limitation below — this is the significant one.

---

## Performance

Test split, 20,000 reviews, measured once with all parameters frozen.

| Slice | n | Accuracy | Macro-F1 |
|---|---|---|---|
| Forced choice (every row) | 20,000 | 0.9201 | 0.9195 |
| Deduplicated | 19,961 | 0.9199 | 0.9194 |
| Confident only (band applied) | 18,079 | 0.9612 | 0.9611 |

Validation macro-F1 was 0.9286, so the generalisation gap is 0.9 points.

### Context — what the transformer would have to beat

| Model | Macro-F1 | Train time |
|---|---|---|
| Majority class | 0.3378 | — |
| TF-IDF word only, 40k | 0.9238 | 12 s |
| **This model** (word + char, 40k) | **0.9286** (val) | 56 s |
| TF-IDF word only, full 160k | 0.9372 | 53 s |

Reported because it is the honest framing: a bag-of-ngrams model trained in under a minute on a laptop CPU reaches 0.93 on this task. Any transformer result should be read as a *delta against this*, not as a standalone figure.

### Calibration

| | Before | After |
|---|---|---|
| ECE | 0.0545 | **0.0077** |
| NLL | 0.2103 | 0.1881 |
| Accuracy | 0.9280 | 0.9280 |

Temperature `T = 0.6268`. **Below 1**, meaning the raw model was *under*-confident — the opposite of the usual transformer behaviour, because L2 regularisation on sparse high-dimensional features shrinks the coefficients. Accuracy is unchanged by construction: temperature scaling cannot move an argmax. Calibration is load-bearing here rather than cosmetic, since both the abstention band and the aggregate correction consume these probabilities.

### Abstention band

`θ_lo = 0.257`, `θ_hi = 0.709` — 9.6% of test reviews fall inside and are returned as `neutre`.

| | Validation | Test |
|---|---|---|
| Accuracy outside band | 0.9610 | 0.9612 |
| Accuracy inside band | 0.6310 | **0.5336** |

Inside the band the model is at coin-flip accuracy. The band is not discarding good predictions; it declines to guess where the text genuinely does not resolve.

**These thresholds are provisional.** They were set by choosing a 10% target abstention rate, not fitted against real 3-class labels — because no such labelled set exists yet (see Limitations).

### Aggregate estimation

`TPR = 0.9631`, `FPR = 0.0426`, measured on a validation holdout disjoint from the calibration set.

Counting positive predictions (*Classify and Count*) is biased whenever classifier errors are asymmetric. The corrected estimator inverts `p̂ = TPR·p + FPR·(1−p)`:

| | Validation | Test |
|---|---|---|
| Classify-and-Count MAE | 0.0195 | 0.0194 |
| Adjusted CC MAE | 0.0058 | **0.0018** |
| Improvement | 3.3× | **10.5×** |
| CI coverage (9 levels) | — | **9 / 9** |

Across true prevalences from 10% to 90%, every reported 95% interval contained the true rate. Naive counting reports a truly 10%-satisfied population as roughly 14%.

### Aspect breakdown

Lexicons derived from frequent terms in negative reviews, with sentiment scored **per sentence** rather than per review — so mixed feedback separates correctly.

| Aspect | Sentences | Corrected satisfaction |
|---|---|---|
| scénario | 2,511 | 35.4% |
| originalité | 755 | 36.9% |
| réalisation | 1,652 | 40.2% |
| émotion | 946 | 41.2% |
| image_son | 2,316 | 51.3% |
| jeu_acteurs | 2,550 | 51.3% |

---

## Training data

**`tblard/allocine`** — French film reviews from Allociné, 160k / 20k / 20k, binary labels.

Audited before use ([`00_audit`](notebooks/00_audit.ipynb)):

| Property | Finding |
|---|---|
| Class balance | 49.6 / 50.4 — no weighting applied |
| Cross-split duplicates | 46 texts; removing them changes macro-F1 by 0.0001 |
| Identical text, conflicting labels | 6 — a small irreducible error floor |
| Token length | p50 91, p90 285, max 518 |
| Truncation at `max_length=288` | 9.8% |
| Casing | 96.3% of reviews contain uppercase — normalisation preserves case |

**Construction note.** Datasets built from star ratings usually drop the middle. Here the hole is in the **ratings**, not the **language**: both classes contain measured, mixed, lukewarm writing (*"Exercice de style intéressant mais fastidieux"* is labelled negative). This is why the abstention band works — the model is genuinely uncertain on such text rather than confidently wrong, confirmed by the 0.5336 in-band accuracy above.

---

## Limitations

**1. Domain. This is the significant one.** The model is trained and measured entirely on **film reviews**. It is presented as a client-satisfaction tool, and no measurement in this card establishes how it performs on product or service feedback.

Concretely, two things degrade off-domain, and the second is worse than the first:

- Accuracy will drop by an unmeasured amount.
- **The confidence interval becomes a false guarantee.** `estimate_with_ci` propagates comment-sampling noise and `TPR`/`FPR` uncertainty from the Allociné holdout. It accounts for **zero** domain shift. Off-domain the app will print `"32% [25–38%]"` with full apparent rigour while the true rate may lie well outside. A wide honest interval is useful; a narrow dishonest one is worse than none.

Closing this requires ~150–300 hand-labelled French product/service comments with 3-class labels. It cannot be auto-generated: labelling with a model would make any agreement figure measure agreement with the system under test.

**2. Truncation.** 9.8% of reviews exceed `max_length=288` and are cut. Review sentiment often lands in the final sentence, so this is not a neutral loss.

**3. Aspect detection is lexicon-based.** 59.9% of sentences contain no lexicon term and are invisible to the method. No implicit aspects, no dependency parsing; a sentence mentioning two aspects is attributed to both with the same sentiment. The lexicons are film-specific and must be rebuilt for another corpus.

**4. Backbone is a bag of n-grams.** No word order beyond bigrams, no negation scope, no sarcasm handling. Sarcasm and mixed valence dominated the error analysis.

**5. No neutral class in training.** The three-way output comes from an abstention band over a binary model, not from a model trained on neutral examples.

---

## How to swap the backbone

`src/predict.py` takes the encoder as a backend; the pipeline around it is shared code. To move to CamemBERT:

1. Train it and push to the Hub.
2. Set `backend`, `model_repo`, `revision` in `inference_config.json`.
3. Re-run [`04_calibration`](notebooks/04_calibration.ipynb) and [`06_aggregate`](notebooks/06_aggregate.ipynb).

Recalibration is **mandatory, not optional**: a new encoder has its own confidence profile and its own error asymmetry, so every learned constant is invalidated. The config validator enforces this — it refuses to aggregate until `TPR`/`FPR` are re-measured.

That would be **v2**, with its own single test evaluation reported separately.

---

## Reproducing

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q
```

57 tests, ~4 seconds, no GPU and no torch required. Notebooks run in order `00 → 01 → 02 → 04 → 06 → 07`, with `08_final_test` last and once.

Environment pinned in [`requirements.txt`](requirements.txt); full freeze in `requirements-full.txt`. Python 3.14.5, scikit-learn 1.8.0, numpy 2.4.6.

## Provenance

Training data is `tblard/allocine` as published on the Hugging Face Hub; consult that dataset's own card for its terms. The model artifact is produced entirely from that corpus by the notebooks in this repo. No other data source contributed to the weights.

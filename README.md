# French Client-Satisfaction Analysis

Estimate **how satisfied clients are** from their written French feedback — with an honest error bar — and identify **what they are unsatisfied about**.

> **Status: Phase 0 complete, Phase 1 partial.** Data audited and CPU baselines measured on real data. No transformer trained yet — that needs a GPU. Tables are filled where real numbers exist and marked *pending* elsewhere.

---

## The idea in one paragraph

Most sentiment projects measure per-comment accuracy and report it as if it answered the business question. It doesn't. *"Is this review positive?"* and *"what share of clients are satisfied?"* are different questions with different failure modes: the second one breaks when the classifier's errors are **asymmetric**, which no amount of accuracy reporting will reveal. This project builds the classifier, then builds the **estimator** on top of it, and measures both separately.

## Headline: counting predictions gives the wrong answer

With true positive prevalence `p`, a classifier predicts positive at rate `p̂ = TPR·p + FPR·(1−p)`. Classify-and-Count reports `p̂` and calls it `p`. Those are equal only when `TPR = 1` and `FPR = 0`.

For a realistic classifier (TPR 0.94, FPR 0.06) facing a genuinely dissatisfied client base (`p = 0.30`), naive counting reports **0.324**. The bias grows as the true rate moves away from the classifier's training prevalence — exactly when the answer matters most.

The correction is three lines ([`src/estimate.py`](src/estimate.py)), verified against synthetic data in [`tests/test_estimate.py`](tests/test_estimate.py).

**This is not hypothetical — it is already measured.** The TF-IDF baseline in [`01_baselines.ipynb`](notebooks/01_baselines.ipynb) has **TPR 0.9323, FPR 0.0749** on validation. Those real rates imply:

| True satisfaction | Naive count reports | Corrected | Naive error |
|---|---|---|---|
| 10% | 16.1% | 10.0% | **+6.1 pts** |
| 30% | 33.2% | 30.0% | +3.2 pts |
| 50% | 50.4% | 50.0% | +0.4 pts |
| 70% | 67.5% | 70.0% | −2.5 pts |
| 90% | 84.7% | 90.0% | **−5.3 pts** |

A 93%-accurate classifier misreports a badly dissatisfied client base by **six percentage points** — in the optimistic direction. Near 50/50 the bias nearly vanishes, which is exactly why it goes unnoticed: it is invisible on a balanced benchmark and largest when the answer matters most.

*(Full CC-vs-ACC plot from the trained model lands with notebook 06.)*

---

## Results

**Development (validation, n=20,000):**

| Model | Acc | Macro-F1 | Train time | ECE |
|---|---|---|---|---|
| Majority class | 0.5102 | 0.3378 | — | — |
| TF-IDF word, 40k | 0.9238 | 0.9238 | 12 s | — |
| TF-IDF word+char, 40k | **0.9286** | **0.9286** | 56 s | — |
| TF-IDF word, full 160k | **0.9372** | **0.9372** | 53 s | — |
| Off-the-shelf (nlptown) | *pending — needs GPU* | | | |
| Zero-shot mDeBERTa | *pending — needs GPU* | | | |
| CamemBERT fine-tuned (40k) | *pending — needs GPU* | | | |

Two things to carry forward:

- **The bar is 0.9286**, not zero. Whatever CamemBERT scores, the reportable result is the *difference* from a model that trains in 56 seconds on a laptop CPU. If that gap is three points, say three points.
- **The 40k subsample costs ~1.3 points** for TF-IDF (0.9238 → 0.9372 on full data). Transformers are more sample-efficient so the gap should be smaller, but §4.3's budget decision is not free and the final 160k run should quantify it.

**Final (test, measured once):**

| Model | Acc | Macro-F1 (3 seeds) | Latency p50 bs=1 / bs=32 | Size | ECE |
|---|---|---|---|---|---|
| TF-IDF + LogReg | | — | | | |
| CamemBERT (40k) | | ± | | | |
| CamemBERT (160k) | | — | | | |
| XLM-R (40k) | | ± | | | |
| CamemBERT quantized | | ± | | | |

**Aggregate estimation:**

| True prevalence | CC | ACC | CC error | ACC error |
|---|---|---|---|---|
| 0.10 | | | | |
| 0.30 | | | | |
| 0.50 | | | | |
| 0.70 | | | | |
| 0.90 | | | | |
| **MAE** | | | | |

**Real client feedback (n=300, 95% CI):**

| Slice | n | Acc [CI] | Macro-F1 [CI] |
|---|---|---|---|
| All, 3-way | 300 | | |
| Binary only | | | |
| Neutral items | | | |
| Mixed-valence items | | | |

Inter-annotator κ: *pending* · % neutral in real feedback: *pending*

---

## Findings

### The polarity hole is in the ratings, not in the language

The spec predicted that `tblard/allocine`, being built from star ratings, would have **no lukewarm reviews at all** — making neutral text out-of-distribution by construction. Reading the data, that turns out to be **wrong, in a useful way**. Both classes are full of measured, mixed writing:

> `label 0` — *"Exercice de style intéressant mais fastidieux […] je n'ai pas été convaincu"*
> `label 0` — *"Le début est un peu abrupt […] Mais à son crédit il n'omet surtout pas de magnifier la scène de l'abbé Myriel"*
> `label 1` — *"Très bonne comédie […] Il y'a une légère perte de rythme"*

Middle *ratings* were likely dropped at construction; middle *language* was not. A reviewer can rate 4/5 and still write a largely critical paragraph.

**This is better news than feared.** The risk was that the model, never having seen lukewarm text, would be *confidently wrong* on it — in which case a probability-threshold abstention band could not help. Instead, lukewarm text is present but arbitrarily assigned to a side, so the model should be genuinely *uncertain* near the boundary. That is exactly the condition the neutral band needs. Notebook 04 tests it directly: are low-confidence validation items in fact the mixed ones?

### Audit results

| Check | Result |
|---|---|
| Dataset | `tblard/allocine` — 200,000 reviews (160k / 20k / 20k) |
| Class balance | 49.6% négatif / 50.4% positif — **no class weighting needed** |
| Cross-split duplicates | 46 texts; test 20,000 → 19,961 after dedup |
| Identical text, conflicting labels | 6 — a small but non-zero label-noise floor |
| Token length | p50 **91**, p90 **285**, p99 454, max 518 |
| `max_length` | **288** (p90, rounded) — 9.8% of reviews truncated |
| Formatting | 96.3% have uppercase, 98.5% punctuation, 0.1% emoji, 0% HTML |

The formatting row settled a real decision: the corpus is **not** lowercased, so `normalize()` v1 preserves casing — *"NUL"* and *"nul"* are different signals and collapsing them would throw away information.

The length row is the expensive one. At p90 = 285 tokens these reviews are ~3× longer than the tweet-length text the original emotion plan assumed, which is what forces the 40k subsample in §4.3.

### Still pending

- **Transformer vs. TF-IDF** — §5.2. The bar is set at 0.9286; the gap is the finding.
- **Domain transfer** — §6. Movie reviews → real product feedback.
- **Calibration** — §9. ECE before and after temperature scaling.

---

## Repo layout

```
src/
  normalize.py      one text normalisation, imported everywhere (§4.4)
  estimate.py       ACC prevalence correction + bootstrap CI  (§10)  <- the centerpiece
  aspects.py        lexicons + sentence-level attribution     (§11)
  predict.py        the single prediction path                (§3.1)
tests/              52 tests, no GPU required
notebooks/
  00_audit.ipynb    data audit                        (§4)  [ready]
  01_baselines      majority / TF-IDF / off-the-shelf / zero-shot (§5)
  02_camembert      fine-tune                         (§7)
  03_xlmr_seeds     3 seeds x 2 models                (§8)
  04_calibration    temperature + neutral band        (§9)
  05_real_feedback  300 hand-labelled comments        (§6)
  06_aggregate      TPR/FPR + the CC-vs-ACC plot      (§10)
  07_efficiency     quantisation + benchmark          (§13)
  08_final_test     THE ONLY notebook touching test   (§1.1)
space/app.py        Gradio demo: a report, not a label (§14.1)
inference_config.json   T, thresholds, TPR/FPR, pinned revision (§3.2)
```

The [spec](implementation-spec.md) is the authority on *why* each piece is shaped this way. Section numbers above are references into it.

## Two rules the layout enforces

1. **Exactly one notebook reads the test split** (`08_final_test`). Auditable at a glance — grep for `ds["test"]`.
2. **Exactly one code path produces a prediction** (`src/predict.py`), imported by both the evaluation notebooks and the Space. If evaluation and serving diverge, published numbers stop describing the deployed system and *nothing raises an error*.

---

## Setup

```bash
python -m pip install -r requirements.txt
```

```bash
python -m pytest tests/ -q
```

The pure-numpy modules (`normalize`, `estimate`, `aspects`) and their tests run without torch — training happens on Kaggle (T4, ~8 GPU-hours total).

## Limitations

Written before the results exist, because they follow from the design rather than from how the numbers turn out:

- **Trained on movie reviews**, applied to product/service feedback. The gap is measured, not assumed away (§6).
- **No neutral class in training.** Handled with a calibrated abstention band rather than pretended away (§9.2).
- **Aspect detection is lexicon-based.** No implicit aspects, no dependency parsing. Honest about what it misses (§11).
- **Aggregate estimates are reliable; individual predictions are not.** Two human annotators agree only ~0.7 of the time on 3-class sentiment, so per-customer decisions are not supportable at any accuracy this will reach.

## Licence and provenance

Model and data provenance are stated exactly, with nothing claimed that cannot be verified — see the model card once published (§14.2).

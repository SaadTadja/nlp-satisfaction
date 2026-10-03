# French Client-Satisfaction Analysis

Estimate **how satisfied clients are** from their written French feedback — with an honest error bar — and identify **what they are unsatisfied about**.

> **Status: scaffold.** The library and tests are in place; no model is trained yet. Results tables below are empty by design and fill in as the phases complete.

---

## The idea in one paragraph

Most sentiment projects measure per-comment accuracy and report it as if it answered the business question. It doesn't. *"Is this review positive?"* and *"what share of clients are satisfied?"* are different questions with different failure modes: the second one breaks when the classifier's errors are **asymmetric**, which no amount of accuracy reporting will reveal. This project builds the classifier, then builds the **estimator** on top of it, and measures both separately.

## Headline: counting predictions gives the wrong answer

With true positive prevalence `p`, a classifier predicts positive at rate `p̂ = TPR·p + FPR·(1−p)`. Classify-and-Count reports `p̂` and calls it `p`. Those are equal only when `TPR = 1` and `FPR = 0`.

For a realistic classifier (TPR 0.94, FPR 0.06) facing a genuinely dissatisfied client base (`p = 0.30`), naive counting reports **0.324**. The bias grows as the true rate moves away from the classifier's training prevalence — exactly when the answer matters most.

The correction is three lines ([`src/estimate.py`](src/estimate.py)), and it is verified against synthetic data in [`tests/test_estimate.py`](tests/test_estimate.py) before any GPU time is spent.

*(Headline CC-vs-ACC plot goes here once notebook 06 runs.)*

---

## Results

**Development (validation):**

| Model | Acc | Macro-F1 | ECE |
|---|---|---|---|
| Majority class | | | — |
| TF-IDF + LogReg | | | |
| Off-the-shelf (nlptown) | | | |
| Zero-shot mDeBERTa | | | |
| CamemBERT fine-tuned (40k) | | | |

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

*Filled in as the phases complete. Each of these is a sentence a reader should remember.*

- **Polarity hole** — *pending §4.1.* Whether the training corpus drops middle ratings, making neutral text out-of-distribution by construction.
- **Duplicate leakage** — *pending §4.2.*
- **Transformer vs. TF-IDF** — *pending §5.2.* How many points a fine-tuned encoder buys over a model that trains in twenty seconds on CPU.
- **Domain transfer** — *pending §6.* Movie reviews → real product feedback.
- **Calibration** — *pending §9.* ECE before and after temperature scaling.

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

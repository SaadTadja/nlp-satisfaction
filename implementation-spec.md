# French Client-Satisfaction Analysis — Implementation Spec

**Goal.** Given a body of French customer feedback, estimate **how satisfied clients are** (with an honest error bar) and **what they are unsatisfied about**.

Supersedes [`french-emotion-analysis-plan.md`](french-emotion-analysis-plan.md), which targeted 6-way emotion on a machine-translated dataset. The pivot: emotion is the wrong primary task for a satisfaction question, and real labeled French review data exists. Chapter 2 of the book still applies unchanged — text classification is text classification.

Every phase has **Why** (the reasoning), **Do this** (steps + code), **Done when** (the gate to move on).

---

## 0. What this system answers

### 0.1 The product question is not the model question

**This is the single most important framing in the document.** Almost every sentiment project measures per-comment accuracy and reports it as if it answered the business question. It doesn't.

| | Model question | Product question |
|---|---|---|
| Asks | "Is *this* review positive?" | "What share of clients are satisfied?" |
| Unit | One comment | A population of comments |
| Metric | Accuracy, F1 | Estimated prevalence + confidence interval |
| Fails when | It misclassifies | Its errors are **asymmetric** |

A classifier at 85% accuracy whose errors skew one way can report *"68% satisfied"* when the truth is 56%. Accuracy doesn't surface that, and no amount of fine-tuning fixes it — it needs a different estimator (§10).

So this project has **two deliverables measured two different ways**: a classifier (Phases 3–5, 9) and an estimator built on top of it (Phase 6). Phase 6 is the centerpiece and the reason the project is interesting.

### 0.2 Deliverables

1. A French satisfaction classifier — positive / negative, with a calibrated **neutral** band.
2. An **aggregate estimator** that reports satisfaction rate with a CI, corrected for classifier bias.
3. An **aspect breakdown** — which topics drive the dissatisfaction.
4. A Gradio app that takes a CSV of feedback and returns a report, not just a label.
5. *(Optional)* An emotion layer separating "angry" from "disappointed" among negatives.

---

## 1. The evaluation contract

Binding for the whole project. Read once, follow throughout.

### 1.1 The test set is touched exactly once

**Why.** Score on test at every phase and you make five+ decisions against it; the final number drifts optimistic by 1–3 points. It happens by accident because the split is sitting right there.

**Do this.** All development on validation. Test once, at the end, for all models at once. Make it mechanically hard to cheat — in every notebook except the final one:

```python
ds = load_dataset("allocine")
ds.pop("test")   # the test split does not exist in this notebook
```

**Done when.** Exactly one notebook in the repo contains `ds["test"]`.

### 1.2 Two metric families, never conflated

| Level | Primary | Also report |
|---|---|---|
| Per-comment | Macro-F1 | Accuracy, per-class P/R, confusion matrix |
| **Aggregate** | **Absolute error on estimated satisfaction rate** | CI width, bias under prevalence shift |

Keep them in separate tables. A model can win on one and lose on the other — that's not a contradiction, it's the whole point of §0.1.

### 1.3 Any comparison between two models needs three seeds

**Why.** Run-to-run macro-F1 std for a base encoder is typically 0.5–1.5 points. "CamemBERT beat XLM-R by 0.8" on one seed measures nothing.

**Do this.** Seeds `{13, 42, 1337}`, identical settings, report `mean ± std`. With the §4.3 subsample this costs ~15 minutes per model.

**Done when.** Every head-to-head row carries a ±, and the README states that gaps smaller than the spread aren't treated as real.

### 1.4 Small sets get confidence intervals

At n=300, a 0.70 accuracy has a 95% CI of roughly ±5 points. Report it.

```python
import numpy as np

def bootstrap_ci(y_true, y_pred, metric, n=5000, seed=0):
    rng, idx = np.random.default_rng(seed), np.arange(len(y_true))
    vals = [metric(y_true[s], y_pred[s])
            for s in (rng.choice(idx, len(idx), replace=True) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
```

---

## 2. The stack

Total cost **$0**. Principle: the fewest moving parts that still support the claims in §1.

### 2.1 Compute

| | Primary | Fallback |
|---|---|---|
| Platform | **Kaggle Notebooks** | Google Colab free |
| GPU | T4 ×2 (use one) | T4 |
| Session | 12 h | ~4 h, variable |
| Quota | 30 GPU-h/week, guaranteed | Best-effort, can be refused |

**Why Kaggle.** Colab free disconnects unpredictably and can refuse a GPU at busy times — a real risk now that the dataset is 10× larger than the original plan assumed. Kaggle guarantees 30 GPU-hours/week; this project needs ~8.

Use **T4, not P100** — Pascal has no tensor cores, so `fp16` buys far less there. Use **one** device: multi-GPU changes the effective batch size, which silently breaks the "identical settings" requirement in §1.3.

**`fp16=True`, never `bf16=True`** — T4 is Turing and has no bf16 support.

Sessions die: always `save_strategy="epoch"` to persistent storage or `push_to_hub=True`.

### 2.2 Libraries

```
torch  transformers  datasets  accelerate  tokenizers
sentencepiece  protobuf        # CamemBERT/XLM-R are SentencePiece-based; missing
                               # these gives a confusing mid-notebook ImportError
scikit-learn                   # TF-IDF baseline + all metrics
numpy  pandas  matplotlib
```

**Why `sklearn` for metrics, not `evaluate`.** `evaluate` downloads metric scripts from the Hub at call time — a network dependency inside your eval loop that breaks offline and inside a Space. `f1_score(average="macro")` is the same number with no surprises.

No `seaborn` — `ConfusionMatrixDisplay` plus matplotlib covers every figure here.

### 2.3 Pinning

Two files, which solves "a full `pip freeze` on Kaggle is 400 noisy lines":

- **`requirements.txt`** — the ~11 direct deps, exact versions. For humans.
- **`requirements-full.txt`** — complete freeze. For reproduction.

Renames to expect coming from the book:

| Book code | Current |
|---|---|
| `TrainingArguments(evaluation_strategy=)` | `eval_strategy=` |
| `Trainer(tokenizer=)` | `processing_class=` |
| `torch.quantization.quantize_dynamic` | `torch.ao.quantization.quantize_dynamic` |

### 2.4 Experiment tracking

**Why it isn't optional.** §1.3 implies ~15 training runs. A cell printing final metrics won't survive that, which defeats the point of seeding.

```python
args = TrainingArguments(..., report_to="wandb", run_name=f"camembert-s{seed}")
```

Log seed, model, and data size as config fields so the three-seed `mean ± std` comes out of the UI rather than hand-transcription. A public W&B report in the README shows the runs you *didn't* ship — that's evidence of method. No account? `report_to="tensorboard"` and commit `runs/`.

### 2.5 Storage

| Artifact | Where | Why |
|---|---|---|
| Notebooks, README, requirements | GitHub | What a reader actually opens |
| Fine-tuned models | HF Hub | `push_to_hub=True`; free model card |
| 300-item real-feedback set (§6) | HF Hub dataset | The most original thing you'll produce |
| Checkpoints | Kaggle working dir | Session-death insurance |

**Commit notebooks with outputs.** Contradicts normal hygiene, correct here: GitHub renders executed notebooks, so a reader sees your figures without running anything. Don't install `nbstripout`.

HF token with **write** scope, stored as a Kaggle Secret — never in a committed cell.

### 2.6 Labeling and serving

**Labeling the 300 real comments: Google Sheets.** Not Argilla, not Label Studio. 300 items, 3 classes, one collaborator. Dropdowns via data validation, five minutes to set up. Standing up a labeling server for 300 rows is yak-shaving.

**Serving: Gradio on a Hugging Face Space, CPU basic** (2 vCPU, 16 GB). Three things that matter:

1. **Pin Gradio exactly.** Spaces track Gradio's default; major versions change component APIs. An unpinned app breaks on its own, months after you stop touching it.
2. **CPU-only torch wheel** — the default pulls ~2.5 GB of CUDA libraries that never execute on a CPU Space:
   ```
   --extra-index-url https://download.pytorch.org/whl/cpu
   torch==X.Y.Z+cpu
   ```
3. **Quantize at startup; don't serialize the quantized model.** Serialized dynamically-quantized models are fragile across torch versions. Push fp32, quantize in `app.py` — a few seconds at boot, removes a whole failure class.

### 2.7 Deliberately not in the stack

Docker (Spaces builds the image) · FastAPI/Streamlit (Gradio is ~40 lines) · PyTorch Lightning (`Trainer` is already the abstraction) · Hydra (a dict is fine) · DVC (HF Hub versions datasets) · MLflow (W&B is less setup) · Poetry/uv (hosted notebooks) · vector DB (no retrieval anywhere).

*"I considered these and chose not to"* is worth a README line. An over-engineered stack reads as résumé-driven development.

---

## 3. System design

**What this is.** An offline batch pipeline producing one artifact, plus a stateless demo serving it. No persistent state, no concurrency problem, no scaling requirement.

The architecture question is narrow: **how do you guarantee the thing you measured and the thing you serve are the same thing?**

### 3.1 One prediction path, used everywhere

**Why.** Calibration, the neutral band, the aggregate correction, the real-feedback numbers, the final test table — all are only true if eval-time and serve-time are the identical function. Described as fragments, they drift. If the Space normalizes and the eval notebook doesn't, your published numbers no longer describe the deployed artifact and **nothing errors**. You just quietly become wrong.

```
raw text
  ├─ normalize()                 §4.4   one function, one definition
  ├─ tokenize(max_length)        §4.3
  ├─ encoder → logits
  ├─ logits / T                  §9     temperature scaling
  ├─ softmax → p(positive)
  ├─ neutral band [θ_lo, θ_hi]   §9.2
  └─ label ∈ {positif, neutre, négatif}
       ↓
  per-comment:  {label, p_pos, confident}
  aggregate:    ACC-corrected satisfaction rate + CI   §10
```

**Design rule.** `src/predict.py` exposes `predict(texts)` and `estimate_satisfaction(texts)`. Imported by notebooks 04–08 and `space/app.py`. **No notebook reimplements a stage inline.** Writing `softmax` in a notebook means you've broken the contract.

### 3.2 The model is not just weights

`T`, `θ_lo`, `θ_hi`, `TPR`, `FPR`, and the label order are learned parameters living *outside* the checkpoint, fit in different phases on different splits. Drift them apart from the weights and every output is silently wrong.

```json
{
  "model_repo": "you/camembert-satisfaction-fr",
  "revision": "a1b2c3d4",
  "labels": ["négatif", "positif"],
  "max_length": 192,
  "normalize_version": "v1",
  "temperature": 1.74,
  "theta_lo": 0.35,
  "theta_hi": 0.65,
  "tpr": 0.942,
  "fpr": 0.061
}
```

Commit as `inference_config.json` next to the weights. `predict.py` loads it; nothing is hardcoded.

**Set `id2label` before pushing** — otherwise the Hub widget shows `LABEL_0`/`LABEL_1`, and if you ever reorder `labels`, predictions silently remap:

```python
model.config.id2label = dict(enumerate(labels))
model.config.label2id = {l: i for i, l in enumerate(labels)}
```

**Pin the revision in `app.py`** — a specific SHA, not `main`. A later experimental push would otherwise change the live demo out from under the README.

### 3.3 Stage graph

```
00_audit ──> normalize(), max_length, subsample, polarity-hole finding
              │
01_baselines ─> majority, TF-IDF, zero-shot, off-the-shelf rows
              │
02_camembert ─> fp32 checkpoint (Hub) ─┬─> 03_xlmr_seeds
                                       ├─> 04_calibration ──> T, θ_lo, θ_hi ─┐
                                       └─> 07_efficiency ──> quantized       │
05_real_feedback ──> 300-item CSV ────────> domain-shift numbers ────────────┤
06_aggregate ──> TPR, FPR, shift experiment ─────────────────────────────────┤
                                            inference_config.json <──────────┘
                                                      │
08_final_test ────────────────────────────────────────┴──> final tables (once)
space/app.py ─────────────────────────────────────────┘
```

`inference_config.json` is where four phases converge, and both the final test notebook and the Space hang off it. That's §3.1 as a picture.

### 3.4 Serving runtime

- **Load and quantize at module scope**, not per request.
- **Cold start is ~20–40 s** after a Space sleeps. Acceptable, but *surprising* — say so in the README or your first reader concludes it's broken.
- **Memory:** ~440 MB fp32, both copies resident during conversion, peak ~700 MB. Fine in 16 GB.
- **Do NOT set `torch.set_num_threads(1)` in the Space.** §13.2 pins threads for *benchmark stability*; carrying that into `app.py` halves throughput for no reason. Very easy copy-paste error.
- **Batch mode matters now.** A CSV of 2,000 comments is the main use case — batch at 32, show a progress bar, cap the upload at ~5,000 rows so a free Space can't be wedged.

### 3.5 Input guards

| Input | Behavior |
|---|---|
| Empty / whitespace | Return early, no model call |
| Over `max_length` | Truncate **and say so** |
| **Non-French** | Soft warning, don't block — see below |
| CSV with no text column | Fail with a readable message naming the expected column |

**Non-French is the one to actually handle.** A reader will type English into your Space — this is not hypothetical. CamemBERT on English emits confident nonsense, and "92% satisfait" about an English sentence undercuts the whole repo. A lightweight check (`py3langid`, a few hundred KB) plus *"Ce texte ne semble pas être en français ; les résultats ne sont pas fiables."* **Warn, don't block** — blocking is worse UX and trusts a small classifier to gate your bigger one.

### 3.6 Not designed in

No database (nothing to persist) · no auth (public demo) · no request queue (Gradio's default is enough) · no retraining loop (the model is static) · no API versioning (Gradio's endpoint is incidental, not a product). If any were needed later, `predict()` is already the seam.

---

## 4. Phase 0 — Data audit *(~2.5 hours)*

**Do this before touching a model.** Everything downstream inherits whatever is wrong here.

### 4.1 Confirm the dataset and its construction

**Primary: `allocine`** on Hugging Face — ~200k real French reviews (160k/20k/20k), binary sentiment. Genuinely human-written French, real labels, no machine translation. This is what makes the pivot worthwhile.

**Verify availability and shape on day 1** rather than trusting this document — Hub datasets move and get deprecated. If `allocine` is unavailable, `cardiffnlp/tweet_sentiment_multilingual` (French config, 3-class) is a smaller fallback, and the spec survives with a smaller `n`.

**The finding to look for: the polarity hole.** Review datasets built from star ratings are usually constructed by taking high ratings as positive, low as negative, and **discarding the middle**. Check it:

```python
ds = load_dataset("allocine")
# inspect whatever rating/metadata columns exist; read 30 examples per class
# and judge: are there any genuinely lukewarm reviews, or only strong opinions?
```

If the middle was dropped, then **neutral text is out-of-distribution by construction** — your training data has a hole exactly where real client feedback is densest. That is not a flaw you can train away, it is a property of the data, and it is the direct justification for the neutral band in §9.2 and for the domain-shift test in §6. Write it in the README.

### 4.2 Duplicate leakage

**Why.** Scraped review corpora routinely contain duplicates (reposts, bot reviews, template text). Any text in both train and test inflates your score for free. 20 lines, and it yields a README sentence either way.

```python
import re, unicodedata
from collections import defaultdict

def norm(t):
    t = unicodedata.normalize("NFKC", t).lower().strip()
    return re.sub(r"\s+", " ", t)

seen = defaultdict(list)
for split in ds:
    for i, t in enumerate(ds[split]["review"]):
        seen[norm(t)].append((split, i))

cross = {k: v for k, v in seen.items() if len({s for s, _ in v}) > 1}
print(f"texts in >1 split: {len(cross)}")

conflict = [v for v in seen.values() if len({ds[s][i]["label"] for s, i in v}) > 1]
print(f"identical texts, conflicting labels: {len(conflict)}")
```

Report both raw and deduplicated test numbers — don't silently drop rows. The `conflict` count is an irreducible error floor and tells you how noisy the labels are.

### 4.3 Length, and the subsample decision

**Why this is a budget decision, not a detail.** Allociné reviews are *long* — far longer than the tweets the original plan assumed. At 160k rows × 256 tokens, one epoch on a T4 runs into hours, and §1.3's three-seed requirement becomes unaffordable. That would quietly push you into dropping the seeds, which is the one thing you must not drop.

```python
import numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("almanach/camembert-base")
lens = [len(tok(t).input_ids) for t in ds["train"]["review"][:5000]]
print(np.percentile(lens, [50, 90, 95, 99]))
```

**Do this.** Subsample **40k train rows** (stratified), set `max_length` near p90 — likely 192 — with `DataCollatorWithPadding`. Binary sentiment saturates fast; 40k gets you within a point or so of the full set at a quarter of the cost. Budget: ~6 min/epoch, so 3 seeds × 2 models ≈ 70 minutes total.

**Run the full 160k once, at the end, for the final CamemBERT only**, and report both. "40k reaches X, 160k reaches X+0.4" is a useful data-efficiency note that costs one run.

Also check the **truncation rate** at your chosen `max_length`: if 15% of reviews are cut, say so, and note that the sentiment is often in the final sentence of a review — a real risk worth one sentence in the limitations.

### 4.4 Text format and `normalize()`

**Why.** Whatever formatting the corpus has is baked into your model. If the Space feeds it a different format, part of your measured drop is formatting, not language — and you'd misattribute it.

```python
texts = ds["train"]["review"][:5000]
for name, pat in [("uppercase", None), ("punct", r"[.,!?;:]"),
                  ("emoji", r"[\U0001F300-\U0001FAFF]"), ("html", r"<[^>]+>")]:
    hits = (sum(t != t.lower() for t in texts) if pat is None
            else sum(bool(re.search(pat, t)) for t in texts))
    print(f"{name}: {hits/len(texts):.1%}")
```

Write **one** `normalize()` in `src/normalize.py`, applied in exactly three places: training, evaluation, `app.py`. One function, imported — not three inline copies that drift.

**Done when.** You have: a confirmed dataset, the polarity-hole finding written down, a duplicate report, a 40k stratified subsample, a chosen `max_length` with its truncation rate, and `normalize()` committed.

---

## 5. Phase 1 — Baselines *(~2.5 hours)*

Four baselines. Three are nearly free and they're what make the final table honest.

### 5.1 Majority class *(2 min)*

Anchors the bottom and shows why accuracy alone misleads. Allociné is roughly balanced, so expect ~50%.

### 5.2 TF-IDF + Logistic Regression *(15 min)*

**Why this is the baseline that matters.** Most projects skip it because it's embarrassing: on binary review sentiment, word+char n-grams with logistic regression is **very** strong — typically low-to-mid 90s on Allociné. If your fine-tuned CamemBERT hits 96% and TF-IDF hits 93%, the honest headline is *"the transformer buys three points over a model that trains in twenty seconds on CPU."*

That delta is the justification for the whole project. Omitting it makes the transformer look better and mean less.

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tfidf = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000, class_weight="balanced"),
).fit(train_texts, train_labels)
```

Keep it — it's also your latency and size floor in Phase 7, which is what makes the quantized transformer's numbers legible.

### 5.3 Off-the-shelf French sentiment model *(30 min)*

**Why.** The honest question a reviewer will ask is *"why not just use a model that already exists?"* Answer it with a number.

Use `nlptown/bert-base-multilingual-uncased-sentiment` (multilingual, 5-star product reviews — collapse 1–2★→negative, 4–5★→positive, 3★→neutral). Its training domain is *product* reviews, which is closer to your end use than movie reviews, so it's a genuinely informative comparison.

> ⚠️ **Do not benchmark against a model trained on Allociné** (several exist on the Hub). It has seen your test set; the comparison is meaningless. If you mention such a model, say explicitly why you excluded it — that catch is itself worth a sentence.

### 5.4 Zero-shot *(~1 h)*

Answers a third question: is *labeled data* worth it? `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, candidate labels `["positif", "négatif"]`, on **validation**.

French grammar trap: pick a hypothesis template that is grammatical for every label. `"Ce client est {}."` works; try two or three and report which won — zero-shot swings several points on phrasing, and quietly reporting one template is misleading.

**Done when.** Four validation rows with macro-F1 and per-class P/R.

---

## 6. Phase 2 — Real client-feedback eval set *(~6 hours)*

**Positioned early, deliberately.** It's the highest-value deliverable and has the longest lead time. Built last, it's a number you can no longer act on.

### 6.1 Why this set exists

Allociné is **movie** reviews. Your product question is about **client feedback on a product or service**. Those differ in vocabulary, length, structure, and — critically — in how much neutral content they contain. This set measures that gap, and the gap is the honest headline of the project.

### 6.2 Source and size

Collect **300** genuine French comments about products or services. Prefer sources with clean licensing — public review corpora on the Hub, or public APIs (Mastodon's French timeline, YouTube Data API). Avoid scraping Trustpilot or Amazon: the ToS question is real and this goes in a public repo.

Filter to a sensible length band and record the filter.

**Do not write them yourself.** Text you compose to test a sentiment model will be clean, single-valence and prototypical — exactly the distribution the model already handles. The point is to find failures you can't anticipate.

### 6.3 Label protocol

- **Three classes: positif / neutre / négatif**, even though training is binary. The neutral count is the finding.
- Shuffle, strip metadata, label **before** running any model. You are both annotator and interested party; blind labeling costs nothing and removes the objection.
- Write a one-page codebook first: definition, two examples per class, and a tie-break rule for mixed reviews (e.g. *"label the valence of the overall recommendation, not of individual clauses"*).
- Also record, per item, whether it is **mixed** (both praise and complaint). Mixed reviews are where binary classifiers fail silently and where aspect analysis (§11) earns its place.
- Second annotator on 50–100 items → **Cohen's κ**:

```python
from sklearn.metrics import cohen_kappa_score
print(cohen_kappa_score(me, them))
```

κ around 0.6–0.75 is normal for 3-class sentiment. Reporting it is stronger than any F1 — it shows you know the task has a ceiling and you measured it.

### 6.4 What to report

| Slice | What it isolates |
|---|---|
| All 300, 3-way | Headline, deployment-realistic |
| Binary only (drop neutral) | Comparable to the Allociné test number |
| **Neutral items** | How a binary-trained model handles the hole from §4.1 |
| Mixed-valence items | Where single-label classification breaks down |

**Done when.** 300 labeled items (3-class + mixed flag), a κ figure, saved as CSV and pushed as a Hub dataset. No model predictions seen yet.

---

## 7. Phase 3 — Fine-tune CamemBERT *(~6 hours)*

```python
args = TrainingArguments(
    output_dir="camembert-satisfaction-fr",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=True,                       # T4: fp16 yes, bf16 NO
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    seed=42,
    report_to="wandb",
)
```

**Why these values.** `2e-5` is the standard safe LR for base encoders; above `5e-5` they destabilize. Three epochs with `load_best_model_at_end` lets you *see* the overfitting point rather than guess — binary sentiment on 40k usually peaks at epoch 2. `warmup_ratio=0.1` prevents the early-step collapse that occasionally produces a dead run stuck at the majority class.

No class weighting needed — Allociné is near-balanced. (That's a real simplification the pivot buys you.)

### 7.1 Error analysis

Sort validation examples by per-example loss, inspect the worst 30, and **categorize** rather than list: mislabeled / sarcasm / mixed valence / negation / off-topic / genuine model error. A count table beats ten anecdotes. Expect **sarcasm and mixed valence** to dominate — both are well-known hard cases in review sentiment, and naming them connects directly to §11.

**Done when.** Validation macro-F1 recorded, confusion matrix saved, categorized error table written, model pushed to the Hub with `id2label` set.

---

## 8. Phase 4 — Multilingual comparison *(~1.5 hours)*

`xlm-roberta-base` instead of CamemBERT, everything else byte-identical. **Three seeds each** (§1.3), reported `mean ± std`.

Honest caveat for the README: XLM-R base carries a 250k-token vocabulary against CamemBERT's ~32k, so "equal settings" is not "equal parameters." Naming it yourself beats having it pointed out.

**Done when.** Two rows with ±, and a sentence that explicitly says whether the gap exceeds the spread. If it doesn't, write *"within run-to-run variance"* — that's a result.

---

## 9. Phase 5 — Calibration and the neutral band *(~2 hours)*

### 9.1 Temperature scaling

**Why it's load-bearing here, not cosmetic.** Two later components consume probabilities directly: the neutral band below, and the aggregate estimator in §10. Uncalibrated transformer outputs are badly overconfident — a 0.94 typically means ~0.80 accuracy — so thresholding them would be meaningless.

```python
import torch
from torch import nn, optim

def fit_temperature(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    log_T  = torch.zeros(1, requires_grad=True)     # optimise log T to keep T > 0
    opt, nll = optim.LBFGS([log_T], lr=0.1, max_iter=100), nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(); loss = nll(logits / log_T.exp(), labels)
        loss.backward(); return loss
    opt.step(closure)
    return log_T.exp().item()

def ece(probs, labels, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    acc, bins, e = (pred == labels).astype(float), np.linspace(0, 1, n_bins + 1), 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e
```

Temperature scaling doesn't change argmax, so **accuracy and F1 are unchanged** — say that in the model card so nobody reads it as an accuracy claim.

### 9.2 The neutral band

**Why.** §4.1 established that the training data has no neutral class *by construction*, while real client feedback is full of lukewarm and mixed comments. A binary model will assign every one of them confidently to a side. Rather than pretend, abstain:

```python
if theta_lo <= p_pos <= theta_hi:
    label = "neutre"
```

**Tune `θ_lo`, `θ_hi` on the §6 real-feedback set**, which has genuine 3-class labels — split it 150/150, fit on the first half, report on the second. This is the second payoff from building that set early.

**Done when.** `T`, ECE before/after, a reliability diagram, `θ_lo`/`θ_hi` chosen on held-out real data, 3-class performance reported on the untouched half, and all four values in `inference_config.json`.

---

## 10. Phase 6 — Aggregate satisfaction estimation *(~3 hours)*

**The centerpiece.** This is what turns a classifier into an answer to the client's actual question.

### 10.1 Why counting predictions is wrong

The obvious approach — classify every comment, count the positives, divide — is called **Classify and Count (CC)**, and it is biased whenever the classifier's errors are asymmetric. Formally, with true positive prevalence `p`:

```
observed positive rate  p̂ = TPR·p + FPR·(1 − p)
```

CC reports `p̂` and calls it `p`. Those are equal only if `TPR = 1` and `FPR = 0`. For a realistic classifier (TPR 0.94, FPR 0.06) with a truly dissatisfied client base (`p = 0.30`), CC reports **0.325** — and the bias grows as the true rate moves away from the prevalence the classifier was trained on, which is exactly when you most need the answer.

### 10.2 The correction

Invert the equation. This is **Adjusted Classify and Count (ACC)**:

```python
def acc_prevalence(pred_pos_rate, tpr, fpr):
    """Bias-corrected satisfaction rate. Requires tpr > fpr."""
    return float(np.clip((pred_pos_rate - fpr) / (tpr - fpr), 0.0, 1.0))
```

`TPR` and `FPR` are measured once on a labeled holdout and stored in `inference_config.json`. Three lines of code, and it is the difference between a demo and an instrument.

### 10.3 The experiment that proves it works

**Why.** Don't assert that ACC is better — demonstrate it. This is cheap and it's the most convincing figure in the project.

**Do this.** Resample your validation set to a range of known true prevalences (10%, 30%, 50%, 70%, 90%), and at each one plot CC's estimate and ACC's estimate against truth.

```python
for true_p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    sample = resample_to_prevalence(val, true_p)
    cc  = (predict(sample) == POS).mean()
    acc = acc_prevalence(cc, tpr, fpr)
    print(f"true={true_p:.2f}  CC={cc:.3f}  ACC={acc:.3f}")
```

Expect CC to bow toward the middle and ACC to track the diagonal. That single plot *is* your README headline figure.

### 10.4 Confidence intervals

The estimate has two uncertainty sources: sampling of the comments, and estimation error in `TPR`/`FPR`. Bootstrap **both** — resample the unlabeled batch *and* the labeled holdout, recompute end to end, take the 2.5/97.5 percentiles. Reporting `"62% satisfied [57–67]"` instead of `"62% satisfied"` is the single clearest signal that you understand what you built.

**Done when.** `TPR`/`FPR` stored, the prevalence-shift plot saved, bootstrap CIs implemented in `estimate_satisfaction()`, and ACC's mean absolute error vs CC's reported across the five shift levels.

---

## 11. Phase 7 — Aspect breakdown: the "why" *(~3 hours)*

**Why.** *"62% satisfied"* isn't actionable. *"62% satisfied; among the dissatisfied, 41% mention delivery and 28% mention customer service"* is. This is the half of the business question the classifier alone can't answer.

**Scope honestly.** Full aspect-based sentiment analysis needs aspect-annotated training data you don't have, and the original plan was right to rule it out. There's a cheap, defensible middle ground.

**Do this.**

1. **Define 6–8 aspect lexicons** from your own data, not from imagination: take the most frequent nouns in negative reviews and group them. Typically something like *livraison, prix, qualité, service client, facilité d'utilisation, délais*.
2. **Split each review into sentences**, and attribute an aspect to a sentence if it contains a lexicon term.
3. **Run the sentiment classifier on the sentence, not the whole review.** This is what makes it work on mixed reviews — *"produit super mais livraison catastrophique"* yields positive-quality and negative-delivery instead of one muddled label.
4. **Report sentiment distribution per aspect**, with counts, and **apply the §10 correction per aspect** — the bias argument holds just as much within a subgroup.

**State the limitations plainly:** no implicit aspects (a complaint with no keyword is missed), no aspect–opinion dependency parsing, lexicons are hand-built and domain-specific. Being precise about what the method *doesn't* do is what makes it credible rather than oversold.

**Done when.** A per-aspect table with corrected sentiment rates and counts, plus 2–3 verbatim example quotes per aspect for the report.

---

## 12. Phase 8 — Emotion layer *(optional, ~3 hours)*

**Only if Phases 0–7 are done.** Among negative feedback, *angry* and *disappointed* call for different responses — anger usually signals a service failure needing escalation, disappointment an expectation gap. That distinction is genuinely useful, which is why it's worth keeping.

Fine-tune on `TPM-28/emotion-FR` as originally planned, and apply it **only to comments the satisfaction model scores negative**.

Three caveats that must be loud, or this layer does more harm than good:

- It's trained on **machine-translated introspective social-media text**, not reviews. The domain gap is large.
- That dataset was harvested by emotion-keyword matching, so a large share of rows literally contain *"je me sens…"* — the label is partly given away by surface vocabulary. Real reviews don't talk that way, so the cue is **absent at deployment**. Measure the cue rate in both corpora and report it.
- Its undocumented label mapping needs verifying before use (match per-class counts against the English `dair-ai/emotion` original — a far stronger check than eyeballing rows).

Present this layer as **exploratory**, with its own numbers on the real-feedback set. If it performs badly there, say so and keep it out of the demo's default view. A clearly-labeled negative result is worth more than a quietly shipped bad feature.

---

## 13. Phase 9 — Efficiency *(~2 hours)*

### 13.1 Quantize

```python
import torch
qmodel = torch.ao.quantization.quantize_dynamic(
    model.cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8
)
```

**Set expectations correctly.** Dynamic quantization converts `nn.Linear` only. CamemBERT-base is ~110M params, of which the embedding table (~32k × 768 ≈ 25M) **stays fp32**. Expect roughly **440 MB → ~185 MB (~2.4×)**, not the 4× people assume. Stating the mechanism is worth more than a bigger number.

It matters more here than in the original plan: processing a 2,000-row CSV on a free CPU Space is the main use case, so a 2–3× throughput gain is the difference between usable and not.

### 13.2 Benchmark properly

```python
import time, numpy as np, torch

def latency_ms(model, enc, n_warm=10, n_run=200):
    torch.set_num_threads(1)              # pin: shared-CPU scheduling is the noise
    model.eval()
    with torch.inference_mode():
        for _ in range(n_warm): model(**enc)
        ts = []
        for _ in range(n_run):
            t0 = time.perf_counter(); model(**enc)
            ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), float(np.percentile(np.array(ts), 95))
```

Report **median and p95**, not mean. Measure **two conditions**: batch 1 (single-comment mode) and batch 32 (CSV mode) — they're different products.

**Gate:** quantized macro-F1 within **1.0 point** of fp32 on validation, and — because §10 consumes probabilities — re-measure `TPR`/`FPR` and ECE on the quantized model rather than reusing the fp32 values.

---

## 14. Phase 10 — Ship *(~5 hours)*

### 14.1 The app is a report, not a label

This is the main difference from the original plan, and it's what makes the project read as a product:

- **Tab 1 — Batch (primary).** Upload a CSV of feedback → corrected satisfaction rate with CI, a positive/neutral/negative bar, the per-aspect table from §11, and example quotes. A sample CSV preloaded so a visitor can click once and see the whole thing.
- **Tab 2 — Single comment.** Text box → label, calibrated probability, neutral-band status. For inspection and for the skeptical reader.

Apply `normalize()` before inference in both — the likeliest source of a silent demo/eval mismatch.

### 14.2 Model card

**Provenance, precisely.** State the chain: fine-tuned on `allocine` (real French movie reviews), evaluated additionally on 300 hand-labeled real product/service comments. Note the licensing of each source honestly rather than inventing one. If you publish the 300-item set, say exactly what it is and how it was annotated.

**Intended use and misuse.** Your own κ figure does the work here: if two humans agree only ~0.7 of the time, per-individual decisions aren't supportable at any accuracy you'll reach.

> **Not suitable for:** decisions about individual customers, automated moderation without human review, or feedback outside the French review register it was trained on. The aggregate estimate is reliable; individual predictions are not.

That last sentence is both true and exactly the distinction §0.1 is built on.

Include: the polarity-hole finding, the neutral band and how its thresholds were set, the CC-vs-ACC plot, the domain-shift numbers, κ, and the aspect method's limitations.

### 14.3 Repo manifest

```
README.md                      # headline figure first, limitations second, setup last
requirements.txt               # ~11 direct deps, exact pins
requirements-full.txt
inference_config.json          # T, thresholds, TPR/FPR, revision — §3.2
src/
  normalize.py                 # the ONE normalize() — §4.4
  predict.py                   # the ONE prediction path + estimate_satisfaction() — §3.1
  aspects.py                   # lexicons + sentence attribution — §11
notebooks/
  00_audit.ipynb               # §4
  01_baselines.ipynb           # §5
  02_camembert.ipynb           # §7
  03_xlmr_seeds.ipynb          # §8
  04_calibration.ipynb         # §9
  05_real_feedback.ipynb       # §6 — the only notebook touching the real set
  06_aggregate.ipynb           # §10
  07_efficiency.ipynb          # §13
  08_final_test.ipynb          # the ONLY notebook containing ds["test"]
data/real_feedback_300.csv     # text, label_3way, mixed, annotator_2
space/                         # app.py + its own pinned requirements.txt
```

The §1.1 discipline is visible in the layout: exactly one notebook reads test, auditable at a glance.

**Done when.** A stranger uploads the sample CSV, gets a satisfaction rate with an interval and an aspect breakdown in under thirty seconds, and finds the limitations section without scrolling past marketing.

---

## 15. Results tables

Three tables. Keep them separate — mixing levels is how optimistic numbers sneak in.

**Development (validation, iterate freely):**

| Model | Acc | Macro-F1 | ECE |
|---|---|---|---|
| Majority class | ~0.50 | ~0.33 | — |
| TF-IDF + LogReg | | | |
| Off-the-shelf (nlptown) | | | |
| Zero-shot mDeBERTa | | | |
| CamemBERT fine-tuned (40k) | | | |

**Final (test, filled once, never revisited):**

| Model | Acc | Macro-F1 (3 seeds) | Lat. p50 bs=1 / bs=32 | Size | ECE |
|---|---|---|---|---|---|
| TF-IDF + LogReg | | — | | | |
| CamemBERT (40k) | | ± | | | |
| CamemBERT (160k) | | — | | | |
| XLM-R (40k) | | ± | | | |
| CamemBERT quantized | | ± | | | |

**Aggregate estimation — the headline:**

| True prevalence | CC estimate | ACC estimate | CC error | ACC error |
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

Plus standalone figures: **inter-annotator κ**, **% neutral in real feedback**, the **CC-vs-ACC plot**, and the **per-aspect table**.

---

## 16. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| `allocine` unavailable or changed | Low | Verify day 1; `cardiffnlp/tweet_sentiment_multilingual` FR fallback (§4.1) |
| Full 160k training blows the GPU budget | **High** | 40k stratified subsample (§4.3); full run once at the end |
| TF-IDF nearly matches the transformer | **High** | Expected (§5.2) — the delta is the finding, not a failure |
| Benchmarking against an Allociné-trained model | Medium | Contaminated; exclude and say why (§5.3) |
| Movie→product domain gap is large | **High** | That's what §6 measures; it's the headline, not a surprise |
| Too few neutral items in the real set | Medium | Oversample lukewarm candidates while collecting; report n |
| `T`/θ/TPR/FPR drift from the weights | Medium | `inference_config.json`, pinned revision (§3.2) |
| Label order silently remaps | Low, severe | Set `id2label` before pushing (§3.2) |
| Reader types English into the demo | **Near-certain** | Soft language warning, don't block (§3.5) |
| Session dies mid-training | Medium on Kaggle | `save_strategy="epoch"` to persistent storage |
| Space breaks later on a Gradio bump | Medium | Pin Gradio and torch exactly (§2.6) |

### 16.1 If the fine-tune underperforms

1. **Macro-F1 ≈ 0.33, acc ≈ 0.50** → collapsed to one class. Drop LR to `1e-5`, confirm `warmup_ratio`, check the loss moved.
2. **Accuracy near chance** → label mapping or shuffle bug. Re-check §4.1.
3. **Plausible but below TF-IDF** → check the truncation rate from §4.3, and that `normalize()` is applied identically on both sides.
4. **Good on Allociné, poor on real feedback** → that's not a bug, it's §6 working. Report it.

---

## 17. Schedule

| Weekend | Phases | Hours |
|---|---|---|
| 1 | 0 audit · 1 baselines · start 3 CamemBERT | ~11 |
| 2 | finish 3 · **2 real-feedback set (a full day)** · 4 XLM-R · 5 calibration | ~13 |
| 3 | **6 aggregate** · 7 aspects · 9 efficiency · 10 ship | ~13 |
| *(spare)* | 8 emotion layer, 160k run, ONNX | — |

≈ 37 hours. If you fall behind, cut in this order: the emotion layer (§12), the 160k run, then the XLM-R comparison. **Never cut** the test discipline (§1.1), the three seeds (§1.3), the real-feedback set (§6), or the aggregate estimator (§10).

---

## 18. Why this project is worth doing

Most sentiment projects are a fine-tune and a demo, and they answer the model question while claiming to answer the product question. This one separates the two and is explicit about it.

Five things distinguish it, none expensive: the data is **audited before use** (polarity hole, leakage, truncation), the transformer is **made to justify itself** against a twenty-second TF-IDF baseline and an off-the-shelf model, the protocol is **stated and followed** (test once, comparisons seeded, small sets get intervals), the model is **honest about itself** (calibrated, with a neutral band where the training data had a hole), and the headline number is an **unbiased population estimate with a confidence interval** rather than a count of predictions.

If only one thing survives contact with reality, make it §10. *"Counting your classifier's predictions gives the wrong answer, here is the correction, and here is the plot showing it works"* is a finding about method, not about your model — and it's the part a hiring manager will remember.

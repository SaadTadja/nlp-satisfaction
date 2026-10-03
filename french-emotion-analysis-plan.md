# French Emotion & Sentiment Analysis — Project Plan

*Based on* Natural Language Processing with Transformers *(Tunstall, von Werra, Wolf — O'Reilly)*

---

## Goal

A model that takes a French comment or message and returns:

- **Emotion** — one of 6 classes, with confidence scores
- **Sentiment** — positive / negative / neutral, derived from the emotion

Trained, benchmarked, quantized, and deployed as a public demo on Hugging Face.

---

## The key shortcut: the book's dataset already exists in French

`TPM-28/emotion-FR` on Hugging Face:

- 20,000 rows — 16k train / 2k validation / 2k test
- Labels 0–5, single-label
- Machine-translated version of the English emotion dataset used in **Chapter 2** of the book

This means the Chapter 2 code runs almost unchanged: swap the dataset name and the model.

**Catch:** the dataset README is empty, so the label names are not documented. Based on matching samples to the original English dataset, the mapping is:

```python
labels = ["tristesse", "joie", "amour", "colère", "peur", "surprise"]
# 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
```

> ⚠️ Verify this yourself on ~20 rows before trusting it. It is an inference, not documentation.

---

## Which book chapters apply

| Chapter | Topic | Used? |
|---|---|---|
| 1 | Hello Transformers / pipelines | ✅ Step 1 |
| 2 | Text classification | ✅ Step 2 — the core |
| 3 | Transformer anatomy | ✅ Step 4 (optional) |
| 4 | Multilingual models | ✅ Step 3 |
| 5 | Text generation | ❌ different task |
| 6 | Summarization | ❌ different task |
| 7 | Question answering | ❌ different task |
| 8 | Making transformers efficient | ✅ Step 5 |
| 9 | Few to no labels | ✅ Step 1 |
| 10 | Training from scratch | ❌ different task |
| 11 | Future directions | ❌ |

Skipping the unrelated chapters is deliberate. Forcing them in would make the project harder and messier, not more complete.

---

## Setup

- **Google Colab**, free T4 GPU
- One notebook per step
- Pin the `transformers` version at the top of every notebook and record it in the README

> The book targets an older `transformers` version. Expect small renames. The most common: `evaluation_strategy` is now `eval_strategy` in `TrainingArguments`.

---

## Step 1 — Zero-shot baseline *(Ch 9 + Ch 1)*

Before training anything, measure what you get with no training at all.

- [ ] Load the zero-shot classification pipeline with `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- [ ] Use the six French emotion words as candidate labels
- [ ] Score on the test set: accuracy + macro-F1

**Output:** the floor. Every later step must beat this number.
**Time:** ~1 hour

---

## Step 2 — Fine-tune CamemBERT *(Ch 2 — the core)*

Follow Chapter 2 almost line by line, replacing DistilBERT with `almanach/camembert-base`.

- [ ] Load `TPM-28/emotion-FR` and apply the label names
- [ ] Plot the class distribution — it is imbalanced (joy and sadness dominate), which is why macro-F1 matters more than accuracy
- [ ] Tokenize with the CamemBERT tokenizer
- [ ] **Feature extraction:** frozen CamemBERT hidden states + scikit-learn `LogisticRegression`
- [ ] **Fine-tuning:** full fine-tune with the `Trainer`
- [ ] Confusion matrix + macro-F1 for both approaches
- [ ] **Error analysis:** sort validation examples by loss and inspect the worst ones — expect translation artifacts and mislabels; this makes a strong README section
- [ ] Push the fine-tuned model to the Hugging Face Hub

**Output:** two results rows (frozen features, fine-tuned) + error analysis notes.
**Time:** one weekend — this is ~80% of the project.

---

## Step 3 — Multilingual comparison *(Ch 4)*

- [ ] Change one line: `xlm-roberta-base` instead of CamemBERT
- [ ] Rerun the Step 2 fine-tuning with identical settings
- [ ] Compare: does a French-specific model beat a multilingual one?

**Output:** one extra results row.

---

## Step 4 — Attention visualization *(Ch 3 — optional)*

- [ ] Use `bertviz` on 3–4 French sentences
- [ ] Save screenshots for the README

Cheap, visual, and shows you understand what's inside the model.

---

## Step 5 — Make it fast *(Ch 8)*

- [ ] Reuse the chapter's `PerformanceBenchmark` class (accuracy, latency, model size)
- [ ] Apply dynamic quantization: `torch.quantization.quantize_dynamic`
- [ ] Benchmark before vs. after on CPU
- [ ] *Stretch:* distill into `cmarkea/distilcamembert-base`

**Output:** a quantized model with latency and size numbers. Quantization alone is enough.

---

## Step 6 — Sentiment for free

Don't train a separate sentiment model. Map the emotions:

| Emotion | Sentiment |
|---|---|
| joie, amour | positive |
| tristesse, colère, peur | negative |
| surprise | neutral |

```python
SENTIMENT = {
    "joie": "positive", "amour": "positive",
    "tristesse": "negative", "colère": "negative", "peur": "negative",
    "surprise": "neutral",
}
```

One model returns both emotion and sentiment, at zero extra training cost.

---

## Step 7 — Reality check on real French

The training data is machine-translated. Test whether the model works on French that real people wrote.

- [ ] Collect or write **100 genuine French comments** — reviews, messages, the kind of text this project claims to analyze
- [ ] Label them yourself with the 6 emotions
- [ ] Run your best model on them
- [ ] Report the score drop vs. the translated test set

This single honest number is what separates this project from the many identical emotion classifiers on the Hub.

**Time:** one evening

---

## Step 8 — Ship

- [ ] **Gradio app on a Hugging Face Space** (~15 lines): text box in → emotion scores + sentiment out
- [ ] **Model card** with the full results table
- [ ] **GitHub repo** with all notebooks, pinned requirements, and a README written for a hiring manager

> Gradio on a Space is much simpler than FastAPI + Streamlit and is enough to demonstrate the work.

---

## Final results table (README centerpiece)

| Model | Accuracy | Macro-F1 | Latency (CPU) | Size |
|---|---|---|---|---|
| Zero-shot mDeBERTa | | | | |
| CamemBERT frozen + LogReg | | | | |
| CamemBERT fine-tuned | | | | |
| XLM-R fine-tuned | | | | |
| CamemBERT quantized | | | | |
| **Best model on 100 real comments** | | | — | — |

---

## Timeline

| Weekend | Steps |
|---|---|
| 1 | Steps 1–2 |
| 2 | Steps 3–5 |
| 3 | Steps 6–8 |

---

## Explicitly out of scope

- Aspect-based sentiment analysis (not covered by the book; different task)
- Per-language routing architecture
- PostgreSQL database
- React dashboard
- Separate sentiment model training

Stating these as non-goals keeps the project finishable.

---

## CV line

> *Fine-tuned and benchmarked French emotion classifiers (CamemBERT, XLM-R), quantized for CPU deployment, and measured performance on real human-written French versus machine-translated training data. Deployed as a public Hugging Face demo.*

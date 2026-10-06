"""Gradio demo — English hotel-review satisfaction.

Run locally:

    python space/app_en.py

Two models, deliberately, because there are two units of analysis and each was
evaluated on its own:

* **review model** — trained on review-level text with real star ratings.
  Best on the review task (macro-F1 0.6258). Drives the batch verdict and the
  aggregate estimate.
* **sentence model** — trained on sentence-level labels. Worse when rolled up
  to reviews (0.5227) but correct on standalone prose (probe 7/8 vs 6/8).
  Drives the aspect breakdown, which is sentence-level by construction, and
  the single-comment tab, where the input *is* a standalone sentence.

Using one model for both would mean applying it to a unit it was never
measured on. The shared pipeline (normalise -> logits -> temperature ->
label) is still defined once, in ``src/predict.py``; only the weights differ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.estimate import acc_prevalence_multiclass, predicted_distribution  # noqa: E402
from src.normalize import is_empty  # noqa: E402
from src.predict import InferenceConfig, SatisfactionModel, load_config  # noqa: E402

MAX_ROWS = 5_000
LABELS = ["negative", "neutral", "positive"]
PRETTY = {"negative": "Negative", "neutral": "Neutral / mixed", "positive": "Positive"}

LOAD_ERROR: str | None = None
REVIEW_MODEL = SENTENCE_MODEL = None
CFG = None
try:
    CFG = load_config(REPO / "inference_config_en.json")
    REVIEW_MODEL = SatisfactionModel(CFG)

    raw = json.loads((REPO / "inference_config_en.json").read_text(encoding="utf-8"))
    scfg = InferenceConfig(
        model_repo=raw["sentence_model_repo"], revision="en-sentence-v1",
        labels=LABELS, normalize_version=raw["normalize_version"],
        temperature=1.0, mode="multiclass", backend="sklearn",
        class_order=raw["sentence_class_order"],
        confusion_matrix=CFG.confusion_matrix, aggregation_ready=True,
    )
    scfg.validate()
    SENTENCE_MODEL = SatisfactionModel(scfg)
except Exception as exc:  # noqa: BLE001
    LOAD_ERROR = f"{type(exc).__name__}: {exc}"

ASPECTS: pd.DataFrame | None = None
_ap = REPO / "data_en" / "sentences.csv"
if _ap.exists():
    ASPECTS = pd.read_csv(_ap, usecols=["phrase", "aspect_label"])


def _aspect_of(text: str) -> set[str]:
    """Nearest-aspect lookup by shared vocabulary with the labelled corpus.

    A stand-in for the dataset's own BERTopic assignment, which is not
    available for text the corpus has never seen.
    """
    if ASPECTS is None:
        return set()
    words = {w.lower().strip(".,!?;:") for w in text.split() if len(w) > 3}
    if not words:
        return set()
    hit = ASPECTS[ASPECTS.phrase.fillna("").str.lower().apply(
        lambda p: len(words & set(p.split())) >= 2)]
    return set(hit.aspect_label.value_counts().head(1).index)


# --------------------------------------------------------------------------
def analyse_one(text: str):
    if LOAD_ERROR:
        return {}, f"⚠️ Model not loaded — {LOAD_ERROR}"
    if is_empty(text):
        return {}, "Enter a review."

    p = SENTENCE_MODEL.predict([text])[0]
    scores = {PRETTY[k]: v for k, v in p.probs.items()}
    notes = [f"### {PRETTY[p.label]}  ·  {p.confidence:.0%} confidence"]

    asp = _aspect_of(text)
    if asp:
        notes.append(f"**Topic:** {', '.join(sorted(asp))}")
    if p.confidence < 0.5:
        notes.append(
            "ℹ️ Low confidence. Mixed reviews — praise and complaint in the "
            "same text — land here, and they are genuinely ambiguous rather "
            "than a model failure."
        )
    low = text.lower()
    if " not " in f" {low} " or "n't" in low:
        notes.append(
            "⚠️ This text contains a negation. The backbone is a bag of "
            "n-grams and cannot scope *not*, so phrases like *“could not have "
            "been better”* are read as negative. A known structural limit."
        )
    return scores, "\n\n".join(notes)


def analyse_batch(file):
    if LOAD_ERROR:
        return f"⚠️ Model not loaded — {LOAD_ERROR}", None, None
    if file is None:
        return "Upload a CSV first.", None, None
    try:
        df = pd.read_csv(file.name)
    except Exception as exc:  # noqa: BLE001
        return f"Could not read the CSV: {exc}", None, None

    col = next((c for c in ("text", "review", "comment", "feedback", "phrase")
                if c in df.columns), None)
    if col is None:
        return (f"No text column found. Expected one of: text, review, comment, "
                f"feedback, phrase. Found: {list(df.columns)}"), None, None

    texts = [t for t in df[col].astype(str).tolist() if not is_empty(t)][:MAX_ROWS]
    if not texts:
        return "No usable text in that file.", None, None

    preds = REVIEW_MODEL.predict(texts, batch_size=256)
    labs = np.array([p.label for p in preds])
    naive = predicted_distribution(labs, LABELS)
    corrected = acc_prevalence_multiclass(naive, CFG.confusion_matrix)

    _, dist, lo, hi, _ = REVIEW_MODEL.estimate_distribution(texts, n_boot=600)
    i = LABELS.index("positive")
    summary = (
        f"## {dist[i]:.0%} satisfied\n\n"
        f"**95% interval: {lo[i]:.0%} – {hi[i]:.0%}**\n\n"
        f"- Reviews analysed: **{len(preds):,}**\n"
        f"- Raw count, uncorrected: {naive[i]:.0%} "
        f"— the correction moved it by {abs(dist[i]-naive[i])*100:.1f} points\n\n"
        f"> Counting predictions is biased whenever the classifier's errors are "
        f"asymmetric. The figure above inverts the measured confusion matrix "
        f"(Adjusted Classify and Count), which cut estimation error 4.5× on "
        f"held-out data."
    )

    breakdown = pd.DataFrame({
        "label": [PRETTY[l] for l in LABELS],
        "counted": [f"{v:.0%}" for v in naive],
        "corrected": [f"{v:.0%}" for v in corrected],
        "95% interval": [f"{lo[k]:.0%} – {hi[k]:.0%}" for k in range(3)],
    })

    sp = SENTENCE_MODEL.predict(texts, batch_size=256)
    ex = pd.DataFrame({
        "verdict": [PRETTY[p.label] for p in sp[:12]],
        "confidence": [f"{p.confidence:.0%}" for p in sp[:12]],
        "text": [p.text[:140] for p in sp[:12]],
    })
    return summary, breakdown, ex


with gr.Blocks(title="Hotel Satisfaction Analysis") as demo:
    gr.Markdown(
        "# Hotel guest satisfaction\n"
        "Estimates **what share of guests are satisfied** from their written "
        "reviews, with a confidence interval, and shows the breakdown."
    )
    if LOAD_ERROR:
        gr.Markdown(f"> ⚠️ **Model not loaded.** `{LOAD_ERROR}`")

    with gr.Tab("Batch (CSV)"):
        gr.Markdown(
            "Upload a CSV with a `text` column (or `review`, `comment`, "
            f"`feedback`, `phrase`). Up to {MAX_ROWS:,} rows.\n\n"
            "**Try `data_en/example_reviews.csv`** — 1,286 held-out reviews. "
            "Their true satisfied share is **51.9%**, so you can check the "
            "estimate against a known answer."
        )
        f_in = gr.File(label="CSV file", file_types=[".csv"])
        btn = gr.Button("Analyse", variant="primary")
        out_md = gr.Markdown()
        out_tbl = gr.Dataframe(label="Corrected breakdown")
        out_ex = gr.Dataframe(label="First rows")
        btn.click(analyse_batch, f_in, [out_md, out_tbl, out_ex])

    with gr.Tab("Single review"):
        t_in = gr.Textbox(label="Review", lines=4,
                          placeholder="Paste a hotel review in English…")
        t_btn = gr.Button("Analyse", variant="primary")
        t_scores = gr.Label(label="Calibrated probabilities")
        t_notes = gr.Markdown()
        t_btn.click(analyse_one, t_in, [t_scores, t_notes])
        gr.Examples([
            "The room was spotless and the staff were incredibly welcoming.",
            "Freezing cold room, the heating never worked and nobody came to fix it.",
            "Clean comfy beds. Receptionists clueless about the area. Rooms very small.",
            "Arrived late, checked in, went to sleep. Nothing special.",
        ], t_in)

    gr.Markdown(
        "---\n"
        "**Limitations.** Trained on ~5,900 reviews of 15 Ibis hotels in Morocco; "
        "transfer to other chains or regions is unmeasured. The source text is "
        "**fragmentary** — sentences that matched no topic were dropped upstream, "
        "so a 'review' here averages 31 words. The backbone is a bag of n-grams "
        "and cannot handle negation. **The aggregate estimate is the reliable "
        "output; individual verdicts are not** and should not drive action on a "
        "single guest."
    )

if __name__ == "__main__":
    demo.launch()

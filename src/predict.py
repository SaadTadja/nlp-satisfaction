"""The single prediction path (§3.1).

Imported by the evaluation notebooks and by ``space/app.py``. Nothing else may
reimplement any stage of it. Calibration, the neutral band, the aggregate
correction and every published number are only valid if evaluation-time and
serve-time are the identical function -- and when they diverge, nothing raises
an error, the results just quietly stop describing the deployed system.

    raw text -> normalize -> [backend] -> logits
             -> /T -> softmax -> neutral band -> label

**Backends.** The encoder is swappable; the pipeline around it is not. Two
backends implement the same one-method interface:

* :class:`SklearnBackend`  -- a TF-IDF + LogisticRegression pipeline. Trains in
  under a minute on CPU, so the full system (calibration, neutral band,
  aggregate correction, aspects, demo) can be built and validated end to end
  without a GPU.
* :class:`TransformerBackend` -- CamemBERT or XLM-R. A drop-in upgrade: train
  it, change two fields in ``inference_config.json``, and every downstream
  number is recomputed by re-running notebooks 04 and 06.

Keeping the stages identical across backends is what makes that swap safe. If
the transformer had its own copy of the pipeline, the published numbers would
stop describing the deployed system the moment the two drifted.

torch is imported lazily, inside ``TransformerBackend`` only, so everything
here stays testable on a machine with no deep-learning stack installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .estimate import SatisfactionEstimate, acc_prevalence, estimate_with_ci
from .normalize import NORMALIZE_VERSION, normalize

__all__ = [
    "Prediction",
    "InferenceConfig",
    "load_config",
    "apply_temperature",
    "label_from_prob",
    "SklearnBackend",
    "TransformerBackend",
    "SatisfactionModel",
    "estimate_with_ci",
]

CONFIG_PATH = Path(__file__).resolve().parents[1] / "inference_config.json"

NEGATIVE, NEUTRAL, POSITIVE = "négatif", "neutre", "positif"

# Which notebook is responsible for each config field, so running the pipeline
# out of order gives an actionable message instead of a KeyError.
#
# Split in two because notebook 06 has a genuine bootstrap problem: it must
# load the model and run predictions in order to *measure* tpr/fpr, so it
# cannot be required to supply them first. Prediction never reads those two
# fields -- only the aggregate correction does.
_PREDICTION_FIELDS = {
    "model_repo": "02_tfidf_model.ipynb (or 02_camembert.ipynb)",
    "revision": "02_tfidf_model.ipynb (or 02_camembert.ipynb)",
    "labels": "02_tfidf_model.ipynb",
    "max_length": "00_audit.ipynb",
    "normalize_version": "00_audit.ipynb",
    "temperature": "04_calibration.ipynb",
    "theta_lo": "04_calibration.ipynb",
    "theta_hi": "04_calibration.ipynb",
}
_AGGREGATION_FIELDS = {
    "tpr": "06_aggregate.ipynb",
    "fpr": "06_aggregate.ipynb",
}
_FIELD_OWNERS = {**_PREDICTION_FIELDS, **_AGGREGATION_FIELDS}


@dataclass(frozen=True)
class Prediction:
    """One comment's result."""

    text: str
    label: str           # négatif | neutre | positif
    p_pos: float         # calibrated P(positive)
    confident: bool      # False when the neutral band was applied
    truncated: bool = False


@dataclass(frozen=True)
class InferenceConfig:
    """Parameters that travel with the weights (§3.2).

    ``temperature``, ``theta_lo``, ``theta_hi``, ``tpr`` and ``fpr`` are
    learned on specific splits in specific phases. They are as much a part of
    the model as the weights; if they drift apart, every output is silently
    wrong.
    """

    model_repo: str
    revision: str
    labels: list[str]
    max_length: int
    normalize_version: str
    temperature: float
    theta_lo: float
    theta_hi: float
    tpr: float
    fpr: float
    backend: str = "transformer"     # "sklearn" | "transformer"
    aggregation_ready: bool = True   # False while notebook 06 is measuring

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.theta_lo <= self.theta_hi <= 1.0:
            raise ValueError("require 0 <= theta_lo <= theta_hi <= 1")
        if self.aggregation_ready and not self.tpr > self.fpr:
            raise ValueError("require tpr > fpr for the §10 correction")
        if self.backend not in ("sklearn", "transformer"):
            raise ValueError(f"unknown backend {self.backend!r}")
        if self.normalize_version != NORMALIZE_VERSION:
            raise ValueError(
                f"config was built for normalize {self.normalize_version!r} but "
                f"src/normalize.py is {NORMALIZE_VERSION!r}. Re-run calibration "
                "and the aggregate notebook, or the served text will differ "
                "from the evaluated text."
            )


def load_config(
    path: str | Path | None = None, *, require_aggregation: bool = True
) -> InferenceConfig:
    """Load and validate ``inference_config.json``.

    Raises a message naming the notebook that produces a missing value, which
    is more useful than a KeyError when the pipeline is run out of order.

    Pass ``require_aggregation=False`` to load a config whose ``tpr``/``fpr``
    are not measured yet. Only notebook 06 needs this, and only because it has
    to run predictions in order to measure them. The returned config predicts
    normally but refuses to aggregate, so a half-built config cannot silently
    produce a satisfaction rate.
    """
    path = CONFIG_PATH if path is None else Path(path)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    required = dict(_PREDICTION_FIELDS)
    if require_aggregation:
        required.update(_AGGREGATION_FIELDS)

    missing = [k for k in required if raw.get(k) is None]
    if missing:
        lines = [f"  - {k}: produced by notebooks/{required[k]}" for k in missing]
        raise ValueError(
            "inference_config.json has unfilled fields:\n" + "\n".join(lines)
        )

    has_rates = raw.get("tpr") is not None and raw.get("fpr") is not None
    cfg = InferenceConfig(
        model_repo=raw["model_repo"],
        revision=raw["revision"],
        labels=list(raw["labels"]),
        max_length=int(raw["max_length"]),
        normalize_version=raw["normalize_version"],
        temperature=float(raw["temperature"]),
        theta_lo=float(raw["theta_lo"]),
        theta_hi=float(raw["theta_hi"]),
        tpr=float(raw["tpr"]) if has_rates else float("nan"),
        fpr=float(raw["fpr"]) if has_rates else float("nan"),
        backend=raw.get("backend", "transformer"),
        aggregation_ready=has_rates,
    )
    cfg.validate()
    return cfg


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled softmax over the last axis (§9.1).

    Division by T does not change the argmax, so accuracy and F1 are
    unaffected -- only the probabilities become trustworthy. Both the neutral
    band and the aggregate estimator consume those probabilities, which is why
    calibration is load-bearing here rather than cosmetic.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)      # stabilise before exp
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def label_from_prob(p_pos: float, theta_lo: float, theta_hi: float) -> tuple[str, bool]:
    """Map a calibrated P(positive) to a three-way label (§9.2).

    Returns ``(label, confident)``.
    """
    if theta_lo <= p_pos <= theta_hi:
        return NEUTRAL, False
    return (POSITIVE if p_pos > theta_hi else NEGATIVE), True


# --------------------------------------------------------------------------
# Backends: the only part that differs between TF-IDF and a transformer
# --------------------------------------------------------------------------
class SklearnBackend:
    """TF-IDF + LogisticRegression, loaded from joblib.

    ``decision_function`` returns the log-odds of the positive class, so
    ``[0, d]`` is a valid logit pair: ``softmax([0, d])[1] == sigmoid(d)``.
    Expressing it this way means temperature scaling applies identically here
    and in the transformer backend -- dividing by T is then exactly Platt
    recalibration of the logistic model.
    """

    needs_tokenizer = False

    def __init__(self, model_path: str | Path):
        import joblib  # noqa: PLC0415 - optional at import time

        self.pipeline = joblib.load(model_path)

    def logits(self, texts: list[str], max_length: int) -> np.ndarray:
        d = self.pipeline.decision_function(texts)
        return np.column_stack([np.zeros_like(d), d])

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [0] * len(texts)      # no truncation concept for bag-of-ngrams


class TransformerBackend:
    """CamemBERT / XLM-R via transformers, quantised for CPU serving."""

    needs_tokenizer = True

    def __init__(self, model_repo: str, revision: str, quantize: bool = True,
                 device: str = "cpu"):
        import torch  # noqa: PLC0415 - lazy: keeps the rest importable
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_repo, revision=revision
        ).eval()

        if quantize:
            # Converts nn.Linear only; the ~25M-parameter embedding table stays
            # fp32, so expect roughly 2.4x smaller, not the 4x people assume.
            model = torch.ao.quantization.quantize_dynamic(
                model.cpu(), {torch.nn.Linear}, dtype=torch.qint8
            )
            device = "cpu"
        else:
            model = model.to(device)

        self.model = model
        self.device = device

    def logits(self, texts: list[str], max_length: int) -> np.ndarray:
        torch = self._torch
        enc = self.tokenizer(texts, truncation=True, max_length=max_length,
                             padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            return self.model(**enc).logits.cpu().numpy()

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [len(self.tokenizer(t).input_ids) for t in texts]


def _build_backend(cfg: InferenceConfig, quantize: bool):
    if cfg.backend == "sklearn":
        return SklearnBackend(cfg.model_repo)
    return TransformerBackend(cfg.model_repo, cfg.revision, quantize=quantize)


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
class SatisfactionModel:
    """Loads a backend once and serves single and batch prediction.

    Construct at module scope in the Space, never per request (§3.4).
    """

    def __init__(self, config: InferenceConfig | None = None,
                 quantize: bool = True, backend=None):
        self.cfg = config or load_config()
        self.backend = backend or _build_backend(self.cfg, quantize)
        self._pos_index = self.cfg.labels.index("positif")

    def predict(self, texts, batch_size: int = 32) -> list[Prediction]:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        out: list[Prediction] = []
        for start in range(0, len(texts), batch_size):
            out.extend(self._predict_batch(texts[start : start + batch_size]))
        return out

    def _predict_batch(self, chunk: list[str]) -> list[Prediction]:
        cleaned = [normalize(t) for t in chunk]
        live_idx = [i for i, t in enumerate(cleaned) if t]

        results: list[Prediction | None] = [None] * len(chunk)
        for i, t in enumerate(chunk):
            if i not in live_idx:          # §3.5: empty never reaches the model
                results[i] = Prediction(t, NEUTRAL, 0.5, False)
        if not live_idx:
            return [r for r in results if r is not None]

        live = [cleaned[i] for i in live_idx]
        logits = self.backend.logits(live, self.cfg.max_length)
        probs = apply_temperature(logits, self.cfg.temperature)
        lengths = self.backend.token_lengths(live)

        for i, p, n_tok in zip(live_idx, probs, lengths):
            p_pos = float(p[self._pos_index])
            label, confident = label_from_prob(
                p_pos, self.cfg.theta_lo, self.cfg.theta_hi
            )
            results[i] = Prediction(
                text=chunk[i], label=label, p_pos=p_pos, confident=confident,
                truncated=n_tok > self.cfg.max_length,
            )
        return [r for r in results if r is not None]

    def prob_positive(self, texts, batch_size: int = 32) -> np.ndarray:
        """Calibrated P(positive) only -- convenient for sweeps and plots."""
        return np.array([p.p_pos for p in self.predict(texts, batch_size)])

    def estimate_satisfaction(self, texts, batch_size: int = 32,
                              n_boot: int = 2000) -> SatisfactionEstimate:
        """Bias-corrected satisfaction rate for a body of feedback (§10).

        Neutral predictions are excluded from the base: the question is what
        share of *opinionated* feedback is positive. Folding abstentions into
        the denominator would silently drag every estimate toward zero. The
        caller reports the neutral count separately.
        """
        if not self.cfg.aggregation_ready:
            raise ValueError(
                "tpr/fpr are not measured yet, so the correction cannot be "
                "applied. Run notebooks/06_aggregate.ipynb first."
            )
        preds = self.predict(texts, batch_size=batch_size)
        decided = [p for p in preds if p.confident]
        if not decided:
            raise ValueError("no confident predictions; cannot estimate a rate")

        flags = np.array([p.label == POSITIVE for p in decided], dtype=float)
        point = acc_prevalence(float(flags.mean()), self.cfg.tpr, self.cfg.fpr)

        # Interval from the comment sample alone. For the full interval that
        # also propagates TPR/FPR uncertainty, call estimate_with_ci directly
        # with the labelled holdout -- notebook 06 does this.
        rng = np.random.default_rng(0)
        draws = [
            acc_prevalence(flags[rng.integers(0, len(flags), len(flags))].mean(),
                           self.cfg.tpr, self.cfg.fpr)
            for _ in range(n_boot)
        ]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return SatisfactionEstimate(point, float(lo), float(hi),
                                    float(flags.mean()), len(decided))

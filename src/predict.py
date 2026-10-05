"""The single prediction path (§3.1).

Imported by the evaluation notebooks and by ``space/app.py``. Nothing else may
reimplement any stage of it. Calibration, the neutral band, the aggregate
correction and every published number are only valid if evaluation-time and
serve-time are the identical function -- and when they diverge, nothing raises
an error, the results just quietly stop describing the deployed system.

    raw text -> normalize -> [backend] -> logits -> /T -> softmax
             -> label (+ neutral band, binary mode only)

**Two task modes**, because the two datasets differ in what labels exist:

* ``binary``  -- two classes plus a calibrated abstention band standing in for
  neutral. Used by the French film-review track, where the corpus was polarised
  by construction and no neutral label existed to train on.
* ``multiclass`` -- three real classes from star ratings. Used by the English
  hotel track. No band: neutral is a trained output, and notebook 12 measured
  that training it beats inferring it from a band.

**Two backends**, because the encoder is swappable and the pipeline is not:
``SklearnBackend`` and ``TransformerBackend`` expose the same one method.

torch is imported lazily, inside ``TransformerBackend`` only, so everything
here stays testable on a machine with no deep-learning stack installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .estimate import (
    SatisfactionEstimate,
    acc_prevalence,
    acc_prevalence_multiclass,
    estimate_with_ci,
    predicted_distribution,
)
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

# Which notebook produces each field, so running out of order gives an
# actionable message instead of a KeyError. Split by mode, and split again by
# whether the field is needed to *predict* or only to *aggregate* -- notebook
# 06/13 has to run predictions in order to measure the aggregation fields, so
# it cannot be required to supply them first.
_COMMON = {
    "model_repo": "02_tfidf_model / 12_en_calibration",
    "revision": "02_tfidf_model / 12_en_calibration",
    "labels": "02_tfidf_model / 12_en_calibration",
    "normalize_version": "00_audit / 10_en_audit",
    "temperature": "04_calibration / 12_en_calibration",
}
_BINARY_PREDICT = {"max_length": "00_audit",
                   "theta_lo": "04_calibration", "theta_hi": "04_calibration"}
_BINARY_AGG = {"tpr": "06_aggregate", "fpr": "06_aggregate"}
_MULTI_PREDICT = {"class_order": "12_en_calibration"}
_MULTI_AGG = {"confusion_matrix": "13_en_aggregate"}


@dataclass(frozen=True)
class Prediction:
    """One comment's result."""

    text: str
    label: str
    confidence: float                 # calibrated P(assigned label)
    confident: bool                   # False when the band / threshold applied
    probs: dict = field(default_factory=dict)
    truncated: bool = False

    @property
    def p_pos(self) -> float:
        """Calibrated probability of the positive class, whatever it is named."""
        for k, v in self.probs.items():
            if k in (POSITIVE, "positive"):
                return float(v)
        return float("nan")


@dataclass(frozen=True)
class InferenceConfig:
    """Parameters that travel with the weights (§3.2).

    Temperature, the band thresholds and the error rates are learned on
    specific splits in specific phases. They are as much a part of the model as
    the coefficients; if they drift apart, every output is silently wrong.
    """

    model_repo: str
    revision: str
    labels: list[str]
    normalize_version: str
    temperature: float
    mode: str                          # "binary" | "multiclass"
    backend: str = "sklearn"
    max_length: int = 0
    class_order: list[str] | None = None
    theta_lo: float = 0.0
    theta_hi: float = 0.0
    tpr: float = float("nan")
    fpr: float = float("nan")
    confusion_matrix: np.ndarray | None = None
    aggregation_ready: bool = True

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.backend not in ("sklearn", "transformer"):
            raise ValueError(f"unknown backend {self.backend!r}")
        if self.mode == "binary":
            if not 0.0 <= self.theta_lo <= self.theta_hi <= 1.0:
                raise ValueError("require 0 <= theta_lo <= theta_hi <= 1")
            if self.aggregation_ready and not self.tpr > self.fpr:
                raise ValueError("require tpr > fpr for the correction")
        elif self.mode == "multiclass":
            if len(self.labels) < 3:
                raise ValueError("multiclass mode needs at least 3 labels")
            if self.aggregation_ready:
                M = np.asarray(self.confusion_matrix, dtype=float)
                if M.shape != (len(self.labels),) * 2:
                    raise ValueError("confusion_matrix must be square over labels")
                if not np.allclose(M.sum(axis=0), 1.0, atol=1e-6):
                    raise ValueError(
                        "confusion_matrix columns must sum to 1 "
                        "(M[i, j] = P(predict i | true j))"
                    )
        else:
            raise ValueError(f"unknown mode {self.mode!r}")
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
    """Load and validate an inference config, in either task mode.

    Mode is inferred from the number of labels rather than declared, so a
    config cannot claim one shape and carry another.

    ``require_aggregation=False`` loads a config whose correction parameters
    are not measured yet -- needed only by the aggregate notebook, which must
    run predictions in order to measure them. The result predicts normally but
    refuses to aggregate.
    """
    path = CONFIG_PATH if path is None else Path(path)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    labels = list(raw.get("labels") or [])
    mode = "multiclass" if len(labels) >= 3 else "binary"

    required = dict(_COMMON)
    required.update(_MULTI_PREDICT if mode == "multiclass" else _BINARY_PREDICT)
    agg_fields = _MULTI_AGG if mode == "multiclass" else _BINARY_AGG
    if require_aggregation:
        required.update(agg_fields)

    missing = [k for k in required if raw.get(k) is None]
    if missing:
        lines = [f"  - {k}: produced by notebooks/{required[k]}" for k in missing]
        raise ValueError(
            f"{path.name} has unfilled fields ({mode} mode):\n" + "\n".join(lines)
        )

    has_agg = all(raw.get(k) is not None for k in agg_fields)
    M = raw.get("confusion_matrix")
    cfg = InferenceConfig(
        model_repo=raw["model_repo"],
        revision=raw["revision"],
        labels=labels,
        normalize_version=raw["normalize_version"],
        temperature=float(raw["temperature"]),
        mode=mode,
        backend=raw.get("backend", "sklearn"),
        max_length=int(raw.get("max_length") or 0),
        class_order=raw.get("class_order"),
        theta_lo=float(raw.get("theta_lo") or 0.0),
        theta_hi=float(raw.get("theta_hi") or 0.0),
        tpr=float(raw["tpr"]) if raw.get("tpr") is not None else float("nan"),
        fpr=float(raw["fpr"]) if raw.get("fpr") is not None else float("nan"),
        confusion_matrix=np.asarray(M, dtype=float) if M is not None else None,
        aggregation_ready=has_agg,
    )
    cfg.validate()
    return cfg


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled softmax over the last axis (§9.1).

    Division by T cannot change the argmax, so accuracy and F1 are unaffected
    -- only the probabilities become trustworthy. Both the band and the
    aggregate correction consume those probabilities, which is why calibration
    is load-bearing here rather than cosmetic.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def label_from_prob(p_pos: float, theta_lo: float, theta_hi: float) -> tuple[str, bool]:
    """Binary mode only: map a calibrated P(positive) to a three-way label."""
    if theta_lo <= p_pos <= theta_hi:
        return NEUTRAL, False
    return (POSITIVE if p_pos > theta_hi else NEGATIVE), True


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class SklearnBackend:
    """TF-IDF + LogisticRegression, loaded from joblib.

    For a binary model ``decision_function`` returns log-odds, so ``[0, d]`` is
    a valid logit pair (``softmax([0, d])[1] == sigmoid(d)``) and temperature
    scaling is then exactly Platt recalibration. For a multiclass model it
    already returns one column per class, reordered here to the canonical
    label order so the rest of the pipeline never has to care.
    """

    needs_tokenizer = False

    def __init__(self, model_path: str | Path, class_order=None, labels=None):
        import joblib  # noqa: PLC0415

        self.pipeline = joblib.load(model_path)
        self.order = None
        if class_order and labels:
            self.order = [list(class_order).index(l) for l in labels]

    def logits(self, texts: list[str], max_length: int) -> np.ndarray:
        d = self.pipeline.decision_function(texts)
        d = np.asarray(d)
        if d.ndim == 1:
            return np.column_stack([np.zeros_like(d), d])
        return d[:, self.order] if self.order else d

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [0] * len(texts)


class TransformerBackend:
    """CamemBERT / RoBERTa via transformers, quantised for CPU serving."""

    needs_tokenizer = True

    def __init__(self, model_repo: str, revision: str, quantize: bool = True,
                 device: str = "cpu"):
        import torch  # noqa: PLC0415
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
            model = torch.ao.quantization.quantize_dynamic(
                model.cpu(), {torch.nn.Linear}, dtype=torch.qint8
            )
            device = "cpu"
        else:
            model = model.to(device)
        self.model, self.device = model, device

    def logits(self, texts: list[str], max_length: int) -> np.ndarray:
        torch = self._torch
        enc = self.tokenizer(texts, truncation=True, max_length=max_length or 512,
                             padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            return self.model(**enc).logits.cpu().numpy()

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [len(self.tokenizer(t).input_ids) for t in texts]


def _build_backend(cfg: InferenceConfig, quantize: bool):
    if cfg.backend == "sklearn":
        return SklearnBackend(cfg.model_repo, cfg.class_order, cfg.labels)
    return TransformerBackend(cfg.model_repo, cfg.revision, quantize=quantize)


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
class SatisfactionModel:
    """Loads a backend once and serves single and batch prediction.

    Construct at module scope in the Space, never per request (§3.4).
    """

    def __init__(self, config: InferenceConfig | None = None,
                 quantize: bool = True, backend=None, min_confidence: float = 0.0):
        self.cfg = config or load_config()
        self.backend = backend or _build_backend(self.cfg, quantize)
        self.min_confidence = min_confidence
        self._pos_index = next(
            (i for i, l in enumerate(self.cfg.labels) if l in (POSITIVE, "positive")),
            len(self.cfg.labels) - 1,
        )

    # -- prediction --------------------------------------------------------
    def predict(self, texts, batch_size: int = 64) -> list[Prediction]:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        out: list[Prediction] = []
        for s in range(0, len(texts), batch_size):
            out.extend(self._predict_batch(texts[s : s + batch_size]))
        return out

    def _neutral_name(self) -> str:
        for l in self.cfg.labels:
            if l in (NEUTRAL, "neutral"):
                return l
        return self.cfg.labels[len(self.cfg.labels) // 2]

    def _predict_batch(self, chunk: list[str]) -> list[Prediction]:
        cleaned = [normalize(t) for t in chunk]
        live = [i for i, t in enumerate(cleaned) if t]

        results: list[Prediction | None] = [None] * len(chunk)
        for i, t in enumerate(chunk):
            if i not in live:          # §3.5: empty never reaches the model
                results[i] = Prediction(t, self._neutral_name(), 0.0, False, {})
        if not live:
            return [r for r in results if r is not None]

        texts = [cleaned[i] for i in live]
        probs = apply_temperature(
            self.backend.logits(texts, self.cfg.max_length), self.cfg.temperature
        )
        lengths = self.backend.token_lengths(texts)

        for i, p, n_tok in zip(live, probs, lengths):
            pd_ = {l: float(v) for l, v in zip(self.cfg.labels, p)}
            if self.cfg.mode == "binary":
                label, confident = label_from_prob(
                    float(p[self._pos_index]), self.cfg.theta_lo, self.cfg.theta_hi
                )
                conf = float(max(p))
            else:
                k = int(np.argmax(p))
                label, conf = self.cfg.labels[k], float(p[k])
                confident = conf >= self.min_confidence
            results[i] = Prediction(
                text=chunk[i], label=label, confidence=conf, confident=confident,
                probs=pd_, truncated=n_tok > self.cfg.max_length > 0,
            )
        return [r for r in results if r is not None]

    # -- aggregation -------------------------------------------------------
    def estimate_satisfaction(self, texts, batch_size: int = 64,
                              n_boot: int = 1500) -> SatisfactionEstimate:
        """Bias-corrected share of satisfied customers (§10).

        Binary mode excludes abstentions from the base: the question is what
        share of *opinionated* feedback is positive, and folding neutrals into
        the denominator would drag every estimate toward zero.

        Multiclass mode keeps the whole population, because neutral is a real
        class with its own column in the confusion matrix. The returned rate is
        the positive share; use :meth:`estimate_distribution` for all three.
        """
        if not self.cfg.aggregation_ready:
            raise ValueError(
                "correction parameters are not measured yet. Run "
                "notebooks/06_aggregate (binary) or 13_en_aggregate (multiclass)."
            )
        preds = self.predict(texts, batch_size=batch_size)

        if self.cfg.mode == "multiclass":
            dist, lo, hi = self._estimate_distribution(preds, n_boot)
            k = self._pos_index
            naive = predicted_distribution(
                np.array([p.label for p in preds]), self.cfg.labels
            )[k]
            return SatisfactionEstimate(float(dist[k]), float(lo[k]), float(hi[k]),
                                        float(naive), len(preds))

        decided = [p for p in preds if p.confident]
        if not decided:
            raise ValueError("no confident predictions; cannot estimate a rate")
        flags = np.array([p.label in (POSITIVE, "positive") for p in decided], float)
        point = acc_prevalence(float(flags.mean()), self.cfg.tpr, self.cfg.fpr)
        rng = np.random.default_rng(0)
        draws = [acc_prevalence(flags[rng.integers(0, len(flags), len(flags))].mean(),
                                self.cfg.tpr, self.cfg.fpr) for _ in range(n_boot)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return SatisfactionEstimate(point, float(lo), float(hi),
                                    float(flags.mean()), len(decided))

    def estimate_distribution(self, texts, batch_size: int = 64, n_boot: int = 1500):
        """Multiclass only: corrected share of every class, with intervals.

        Returns ``(labels, corrected, ci_low, ci_high, naive)``.
        """
        if self.cfg.mode != "multiclass":
            raise ValueError("estimate_distribution is multiclass-only")
        preds = self.predict(texts, batch_size=batch_size)
        dist, lo, hi = self._estimate_distribution(preds, n_boot)
        naive = predicted_distribution(
            np.array([p.label for p in preds]), self.cfg.labels
        )
        return self.cfg.labels, dist, lo, hi, naive

    def _estimate_distribution(self, preds, n_boot: int):
        M = self.cfg.confusion_matrix
        y = np.array([p.label for p in preds])
        point = acc_prevalence_multiclass(predicted_distribution(y, self.cfg.labels), M)

        # Resample the predictions only. Uncertainty in M itself needs the
        # labelled holdout and is computed in notebook 13; this interval is
        # therefore the narrower of the two and is the one the demo shows.
        rng = np.random.default_rng(0)
        draws = np.array([
            acc_prevalence_multiclass(
                predicted_distribution(y[rng.integers(0, len(y), len(y))],
                                       self.cfg.labels), M)
            for _ in range(n_boot)
        ])
        return point, np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)

"""The single prediction path (§3.1).

Imported by the evaluation notebooks and by ``space/app.py``. Nothing else may
reimplement any stage of it. Calibration, the neutral band, the aggregate
correction and every published number are only valid if evaluation-time and
serve-time are the identical function -- and when they diverge, nothing raises
an error, the results just quietly stop describing the deployed system.

    raw text -> normalize -> tokenize -> encoder -> logits
             -> /T -> softmax -> neutral band -> label

torch and transformers are imported lazily inside :class:`SatisfactionModel`,
so the pure decision logic below (and the whole of ``estimate.py``) stays
testable on a machine with no deep-learning stack installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .estimate import SatisfactionEstimate, acc_prevalence, estimate_with_ci
from .normalize import NORMALIZE_VERSION, is_empty, normalize

__all__ = [
    "Prediction",
    "InferenceConfig",
    "load_config",
    "apply_temperature",
    "label_from_prob",
    "SatisfactionModel",
]

CONFIG_PATH = Path(__file__).resolve().parents[1] / "inference_config.json"

NEGATIVE, NEUTRAL, POSITIVE = "négatif", "neutre", "positif"

# Which notebook is responsible for measuring each config field. Used to give
# an actionable error instead of a KeyError when the pipeline is run before
# its inputs exist.
_FIELD_OWNERS = {
    "model_repo": "02_camembert.ipynb",
    "revision": "02_camembert.ipynb",
    "labels": "02_camembert.ipynb",
    "max_length": "00_audit.ipynb",
    "normalize_version": "00_audit.ipynb",
    "temperature": "04_calibration.ipynb",
    "theta_lo": "04_calibration.ipynb",
    "theta_hi": "04_calibration.ipynb",
    "tpr": "06_aggregate.ipynb",
    "fpr": "06_aggregate.ipynb",
}


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

    def validate(self) -> None:
        if not 0.0 < self.temperature:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.theta_lo <= self.theta_hi <= 1.0:
            raise ValueError("require 0 <= theta_lo <= theta_hi <= 1")
        if not self.tpr > self.fpr:
            raise ValueError("require tpr > fpr for the §10 correction")
        if self.normalize_version != NORMALIZE_VERSION:
            raise ValueError(
                f"config was built for normalize {self.normalize_version!r} but "
                f"src/normalize.py is {NORMALIZE_VERSION!r}. Re-run calibration "
                "and the aggregate notebook, or the served text will differ "
                "from the evaluated text."
            )


def load_config(path: str | Path | None = None) -> InferenceConfig:
    """Load and validate ``inference_config.json``.

    Raises a message naming the notebook that produces a missing value, which
    is more useful than a KeyError when the pipeline is run out of order.
    """
    path = CONFIG_PATH if path is None else Path(path)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    missing = [k for k, v in _FIELD_OWNERS.items() if raw.get(k) is None]
    if missing:
        lines = [f"  - {k}: produced by notebooks/{_FIELD_OWNERS[k]}" for k in missing]
        raise ValueError(
            "inference_config.json has unfilled fields:\n" + "\n".join(lines)
        )

    cfg = InferenceConfig(
        model_repo=raw["model_repo"],
        revision=raw["revision"],
        labels=list(raw["labels"]),
        max_length=int(raw["max_length"]),
        normalize_version=raw["normalize_version"],
        temperature=float(raw["temperature"]),
        theta_lo=float(raw["theta_lo"]),
        theta_hi=float(raw["theta_hi"]),
        tpr=float(raw["tpr"]),
        fpr=float(raw["fpr"]),
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

    The training corpus is polarised by construction -- reviews built from
    star ratings usually drop the middle -- so neutral text is out of
    distribution and a binary model will assign it confidently to a side.
    Abstaining inside the band is more honest than pretending.

    Returns ``(label, confident)``.
    """
    if theta_lo <= p_pos <= theta_hi:
        return NEUTRAL, False
    return (POSITIVE if p_pos > theta_hi else NEGATIVE), True


class SatisfactionModel:
    """Loads the model once and serves both single and batch prediction.

    Construct at module scope in the Space, never per request (§3.4).
    """

    def __init__(
        self,
        config: InferenceConfig | None = None,
        quantize: bool = True,
        device: str = "cpu",
    ):
        import torch  # noqa: PLC0415 -- lazy: keeps pure logic importable
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.cfg = config or load_config()
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_repo, revision=self.cfg.revision
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_repo, revision=self.cfg.revision
        ).eval()

        if quantize:
            # Dynamic quantisation converts nn.Linear only; the ~25M-parameter
            # embedding table stays fp32, so expect roughly 2.4x smaller, not
            # the 4x people assume.
            model = torch.ao.quantization.quantize_dynamic(
                model.cpu(), {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            model = model.to(device)

        self.model = model
        self.device = "cpu" if quantize else device
        self._pos_index = self.cfg.labels.index("positif")

    def predict(self, texts, batch_size: int = 32) -> list[Prediction]:
        """Run the full pipeline over a list of raw texts."""
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        out: list[Prediction] = []

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            out.extend(self._predict_batch(chunk))
        return out

    def _predict_batch(self, chunk: list[str]) -> list[Prediction]:
        torch = self._torch
        cleaned = [normalize(t) for t in chunk]

        # Guard: empty input never reaches the model (§3.5).
        live_idx = [i for i, t in enumerate(cleaned) if t]
        results: list[Prediction | None] = [None] * len(chunk)
        for i, t in enumerate(chunk):
            if i not in live_idx:
                results[i] = Prediction(t, NEUTRAL, 0.5, False)

        if not live_idx:
            return [r for r in results if r is not None]

        live = [cleaned[i] for i in live_idx]
        enc = self.tokenizer(
            live,
            truncation=True,
            max_length=self.cfg.max_length,
            padding=True,
            return_tensors="pt",
        )
        # Flag truncation so the UI can say so rather than silently dropping
        # text -- review sentiment often lands in the final sentence.
        lengths = [len(self.tokenizer(t).input_ids) for t in live]

        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits.cpu().numpy()

        probs = apply_temperature(logits, self.cfg.temperature)
        for slot, (i, p, n_tok) in enumerate(zip(live_idx, probs, lengths)):
            p_pos = float(p[self._pos_index])
            label, confident = label_from_prob(
                p_pos, self.cfg.theta_lo, self.cfg.theta_hi
            )
            results[i] = Prediction(
                text=chunk[i],
                label=label,
                p_pos=p_pos,
                confident=confident,
                truncated=n_tok > self.cfg.max_length,
            )
        return [r for r in results if r is not None]

    def estimate_satisfaction(
        self, texts, batch_size: int = 32, n_boot: int = 2000
    ) -> SatisfactionEstimate:
        """Bias-corrected satisfaction rate for a body of feedback (§10).

        Neutral predictions are excluded from the base: the question is what
        share of *opinionated* feedback is positive. The neutral count is
        reported separately by the caller -- folding abstentions into the
        denominator would silently drag every estimate toward zero.
        """
        preds = self.predict(texts, batch_size=batch_size)
        decided = [p for p in preds if p.confident]
        if not decided:
            raise ValueError("no confident predictions; cannot estimate a rate")

        pos_rate = float(np.mean([p.label == POSITIVE for p in decided]))
        point = acc_prevalence(pos_rate, self.cfg.tpr, self.cfg.fpr)

        # Interval from the comment sample alone. For the full interval that
        # also propagates TPR/FPR uncertainty, call estimate_with_ci directly
        # with the labelled holdout (notebook 06 does this).
        rng = np.random.default_rng(0)
        flags = np.array([p.label == POSITIVE for p in decided], dtype=float)
        draws = [
            acc_prevalence(
                flags[rng.integers(0, len(flags), len(flags))].mean(),
                self.cfg.tpr,
                self.cfg.fpr,
            )
            for _ in range(n_boot)
        ]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return SatisfactionEstimate(point, float(lo), float(hi), pos_rate, len(decided))


# Re-exported so notebooks can reach the full-uncertainty version without a
# second import (§10.4).
__all__.append("estimate_with_ci")

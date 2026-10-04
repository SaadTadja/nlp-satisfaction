"""Tests for the decision logic in the prediction path (§3.1, §9).

Only the torch-free parts are covered here: temperature scaling, the neutral
band, and config validation. ``SatisfactionModel`` needs a trained checkpoint
and is exercised in notebook 04 instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.normalize import NORMALIZE_VERSION  # noqa: E402
from src.predict import (  # noqa: E402
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    apply_temperature,
    label_from_prob,
    load_config,
)

GOOD_CONFIG = {
    "model_repo": "you/camembert-satisfaction-fr",
    "revision": "a1b2c3d4",
    "labels": ["négatif", "positif"],
    "max_length": 192,
    "normalize_version": NORMALIZE_VERSION,
    "temperature": 1.74,
    "theta_lo": 0.35,
    "theta_hi": 0.65,
    "tpr": 0.942,
    "fpr": 0.061,
}


def write_config(tmp_path: Path, **overrides) -> Path:
    cfg = {**GOOD_CONFIG, **overrides}
    p = tmp_path / "inference_config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# Temperature scaling (§9.1)
# --------------------------------------------------------------------------


def test_softmax_rows_sum_to_one():
    probs = apply_temperature(np.array([[2.0, -1.0], [0.5, 0.4]]), 1.3)
    assert np.allclose(probs.sum(axis=-1), 1.0)


def test_temperature_does_not_change_argmax():
    """The central property: calibration fixes probabilities, not decisions,
    so accuracy and F1 are unchanged."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 2))
    base = apply_temperature(logits, 1.0).argmax(-1)
    for t in (0.5, 1.7, 4.0):
        assert np.array_equal(apply_temperature(logits, t).argmax(-1), base)


def test_higher_temperature_softens_confidence():
    logits = np.array([[4.0, -4.0]])
    assert apply_temperature(logits, 3.0).max() < apply_temperature(logits, 1.0).max()


def test_is_numerically_stable_on_large_logits():
    probs = apply_temperature(np.array([[900.0, -900.0]]), 1.0)
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0)


def test_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        apply_temperature(np.array([[1.0, 0.0]]), 0.0)


# --------------------------------------------------------------------------
# Neutral band (§9.2)
# --------------------------------------------------------------------------


def test_band_assigns_three_way_labels():
    assert label_from_prob(0.95, 0.35, 0.65) == (POSITIVE, True)
    assert label_from_prob(0.05, 0.35, 0.65) == (NEGATIVE, True)
    assert label_from_prob(0.50, 0.35, 0.65) == (NEUTRAL, False)


def test_band_edges_are_inclusive():
    """Boundary values fall inside the band -- abstain rather than guess."""
    assert label_from_prob(0.35, 0.35, 0.65)[0] == NEUTRAL
    assert label_from_prob(0.65, 0.35, 0.65)[0] == NEUTRAL


def test_degenerate_band_still_works():
    """A zero-width band reduces cleanly to binary behaviour."""
    assert label_from_prob(0.7, 0.5, 0.5)[0] == POSITIVE
    assert label_from_prob(0.3, 0.5, 0.5)[0] == NEGATIVE


# --------------------------------------------------------------------------
# Config validation (§3.2)
# --------------------------------------------------------------------------


def test_loads_a_complete_config(tmp_path):
    cfg = load_config(write_config(tmp_path))
    assert cfg.temperature == 1.74
    assert cfg.labels == ["négatif", "positif"]


def test_unfilled_field_names_the_responsible_notebook(tmp_path):
    """Running the pipeline out of order must give an actionable message."""
    with pytest.raises(ValueError, match="04_calibration"):
        load_config(write_config(tmp_path, temperature=None))


def test_lists_every_unfilled_field(tmp_path):
    with pytest.raises(ValueError) as exc:
        load_config(write_config(tmp_path, tpr=None, fpr=None))
    assert "tpr" in str(exc.value) and "fpr" in str(exc.value)


def test_rejects_inverted_band(tmp_path):
    with pytest.raises(ValueError, match="theta_lo <= theta_hi"):
        load_config(write_config(tmp_path, theta_lo=0.8, theta_hi=0.2))


def test_rejects_useless_classifier_rates(tmp_path):
    with pytest.raises(ValueError, match="tpr > fpr"):
        load_config(write_config(tmp_path, tpr=0.4, fpr=0.6))


def test_rejects_stale_normalize_version(tmp_path):
    """The drift alarm: a config built against different normalisation must
    not silently pair with current code."""
    with pytest.raises(ValueError, match="normalize"):
        load_config(write_config(tmp_path, normalize_version="v0-stale"))


def test_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="unknown backend"):
        load_config(write_config(tmp_path, backend="magic"))


# --------------------------------------------------------------------------
# The notebook-06 bootstrap: predict before tpr/fpr exist
# --------------------------------------------------------------------------


def test_partial_config_loads_without_rates(tmp_path):
    """Notebook 06 must run predictions in order to MEASURE tpr/fpr, so it
    cannot be required to supply them first."""
    cfg = load_config(
        write_config(tmp_path, tpr=None, fpr=None), require_aggregation=False
    )
    assert cfg.aggregation_ready is False
    assert cfg.temperature == 1.74          # prediction fields still enforced


def test_partial_config_still_requires_prediction_fields(tmp_path):
    with pytest.raises(ValueError, match="04_calibration"):
        load_config(
            write_config(tmp_path, temperature=None, tpr=None, fpr=None),
            require_aggregation=False,
        )


def test_partial_config_refuses_to_aggregate(tmp_path):
    """A half-built config must not silently produce a satisfaction rate."""
    from src.predict import SatisfactionModel

    cfg = load_config(
        write_config(tmp_path, tpr=None, fpr=None), require_aggregation=False
    )

    class StubBackend:
        def logits(self, texts, max_length):
            return np.zeros((len(texts), 2))

        def token_lengths(self, texts):
            return [0] * len(texts)

    model = SatisfactionModel(cfg, backend=StubBackend())
    model.predict(["ça marche"])            # prediction is fine
    with pytest.raises(ValueError, match="06_aggregate"):
        model.estimate_satisfaction(["ça marche"])


def test_full_config_is_aggregation_ready(tmp_path):
    assert load_config(write_config(tmp_path)).aggregation_ready is True

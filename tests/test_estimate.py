"""Tests for the aggregate estimator (§10).

The headline test is ``test_acc_beats_cc_under_prevalence_shift``: it simulates
a classifier with known TPR/FPR, sweeps the true satisfaction rate, and asserts
that the correction substantially outperforms naive counting. That is the
project's central claim, verified here with synthetic data so it is known to
hold before any GPU time is spent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.estimate import (  # noqa: E402
    acc_prevalence,
    classify_and_count,
    confusion_rates,
    estimate_with_ci,
    resample_to_prevalence,
)

TPR, FPR = 0.94, 0.06


def simulate(true_prevalence: float, n: int, rng, tpr=TPR, fpr=FPR):
    """A classifier with exactly the given TPR/FPR, applied to a population
    with the given true prevalence. Returns (y_true, y_pred)."""
    y_true = (rng.random(n) < true_prevalence).astype(int)
    flip = rng.random(n)
    y_pred = np.where(y_true == 1, (flip < tpr).astype(int), (flip < fpr).astype(int))
    return y_true, y_pred


# --------------------------------------------------------------------------
# The central claim
# --------------------------------------------------------------------------


def test_acc_beats_cc_under_prevalence_shift():
    """ACC tracks truth across the prevalence range; CC bows toward the middle."""
    rng = np.random.default_rng(0)
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    cc_errors, acc_errors = [], []
    for true_p in levels:
        y_true, y_pred = simulate(true_p, 40_000, rng)
        cc = classify_and_count(y_pred)
        tpr, fpr = confusion_rates(y_true, y_pred)
        acc = acc_prevalence(cc, tpr, fpr)
        cc_errors.append(abs(cc - true_p))
        acc_errors.append(abs(acc - true_p))

    # ACC should be near-exact; CC is off by several points on average.
    assert np.mean(acc_errors) < 0.005, f"ACC MAE too high: {np.mean(acc_errors):.4f}"
    assert np.mean(cc_errors) > 0.02, "CC should be visibly biased with these rates"
    assert np.mean(acc_errors) < np.mean(cc_errors) / 5


def test_cc_bias_is_worst_at_extreme_prevalence():
    """The bias grows as the true rate moves away from the middle -- which is
    exactly when a satisfaction answer matters most."""
    rng = np.random.default_rng(1)
    _, pred_mid = simulate(0.5, 60_000, rng)
    _, pred_low = simulate(0.1, 60_000, rng)

    bias_mid = abs(classify_and_count(pred_mid) - 0.5)
    bias_low = abs(classify_and_count(pred_low) - 0.1)
    assert bias_low > bias_mid


# --------------------------------------------------------------------------
# acc_prevalence
# --------------------------------------------------------------------------


def test_perfect_classifier_is_identity():
    assert acc_prevalence(0.42, tpr=1.0, fpr=0.0) == pytest.approx(0.42)


def test_result_is_clipped_to_unit_interval():
    assert acc_prevalence(0.01, tpr=0.9, fpr=0.1) == 0.0
    assert acc_prevalence(0.99, tpr=0.9, fpr=0.1) == 1.0


def test_useless_classifier_is_rejected():
    with pytest.raises(ValueError, match="no usable signal"):
        acc_prevalence(0.5, tpr=0.5, fpr=0.5)
    with pytest.raises(ValueError):
        acc_prevalence(0.5, tpr=0.3, fpr=0.7)


def test_inverts_the_forward_relation():
    """acc_prevalence must be the exact inverse of p_hat = TPR*p + FPR*(1-p)."""
    for p in [0.0, 0.17, 0.5, 0.83, 1.0]:
        p_hat = TPR * p + FPR * (1 - p)
        assert acc_prevalence(p_hat, TPR, FPR) == pytest.approx(p, abs=1e-9)


# --------------------------------------------------------------------------
# confusion_rates
# --------------------------------------------------------------------------


def test_confusion_rates_recovers_known_rates():
    rng = np.random.default_rng(2)
    y_true, y_pred = simulate(0.4, 200_000, rng)
    tpr, fpr = confusion_rates(y_true, y_pred)
    assert tpr == pytest.approx(TPR, abs=0.01)
    assert fpr == pytest.approx(FPR, abs=0.01)


def test_confusion_rates_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        confusion_rates([1, 1, 1], [1, 0, 1])


def test_confusion_rates_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        confusion_rates([1, 0, 1], [1, 0])


# --------------------------------------------------------------------------
# estimate_with_ci
# --------------------------------------------------------------------------


def test_ci_contains_truth_and_point_is_accurate():
    rng = np.random.default_rng(3)
    y_true, y_pred = simulate(0.35, 3_000, rng)
    hold_true, hold_pred = simulate(0.5, 4_000, rng)

    est = estimate_with_ci(y_pred, hold_true, hold_pred, n_boot=800, seed=0)
    assert est.rate == pytest.approx(0.35, abs=0.03)
    assert est.ci_low < 0.35 < est.ci_high
    # The correction should have moved the answer away from the naive count.
    assert abs(est.rate - 0.35) < abs(est.naive_rate - 0.35)
    assert est.n == 3_000


def test_ci_narrows_with_more_data():
    rng = np.random.default_rng(4)
    hold_true, hold_pred = simulate(0.5, 20_000, rng)

    widths = []
    for n in (500, 10_000):
        _, y_pred = simulate(0.4, n, rng)
        est = estimate_with_ci(y_pred, hold_true, hold_pred, n_boot=600, seed=1)
        widths.append(est.ci_high - est.ci_low)
    assert widths[1] < widths[0] / 2


def test_small_holdout_widens_the_interval():
    """Uncertainty in TPR/FPR must propagate -- a tiny holdout should visibly
    widen the interval even when the batch itself is large."""
    rng = np.random.default_rng(5)
    _, y_pred = simulate(0.4, 10_000, rng)

    big_true, big_pred = simulate(0.5, 20_000, rng)
    small_true, small_pred = simulate(0.5, 150, rng)

    wide = estimate_with_ci(y_pred, small_true, small_pred, n_boot=800, seed=2)
    tight = estimate_with_ci(y_pred, big_true, big_pred, n_boot=800, seed=2)
    assert (wide.ci_high - wide.ci_low) > (tight.ci_high - tight.ci_low)


def test_estimate_str_is_report_ready():
    rng = np.random.default_rng(6)
    _, y_pred = simulate(0.6, 1_000, rng)
    hold_true, hold_pred = simulate(0.5, 2_000, rng)
    text = str(estimate_with_ci(y_pred, hold_true, hold_pred, n_boot=400))
    assert "satisfied" in text and "%" in text and "n=1000" in text


# --------------------------------------------------------------------------
# resample_to_prevalence
# --------------------------------------------------------------------------


def test_resample_hits_target_prevalence():
    y_true = np.array([1] * 300 + [0] * 700)
    for target in [0.0, 0.1, 0.5, 0.9, 1.0]:
        idx = resample_to_prevalence(y_true, target, n=1_000, seed=7)
        assert len(idx) == 1_000
        assert y_true[idx].mean() == pytest.approx(target, abs=0.001)


def test_resample_exceeds_minority_class_size():
    """With-replacement sampling must allow n larger than the smaller class."""
    y_true = np.array([1] * 10 + [0] * 990)
    idx = resample_to_prevalence(y_true, 0.9, n=1_000, seed=8)
    assert y_true[idx].mean() == pytest.approx(0.9, abs=0.001)


def test_resample_rejects_bad_target():
    with pytest.raises(ValueError):
        resample_to_prevalence([0, 1], 1.5)

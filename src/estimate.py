"""Aggregate satisfaction estimation — the project's centerpiece (§10).

The business question is "what share of clients are satisfied?", which is a
question about a *population*, not about any single comment. The obvious
approach — classify everything and count the positives — is called
Classify and Count (CC) and it is **biased** whenever the classifier's errors
are asymmetric.

With true positive prevalence ``p``, the rate at which the classifier predicts
positive is::

    p_hat = TPR * p + FPR * (1 - p)

CC reports ``p_hat`` and calls it ``p``. Those are equal only when TPR == 1 and
FPR == 0. Inverting the relation gives the Adjusted Classify and Count (ACC)
estimator implemented here, which is unbiased given accurate TPR/FPR.

Worked example of the bias CC introduces (TPR=0.94, FPR=0.06):

    true p = 0.30  ->  CC reports 0.324   (+0.024)
    true p = 0.10  ->  CC reports 0.148   (+0.048)

The error grows as the true rate moves away from the classifier's training
prevalence — precisely when the answer matters most.

No torch dependency: this module is pure numpy so it can be tested without a
GPU or a trained model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SatisfactionEstimate",
    "confusion_rates",
    "classify_and_count",
    "acc_prevalence",
    "estimate_with_ci",
    "resample_to_prevalence",
    "confusion_matrix_rates",
    "predicted_distribution",
    "acc_prevalence_multiclass",
    "project_to_simplex",
]


@dataclass(frozen=True)
class SatisfactionEstimate:
    """Result of an aggregate satisfaction estimate.

    Attributes
    ----------
    rate:
        Bias-corrected share of satisfied clients, in [0, 1].
    ci_low, ci_high:
        Bootstrap confidence bounds propagating *both* sources of uncertainty:
        sampling of the comments, and estimation error in TPR/FPR.
    naive_rate:
        The uncorrected Classify-and-Count figure, kept for comparison so a
        report can show how much the correction moved the answer.
    n:
        Number of comments the estimate is based on.
    """

    rate: float
    ci_low: float
    ci_high: float
    naive_rate: float
    n: int

    def __str__(self) -> str:
        return (
            f"{self.rate:.1%} satisfied "
            f"[{self.ci_low:.1%}-{self.ci_high:.1%}], n={self.n} "
            f"(uncorrected: {self.naive_rate:.1%})"
        )


def confusion_rates(y_true, y_pred, positive=1) -> tuple[float, float]:
    """Measure TPR and FPR on a labelled holdout.

    These are the two numbers the correction depends on. They are measured
    once on labelled data and stored in ``inference_config.json`` (§3.2) so
    they travel with the weights they were measured for.

    Returns
    -------
    (tpr, fpr)
        TPR = P(predict positive | truly positive).
        FPR = P(predict positive | truly negative).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    pos, neg = y_true == positive, y_true != positive
    if not pos.any() or not neg.any():
        raise ValueError("holdout must contain both classes to measure TPR and FPR")

    tpr = float((y_pred[pos] == positive).mean())
    fpr = float((y_pred[neg] == positive).mean())
    return tpr, fpr


def classify_and_count(y_pred, positive=1) -> float:
    """The naive (biased) estimator: the share of positive predictions.

    Provided so reports can show CC alongside ACC rather than only asserting
    that the correction matters.
    """
    return float((np.asarray(y_pred) == positive).mean())


def acc_prevalence(pred_pos_rate: float, tpr: float, fpr: float) -> float:
    """Adjusted Classify and Count: invert the bias relation.

        p = (p_hat - FPR) / (TPR - FPR)

    The result is clipped to [0, 1] because sampling noise can push the raw
    solution slightly outside the valid range, especially at extreme
    prevalences or small n.

    Raises
    ------
    ValueError
        If ``tpr <= fpr``. The correction is undefined there, and a classifier
        in that regime carries no usable signal (it is at or below chance).
    """
    if not tpr > fpr:
        raise ValueError(
            f"tpr ({tpr:.3f}) must exceed fpr ({fpr:.3f}); "
            "the classifier carries no usable signal otherwise"
        )
    return float(np.clip((pred_pos_rate - fpr) / (tpr - fpr), 0.0, 1.0))


def estimate_with_ci(
    y_pred,
    holdout_y_true,
    holdout_y_pred,
    positive=1,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> SatisfactionEstimate:
    """Bias-corrected satisfaction rate with a bootstrap confidence interval.

    Two independent sources of uncertainty are propagated, which is why the
    labelled holdout is resampled alongside the unlabelled batch:

    1. **Sampling of the comments** — the batch is a finite sample of client
       feedback, so ``p_hat`` is noisy.
    2. **Estimation error in TPR/FPR** — these come from a finite labelled
       holdout, and the correction divides by ``(tpr - fpr)``, so error there
       is amplified when the classifier is weak.

    Reporting only the first (the common shortcut) understates the interval,
    sometimes badly when the holdout is small.

    Parameters
    ----------
    y_pred:
        Predicted labels for the unlabelled batch being summarised.
    holdout_y_true, holdout_y_pred:
        Gold and predicted labels on a labelled holdout, used for TPR/FPR.
    n_boot:
        Bootstrap replicates. 2000 is ample for a 95% interval.
    alpha:
        1 - confidence. 0.05 gives a 95% interval.
    """
    y_pred = np.asarray(y_pred)
    holdout_y_true = np.asarray(holdout_y_true)
    holdout_y_pred = np.asarray(holdout_y_pred)

    tpr, fpr = confusion_rates(holdout_y_true, holdout_y_pred, positive)
    naive = classify_and_count(y_pred, positive)
    point = acc_prevalence(naive, tpr, fpr)

    rng = np.random.default_rng(seed)
    n, m = len(y_pred), len(holdout_y_true)
    pos_flags = (y_pred == positive).astype(float)

    draws = np.empty(n_boot, dtype=float)
    kept = 0
    for _ in range(n_boot):
        p_hat_b = pos_flags[rng.integers(0, n, n)].mean()
        idx = rng.integers(0, m, m)
        try:
            tpr_b, fpr_b = confusion_rates(
                holdout_y_true[idx], holdout_y_pred[idx], positive
            )
            draws[kept] = acc_prevalence(p_hat_b, tpr_b, fpr_b)
        except ValueError:
            # A degenerate resample (one class absent, or tpr <= fpr) carries
            # no information. Skip it rather than letting it distort the
            # interval; with a sane holdout this is rare.
            continue
        kept += 1

    if kept < n_boot // 2:
        raise RuntimeError(
            f"only {kept}/{n_boot} bootstrap replicates were usable; "
            "the holdout is likely too small or too imbalanced"
        )

    lo, hi = np.percentile(draws[:kept], [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return SatisfactionEstimate(point, float(lo), float(hi), naive, n)


# --------------------------------------------------------------------------
# Multiclass generalisation
#
# The two-class formula above is the special case of a general relation: the
# distribution of *predicted* labels is the confusion matrix times the
# distribution of *true* labels,
#
#     p_hat = M @ p        where  M[i, j] = P(predict i | true j)
#
# so the correction is a linear solve rather than a ratio. This is needed the
# moment a dataset has a real neutral class instead of a binary split.
# --------------------------------------------------------------------------


def confusion_matrix_rates(y_true, y_pred, labels) -> np.ndarray:
    """Column-stochastic confusion matrix ``M[i, j] = P(predict i | true j)``.

    Note the orientation: **columns** are the true class and sum to 1. This is
    the transpose of sklearn's ``confusion_matrix`` convention, and it is the
    one that makes ``p_hat = M @ p`` hold.

    Raises if any true class is absent from the holdout — its column would be
    undefined, and the solve would silently produce nonsense.
    """
    y_true, y_pred, labels = np.asarray(y_true), np.asarray(y_pred), list(labels)
    k = len(labels)
    M = np.zeros((k, k), dtype=float)
    for j, true_lab in enumerate(labels):
        mask = y_true == true_lab
        if not mask.any():
            raise ValueError(
                f"class {true_lab!r} does not appear in the holdout; "
                "its confusion column cannot be estimated"
            )
        for i, pred_lab in enumerate(labels):
            M[i, j] = (y_pred[mask] == pred_lab).mean()
    return M


def predicted_distribution(y_pred, labels) -> np.ndarray:
    """Share of predictions falling in each label — the multiclass ``p_hat``.

    This is multiclass Classify-and-Count: biased in exactly the same way as
    the binary version, and reported alongside the correction for comparison.
    """
    y_pred = np.asarray(y_pred)
    n = len(y_pred)
    if n == 0:
        raise ValueError("no predictions to summarise")
    return np.array([(y_pred == lab).mean() for lab in labels], dtype=float)


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Nearest point to ``v`` on the probability simplex (Euclidean).

    The linear solve can land slightly outside the simplex — small negative
    entries, or a sum a little off 1 — because ``p_hat`` is a noisy estimate.
    Clipping at zero and renormalising is the common shortcut but is not the
    nearest valid point; this is, and it costs eight lines.
    """
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def acc_prevalence_multiclass(
    pred_dist: np.ndarray, M: np.ndarray, rcond: float = 1e-10
) -> np.ndarray:
    """Bias-corrected class prevalences: solve ``p_hat = M @ p`` for ``p``.

    Falls back to a least-squares solve when ``M`` is singular or
    ill-conditioned — which happens when two classes are confused so heavily
    that they are not separable. The result is projected onto the simplex so
    it is always a valid distribution.

    A badly conditioned ``M`` is worth noticing rather than silently
    absorbing: it means the correction is extrapolating from a classifier that
    cannot really tell those classes apart, and the output deserves wide error
    bars.
    """
    pred_dist = np.asarray(pred_dist, dtype=float).ravel()
    M = np.asarray(M, dtype=float)
    if M.shape[0] != M.shape[1] or M.shape[0] != pred_dist.size:
        raise ValueError("M must be square and match the length of pred_dist")

    try:
        cond = np.linalg.cond(M)
    except np.linalg.LinAlgError:
        cond = np.inf
    if not np.isfinite(cond) or cond > 1.0 / rcond:
        p, *_ = np.linalg.lstsq(M, pred_dist, rcond=None)
    else:
        p = np.linalg.solve(M, pred_dist)
    return project_to_simplex(p)


def resample_to_prevalence(
    y_true, target_prevalence: float, n: int | None = None, positive=1, seed: int = 0
) -> np.ndarray:
    """Indices of a resample whose positive prevalence is ``target_prevalence``.

    This drives the §10.3 experiment: sweep the true prevalence across a range,
    then plot CC and ACC against truth. CC bows toward the classifier's
    training prevalence; ACC tracks the diagonal. That plot is the project's
    headline figure, so it is worth generating from real predictions rather
    than asserting the result.

    Sampling is with replacement, so ``n`` is not limited by the size of the
    smaller class — important at extreme target prevalences.
    """
    if not 0.0 <= target_prevalence <= 1.0:
        raise ValueError("target_prevalence must be in [0, 1]")

    y_true = np.asarray(y_true)
    pos_idx = np.flatnonzero(y_true == positive)
    neg_idx = np.flatnonzero(y_true != positive)
    if n is None:
        n = len(y_true)

    n_pos = int(round(n * target_prevalence))
    n_neg = n - n_pos
    if n_pos and len(pos_idx) == 0:
        raise ValueError("no positive examples available to resample")
    if n_neg and len(neg_idx) == 0:
        raise ValueError("no negative examples available to resample")

    rng = np.random.default_rng(seed)
    parts = []
    if n_pos:
        parts.append(rng.choice(pos_idx, n_pos, replace=True))
    if n_neg:
        parts.append(rng.choice(neg_idx, n_neg, replace=True))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out

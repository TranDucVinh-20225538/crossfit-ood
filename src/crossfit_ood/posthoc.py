"""POST-HOC helper: does Delta track a leakage surrogate once group count is accounted for?

Exploratory only (not preregistered). Delta depends on the number of groups (fewer
groups -> larger Delta at the same ICC), so a raw correlation between Delta and a
surrogate (ICC, probe accuracy for group identity, ...) across datasets/backbones can
be driven by group count. This helper reports the correlation with and without
adjusting for ``1 / n_groups``.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = ["posthoc_delta_regression"]


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank); no scipy needed."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return sums[inv] / counts[inv]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else float("nan")


def _resid(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    X = np.c_[np.ones(len(z)), z]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def _ols(y: np.ndarray, cols: Dict[str, np.ndarray]) -> Dict[str, float]:
    X = np.column_stack([np.ones(len(y)), *cols.values()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    r2 = 1.0 - (res @ res) / np.sum((y - y.mean()) ** 2)
    out = {"intercept": float(beta[0]), "r2": float(r2)}
    out.update({k: float(b) for k, b in zip(cols, beta[1:])})
    return out


def posthoc_delta_regression(delta: Sequence[float], surrogate: Sequence[float],
                             n_groups: Sequence[float]) -> Dict[str, object]:
    """POST-HOC: Delta vs surrogate, with and without the ``1/n_groups`` term.

    Returns a dict with
      * ``pearson_raw``, ``spearman_raw``: corr(Delta, surrogate);
      * ``pearson_partial``, ``spearman_partial``: correlation of the residuals of
        Delta and of the surrogate after regressing each on ``1/n_groups``
        (Spearman: on ranks, i.e. rank-partial correlation);
      * ``ols_surrogate``: Delta ~ 1 + surrogate;
        ``ols_surrogate_inv_n``: Delta ~ 1 + surrogate + 1/n_groups;
      * ``n``, ``label = "POST-HOC (exploratory)"``.
    With few points (one per dataset x backbone) these are descriptive only.
    """
    d = np.asarray(delta, float)
    s = np.asarray(surrogate, float)
    inv_n = 1.0 / np.asarray(n_groups, float)
    if not (len(d) == len(s) == len(inv_n)) or len(d) < 4:
        raise ValueError("need equal-length inputs with at least 4 points")
    rd, rs, rn = _rank(d), _rank(s), _rank(inv_n)
    return {
        "label": "POST-HOC (exploratory)",
        "n": int(len(d)),
        "pearson_raw": _pearson(d, s),
        "spearman_raw": _pearson(rd, rs),
        "pearson_partial": _pearson(_resid(d, inv_n), _resid(s, inv_n)),
        "spearman_partial": _pearson(_resid(rd, rn), _resid(rs, rn)),
        "ols_surrogate": _ols(d, {"surrogate": s}),
        "ols_surrogate_inv_n": _ols(d, {"surrogate": s, "inv_n_groups": inv_n}),
    }

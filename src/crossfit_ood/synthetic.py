"""Synthetic grouped features with a group random effect (for tests and demos).

Generative model (per feature dimension, total within+between variance = 1)::

    x = offset + mu_class + u_group + eps,
    u_group ~ N(0, icc * I),   eps ~ N(0, (1 - icc) * I)

so ``icc`` is the intra-class (intra-group) correlation of every dimension.
OOD samples use the same model with a mean shift of norm ``ood_shift`` along a
random direction and their own groups. Train / ID-eval are a *sample-level*
split within each training group (full group overlap, i.e. the leaky setting);
``n_new_groups`` adds ID-eval groups absent from train; ``features_fresh`` are
samples from entirely new ID groups (an oracle for "true" new-group AUROC).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

__all__ = ["GroupedData", "make_grouped_features", "make_features_from_sizes"]


@dataclass
class GroupedData:
    features_train: np.ndarray
    groups_train: np.ndarray
    labels_train: np.ndarray
    features_id_eval: np.ndarray
    groups_id_eval: np.ndarray
    labels_id_eval: np.ndarray
    features_ood: np.ndarray
    groups_ood: np.ndarray
    features_fresh: np.ndarray
    groups_fresh: np.ndarray
    icc: float


def make_grouped_features(
    icc: float = 0.5,
    *,
    n_groups: int = 24,
    n_per_group: Union[int, Tuple[int, int]] = 40,
    eval_fraction: float = 0.5,
    n_new_groups: int = 0,
    n_features: int = 32,
    n_classes: int = 2,
    class_sep: float = 1.0,
    ood_shift: float = 3.0,
    n_ood_groups: int = 24,
    n_fresh_groups: int = 24,
    offset: float = 2.0,
    random_state: Optional[int] = 0,
    population_seed: Optional[int] = None,
) -> GroupedData:
    """Draw a synthetic dataset; see module docstring. Each group has one class.

    ``n_per_group`` may be an int (equal group sizes) or ``(lo, hi)``: each group
    size drawn uniformly from ``lo..hi`` (unequal groups, e.g. slides with
    different patch counts).

    ``population_seed``: if given, the *population* parameters (class means, OOD
    direction) come from this seed and ``random_state`` only drives sampling
    (groups, noise, splits). Use a fixed ``population_seed`` across replicates
    for coverage studies, so that the estimand is the same in every replicate.
    If None (default), both come from ``random_state`` (each seed is a new
    population).
    """
    if not 0.0 <= icc < 1.0:
        raise ValueError("icc must be in [0, 1)")
    rng = np.random.default_rng(random_state)
    prng = rng if population_seed is None else np.random.default_rng(population_seed)
    d = n_features
    mu = prng.normal(0.0, class_sep / np.sqrt(d), size=(n_classes, d)) * np.sqrt(d)
    mu *= class_sep / np.maximum(np.linalg.norm(mu, axis=1, keepdims=True), 1e-12)
    direction = prng.normal(size=d)
    direction /= np.linalg.norm(direction)

    def size() -> int:
        if isinstance(n_per_group, (tuple, list)):
            return int(rng.integers(n_per_group[0], n_per_group[1] + 1))
        return int(n_per_group)

    def draw(n_g: int, base: np.ndarray, gid0: int, cls: Optional[np.ndarray]):
        xs, gs, ys = [], [], []
        for g in range(n_g):
            n_g_i = size()
            c = int(rng.integers(n_classes)) if cls is None else int(cls[g])
            u = rng.normal(0.0, np.sqrt(icc), size=d)
            e = rng.normal(0.0, np.sqrt(1.0 - icc), size=(n_g_i, d))
            xs.append(offset + base + (mu[c] if cls is not False else 0) + u + e)
            gs.append(np.full(n_g_i, gid0 + g))
            ys.append(np.full(n_g_i, c))
        return np.concatenate(xs), np.concatenate(gs), np.concatenate(ys)

    zero = np.zeros(d)
    x, g, y = draw(n_groups, zero, 0, None)
    is_eval = np.zeros(len(x), bool)
    for gg in range(n_groups):
        idx = np.flatnonzero(g == gg)
        is_eval[rng.choice(idx, int(round(eval_fraction * len(idx))), replace=False)] = True
    xtr, gtr, ytr = x[~is_eval], g[~is_eval], y[~is_eval]
    xev, gev, yev = x[is_eval], g[is_eval], y[is_eval]
    if n_new_groups:
        xn, gn, yn = draw(n_new_groups, zero, n_groups, None)
        keep = rng.random(len(xn)) < eval_fraction
        xev, gev, yev = (np.concatenate([xev, xn[keep]]), np.concatenate([gev, gn[keep]]),
                         np.concatenate([yev, yn[keep]]))
    xo, go, _ = draw(n_ood_groups, ood_shift * direction, 10_000, None)
    xf, gf, _ = draw(n_fresh_groups, zero, 20_000, None)
    return GroupedData(xtr, gtr, ytr, xev, gev, yev, xo, go, xf, gf, icc)


def make_features_from_sizes(
    icc: float,
    train_sizes,
    eval_sizes,
    ood_sizes,
    *,
    n_features: int = 32,
    n_classes: int = 2,
    class_sep: float = 1.0,
    ood_shift: float = 3.0,
    offset: float = 2.0,
    random_state: Optional[int] = 0,
    population_seed: Optional[int] = None,
) -> GroupedData:
    """Same generative model as :func:`make_grouped_features`, explicit design.

    ``train_sizes[g]`` / ``eval_sizes[g]``: train and ID-eval samples of ID group
    ``g`` (``eval_sizes[g] = 0`` -> train-only group; a group with
    ``train_sizes[g] = 0`` is an ID-eval group absent from train).
    ``ood_sizes[j]``: samples of OOD group ``j``. Used to mimic real designs
    (group counts and skewed per-group sizes) in coverage studies. The design
    (sizes) is fixed by the caller; ``random_state`` drives group effects and
    noise; ``population_seed`` fixes class means and the OOD direction.
    ``features_fresh`` / ``groups_fresh`` are empty.
    """
    if not 0.0 <= icc < 1.0:
        raise ValueError("icc must be in [0, 1)")
    ts, es, os_ = (np.asarray(a, dtype=int) for a in (train_sizes, eval_sizes, ood_sizes))
    if ts.shape != es.shape or ts.ndim != 1:
        raise ValueError("train_sizes and eval_sizes must be 1-d arrays of equal length")
    if (ts < 0).any() or (es < 0).any() or (os_ < 1).any():
        raise ValueError("sizes must be >= 0 (ID) / >= 1 (OOD)")
    rng = np.random.default_rng(random_state)
    prng = rng if population_seed is None else np.random.default_rng(population_seed)
    d = n_features
    mu = prng.normal(0.0, class_sep / np.sqrt(d), size=(n_classes, d)) * np.sqrt(d)
    mu *= class_sep / np.maximum(np.linalg.norm(mu, axis=1, keepdims=True), 1e-12)
    direction = prng.normal(size=d)
    direction /= np.linalg.norm(direction)
    sd_u, sd_e = np.sqrt(icc), np.sqrt(1.0 - icc)

    G = len(ts)
    cls = rng.integers(n_classes, size=G)
    u = rng.normal(0.0, sd_u, size=(G, d))
    tot = ts + es
    g_all = np.repeat(np.arange(G), tot)
    x_all = offset + mu[cls[g_all]] + u[g_all] + rng.normal(0.0, sd_e, size=(tot.sum(), d))
    # first ts[g] samples of each group -> train, rest -> eval
    start = np.concatenate([[0], np.cumsum(tot)[:-1]])
    pos = np.arange(tot.sum()) - np.repeat(start, tot)
    is_tr = pos < ts[g_all]
    xtr, gtr = x_all[is_tr], g_all[is_tr]
    xev, gev = x_all[~is_tr], g_all[~is_tr]
    J = len(os_)
    uo = rng.normal(0.0, sd_u, size=(J, d))
    co = rng.integers(n_classes, size=J)
    go = np.repeat(np.arange(J), os_)
    xo = (offset + ood_shift * direction + mu[co[go]] + uo[go]
          + rng.normal(0.0, sd_e, size=(os_.sum(), d)))
    empty = np.zeros((0, d))
    return GroupedData(xtr, gtr, cls[gtr], xev, gev, cls[gev], xo, go + 1_000_000,
                       empty, np.zeros(0, int), icc)

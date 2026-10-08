"""Leaky vs cross-fitted AUROC for feature-based OOD scorers under group overlap.

Terminology
-----------
*group*   : unit of dependence (patient, slide, hospital, camera trap ...).
*train*   : samples the scorer is fitted on (``features_train``).
*ID-eval* : held-out in-distribution samples (``features_id_eval``); with a
            sample-level split their groups usually also occur in train.
*OOD*     : out-of-distribution samples (``features_ood``).

Score convention: higher = more ID. AUROC = P(score_ID > score_OOD) + 0.5 P(tie).
See :func:`crossfit_auroc` for the two protocols and the three uncertainty methods.
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .scorers import DEFAULT_SCORERS, Scorer, get_scorer

__all__ = [
    "OverlapStats",
    "TPRTransfer",
    "ScoreResult",
    "CrossfitReport",
    "group_overlap",
    "make_group_folds",
    "default_n_splits",
    "crossfit_auroc",
    "PROTOCOLS",
    "POOLINGS",
    "UNCERTAINTY_METHODS",
    "LOGO_MAX_GROUPS",
]

ScorerSpec = Union[str, Scorer]
PROTOCOLS = ("default", "paper_2fold")
POOLINGS = ("per_fit", "naive")
UNCERTAINTY_METHODS = ("jackknife", "subsample", "bootstrap")
LOGO_MAX_GROUPS = 50
_K_LARGE = 10
_CHUNK = 200


# =========================================================================== results
@dataclass(frozen=True)
class OverlapStats:
    """Group overlap between train and ID-eval.

    ``sample_rate``: fraction of ID-eval *samples* whose group occurs in train.
    ``group_rate``: fraction of ID-eval *groups* that occur in train.
    """

    sample_rate: float
    group_rate: float
    n_id_eval_groups: int
    n_shared_groups: int


@dataclass(frozen=True)
class TPRTransfer:
    """Threshold transfer from seen to new groups, point estimate.

    For every fitted model: threshold = ``1 - target_tpr`` quantile of the scores
    of *seen* ID-eval samples (group in that model's fit set); ``tpr_new`` = pooled
    fraction of *new*-group ID-eval samples passing it. ``false_alarm_rate = 1 -
    tpr_new``; ``false_alarm_ratio = false_alarm_rate / (1 - target_tpr)`` (1 =
    nominal); ``fpr_ood`` = fraction of OOD passing.
    """

    target_tpr: float
    tpr_seen: float
    tpr_new: float
    false_alarm_rate: float
    false_alarm_ratio: float
    fpr_ood: float
    n_seen: int
    n_new: int
    n_folds_used: int


@dataclass(frozen=True)
class ScoreResult:
    """Per-scorer result.

    Protocol ``default``: ``auroc_leaky`` = scorer fitted on all train;
    ``auroc_crossfit`` = group-cross-fitted. Protocol ``paper_2fold``:
    ``auroc_leaky`` = mean AUROC of *seen*-group ID vs OOD, ``auroc_crossfit`` =
    mean AUROC of *unseen*-group ID vs OOD (same fitted model per fold).
    ``delta = auroc_leaky - auroc_crossfit``. ``extra`` holds method-specific
    diagnostics (e.g. both jackknife SEs, both subsampling scalings).
    """

    name: str
    auroc_leaky: float
    auroc_crossfit: float
    delta: float
    delta_ci: Tuple[float, float]
    auroc_leaky_ci: Tuple[float, float]
    auroc_crossfit_ci: Tuple[float, float]
    delta_se: float
    tpr_transfer: TPRTransfer
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class CrossfitReport:
    overlap: OverlapStats
    results: Dict[str, ScoreResult]
    protocol: str
    uncertainty: str
    n_splits: int
    n_repeats: int
    match_fit_size: bool
    ci_level: float
    n_groups_train: int = 0
    n_groups_id_eval: int = 0      # ID-eval groups actually used by the protocol
    n_groups_shared: int = 0       # used ID-eval groups that also occur in train
    n_groups_fit: Tuple[int, int] = (0, 0)  # (min, max) train groups per cross-fit / paper_2fold fit
    pooling: str = "per_fit"
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """Plain-text table (no pandas dependency)."""
        o = self.overlap
        lab = ("leaky", "xfit") if self.protocol == "default" else ("seen", "unseen")
        g0, g1 = self.n_groups_fit
        w = max([12] + [len(n) + 1 for n in self.results])
        lines = [
            f"protocol={self.protocol}  K={self.n_splits}  repeats={self.n_repeats}  "
            f"uncertainty={self.uncertainty}  pooling={self.pooling}",
            f"groups: train={self.n_groups_train}  ID-eval used={self.n_groups_id_eval}  "
            f"shared={self.n_groups_shared}  per fit={g0 if g0 == g1 else f'{g0}-{g1}'}  "
            f"(overlap: {o.sample_rate:.1%} of all ID-eval samples)",
            "note: Delta depends on the number of groups per fit; compare only at equal groups per fit",
            f"{'score':<{w}}{lab[0]:>8}{lab[1]:>8}{'delta':>8}  {int(self.ci_level * 100)}% CI"
            f"{'':>10}{'TPR_new':>8}{'FA x':>7}",
        ]
        for r in self.results.values():
            t = r.tpr_transfer
            lines.append(
                f"{r.name:<{w}}{r.auroc_leaky:8.3f}{r.auroc_crossfit:8.3f}{r.delta:+8.3f}"
                f"  [{r.delta_ci[0]:+.3f}, {r.delta_ci[1]:+.3f}]"
                f"{t.tpr_new:8.3f}{t.false_alarm_ratio:7.2f}"
            )
        return "\n".join(lines)


# =========================================================================== helpers
def group_overlap(groups_train: Sequence, groups_id_eval: Sequence) -> OverlapStats:
    """Fraction of ID-eval samples / groups whose group also occurs in train."""
    gtr = np.asarray(groups_train)
    gev = np.asarray(groups_id_eval)
    seen = np.isin(gev, gtr)
    uniq = np.unique(gev)
    shared = np.isin(uniq, gtr)
    return OverlapStats(
        sample_rate=float(seen.mean()) if len(gev) else float("nan"),
        group_rate=float(shared.mean()) if len(uniq) else float("nan"),
        n_id_eval_groups=int(len(uniq)),
        n_shared_groups=int(shared.sum()),
    )


def _random_fold_map(unique_groups: np.ndarray, n_splits: int, seed: Optional[int]) -> dict:
    k = int(min(n_splits, len(unique_groups)))
    if k < 2:
        raise ValueError("need at least 2 distinct groups to build folds")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_groups))
    out = {}
    for f, chunk in enumerate(np.array_split(perm, k)):
        for i in chunk:
            out[unique_groups[i].item() if hasattr(unique_groups[i], "item") else unique_groups[i]] = f
    return out


def _apply_map(fmap: dict, groups: np.ndarray) -> np.ndarray:
    """Fold per sample (-1 if the group is not in the map)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    vals = np.array([fmap.get(u.item() if hasattr(u, "item") else u, -1) for u in uniq], dtype=int)
    return vals[inv] if len(groups) else np.empty(0, int)


def default_n_splits(n_groups: int) -> int:
    """Default fold count: leave-one-group-out if ``n_groups <= 50``, otherwise K = 10.

    Delta depends on the number of groups per fit, so many folds (each fit sees
    almost all groups) keep the cross-fitted model close to the leaky one.
    """
    n_groups = int(n_groups)
    return n_groups if n_groups <= LOGO_MAX_GROUPS else _K_LARGE


def make_group_folds(groups: Sequence, n_splits: Optional[int] = None,
                     random_state: Optional[int] = 0) -> np.ndarray:
    """Assign each sample to one of ``K = min(n_splits, n_groups)`` folds, whole groups per fold.

    Groups are shuffled with ``random_state`` and split into K near-equal chunks
    *by group count*. ``n_splits=None``: :func:`default_n_splits`. Returns an int
    array of length ``len(groups)``.
    """
    g = np.asarray(groups)
    uniq = np.unique(g)
    k = default_n_splits(len(uniq)) if n_splits is None else n_splits
    return _apply_map(_random_fold_map(uniq, k, random_state), g)


def _fold_ids_to_maps(fold_ids, gid: np.ndarray) -> List[dict]:
    if isinstance(fold_ids, np.ndarray) and fold_ids.ndim == 1:
        fold_ids = [fold_ids]
    maps = []
    for f in fold_ids:
        f = np.asarray(f, dtype=int)
        if f.shape != gid.shape or (len(f) and f.min() < 0):
            raise ValueError("each fold_ids array must be non-negative ints, one per ID-eval sample")
        m = {}
        for g in np.unique(gid):
            vals = np.unique(f[gid == g])
            if len(vals) != 1:
                raise ValueError(f"group {g!r} is split across folds")
            m[g.item() if hasattr(g, "item") else g] = int(vals[0])
        maps.append(m)
    return maps


def _check_features(name: str, x, n_features: Optional[int] = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN/inf")
    if n_features is not None and x.shape[1] != n_features:
        raise ValueError(f"{name} has {x.shape[1]} features, expected {n_features}")
    return x


def _resolve_scorers(scorers) -> Dict[str, Scorer]:
    if isinstance(scorers, dict):
        return dict(scorers)
    if isinstance(scorers, str) or not isinstance(scorers, Iterable):
        scorers = [scorers]
    out: Dict[str, Scorer] = {}
    for s in scorers:
        if isinstance(s, str):
            out[s] = get_scorer(s)
        else:
            out[getattr(s, "name", type(s).__name__)] = s
    return out


def _percentile_ci(x: np.ndarray, level: float) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    a = (1.0 - level) / 2.0
    return (float(np.quantile(x, a)), float(np.quantile(x, 1.0 - a)))


def _cluster_weights(codes: np.ndarray, n_clusters: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    counts = np.stack([np.bincount(d, minlength=n_clusters) for d in draws]).astype(np.float64)
    return counts[:, codes]


def _parallel(fn, args_list, n_jobs: int):
    if n_jobs == 1 or len(args_list) <= 1:
        return [fn(*a) for a in args_list]
    from joblib import Parallel, delayed  # joblib ships with scikit-learn

    return Parallel(n_jobs=n_jobs)(delayed(fn)(*a) for a in args_list)


# =========================================================================== AUROC arms
class _Arm:
    """ID-eval subset ``idx`` scored by one model vs *all* OOD scored by the same model.

    OOD scores are sorted once, so any (bootstrap / jackknife) weighting only needs
    a cumulative sum: O(B * (n_id + n_ood)).
    """

    def __init__(self, idx: np.ndarray, id_scores: np.ndarray, ood_scores: np.ndarray) -> None:
        self.idx = idx
        self.order = np.argsort(ood_scores, kind="mergesort")
        s = ood_scores[self.order]
        self.left = np.searchsorted(s, id_scores, side="left")
        self.right = np.searchsorted(s, id_scores, side="right")

    def num_wsum(self, w_id: np.ndarray, w_ood: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = np.concatenate([np.zeros((w_ood.shape[0], 1)), np.cumsum(w_ood[:, self.order], axis=1)], axis=1)
        below = c[:, self.left]
        ties = c[:, self.right] - below
        wi = w_id[:, self.idx]
        return np.sum(wi * (below + 0.5 * ties), axis=1), wi.sum(axis=1)


def _combine(arms: List[_Arm], w_id: np.ndarray, w_ood: np.ndarray, how: str) -> np.ndarray:
    vsum = w_ood.sum(axis=1)
    parts = [a.num_wsum(w_id, w_ood) for a in arms if len(a.idx)]
    with np.errstate(invalid="ignore", divide="ignore"):
        if how == "pooled":  # each ID sample weighted equally (sample-weighted over arms)
            num = sum(p[0] for p in parts)
            ws = sum(p[1] for p in parts)
            return num / (ws * vsum)
        per_arm = np.stack([p[0] / (p[1] * vsum) for p in parts])  # "mean": arms weighted equally
        return np.nanmean(per_arm, axis=0)


@dataclass
class _ScorerFit:
    """Fitted arms for one scorer; ``stats`` maps weights -> (AUROC_leaky, AUROC_xfit)."""

    leaky: List[Tuple[List[_Arm], str]]  # one entry per repeat (or a single shared one)
    xfit: List[Tuple[List[_Arm], str]]
    tpr_counts: Tuple[int, int, int, int, int, int, int]  # hits_seen,n_seen,hits_new,n_new,hits_ood,n_ood,used

    def stats(self, w_id: np.ndarray, w_ood: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a_l = np.mean([_combine(a, w_id, w_ood, h) for a, h in self.leaky], axis=0)
        a_x = np.mean([_combine(a, w_id, w_ood, h) for a, h in self.xfit], axis=0)
        return a_l, a_x

    def point(self, n_id: int, n_ood: int) -> Tuple[float, float]:
        a_l, a_x = self.stats(np.ones((1, n_id)), np.ones((1, n_ood)))
        return float(a_l[0]), float(a_x[0])


def _tpr_update(acc: list, s_all: np.ndarray, seen: np.ndarray, new: np.ndarray,
                s_ood: np.ndarray, target_tpr: float) -> None:
    if not seen.any() or not new.any():
        return
    thr = np.quantile(s_all[seen], 1.0 - target_tpr)
    acc[0] += int((s_all[seen] >= thr).sum()); acc[1] += int(seen.sum())
    acc[2] += int((s_all[new] >= thr).sum()); acc[3] += int(new.sum())
    acc[4] += int((s_ood >= thr).sum()); acc[5] += len(s_ood)
    acc[6] += 1


def _fit(proto: Scorer, x: np.ndarray, y: Optional[np.ndarray]) -> Scorer:
    return copy.deepcopy(proto).fit(x, y)


def _fit_scorer(proto, xtr, gtr, ytr, xid, gid, xood, protocol, fold_maps, match_fit_size,
                target_tpr, seed, pooling="per_fit") -> _ScorerFit:
    """Fit all models of one protocol for one scorer and build the AUROC arms.

    ``pooling="per_fit"``: each ID-eval sample is compared only with OOD scored by
    the same fitted model. ``"naive"`` (default protocol only): every ID-eval
    sample, scored by its cross-fit model, is compared with the OOD scores
    averaged over the cross-fit models.
    """
    acc = [0] * 7
    leaky_arms, xfit_arms = [], []
    all_id = np.arange(len(xid))
    if protocol == "default":
        leaky = _fit(proto, xtr, ytr)
        s_id_leaky, s_ood_leaky = leaky.score(xid), leaky.score(xood)
        if not match_fit_size:
            leaky_arms.append(([_Arm(all_id, s_id_leaky, s_ood_leaky)], "pooled"))
        fit_rng = np.random.default_rng(seed)
        for fmap in fold_maps:
            fold_id = _apply_map(fmap, gid)
            arms_x, arms_l = [], []
            naive_id, naive_ood = np.empty(len(xid)), []
            for f in np.unique(fold_id):
                in_f = fold_id == f
                excl = np.isin(gtr, np.unique(gid[in_f]))
                if excl.all():
                    raise ValueError(f"fold {f}: no train samples left after excluding its groups")
                if excl.any():
                    model = _fit(proto, xtr[~excl], None if ytr is None else ytr[~excl])
                    s_all, s_ood = model.score(xid), model.score(xood)
                else:
                    s_all, s_ood = s_id_leaky, s_ood_leaky
                idx = np.flatnonzero(in_f)
                arms_x.append(_Arm(idx, s_all[idx], s_ood))
                naive_id[idx] = s_all[idx]
                naive_ood.append(s_ood)
                seen = ~in_f & np.isin(gid, gtr[~excl])
                _tpr_update(acc, s_all, seen, in_f, s_ood, target_tpr)
                if match_fit_size:
                    n_keep = int((~excl).sum())
                    if n_keep == len(xtr):
                        sl_id, sl_ood = s_id_leaky[idx], s_ood_leaky
                    else:
                        sub = fit_rng.choice(len(xtr), n_keep, replace=False)
                        lm = _fit(proto, xtr[sub], None if ytr is None else ytr[sub])
                        sl_id, sl_ood = lm.score(xid[idx]), lm.score(xood)
                    arms_l.append(_Arm(idx, sl_id, sl_ood))
            if pooling == "naive":
                xfit_arms.append(([_Arm(all_id, naive_id, np.mean(naive_ood, axis=0))], "pooled"))
            else:
                xfit_arms.append((arms_x, "pooled"))
            if match_fit_size:
                leaky_arms.append((arms_l, "pooled"))
    else:  # paper_2fold
        for fmap in fold_maps:
            tf, idf = _apply_map(fmap, gtr), _apply_map(fmap, gid)
            arms_seen, arms_unseen = [], []
            for f in np.unique(tf[tf >= 0]):
                fit_mask = tf == f
                model = _fit(proto, xtr[fit_mask], None if ytr is None else ytr[fit_mask])
                s_all, s_ood = model.score(xid), model.score(xood)
                seen = idf == f
                unseen = (idf >= 0) & (idf != f)
                arms_seen.append(_Arm(np.flatnonzero(seen), s_all[seen], s_ood))
                arms_unseen.append(_Arm(np.flatnonzero(unseen), s_all[unseen], s_ood))
                _tpr_update(acc, s_all, seen, unseen, s_ood, target_tpr)
            leaky_arms.append((arms_seen, "mean"))
            xfit_arms.append((arms_unseen, "mean"))
    return _ScorerFit(leaky_arms, xfit_arms, tuple(acc))


def _new_fold_maps(protocol, gtr, gid, n_splits, n_repeats, seed) -> List[dict]:
    base = np.random.default_rng(seed)
    seeds = base.integers(0, 2**31 - 1, size=n_repeats)
    if protocol == "default":
        units, k = np.unique(gid), n_splits
    else:
        units, k = np.unique(gtr), 2
    return [_random_fold_map(units, k, int(s)) for s in seeds]


def _n_groups_fit(protocol, gtr, gid, fold_maps) -> Tuple[int, int]:
    """(min, max) number of train groups per fitted model (cross-fit / paper_2fold fits)."""
    ug = np.unique(gtr)
    counts = []
    for fmap in fold_maps:
        if protocol == "default":
            fold_id = _apply_map(fmap, gid)
            for f in np.unique(fold_id):
                counts.append(int((~np.isin(ug, np.unique(gid[fold_id == f]))).sum()))
        else:
            tf = _apply_map(fmap, ug)
            counts += [int((tf == f).sum()) for f in np.unique(tf[tf >= 0])]
    return (min(counts), max(counts)) if counts else (0, 0)


def _point_estimates(scorers: Dict[str, Scorer], xtr, gtr, ytr, xid, gid, xood, protocol,
                     fold_maps, match_fit_size, target_tpr, seed,
                     pooling="per_fit") -> Dict[str, Tuple[float, float, float]]:
    out = {}
    for name, proto in scorers.items():
        sf = _fit_scorer(proto, xtr, gtr, ytr, xid, gid, xood, protocol, fold_maps,
                         match_fit_size, target_tpr, seed, pooling)
        a_l, a_x = sf.point(len(xid), len(xood))
        out[name] = (a_l, a_x, a_l - a_x)
    return out


# =========================================================================== jackknife
def _jk_var_equal(theta_del: np.ndarray) -> np.ndarray:
    """Classic delete-a-block jackknife: (g-1)/g * sum (theta_j - mean)^2 (equal blocks)."""
    g = theta_del.shape[0]
    return (g - 1) / g * np.sum((theta_del - theta_del.mean(axis=0)) ** 2, axis=0)


def _jk_var_weighted(theta_full: np.ndarray, theta_del: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Weighted delete-m_j jackknife (Busing, Meijer & van der Leeden, 1999).

    ``m``: size of each deleted block in *independent units* (groups).

    h_j = n / m_j;  pseudo_j = h_j * theta - (h_j - 1) * theta_(-j);
    theta_J = g * theta - sum_j (1 - m_j / n) theta_(-j);
    V = (1/g) * sum_j (pseudo_j - theta_J)^2 / (h_j - 1).
    Reduces to the classic formula when all m_j are equal.
    """
    m = np.asarray(m, dtype=np.float64)
    n, g = m.sum(), len(m)
    h = (n / m)[:, None]
    theta_j = g * theta_full - np.sum((1.0 - m / n)[:, None] * theta_del, axis=0)
    pseudo = h * theta_full - (h - 1.0) * theta_del
    return np.sum((pseudo - theta_j) ** 2 / (h - 1.0), axis=0) / g


def _blocks(units: np.ndarray, n_blocks: int, rng: np.random.Generator) -> List[np.ndarray]:
    perm = units[rng.permutation(len(units))]
    return [b for b in np.array_split(perm, min(n_blocks, len(units))) if len(b)]


# =========================================================================== main API
def crossfit_auroc(
    features_train: np.ndarray,
    groups_train: Sequence,
    features_id_eval: np.ndarray,
    groups_id_eval: Sequence,
    features_ood: np.ndarray,
    *,
    labels_train: Optional[Sequence] = None,
    scorers: Union[ScorerSpec, Iterable[ScorerSpec], Dict[str, Scorer]] = DEFAULT_SCORERS,
    groups_ood: Optional[Sequence] = None,
    protocol: str = "default",
    n_splits: Optional[int] = None,
    n_repeats: int = 1,
    pooling: str = "per_fit",
    match_fit_size: bool = False,
    fold_ids: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    fold_ids_eval: Optional[np.ndarray] = None,
    uncertainty: str = "jackknife",
    ci_level: float = 0.95,
    n_jackknife_blocks: int = 50,
    jackknife_formula: str = "weighted",
    n_subsamples: int = 200,
    subsample_size: Optional[Union[int, float]] = None,
    subsample_scaling: str = "fpc",
    subsample_ci: str = "root",
    n_bootstrap: int = 1000,
    target_tpr: float = 0.95,
    n_jobs: int = 1,
    random_state: Optional[int] = 0,
) -> CrossfitReport:
    """Compare AUROC of feature-based OOD scorers with and without group leakage.

    Protocols
    ---------
    ``protocol="default"`` (leaky vs cross-fit on the *same* ID-eval samples):

    * **Leaky**: one scorer fitted on *all* train; scores every ID-eval and OOD
      sample (what a standard sample-level split reports).
    * **Cross-fit**: ID-eval *groups* are split into ``K`` folds. For fold k a
      fresh scorer is fitted on the train samples whose group is **not** among
      fold k's ID-eval groups; it scores fold k's ID-eval samples and all OOD
      samples. Per-fit (matched) pooling compares each ID sample only with OOD
      scored by the *same* model:
      ``AUROC_xfit = sum_k n_k AUROC(ID_k vs OOD | model_k) / sum_k n_k``,
      averaged over ``n_repeats`` fold partitions.
    * ``n_splits=None`` (default): leave-one-group-out if there are at most 50
      ID-eval groups, otherwise K = 10 (:func:`default_n_splits`). Delta
      depends on the number of groups per fit, so many folds keep each
      cross-fitted model close to the leaky one; an integer gives
      ``K = min(n_splits, n_id_eval_groups)``.
    * ``pooling="naive"`` (NOT recommended, warns): compares every ID sample
      with the OOD scores *averaged over* the K cross-fit models. Averaging
      shrinks the spread of the OOD scores, so the result is not an AUROC of any
      fitted model and can be biased. Kept only to quantify that bias.
    * Cross-fit models see ~(K-1)/K of train; ``match_fit_size=True`` replaces
      the leaky arm by size-matched random-subset fits pooled the same way.

    ``protocol="paper_2fold"`` (seen vs unseen groups under the *same* model;
    reproduces the paper's Delta_fit):

    * The *train groups* are split into 2 folds (random, by group count). For
      each fold f one scorer is fitted on the train samples of fold-f groups.
      **Seen arm**: ID-eval samples whose group is in fold f, vs OOD.
      **Unseen arm**: ID-eval samples whose group is in the other fold, vs OOD.
      Both arms use the same model and OOD scores. AUROCs are averaged with
      equal weight over the 2 folds (and over ``n_repeats`` partitions).
      ``auroc_leaky`` := seen, ``auroc_crossfit`` := unseen.
    * Seen definition: *strict* by default, i.e. an ID-eval sample is seen by
      the model whose fit set contains its group. ID-eval samples whose group
      never occurs in train are not used (``meta['n_id_eval_unused']``). The
      paper's alternative definition, in which such groups are assigned to a
      fold and counted as seen for it, is selected with ``fold_ids_eval``.
      ``n_splits``, ``pooling`` and ``match_fit_size`` do not apply.
    * Differences to ``default`` (numbers differ by design): models are fitted
      on half of the train groups (default: all / (K-1)/K); the two arms
      compare *different* ID samples (different groups, so group heterogeneity
      enters the contrast) under the *same* model, whereas default compares the
      *same* ID samples under *different* models; no sample-size confound
      between arms in paper_2fold.

    Only the scorer is refitted; features (backbone) are taken as given, so
    both protocols measure the **scorer-fit channel** only.

    Uncertainty (``uncertainty=``)
    ------------------------------
    The CI target is the sampling distribution of Delta over datasets with new
    groups, i.e. the whole pipeline (fits and fold partition) is random.

    * ``"jackknife"`` (default): refit-aware grouped delete-a-block jackknife.
      In synthetic simulations it reached nominal coverage of the true Delta
      (conservative near Delta = 0), unlike the fixed-score bootstrap.
      ID side: units = groups (a group's train *and* ID-eval samples together;
      for paper_2fold only train groups), split into ``min(n_jackknife_blocks,
      n_units)`` random blocks; each block is deleted and the whole pipeline is
      refitted with fold membership of the remaining groups held fixed. OOD
      side: OOD groups (or samples) in blocks, deleted with fitted scores held
      fixed (exact: OOD never enters a fit). ``V = V_id + V_ood``; CI =
      estimate +/- z*sqrt(V). ``jackknife_formula="weighted"`` (default) uses
      the weighted delete-m_j jackknife of Busing et al. (1999) with m_j =
      number of *units* (groups) in block j, n = total units: correct when the
      blocks hold unequal numbers of groups (n_units > n_jackknife_blocks and
      not divisible). It equals ``"equal"`` ((g-1)/g * sum (theta_j - mean)^2)
      whenever all blocks hold the same number of groups, e.g. one group per
      block. ``"weighted_samples"`` (m_j = samples per block) is kept only as a
      diagnostic: groups, not samples, are the independent units, and this
      variant under-estimates the variance with unequal group sizes. All three
      SEs are reported in ``extra``. Cost: ``n_blocks * fits`` extra fits.
    * ``"subsample"``: group subsampling without replacement. Each of
      ``n_subsamples`` replicates draws ``m`` of the n ID-side units (default
      m = n // 2; ``subsample_size`` int, or float fraction) and the same
      fraction of OOD units, **recomputes the fold partition inside the
      subsample** (random, so partition noise is part of the variability; with
      user ``fold_ids`` the given map is restricted instead) and refits
      everything. Root-of-quantiles CI (Politis & Romano, 1994): with
      ``d_b = theta_b - theta`` and scale ``c``,
      ``CI = [theta - c*q_{1-a/2}(d), theta - c*q_{a/2}(d)]``, ``SE = c*sd(d)``.
      ``subsample_scaling="fpc"`` (default): ``c = sqrt(m/(n-m))`` — finite-
      sampling-fraction correction, exact for linear statistics because
      Var(theta_m - theta_n) = sigma^2 (1/m - 1/n). ``"politis_romano"``:
      ``c = sqrt(m/n)``, the m/n -> 0 asymptotic form, which is too narrow by
      sqrt(1 - m/n) when m = n/2. Both root CIs are reported in ``extra``.
      ``subsample_ci="normal"``: theta +/- z*c*sd(d) instead (centred; ignores
      the mean of d). WARNING: Delta depends on the number of groups (its
      expectation at m = n/2 groups differs from that at n), so the root
      ``d_b`` is not centred at 0 and the root CI is shifted. ``meta`` reports
      the mean shift of the subsample Delta. In synthetic simulations the root
      CI under-covered badly when Delta > 0. Kept for reference, NOT recommended.
    * ``"bootstrap"`` (opt-in; fast, under-covers when Delta > 0): paired
      group-cluster bootstrap with *fitted scores held fixed* (ID-eval groups
      resampled; OOD groups if ``groups_ood``, else OOD samples); percentile
      CI. It ignores refit variability, so its CI is too narrow whenever
      Delta > 0. Use for quick looks, not for reported CIs; a warning is issued.

    Parameters
    ----------
    features_train, groups_train : (n_tr, d) array, (n_tr,) group ids
    features_id_eval, groups_id_eval : (n_id, d) array, (n_id,) group ids
    features_ood : (n_ood, d) array
    labels_train : optional class labels (class-conditional Mahalanobis). The
        paper's Delta_fit tables use class-agnostic ``mahalanobis_l2``, i.e. no labels.
    scorers : registry name(s), scorer instance(s) or ``{name: scorer}``;
        instances are deep-copied before every fit. Default
        ``("mahalanobis_l2", "knn_mean_cosine")``.
    groups_ood : optional OOD group ids (resampling / jackknife units)
    protocol : ``"default"`` or ``"paper_2fold"`` (see above)
    n_splits : folds over ID-eval groups (default protocol); None = leave-one-group-out
        if <= 50 ID-eval groups, else 10
    n_repeats : number of independent fold partitions (both protocols)
    pooling : ``"per_fit"`` (default) or ``"naive"`` (default protocol only; warns)
    match_fit_size : size-matched leaky arm (default protocol only)
    fold_ids : user fold per ID-eval sample (array, or list per repeat; must be
        constant within groups); for protocol="paper_2fold": fold 0 / 1 per TRAIN
        sample (e.g. the paper's fixed folds); the jackknife keeps this map
    fold_ids_eval : protocol="paper_2fold" with ``fold_ids`` only: fold (0 / 1, or -1 = none)
        per ID-eval sample. ID-eval groups absent from the training set but given a fold here
        are kept and counted as seen for that fold (and become jackknife units); train groups
        must get the same fold as in ``fold_ids``
    uncertainty, ci_level, n_jackknife_blocks, jackknife_formula, n_subsamples,
    subsample_size, subsample_scaling, subsample_ci, n_bootstrap : see *Uncertainty*
    target_tpr : seen-group TPR defining the transfer threshold
    n_jobs : parallel workers (joblib) for jackknife / subsample refits
    random_state : seed; results are deterministic given it (any n_jobs)

    Returns
    -------
    CrossfitReport (``.results[name]`` -> :class:`ScoreResult`, ``.summary()``,
    ``.to_dict()``).
    """
    # ---------------------------------------------------------------- validation
    xtr = _check_features("features_train", features_train)
    d = xtr.shape[1]
    xid = _check_features("features_id_eval", features_id_eval, d)
    xood = _check_features("features_ood", features_ood, d)
    gtr, gid = np.asarray(groups_train), np.asarray(groups_id_eval)
    if len(gtr) != len(xtr):
        raise ValueError("groups_train length does not match features_train")
    if len(gid) != len(xid):
        raise ValueError("groups_id_eval length does not match features_id_eval")
    ytr = None if labels_train is None else np.asarray(labels_train)
    if ytr is not None and len(ytr) != len(xtr):
        raise ValueError("labels_train length does not match features_train")
    good = None if groups_ood is None else np.asarray(groups_ood)
    if good is not None and len(good) != len(xood):
        raise ValueError("groups_ood length does not match features_ood")
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    if uncertainty not in UNCERTAINTY_METHODS:
        raise ValueError(f"uncertainty must be one of {UNCERTAINTY_METHODS}")
    if jackknife_formula not in ("weighted", "equal", "weighted_samples"):
        raise ValueError("jackknife_formula must be 'weighted', 'equal' or 'weighted_samples'")
    if subsample_ci not in ("root", "normal"):
        raise ValueError("subsample_ci must be 'root' or 'normal'")
    if subsample_scaling not in ("fpc", "politis_romano"):
        raise ValueError("subsample_scaling must be 'fpc' or 'politis_romano'")
    if not 0.0 < target_tpr < 1.0:
        raise ValueError("target_tpr must be in (0, 1)")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}")
    if pooling == "naive":
        if protocol != "default" or match_fit_size:
            raise ValueError("pooling='naive' is only defined for protocol='default' without match_fit_size")
        warnings.warn("pooling='naive' compares ID scores with OOD scores averaged over the cross-fit "
                      "models; this is not the AUROC of any fitted model and can be biased. Use "
                      "pooling='per_fit' for reported results.", UserWarning, stacklevel=2)
    if uncertainty == "bootstrap" and n_bootstrap > 0:
        warnings.warn("uncertainty='bootstrap' (fixed-score bootstrap) is fast but under-covers when "
                      "Delta > 0; use the default 'jackknife' for reported intervals.", UserWarning,
                      stacklevel=2)
    if n_splits is not None and n_splits < 2:
        raise ValueError("n_splits must be >= 2 (or None for the default fold rule)")
    scorer_map = _resolve_scorers(scorers)
    meta: Dict[str, object] = {"n_train": len(xtr), "n_id_eval": len(xid), "n_ood": len(xood),
                               "n_features": d}

    paper_user_map = None
    if protocol == "paper_2fold":
        if match_fit_size:
            raise ValueError("match_fit_size is not supported with protocol='paper_2fold'")
        if fold_ids_eval is not None and fold_ids is None:
            raise ValueError("fold_ids_eval needs fold_ids (per train sample)")
        fe = None
        if fold_ids_eval is not None:
            fe = np.asarray(fold_ids_eval, dtype=int)
            if fe.shape != gid.shape or not np.isin(fe, (-1, 0, 1)).all():
                raise ValueError("fold_ids_eval must be -1 / 0 / 1, one per ID-eval sample")
        used = np.isin(gid, gtr) | (fe >= 0 if fe is not None else False)
        meta["n_id_eval_unused"] = int((~used).sum())
        xid, gid = xid[used], gid[used]
        if fe is not None:
            fe = fe[used]
        if len(np.unique(gtr)) < 2:
            raise ValueError("paper_2fold needs at least 2 train groups")
        if fold_ids is not None:
            # paper_2fold: fold_ids = fold (0 / 1) per TRAIN sample, whole groups per fold
            paper_user_map = _fold_ids_to_maps(fold_ids, gtr)
            if any(sorted(set(m.values())) != [0, 1] for m in paper_user_map):
                raise ValueError("paper_2fold fold_ids must use exactly the folds 0 and 1")
            if fe is not None:
                if len(paper_user_map) != 1:
                    raise ValueError("fold_ids_eval supports a single fold_ids array")
                m = paper_user_map[0]
                known = _apply_map(m, gid)
                if np.any((known >= 0) & (fe >= 0) & (known != fe)):
                    raise ValueError("fold_ids_eval disagrees with fold_ids on a train group")
                new = known < 0
                ev = _fold_ids_to_maps(fe[new], gid[new])[0] if new.any() else {}
                m.update(ev)
                meta["n_id_eval_orphan"] = int(new.sum())
                meta["n_groups_orphan"] = len(ev)
    user_folds = fold_ids is not None
    if paper_user_map is not None:
        fold_maps = paper_user_map
    elif user_folds:
        fold_maps = _fold_ids_to_maps(fold_ids, gid)
    if protocol == "default" and n_splits is None and not user_folds:
        n_splits = default_n_splits(len(np.unique(gid)))
        meta["fold_rule"] = ("leave-one-group-out" if n_splits == len(np.unique(gid))
                             else f"K={n_splits} (more than {LOGO_MAX_GROUPS} ID-eval groups)")
    if not user_folds:
        fold_maps = _new_fold_maps(protocol, gtr, gid, n_splits, n_repeats, random_state)
    n_rep = len(fold_maps)
    k = max(fold_maps[0].values()) + 1
    fit_seed = None if random_state is None else random_state + 2

    # ---------------------------------------------------------------- point estimates
    fits = {name: _fit_scorer(p, xtr, gtr, ytr, xid, gid, xood, protocol, fold_maps,
                              match_fit_size, target_tpr, fit_seed, pooling)
            for name, p in scorer_map.items()}
    point = {n: f.point(len(xid), len(xood)) for n, f in fits.items()}
    theta = {n: np.array([p[0], p[1], p[0] - p[1]]) for n, p in point.items()}
    ci: Dict[str, np.ndarray] = {}   # name -> (3, 2) lo/hi for (leaky, xfit, delta)
    se: Dict[str, float] = {}
    extra: Dict[str, Dict[str, float]] = {n: {} for n in fits}
    rng = np.random.default_rng(None if random_state is None else random_state + 1)
    ood_units_lab = np.arange(len(xood)) if good is None else good
    ood_units = np.unique(ood_units_lab)
    id_units = (np.unique(np.concatenate([gtr, gid])) if protocol == "default" or fold_ids_eval is not None
                else np.unique(gtr))

    # ---------------------------------------------------------------- uncertainty
    if uncertainty == "bootstrap" and n_bootstrap > 0:
        _, id_codes = np.unique(gid, return_inverse=True)
        w_id = _cluster_weights(id_codes, int(id_codes.max()) + 1, n_bootstrap, rng)
        _, ood_codes = np.unique(ood_units_lab, return_inverse=True)
        w_ood = _cluster_weights(ood_codes, int(ood_codes.max()) + 1, n_bootstrap, rng)
        for n, f in fits.items():
            bl, bx = [], []
            for i in range(0, n_bootstrap, _CHUNK):
                a_l, a_x = f.stats(w_id[i:i + _CHUNK], w_ood[i:i + _CHUNK])
                bl.append(a_l); bx.append(a_x)
            reps = np.stack([np.concatenate(bl), np.concatenate(bx)], axis=1)
            reps = np.concatenate([reps, reps[:, :1] - reps[:, 1:2]], axis=1)
            ci[n] = np.array([_percentile_ci(reps[:, j], ci_level) for j in range(3)])
            se[n] = float(np.nanstd(reps[:, 2], ddof=1))
        meta["n_bootstrap"] = n_bootstrap

    elif uncertainty == "jackknife":
        id_blocks = _blocks(id_units, n_jackknife_blocks, rng)
        ood_blocks = _blocks(ood_units, n_jackknife_blocks, rng)

        def _del_id(blk):
            kt, ke = ~np.isin(gtr, blk), ~np.isin(gid, blk)
            return _point_estimates(scorer_map, xtr[kt], gtr[kt], None if ytr is None else ytr[kt],
                                    xid[ke], gid[ke], xood, protocol, fold_maps, match_fit_size,
                                    target_tpr, fit_seed, pooling)

        id_res = _parallel(_del_id, [(b,) for b in id_blocks], n_jobs)
        m_id_u = np.array([len(b) for b in id_blocks], dtype=float)  # groups per block
        m_id_s = np.array([np.isin(gtr, b).sum() + np.isin(gid, b).sum() for b in id_blocks], dtype=float)
        w_ood_del = np.ones((len(ood_blocks), len(xood)))
        for j, b in enumerate(ood_blocks):
            w_ood_del[j, np.isin(ood_units_lab, b)] = 0.0
        m_ood_u = np.array([len(b) for b in ood_blocks], dtype=float)
        m_ood_s = len(xood) - w_ood_del.sum(axis=1)
        z = NormalDist().inv_cdf(0.5 + ci_level / 2)
        for n, f in fits.items():
            th_id = np.array([[r[n][0], r[n][1], r[n][2]] for r in id_res])
            a_l, a_x = f.stats(np.ones((len(ood_blocks), len(xid))), w_ood_del)
            th_ood = np.stack([a_l, a_x, a_l - a_x], axis=1)
            v = {
                "equal": _jk_var_equal(th_id) + _jk_var_equal(th_ood),
                "weighted": _jk_var_weighted(theta[n], th_id, m_id_u)
                + _jk_var_weighted(theta[n], th_ood, m_ood_u),
                "weighted_samples": _jk_var_weighted(theta[n], th_id, m_id_s)
                + _jk_var_weighted(theta[n], th_ood, m_ood_s),
            }
            sd = np.sqrt(v[jackknife_formula])
            ci[n] = np.stack([theta[n] - z * sd, theta[n] + z * sd], axis=1)
            se[n] = float(sd[2])
            extra[n].update({f"jk_se_{k}": float(np.sqrt(vv[2])) for k, vv in v.items()})
            extra[n].update(jk_se_id_equal=float(np.sqrt(_jk_var_equal(th_id)[2])),
                            jk_se_ood_equal=float(np.sqrt(_jk_var_equal(th_ood)[2])))
        meta.update(n_jackknife_blocks_id=len(id_blocks), n_jackknife_blocks_ood=len(ood_blocks),
                    jackknife_formula=jackknife_formula)

    elif uncertainty == "subsample":
        n_u = len(id_units)
        if subsample_size is None:
            m_u = n_u // 2
        elif isinstance(subsample_size, float) and 0 < subsample_size < 1:
            m_u = int(round(subsample_size * n_u))
        else:
            m_u = int(subsample_size)
        if not 2 <= m_u < n_u:
            raise ValueError(f"subsample_size must give 2 <= m < n_units={n_u}, got {m_u}")
        frac = m_u / n_u
        m_o = max(1, int(round(frac * len(ood_units))))
        seeds = rng.integers(0, 2**31 - 1, size=(n_subsamples, 3))

        def _sub(s0, s1, s2):
            r = np.random.default_rng(s0)
            us = id_units[r.choice(n_u, m_u, replace=False)]
            os_ = ood_units[np.random.default_rng(s1).choice(len(ood_units), m_o, replace=False)]
            kt, ke, ko = np.isin(gtr, us), np.isin(gid, us), np.isin(ood_units_lab, os_)
            if len(np.unique(gid[ke])) < 2 or kt.sum() < 2:
                return None
            maps = fold_maps if user_folds else _new_fold_maps(protocol, gtr[kt], gid[ke], n_splits,
                                                              n_rep, int(s2))
            try:
                return _point_estimates(scorer_map, xtr[kt], gtr[kt], None if ytr is None else ytr[kt],
                                        xid[ke], gid[ke], xood[ko], protocol, maps, match_fit_size,
                                        target_tpr, fit_seed, pooling)
            except ValueError:
                return None

        sub_res = [r for r in _parallel(_sub, [tuple(int(v) for v in s) for s in seeds], n_jobs)
                   if r is not None]
        c_fpc, c_pr = np.sqrt(m_u / (n_u - m_u)), np.sqrt(m_u / n_u)
        a = (1.0 - ci_level) / 2.0
        for n in fits:
            dev = np.array([[r[n][0], r[n][1], r[n][2]] for r in sub_res]) - theta[n]
            q_lo, q_hi = np.quantile(dev, a, axis=0), np.quantile(dev, 1 - a, axis=0)
            sd = np.std(dev, axis=0, ddof=1)
            ci_f = np.stack([theta[n] - c_fpc * q_hi, theta[n] - c_fpc * q_lo], axis=1)
            ci_p = np.stack([theta[n] - c_pr * q_hi, theta[n] - c_pr * q_lo], axis=1)
            c_sel = c_fpc if subsample_scaling == "fpc" else c_pr
            if subsample_ci == "root":
                ci[n] = ci_f if subsample_scaling == "fpc" else ci_p
            else:
                zq = NormalDist().inv_cdf(0.5 + ci_level / 2)
                ci[n] = np.stack([theta[n] - zq * c_sel * sd, theta[n] + zq * c_sel * sd], axis=1)
            se[n] = float(c_sel * sd[2])
            extra[n].update(ss_se_fpc=float(c_fpc * sd[2]), ss_se_pr=float(c_pr * sd[2]),
                            ss_ci_fpc_lo=float(ci_f[2, 0]), ss_ci_fpc_hi=float(ci_f[2, 1]),
                            ss_ci_pr_lo=float(ci_p[2, 0]), ss_ci_pr_hi=float(ci_p[2, 1]))
        meta.update(n_subsamples=n_subsamples, n_subsamples_used=len(sub_res), subsample_m=m_u,
                    subsample_n=n_u, subsample_scaling=subsample_scaling, subsample_ci=subsample_ci,
                    subsample_mean_shift_delta={n: float(np.mean([r[n][2] for r in sub_res]) - theta[n][2])
                                                for n in fits})

    # ---------------------------------------------------------------- assemble
    results: Dict[str, ScoreResult] = {}
    nan = float("nan")
    for n, f in fits.items():
        hs, ns, hn, nn, ho, no, used_f = f.tpr_counts
        tpr_new = hn / nn if nn else nan
        tt = TPRTransfer(target_tpr, hs / ns if ns else nan, tpr_new, 1.0 - tpr_new,
                         (1.0 - tpr_new) / (1.0 - target_tpr), ho / no if no else nan, ns, nn, used_f)
        c = ci.get(n, np.full((3, 2), nan))
        results[n] = ScoreResult(
            name=n, auroc_leaky=float(theta[n][0]), auroc_crossfit=float(theta[n][1]),
            delta=float(theta[n][2]), delta_ci=(float(c[2, 0]), float(c[2, 1])),
            auroc_leaky_ci=(float(c[0, 0]), float(c[0, 1])),
            auroc_crossfit_ci=(float(c[1, 0]), float(c[1, 1])),
            delta_se=se.get(n, nan), tpr_transfer=tt, extra=extra[n])
    meta["n_id_eval_seen_in_train"] = int(np.isin(gid, gtr).sum())
    return CrossfitReport(overlap=group_overlap(gtr, np.asarray(groups_id_eval)),
                          results=results, protocol=protocol, uncertainty=uncertainty, n_splits=k,
                          n_repeats=n_rep, match_fit_size=match_fit_size, ci_level=ci_level,
                          n_groups_train=int(len(np.unique(gtr))),
                          n_groups_id_eval=int(len(np.unique(gid))),
                          n_groups_shared=int(np.isin(np.unique(gid), gtr).sum()),
                          n_groups_fit=_n_groups_fit(protocol, gtr, gid, fold_maps),
                          pooling=pooling, meta=meta)

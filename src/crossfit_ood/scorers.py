"""Feature-based OOD scorers and a minimal plug-in registry.

Every scorer follows the :class:`Scorer` protocol::

    scorer.fit(features, labels=None) -> scorer
    scorer.score(features) -> np.ndarray  # shape (n,), HIGHER = MORE IN-DISTRIBUTION

All computations are done in float64. Scorers must be deep-copyable: the
evaluation code calls ``copy.deepcopy`` on an *unfitted* scorer before every
fit (one leaky fit + one fit per cross-fit fold), so constructor arguments act
as the scorer's hyper-parameters.

Registered scorers
------------------
``knn_mean_cosine`` (default)  mean cosine distance to the k = 50 nearest fit features
``mahalanobis_l2`` (default)   Ledoit-Wolf Mahalanobis on L2-normalised features
``knn_kth``                    distance to the k-th nearest fit feature (Sun et al., 2022)
``mahalanobis``                Ledoit-Wolf Mahalanobis on raw features (Lee et al., 2018)
``vim``                        Virtual-logit Matching (Wang et al., 2022); see the warning there
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol, runtime_checkable

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "Scorer",
    "KNNMeanCosineScorer",
    "MahalanobisL2Scorer",
    "KNNScorer",
    "MahalanobisScorer",
    "ViMScorer",
    "DEFAULT_SCORERS",
    "register_scorer",
    "get_scorer",
    "available_scorers",
]

DEFAULT_SCORERS = ("mahalanobis_l2", "knn_mean_cosine")


@runtime_checkable
class Scorer(Protocol):
    """Protocol for OOD scorers (higher score = more ID-like)."""

    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> "Scorer":
        ...

    def score(self, features: np.ndarray) -> np.ndarray:
        ...


_REGISTRY: Dict[str, Callable[..., Scorer]] = {}


def register_scorer(name: str) -> Callable[[Callable[..., Scorer]], Callable[..., Scorer]]:
    """Class/factory decorator that makes a scorer available by name.

    Example::

        @register_scorer("my_score")
        class MyScorer:
            def fit(self, features, labels=None): ...; return self
            def score(self, features): ...
    """

    def deco(factory: Callable[..., Scorer]) -> Callable[..., Scorer]:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"scorer {name!r} is already registered")
        _REGISTRY[key] = factory
        return factory

    return deco


def get_scorer(name: str, **kwargs) -> Scorer:
    """Instantiate a registered scorer by name (case-insensitive)."""
    try:
        factory = _REGISTRY[name.lower()]
    except KeyError:
        raise KeyError(f"unknown scorer {name!r}; available: {available_scorers()}") from None
    return factory(**kwargs)


def available_scorers() -> list:
    """Names of all registered scorers."""
    return sorted(_REGISTRY)


def _as_f64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be 2-D (n_samples, n_features), got shape {x.shape}")
    return x


def _l2(x: np.ndarray) -> np.ndarray:
    """L2 normalisation used by the paper's scorers: ``x / (||x|| + 1e-8)``."""
    x = _as_f64(x)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


# --------------------------------------------------------------------------- optional torch path
_CHUNK = 2048


def _torch(device: str):
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised only without torch
        raise ImportError("device=... needs PyTorch; install with: pip install 'crossfit-ood[gpu]'") from None
    return torch, torch.device(device)


def _torch_knn(fit_x: np.ndarray, query_x: np.ndarray, k: int, device: str, metric: str) -> np.ndarray:
    """Exact brute-force k-NN in float64 with torch (no approximate search).

    ``metric="cosine"``: rows are renormalised to unit length and the distance is
    ``clip(1 - cos, 0, 2)`` as in scikit-learn; returns the mean over the k nearest.
    ``metric="euclidean"``: returns the distance to the k-th nearest.
    """
    torch, dev = _torch(device)
    t = torch.from_numpy(np.ascontiguousarray(fit_x, dtype=np.float64)).to(dev)
    q_all = torch.from_numpy(np.ascontiguousarray(query_x, dtype=np.float64)).to(dev)
    if metric == "cosine":
        t = t / torch.linalg.norm(t, dim=1, keepdim=True)
        q_all = q_all / torch.linalg.norm(q_all, dim=1, keepdim=True)
    else:
        tn = (t * t).sum(1)
    out = torch.empty(q_all.shape[0], dtype=torch.float64, device=dev)
    for i in range(0, q_all.shape[0], _CHUNK):
        q = q_all[i:i + _CHUNK]
        if metric == "cosine":
            sim = torch.topk(q @ t.T, k, dim=1, largest=True).values
            out[i:i + _CHUNK] = torch.clamp(1.0 - sim, 0.0, 2.0).mean(1)
        else:
            d2 = (q * q).sum(1, keepdim=True) - 2.0 * q @ t.T + tn[None, :]
            kth = torch.topk(d2, k, dim=1, largest=False).values[:, -1]
            out[i:i + _CHUNK] = torch.sqrt(torch.clamp(kth, min=0.0))
    return out.cpu().numpy()


# --------------------------------------------------------------------------- k-NN
@register_scorer("knn_mean_cosine")
class KNNMeanCosineScorer:
    """k-NN score used in the paper (default kNN scorer).

    Features are L2-normalised (``x / (||x|| + 1e-8)``);
    ``score = -mean cosine distance to the k nearest fit samples`` (exact search,
    scikit-learn ``NearestNeighbors(metric="cosine")``). ``k`` (default 50) is
    capped at the number of fit samples. Labels are ignored.

    ``device`` (e.g. ``"cuda"``): exact brute-force search with PyTorch in
    float64 instead of scikit-learn (optional extra ``crossfit-ood[gpu]``).
    Same definition; scores agree with the CPU path to floating-point rounding.
    """

    def __init__(self, k: int = 50, device: Optional[str] = None) -> None:
        self.k = k
        self.device = device

    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> "KNNMeanCosineScorer":
        x = _l2(features)
        self.k_ = int(min(self.k, x.shape[0]))
        if self.device is None:
            self.nn_ = NearestNeighbors(n_neighbors=self.k_, metric="cosine", algorithm="auto").fit(x)
        else:
            self.fit_x_ = x
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        q = _l2(features)
        if self.device is None:
            dist, _ = self.nn_.kneighbors(q, n_neighbors=self.k_)
            return -dist.mean(axis=1)
        return -_torch_knn(self.fit_x_, q, self.k_, self.device, "cosine")


@register_scorer("knn_kth")
class KNNScorer:
    """Deep k-NN score (Sun et al., 2022): L2-normalise, ``score = -distance to the k-th NN``.

    Alternative to the default :class:`KNNMeanCosineScorer` (k-th neighbour
    instead of the mean over k neighbours). Normalisation ``x / max(||x||, 1e-12)``;
    Euclidean distance on unit vectors is monotone in cosine distance. ``k`` is
    capped at the number of fit samples. Labels are ignored. ``device``: optional
    exact PyTorch search, as for :class:`KNNMeanCosineScorer`.
    """

    def __init__(self, k: int = 50, normalize: bool = True, device: Optional[str] = None) -> None:
        self.k = k
        self.normalize = normalize
        self.device = device

    def _prep(self, x: np.ndarray) -> np.ndarray:
        x = _as_f64(x)
        if self.normalize:
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return x

    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> "KNNScorer":
        x = self._prep(features)
        self.k_ = int(min(self.k, x.shape[0]))
        if self.device is None:
            self.nn_ = NearestNeighbors(n_neighbors=self.k_).fit(x)
        else:
            self.fit_x_ = x
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        q = self._prep(features)
        if self.device is None:
            dist, _ = self.nn_.kneighbors(q, n_neighbors=self.k_)
            return -dist[:, -1]
        return -_torch_knn(self.fit_x_, q, self.k_, self.device, "euclidean")


# --------------------------------------------------------------------------- Mahalanobis
@register_scorer("mahalanobis_l2")
class MahalanobisL2Scorer:
    """Mahalanobis score on L2-normalised features (default Mahalanobis scorer).

    Features are L2-normalised (``x / (||x|| + 1e-8)``) and handled in float64;
    the covariance is estimated with Ledoit-Wolf shrinkage.

    * With ``labels`` (and ``class_conditional=True``, the default):
      class-conditional means and one covariance shared across classes,
      estimated by Ledoit-Wolf on class-centred features;
      ``score = -sqrt(min_c (x - mu_c)^T P (x - mu_c))``.
    * Without ``labels`` (or ``class_conditional=False``): one class-agnostic
      mean, ``LedoitWolf()`` centred at the mean;
      ``score = -sqrt((x - mu)^T P (x - mu))``. This is the definition used for
      the paper's Delta_fit tables, so pass no labels to reproduce them.

    Negative quadratic forms from rounding are clipped at 0 before the square root.
    Classes absent from the fit set are simply absent (min over present classes).
    """

    def __init__(self, class_conditional: bool = True) -> None:
        self.class_conditional = class_conditional

    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> "MahalanobisL2Scorer":
        x = _l2(features)
        if labels is None or not self.class_conditional:
            lw = LedoitWolf().fit(x)
            self.means_ = np.asarray(lw.location_, dtype=np.float64)[None, :]
        else:
            labels = np.asarray(labels)
            classes = np.unique(labels)
            means = np.stack([x[labels == c].mean(axis=0) for c in classes])
            lw = LedoitWolf(assume_centered=True).fit(x - means[np.searchsorted(classes, labels)])
            self.means_ = means
        self.precision_ = np.asarray(lw.precision_, dtype=np.float64)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        x = _l2(features)
        d2 = np.empty((x.shape[0], self.means_.shape[0]))
        for j, mu in enumerate(self.means_):
            diff = x - mu
            d2[:, j] = np.sum((diff @ self.precision_) * diff, axis=1)
        return -np.sqrt(np.maximum(d2.min(axis=1), 0.0))


@register_scorer("mahalanobis")
class MahalanobisScorer:
    """Mahalanobis score on raw (unnormalised) features with a shared Ledoit-Wolf covariance
    (Lee et al., 2018).

    * With ``labels``: class-conditional means, one covariance shared across
      classes, estimated by Ledoit-Wolf on class-centred features;
      ``score(x) = -min_c (x - mu_c)^T P (x - mu_c)``.
    * Without ``labels``: a single mean (class-agnostic variant).

    Classes absent from the fit set are simply absent (min over present classes).
    """

    def __init__(self) -> None:
        self.means_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> "MahalanobisScorer":
        x = _as_f64(features)
        if labels is None:
            means = x.mean(axis=0, keepdims=True)
            centred = x - means
        else:
            labels = np.asarray(labels)
            classes = np.unique(labels)
            means = np.stack([x[labels == c].mean(axis=0) for c in classes])
            centred = x - means[np.searchsorted(classes, labels)]
        lw = LedoitWolf(assume_centered=True).fit(centred)
        self.means_ = means
        self.precision_ = np.asarray(lw.precision_, dtype=np.float64)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        x = _as_f64(features)
        d2 = np.empty((x.shape[0], self.means_.shape[0]))
        for j, mu in enumerate(self.means_):
            diff = x - mu
            d2[:, j] = np.sum((diff @ self.precision_) * diff, axis=1)
        return -d2.min(axis=1)


# --------------------------------------------------------------------------- ViM
def _default_vim_dim(n_features: int) -> int:
    # Same rule of thumb as the reference ViM implementation (Wang et al., 2022).
    if n_features >= 2048:
        return 1000
    if n_features >= 768:
        return 512
    return n_features // 2


@register_scorer("vim")
class ViMScorer:
    """Virtual-logit Matching (Wang et al., CVPR 2022). Not in the default scorer list.

    What is *fitted* on the fit set (the scorer-fit channel): the principal
    subspace of the (origin-shifted) features and the scale ``alpha``.
    The classifier head ``(weight, bias)`` is treated as part of the frozen
    backbone and is never refitted.

    * With ``weight`` (C, D) and ``bias`` (C,): full ViM. Origin
      ``u = -pinv(W) b``; ``logits = x W^T + b``; ``alpha = mean max-logit /
      mean residual norm`` on the fit set;
      ``score = logsumexp(logits) - alpha * ||residual||``.
    * Without a head: residual-only variant, origin = mean of fit features,
      ``score = -||residual||`` (labels unused).

    ``dim`` (principal-subspace size) defaults to the reference rule
    (1000 if D>=2048, 512 if D>=768, else D//2), capped at ``n_fit - 1``.

    .. warning::
       Numerically unstable at ``dim = D - C`` (D features, C classes): the
       residual is then the span of the C smallest-eigenvalue directions of the
       fit covariance, which are nearly degenerate and are not determined
       stably. Scores can change with the BLAS library or thread count, and
       Delta estimates are not reproducible. Prefer the default ``dim`` and
       report ViM results with this caveat.
    """

    def __init__(self, weight: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None,
                 dim: Optional[int] = None) -> None:
        if (weight is None) != (bias is None):
            raise ValueError("pass both weight and bias, or neither")
        self.weight = weight
        self.bias = bias
        self.dim = dim

    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> "ViMScorer":
        x = _as_f64(features)
        n, d = x.shape
        if self.weight is not None:
            w = np.asarray(self.weight, dtype=np.float64)
            b = np.asarray(self.bias, dtype=np.float64)
            self.origin_ = -np.linalg.pinv(w) @ b
        else:
            self.origin_ = x.mean(axis=0)
        k = self.dim if self.dim is not None else _default_vim_dim(d)
        k = int(max(1, min(k, d - 1, n - 1)))
        z = x - self.origin_
        cov = z.T @ z / max(n - 1, 1)
        evals, evecs = np.linalg.eigh(cov)  # ascending
        self.residual_basis_ = evecs[:, : d - k]  # complement of top-k subspace
        res_norm = np.linalg.norm(z @ self.residual_basis_, axis=1)
        if self.weight is not None:
            max_logit = (x @ w.T + b).max(axis=1)
            self.alpha_ = float(max_logit.mean() / max(res_norm.mean(), 1e-12))
        else:
            self.alpha_ = 1.0
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        x = _as_f64(features)
        res = np.linalg.norm((x - self.origin_) @ self.residual_basis_, axis=1)
        if self.weight is None:
            return -res
        logits = x @ np.asarray(self.weight, np.float64).T + np.asarray(self.bias, np.float64)
        m = logits.max(axis=1, keepdims=True)
        energy = (m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True)))[:, 0]
        return energy - self.alpha_ * res

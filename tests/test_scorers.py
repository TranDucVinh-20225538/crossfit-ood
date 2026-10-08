import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from crossfit_ood import (KNNScorer, MahalanobisScorer, Scorer, ViMScorer, available_scorers,
                          crossfit_auroc, get_scorer, make_grouped_features, register_scorer)


def _easy(rng, d=16, n=400):
    """ID and OOD differ in mean *direction* (so L2-normalised kNN also separates them)."""
    h = d // 2
    xid = rng.normal(size=(n, d))
    xid[:, :h] += 3.0
    y = (xid[:, 0] > 3.0).astype(int)
    xood = rng.normal(size=(n, d))
    xood[:, h:] += 3.0
    return xid, y, xood


def _auroc(s_id, s_ood):
    return roc_auc_score(np.r_[np.ones(len(s_id)), np.zeros(len(s_ood))], np.r_[s_id, s_ood])


@pytest.mark.parametrize("name", ["mahalanobis", "knn_kth", "vim"])
def test_registered_scorers_separate_easy_ood(rng, name):
    xid, y, xood = _easy(rng)
    s = get_scorer(name)
    assert isinstance(s, Scorer)
    s.fit(xid[:200].astype(np.float32), y[:200] if name == "mahalanobis" else None)
    a, b = s.score(xid[200:]), s.score(xood)
    assert a.shape == (200,) and a.dtype == np.float64
    assert _auroc(a, b) > 0.95


def test_mahalanobis_class_conditional_vs_agnostic(rng):
    xid, y, xood = _easy(rng)
    m1 = MahalanobisScorer().fit(xid, y)
    m0 = MahalanobisScorer().fit(xid)
    assert m1.means_.shape[0] == 2 and m0.means_.shape[0] == 1
    assert m1.precision_.dtype == np.float64


def test_vim_with_head_and_residual_only(rng):
    # ID varies strongly in the first 8 dims (principal subspace, where the head lives);
    # OOD is shifted in the low-variance residual dims.
    d, n = 32, 600
    scale = np.r_[np.full(8, 5.0), np.full(d - 8, 0.5)]
    xid = rng.normal(size=(n, d)) * scale
    xood = rng.normal(size=(n, d)) * scale
    xood[:, 8:] += 1.5
    w = np.zeros((3, d))
    w[:, :8] = rng.normal(size=(3, 8))
    b = np.zeros(3)
    full = ViMScorer(weight=w, bias=b, dim=8).fit(xid[:300])
    res = ViMScorer(dim=8).fit(xid[:300])
    assert np.isfinite(full.alpha_) and res.alpha_ == 1.0
    assert _auroc(full.score(xid[300:]), full.score(xood)) > 0.9
    assert _auroc(res.score(xid[300:]), res.score(xood)) > 0.9
    # full ViM = energy - alpha * residual norm
    logits = xid[300:] @ w.T + b
    energy = np.log(np.exp(logits).sum(1))
    resid = np.linalg.norm((xid[300:] - full.origin_) @ full.residual_basis_, axis=1)
    np.testing.assert_allclose(full.score(xid[300:]), energy - full.alpha_ * resid, rtol=1e-10)
    with pytest.raises(ValueError):
        ViMScorer(weight=w)


def test_knn_k_capped(rng):
    x = rng.normal(size=(10, 4))
    s = KNNScorer(k=50).fit(x)
    assert s.k_ == 10 and s.score(x).shape == (10,)


def test_custom_scorer_plugin():
    @register_scorer("test_norm")
    class NormScorer:
        def fit(self, features, labels=None):
            self.mu_ = np.asarray(features, float).mean(0)
            return self

        def score(self, features):
            return -np.linalg.norm(np.asarray(features, float) - self.mu_, axis=1)

    assert "test_norm" in available_scorers()
    with pytest.raises(ValueError):
        register_scorer("test_norm")(NormScorer)
    d = make_grouped_features(0.3, random_state=1)
    rep = crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval, d.groups_id_eval,
                         d.features_ood, scorers=["test_norm", KNNScorer(k=5)],
                         uncertainty="bootstrap", n_bootstrap=50)
    assert set(rep.results) == {"test_norm", "KNNScorer"}
    with pytest.raises(KeyError):
        get_scorer("does_not_exist")

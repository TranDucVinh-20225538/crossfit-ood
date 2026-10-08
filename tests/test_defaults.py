"""Each documented v0.1.0 default is asserted here."""
import inspect

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

import crossfit_ood
from crossfit_ood import (DEFAULT_SCORERS, KNNMeanCosineScorer, KNNScorer, MahalanobisL2Scorer, ViMScorer,
                          available_scorers, crossfit_auroc, default_n_splits, get_scorer,
                          make_grouped_features)
from crossfit_ood.core import PROTOCOLS

SIG = inspect.signature(crossfit_auroc).parameters


def _l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def test_version():
    assert crossfit_ood.__version__ == "0.1.0"


# --------------------------------------------------------------------------- scorers
def test_default_scorer_list():
    assert DEFAULT_SCORERS == ("mahalanobis_l2", "knn_mean_cosine")
    assert SIG["scorers"].default == DEFAULT_SCORERS
    d = make_grouped_features(0.3, n_groups=6, n_ood_groups=6, random_state=0)
    rep = crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval, d.groups_id_eval,
                         d.features_ood, n_jackknife_blocks=3)
    assert list(rep.results) == ["mahalanobis_l2", "knn_mean_cosine"]


def test_knn_default_is_mean_cosine_k50(rng):
    s = get_scorer("knn_mean_cosine")
    assert isinstance(s, KNNMeanCosineScorer) and s.k == 50 and s.device is None
    x, q = rng.normal(size=(200, 6)) * 3, rng.normal(size=(40, 6))
    got = s.fit(x).score(q)
    sims = _l2(q) / np.linalg.norm(_l2(q), axis=1, keepdims=True) @ \
        (_l2(x) / np.linalg.norm(_l2(x), axis=1, keepdims=True)).T
    want = -np.sort(1.0 - sims, axis=1)[:, :50].mean(axis=1)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)


def test_knn_kth_variant_available(rng):
    assert "knn_kth" in available_scorers()
    s = get_scorer("knn_kth")
    assert isinstance(s, KNNScorer) and s.k == 50
    x, q = rng.normal(size=(200, 6)), rng.normal(size=(40, 6))
    xu = x / np.linalg.norm(x, axis=1, keepdims=True)
    qu = q / np.linalg.norm(q, axis=1, keepdims=True)
    want = -NearestNeighbors(n_neighbors=50).fit(xu).kneighbors(qu)[0][:, -1]
    np.testing.assert_allclose(s.fit(x).score(q), want, rtol=0, atol=1e-12)


def test_mahalanobis_default_class_conditional_lw_float64_l2(rng):
    s = get_scorer("mahalanobis_l2")
    assert isinstance(s, MahalanobisL2Scorer) and s.class_conditional
    x = rng.normal(size=(300, 5)).astype(np.float32)
    y = rng.integers(0, 3, 300)
    x[:, 0] += 4 * y
    s.fit(x, y)
    assert s.means_.shape == (3, 5) and s.precision_.dtype == np.float64
    xu = _l2(x.astype(np.float64))
    mu = np.stack([xu[y == c].mean(0) for c in range(3)])
    lw = LedoitWolf(assume_centered=True).fit(xu - mu[y])
    np.testing.assert_allclose(s.precision_, lw.precision_, rtol=1e-12, atol=1e-12)
    q = rng.normal(size=(20, 5))
    d2 = np.stack([np.sum(((_l2(q) - m) @ lw.precision_) * (_l2(q) - m), axis=1) for m in mu], axis=1)
    np.testing.assert_allclose(s.score(q), -np.sqrt(d2.min(axis=1)), rtol=1e-10, atol=1e-12)
    assert s.score(q).dtype == np.float64
    # L2-normalised: invariant to rescaling each sample (up to the 1e-8 guard)
    np.testing.assert_allclose(s.score(7.0 * q), s.score(q), rtol=1e-6, atol=1e-8)


def test_mahalanobis_without_labels_is_class_agnostic(rng):
    x = rng.normal(size=(300, 5))
    y = rng.integers(0, 2, 300)
    a = MahalanobisL2Scorer().fit(x)
    b = MahalanobisL2Scorer(class_conditional=False).fit(x, y)
    assert a.means_.shape == (1, 5)
    np.testing.assert_array_equal(a.score(x), b.score(x))
    lw = LedoitWolf().fit(_l2(x))
    np.testing.assert_allclose(a.precision_, lw.precision_, rtol=1e-12, atol=1e-12)


def test_vim_available_but_not_default():
    assert "vim" in available_scorers() and "vim" not in DEFAULT_SCORERS
    doc = ViMScorer.__doc__
    assert "unstable" in doc.lower() and "dim = D - C" in doc


# --------------------------------------------------------------------------- pooling
def test_default_pooling_is_per_fit():
    assert SIG["pooling"].default == "per_fit"
    d = make_grouped_features(0.3, n_groups=6, n_ood_groups=6, random_state=0)
    args = (d.features_train, d.groups_train, d.features_id_eval, d.groups_id_eval, d.features_ood)
    rep = crossfit_auroc(*args, n_jackknife_blocks=3)
    assert rep.pooling == "per_fit" and "pooling=per_fit" in rep.summary()
    with pytest.warns(UserWarning, match="naive"):
        crossfit_auroc(*args, pooling="naive", uncertainty="bootstrap", n_bootstrap=0)
    with pytest.raises(ValueError):
        crossfit_auroc(*args, pooling="naive", protocol="paper_2fold", n_bootstrap=0)
    with pytest.raises(ValueError):
        crossfit_auroc(*args, pooling="something")


# --------------------------------------------------------------------------- folds
def test_fold_rule_logo_up_to_50_groups_else_10():
    assert SIG["n_splits"].default is None
    assert [default_n_splits(n) for n in (2, 12, 50, 51, 400)] == [2, 12, 50, 10, 10]


@pytest.mark.parametrize("n_groups,expect_k", [(12, 12), (50, 50), (60, 10)])
def test_fold_rule_applied(n_groups, expect_k):
    d = make_grouped_features(0.3, n_groups=n_groups, n_per_group=6, n_features=4, n_ood_groups=6,
                              random_state=0)
    rep = crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval, d.groups_id_eval,
                         d.features_ood, scorers=["mahalanobis_l2"], uncertainty="bootstrap", n_bootstrap=0)
    assert rep.n_splits == expect_k
    assert rep.meta["fold_rule"].startswith("leave-one-group-out" if expect_k == n_groups else "K=10")
    if expect_k == n_groups:
        assert rep.n_groups_fit == (n_groups - 1, n_groups - 1)
    else:
        assert rep.n_groups_fit == (n_groups - 6, n_groups - 6)


# --------------------------------------------------------------------------- intervals
def test_default_interval_is_refit_aware_jackknife_busing():
    assert SIG["uncertainty"].default == "jackknife"
    assert SIG["jackknife_formula"].default == "weighted"
    d = make_grouped_features(0.3, n_groups=8, n_ood_groups=8, random_state=0)
    rep = crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval, d.groups_id_eval,
                         d.features_ood, scorers=["mahalanobis_l2"], n_jackknife_blocks=5)
    assert rep.uncertainty == "jackknife" and rep.meta["jackknife_formula"] == "weighted"
    r = rep.results["mahalanobis_l2"]
    assert r.delta_se == pytest.approx(r.extra["jk_se_weighted"])


def test_bootstrap_is_opt_in_and_warns():
    d = make_grouped_features(0.3, n_groups=6, n_ood_groups=6, random_state=0)
    with pytest.warns(UserWarning, match="under-covers"):
        rep = crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval, d.groups_id_eval,
                             d.features_ood, scorers=["mahalanobis_l2"], uncertainty="bootstrap",
                             n_bootstrap=50)
    assert rep.uncertainty == "bootstrap"
    assert "under-covers" in crossfit_auroc.__doc__


# --------------------------------------------------------------------------- protocols / seen
def test_protocols():
    assert PROTOCOLS == ("default", "paper_2fold")
    assert SIG["protocol"].default == "default"


def test_seen_definition_strict_by_default():
    assert SIG["fold_ids_eval"].default is None
    d = make_grouped_features(0.3, n_groups=8, n_new_groups=3, random_state=0)
    rep = crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval, d.groups_id_eval,
                         d.features_ood, scorers=["mahalanobis_l2"], protocol="paper_2fold",
                         uncertainty="bootstrap", n_bootstrap=0)
    new = ~np.isin(d.groups_id_eval, d.groups_train)
    assert new.sum() > 0 and rep.meta["n_id_eval_unused"] == new.sum()
    assert rep.n_groups_id_eval == 8


# --------------------------------------------------------------------------- ties
class _RoundScorer:
    """Score = rounded first feature minus the fit mean: many exact ties."""

    def fit(self, x, labels=None):
        self.c_ = float(np.round(np.mean(np.asarray(x)[:, 0])))
        return self

    def score(self, x):
        return np.round(np.asarray(x, float)[:, 0]) - self.c_


def test_ties_count_one_half(rng):
    g = np.repeat(np.arange(5), 12)
    x = np.c_[rng.integers(0, 3, len(g)).astype(float), np.zeros(len(g))]
    xood = np.c_[rng.integers(0, 3, 40).astype(float), np.zeros(40)]
    rep = crossfit_auroc(x, g, x, g, xood, scorers={"r": _RoundScorer()}, uncertainty="bootstrap",
                         n_bootstrap=0)
    r = rep.results["r"]
    m = _RoundScorer().fit(x)
    leaky = roc_auc_score(np.r_[np.ones(len(x)), np.zeros(40)], np.r_[m.score(x), m.score(xood)])
    assert r.auroc_leaky == pytest.approx(leaky, abs=1e-12)
    num = 0.0
    for k in range(5):  # leave-one-group-out, per-fit pooling
        mk = _RoundScorer().fit(x[g != k])
        sk = mk.score(x[g == k])
        num += len(sk) * roc_auc_score(np.r_[np.ones(len(sk)), np.zeros(40)], np.r_[sk, mk.score(xood)])
    assert r.auroc_crossfit == pytest.approx(num / len(x), abs=1e-12)

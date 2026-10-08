import numpy as np
import pytest

from crossfit_ood import crossfit_auroc, group_overlap, make_group_folds, make_grouped_features
from conftest import run


class GroupMemoScorer:
    """Pathological scorer: 1 if the row's group code (column 0) was in the fit set, else 0.

    Leaky AUROC must be 1, cross-fit AUROC exactly 0.5 if no ID-eval sample is ever
    scored by a model that saw its group.
    """

    def fit(self, features, labels=None):
        self.seen_ = set(np.asarray(features)[:, 0].astype(int).tolist())
        return self

    def score(self, features):
        return np.array([1.0 if int(v) in self.seen_ else 0.0 for v in np.asarray(features)[:, 0]])


def _memo_data(n_groups=10, n=6):
    g = np.repeat(np.arange(n_groups), n)
    x = np.c_[g, np.zeros(len(g))].astype(float)
    xood = np.c_[np.full(30, 999.0), np.zeros(30)]
    return x, g, xood


def test_group_overlap():
    o = group_overlap([1, 1, 2, 3], [1, 2, 2, 4, 5, 5])
    assert o.sample_rate == pytest.approx(3 / 6)
    assert o.group_rate == pytest.approx(2 / 4)
    assert (o.n_id_eval_groups, o.n_shared_groups) == (4, 2)


def test_make_group_folds_keeps_groups_whole():
    g = np.repeat(np.arange(7), 3)
    f = make_group_folds(g, n_splits=5, random_state=0)
    assert f.max() + 1 == 5
    for gg in range(7):
        assert len(np.unique(f[g == gg])) == 1
    assert make_group_folds(np.repeat([0, 1, 2], 2), n_splits=5).max() + 1 == 3
    with pytest.raises(ValueError):
        make_group_folds([0, 0, 0])


def test_crossfit_never_scores_with_own_group():
    x, g, xood = _memo_data()
    rep = crossfit_auroc(x, g, x, g, xood, scorers={"memo": GroupMemoScorer()},
                         uncertainty="bootstrap", n_bootstrap=100)
    r = rep.results["memo"]
    assert r.auroc_leaky == 1.0
    assert r.auroc_crossfit == 0.5
    assert r.delta == 0.5
    assert r.tpr_transfer.tpr_seen == 1.0 and r.tpr_transfer.tpr_new == 0.0


def test_no_overlap_gives_exactly_zero_delta():
    d = make_grouped_features(0.6, random_state=0)
    gid_new = d.groups_id_eval + 100_000  # pretend all ID-eval groups are new
    rep = crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval, gid_new,
                         d.features_ood, labels_train=d.labels_train, uncertainty="bootstrap",
                         n_bootstrap=200)
    assert rep.overlap.sample_rate == 0.0
    for r in rep.results.values():
        assert r.delta == 0.0
        assert r.delta_ci == (0.0, 0.0)
    rj = crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval, gid_new,
                        d.features_ood, labels_train=d.labels_train, scorers=["mahalanobis"],
                        uncertainty="jackknife")
    assert rj.results["mahalanobis"].delta == 0.0
    assert rj.results["mahalanobis"].delta_se == 0.0


def test_deterministic_and_serialisable():
    d = make_grouped_features(0.3, random_state=2)
    a = run(d, n_bootstrap=100, random_state=5)
    b = run(d, n_bootstrap=100, random_state=5)
    assert a.to_dict() == b.to_dict()
    assert "mahalanobis" in a.summary()
    assert a.meta["n_id_eval_seen_in_train"] == len(d.features_id_eval)


def test_partial_overlap_and_options():
    d = make_grouped_features(0.4, n_new_groups=8, random_state=3)
    rep = run(d, n_bootstrap=100, n_repeats=2, match_fit_size=True)
    assert 0.0 < rep.overlap.sample_rate < 1.0
    assert rep.n_repeats == 2 and rep.match_fit_size
    for r in rep.results.values():
        assert r.delta_ci[0] <= r.delta_ci[1]
        assert r.tpr_transfer.n_folds_used > 0


def test_user_fold_ids():
    d = make_grouped_features(0.3, random_state=4)
    folds = d.groups_id_eval % 3
    rep = run(d, fold_ids=folds, n_bootstrap=0)
    assert rep.n_splits == 3
    bad = folds.copy()
    bad[0] = (bad[0] + 1) % 3
    with pytest.raises(ValueError, match="split across folds"):
        run(d, fold_ids=bad, n_bootstrap=0)


def test_jackknife_ci_contains_estimate():
    d = make_grouped_features(0.4, n_groups=12, n_ood_groups=12, random_state=0)
    rep = run(d, uncertainty="jackknife", n_jackknife_blocks=12)
    assert rep.uncertainty == "jackknife"
    for r in rep.results.values():
        assert r.delta_ci[0] < r.delta < r.delta_ci[1]
        assert r.delta_se > 0
        # one group per block -> group-weighted Busing == classic formula
        assert r.extra["jk_se_weighted"] == pytest.approx(r.extra["jk_se_equal"], rel=1e-9)


def test_jackknife_parallel_matches_serial():
    d = make_grouped_features(0.4, n_groups=10, n_ood_groups=10, random_state=1)
    a = run(d, uncertainty="jackknife", scorers=["mahalanobis", "knn_kth"], n_jobs=1)
    b = run(d, uncertainty="jackknife", scorers=["mahalanobis", "knn_kth"], n_jobs=2)
    assert a.to_dict() == b.to_dict()


def test_jackknife_formulas_known_answer():
    """Delete-a-block jackknife of a cluster mean vs the analytic cluster-robust variance."""
    from crossfit_ood.core import _jk_var_equal, _jk_var_weighted

    rng = np.random.default_rng(0)
    m_eq = np.ones(5)
    th = rng.normal(size=(5, 3))
    np.testing.assert_allclose(_jk_var_weighted(th.mean(0), th, m_eq), _jk_var_equal(th))
    ratios = []
    for _ in range(300):
        g = 23
        cnt = rng.integers(2, 60, g).astype(float)
        tot = np.array([rng.normal(rng.normal(), 1, int(c)).sum() for c in cnt])
        theta = tot.sum() / cnt.sum()
        analytic = g / (g - 1) * np.sum((tot - cnt * theta) ** 2) / cnt.sum() ** 2
        blocks = np.array_split(rng.permutation(g), 10)  # unequal: 3 or 2 groups per block
        th_del = np.array([[(tot.sum() - tot[b].sum()) / (cnt.sum() - cnt[b].sum())] for b in blocks])
        mu = np.array([len(b) for b in blocks], float)
        ratios.append(_jk_var_weighted(np.array([theta]), th_del, mu)[0] / analytic)
    assert abs(np.mean(ratios) - 1.0) < 0.06


def test_subsample_deterministic_and_scalings():
    d = make_grouped_features(0.4, n_groups=12, n_ood_groups=12, random_state=2)
    kw = dict(uncertainty="subsample", n_subsamples=30, scorers=["mahalanobis", "knn_kth"])
    a = run(d, random_state=3, **kw)
    b = run(d, random_state=3, n_jobs=2, **kw)
    assert a.to_dict() == b.to_dict()
    assert a.meta["subsample_m"] == 6 and a.meta["subsample_n"] == 12
    for r in a.results.values():
        e = r.extra
        assert e["ss_se_fpc"] == pytest.approx(e["ss_se_pr"] * np.sqrt(2.0))  # m = n/2
        assert (e["ss_ci_fpc_hi"] - e["ss_ci_fpc_lo"]) > (e["ss_ci_pr_hi"] - e["ss_ci_pr_lo"])
        assert r.delta_ci == (e["ss_ci_fpc_lo"], e["ss_ci_fpc_hi"])
    c = run(d, random_state=3, subsample_size=0.75, **kw)
    assert c.meta["subsample_m"] == 9


def test_input_validation():
    d = make_grouped_features(0.2, random_state=0)
    with pytest.raises(ValueError):
        crossfit_auroc(d.features_train, d.groups_train[:-1], d.features_id_eval,
                       d.groups_id_eval, d.features_ood)
    with pytest.raises(ValueError):
        crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval[:, :5],
                       d.groups_id_eval, d.features_ood)
    bad = d.features_ood.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval, d.groups_id_eval, bad)
    with pytest.raises(ValueError):
        run(d, uncertainty="magic")
    with pytest.raises(ValueError):
        run(d, protocol="magic")
    with pytest.raises(ValueError):
        run(d, uncertainty="subsample", subsample_size=1)


def test_default_uncertainty_is_jackknife():
    d = make_grouped_features(0.3, n_groups=8, n_ood_groups=8, random_state=0)
    rep = crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval, d.groups_id_eval,
                         d.features_ood, scorers=["mahalanobis"])
    assert rep.uncertainty == "jackknife"
    assert rep.meta["jackknife_formula"] == "weighted"


# --------------------------------------------------------------------------- paper_2fold
def test_paper_2fold_memo_scorer():
    """Seen arm uses groups in the fit half (AUROC 1), unseen arm the other half (0.5)."""
    x, g, xood = _memo_data(n_groups=10)
    rep = crossfit_auroc(x, g, x, g, xood, scorers={"memo": GroupMemoScorer()},
                         protocol="paper_2fold", uncertainty="bootstrap", n_bootstrap=50)
    r = rep.results["memo"]
    assert (r.auroc_leaky, r.auroc_crossfit, r.delta) == (1.0, 0.5, 0.5)
    assert rep.n_splits == 2


def test_paper_2fold_matches_manual_computation():
    from sklearn.metrics import roc_auc_score

    from crossfit_ood import get_scorer
    from crossfit_ood.core import _apply_map, _new_fold_maps

    d = make_grouped_features(0.4, n_new_groups=4, random_state=7)
    rep = run(d, protocol="paper_2fold", scorers=["knn_kth"], n_bootstrap=0, n_repeats=2, random_state=11)
    maps = _new_fold_maps("paper_2fold", d.groups_train, d.groups_id_eval, 5, 2, 11)
    used = np.isin(d.groups_id_eval, d.groups_train)
    xid, gid = d.features_id_eval[used], d.groups_id_eval[used]
    seen_l, unseen_l = [], []
    for fmap in maps:
        tf, idf = _apply_map(fmap, d.groups_train), _apply_map(fmap, gid)
        s_seen, s_unseen = [], []
        for f in (0, 1):
            m = get_scorer("knn_kth").fit(d.features_train[tf == f])
            so = m.score(d.features_ood)
            for mask, acc in ((idf == f, s_seen), (idf == 1 - f, s_unseen)):
                si = m.score(xid[mask])
                acc.append(roc_auc_score(np.r_[np.ones(len(si)), np.zeros(len(so))], np.r_[si, so]))
        seen_l.append(np.mean(s_seen)); unseen_l.append(np.mean(s_unseen))
    r = rep.results["knn_kth"]
    assert r.auroc_leaky == pytest.approx(np.mean(seen_l), abs=1e-12)
    assert r.auroc_crossfit == pytest.approx(np.mean(unseen_l), abs=1e-12)
    assert rep.meta["n_id_eval_unused"] == int((~used).sum()) > 0
    assert rep.n_repeats == 2


def test_paper_2fold_null_and_effect():
    sc = ("mahalanobis", "knn_kth", "vim")
    d0 = [run(make_grouped_features(0.0, random_state=s), protocol="paper_2fold", n_bootstrap=0,
              scorers=sc) for s in range(3)]
    d4 = run(make_grouped_features(0.4, random_state=0), protocol="paper_2fold", n_bootstrap=300,
             scorers=sc)
    for name in ("mahalanobis", "knn_kth", "vim"):
        assert abs(np.mean([r.results[name].delta for r in d0])) < 0.03
        assert d4.results[name].delta_ci[0] > 0.0
    with pytest.raises(ValueError):
        run(make_grouped_features(0.0), protocol="paper_2fold", fold_ids=np.zeros(10, int))


def test_paper_2fold_with_jackknife_and_subsample():
    d = make_grouped_features(0.4, n_groups=10, n_ood_groups=10, random_state=5)
    for u, kw in (("jackknife", {}), ("subsample", {"n_subsamples": 20})):
        rep = run(d, protocol="paper_2fold", uncertainty=u, scorers=["mahalanobis"], **kw)
        r = rep.results["mahalanobis"]
        assert np.isfinite(r.delta_se) and r.delta_ci[0] < r.delta_ci[1]


def test_report_shows_groups_K_protocol():
    d = make_grouped_features(0.3, n_new_groups=4, random_state=0)
    rep = run(d, n_bootstrap=0, n_splits=5)
    dd = rep.to_dict()
    assert (dd["n_groups_train"], dd["n_groups_id_eval"], dd["n_groups_shared"]) == (24, 28, 24)
    assert dd["n_splits"] == 5 and dd["protocol"] == "default"
    txt = rep.summary()
    assert "K=5" in txt and "train=24" in txt and "ID-eval used=28" in txt
    rp = run(d, n_bootstrap=0, protocol="paper_2fold")
    assert (rp.n_splits, rp.n_groups_id_eval, rp.protocol) == (2, 24, "paper_2fold")
    assert "K=2" in rp.summary()

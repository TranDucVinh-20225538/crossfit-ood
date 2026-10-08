"""Synthetic dose-response: group random effect (ICC) -> leakage Delta."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from crossfit_ood import get_scorer, make_grouped_features
from conftest import run as _run

SCORERS = ("mahalanobis", "knn_kth", "vim")
SEEDS = (0, 1, 2)


def run(data, **kw):
    kw.setdefault("scorers", SCORERS)
    return _run(data, **kw)


def _mean_delta(icc):
    out = {s: [] for s in SCORERS}
    for seed in SEEDS:
        rep = run(make_grouped_features(icc, random_state=seed), n_bootstrap=0, random_state=seed)
        for s in SCORERS:
            out[s].append(rep.results[s].delta)
    return {s: float(np.mean(v)) for s, v in out.items()}


@pytest.mark.parametrize("seed", SEEDS)
def test_icc0_delta_is_zero_within_ci(seed):
    """No group effect -> no leakage: Delta ~ 0 and the (refit-aware) CI covers 0."""
    rep = run(make_grouped_features(0.0, random_state=seed), uncertainty="jackknife",
              random_state=seed)
    for s in SCORERS:
        r = rep.results[s]
        assert abs(r.delta) < 0.03, (s, r.delta)
        assert r.delta_ci[0] <= 0.0 <= r.delta_ci[1], (s, r.delta, r.delta_ci)


def test_icc0_fixed_bootstrap_delta_small():
    deltas = _mean_delta(0.0)
    assert abs(deltas["mahalanobis"]) < 0.005
    assert abs(deltas["knn_kth"]) < 0.005
    assert abs(deltas["vim"]) < 0.02


def test_delta_increases_monotonically_with_icc():
    iccs = (0.0, 0.1, 0.2, 0.4, 0.6)
    curve = {s: [] for s in SCORERS}
    for icc in iccs:
        for s, v in _mean_delta(icc).items():
            curve[s].append(v)
    for s, ys in curve.items():
        assert np.all(np.diff(ys) > 0), (s, dict(zip(iccs, np.round(ys, 4))))


def test_delta_significant_at_moderate_icc():
    rep = run(make_grouped_features(0.4, random_state=0), n_bootstrap=500)
    for s in SCORERS:
        assert rep.results[s].delta_ci[0] > 0.0


def test_tpr_transfer_to_new_groups():
    r0 = run(make_grouped_features(0.0, random_state=0), n_bootstrap=0)
    r6 = run(make_grouped_features(0.6, random_state=0), n_bootstrap=0)
    for s in SCORERS:
        t0, t6 = r0.results[s].tpr_transfer, r6.results[s].tpr_transfer
        assert abs(t0.tpr_new - 0.95) < 0.03          # nominal when no group effect
        assert t6.tpr_new < t0.tpr_new                 # threshold does not transfer
    assert r6.results["mahalanobis"].tpr_transfer.false_alarm_ratio > 2.0


def test_crossfit_tracks_fresh_group_oracle():
    """Cross-fit AUROC ~ AUROC on genuinely new groups; leaky AUROC overshoots it."""
    gap_x, gap_l = {s: [] for s in SCORERS}, {s: [] for s in SCORERS}
    for seed in SEEDS:
        d = make_grouped_features(0.4, n_groups=80, n_fresh_groups=80, random_state=seed)
        rep = run(d, n_bootstrap=0)
        y = np.r_[np.ones(len(d.features_fresh)), np.zeros(len(d.features_ood))]
        for s in SCORERS:
            m = get_scorer(s).fit(d.features_train, d.labels_train)
            oracle = roc_auc_score(y, np.r_[m.score(d.features_fresh), m.score(d.features_ood)])
            gap_x[s].append(rep.results[s].auroc_crossfit - oracle)
            gap_l[s].append(rep.results[s].auroc_leaky - oracle)
    for s in SCORERS:
        gx, gl = np.mean(gap_x[s]), np.mean(gap_l[s])
        assert abs(gx) < 0.035, (s, gx)
        assert gl > abs(gx) + 0.03, (s, gl, gx)

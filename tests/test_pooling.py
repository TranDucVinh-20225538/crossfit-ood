"""Per-fit (matched) vs naive pooling on a constructed case."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from crossfit_ood import crossfit_auroc


class ShiftScorer:
    """score = x0 - mean(fit x0). Each model shifts ID and OOD by the same constant, so a
    per-fit AUROC ignores the shift, while averaging OOD scores over models does not."""

    def fit(self, x, labels=None):
        self.c_ = float(np.mean(np.asarray(x)[:, 0]))
        return self

    def score(self, x):
        return np.asarray(x, float)[:, 0] - self.c_


def _auc(a, b):
    return roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])


@pytest.fixture
def case():
    # group 0 dominates the fit set, so leaving it out moves the fit mean far more than
    # leaving out any other group; OOD sits in a narrow band between the groups
    rng = np.random.default_rng(0)
    G = 6
    gtr = np.repeat(np.arange(G), [200, 10, 10, 10, 10, 10])
    xtr = np.c_[10.0 * gtr + rng.normal(size=len(gtr)), np.zeros(len(gtr))]
    gid = np.repeat(np.arange(G), 10)
    xid = np.c_[10.0 * gid + rng.normal(size=len(gid)), np.zeros(len(gid))]
    xood = np.c_[rng.normal(25, 3, 60), np.zeros(60)]
    return xtr, gtr, xid, gid, xood


def _manual(xtr, gtr, xid, gid, xood):
    num, s_id, s_ood = 0.0, np.empty(len(xid)), []
    for k in np.unique(gid):  # leave-one-group-out (6 groups <= 50)
        m = ShiftScorer().fit(xtr[gtr != k])
        a, b = m.score(xid[gid == k]), m.score(xood)
        num += len(a) * _auc(a, b)
        s_id[gid == k] = a
        s_ood.append(b)
    return num / len(xid), _auc(s_id, np.mean(s_ood, axis=0))


def test_per_fit_and_naive_differ_and_default_is_per_fit(case):
    per_fit, naive = _manual(*case)
    kw = dict(scorers={"shift": ShiftScorer()}, uncertainty="bootstrap", n_bootstrap=0)
    default = crossfit_auroc(*case, **kw)
    explicit = crossfit_auroc(*case, pooling="per_fit", **kw)
    with pytest.warns(UserWarning):
        nv = crossfit_auroc(*case, pooling="naive", **kw)
    assert default.pooling == "per_fit" and default.n_splits == 6
    assert default.results["shift"].auroc_crossfit == pytest.approx(per_fit, abs=1e-12)
    assert explicit.results["shift"].auroc_crossfit == default.results["shift"].auroc_crossfit
    assert nv.results["shift"].auroc_crossfit == pytest.approx(naive, abs=1e-12)
    assert abs(per_fit - naive) > 0.05
    # the leaky arm does not depend on the pooling rule
    assert nv.results["shift"].auroc_leaky == default.results["shift"].auroc_leaky


def test_naive_pooling_has_jackknife_interval(case):
    with pytest.warns(UserWarning):
        rep = crossfit_auroc(*case, scorers={"shift": ShiftScorer()}, pooling="naive",
                             n_jackknife_blocks=6)
    r = rep.results["shift"]
    assert np.isfinite(r.delta_se) and r.delta_ci[0] <= r.delta <= r.delta_ci[1]

"""paper_2fold with fold_ids_eval: ID-eval groups absent from training but given a fold count as seen."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from crossfit_ood import crossfit_auroc, get_scorer


def _data(seed=0):
    r = np.random.default_rng(seed)
    gtr = np.repeat(np.arange(20), 15)
    xtr = r.normal(size=(len(gtr), 8)) + r.normal(size=(20, 8))[gtr]
    gid = np.r_[np.repeat(np.arange(20), 3), np.repeat([100, 101, 102, 103], 4)]
    xid = r.normal(size=(len(gid), 8))
    xid[:60] += r.normal(size=(20, 8))[gid[:60]] * 0.5
    xood = r.normal(loc=1.0, size=(40, 8))
    ftr = (gtr % 2).astype(int)
    fe = np.r_[(gid[:60] % 2), [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1]].astype(int)
    return xtr, gtr, xid, gid, xood, ftr, fe


def _manual(xtr, gtr, xid, xood, ftr, fs):
    same, dis = [], []
    for f in (0, 1):
        s = get_scorer("mahalanobis_l2").fit(xtr[ftr == f])
        si, so = s.score(xid), s.score(xood)
        auc = lambda m: roc_auc_score(np.r_[np.ones(m.sum()), np.zeros(len(so))], np.r_[si[m], so])  # noqa: E731
        same.append(auc(fs == f)); dis.append(auc(fs == 1 - f))
    return np.mean(same) - np.mean(dis)


def test_orphans_counted_as_seen():
    xtr, gtr, xid, gid, xood, ftr, fe = _data()
    kw = dict(scorers={"m": get_scorer("mahalanobis_l2")}, protocol="paper_2fold", fold_ids=ftr,
              uncertainty="bootstrap", n_bootstrap=0)
    r = crossfit_auroc(xtr, gtr, xid, gid, xood, fold_ids_eval=fe, **kw)
    assert r.meta["n_id_eval_orphan"] == 12 and r.meta["n_groups_orphan"] == 3
    assert r.results["m"].delta == pytest.approx(_manual(xtr, gtr, xid, xood, ftr, fe), abs=1e-12)
    strict = crossfit_auroc(xtr, gtr, xid, gid, xood, **kw)
    fs_strict = np.where(np.isin(gid, gtr), fe, -1)
    assert strict.results["m"].delta == pytest.approx(_manual(xtr, gtr, xid, xood, ftr, fs_strict), abs=1e-12)


def test_orphans_are_jackknife_units():
    xtr, gtr, xid, gid, xood, ftr, fe = _data()
    r = crossfit_auroc(xtr, gtr, xid, gid, xood, scorers={"m": get_scorer("mahalanobis_l2")},
                       protocol="paper_2fold", fold_ids=ftr, fold_ids_eval=fe, uncertainty="jackknife",
                       n_jackknife_blocks=50)
    assert np.isfinite(r.results["m"].delta_se)


def test_fold_ids_eval_validation():
    xtr, gtr, xid, gid, xood, ftr, fe = _data()
    kw = dict(scorers=["mahalanobis_l2"], protocol="paper_2fold", uncertainty="bootstrap", n_bootstrap=0)
    with pytest.raises(ValueError, match="needs fold_ids"):
        crossfit_auroc(xtr, gtr, xid, gid, xood, fold_ids_eval=fe, **kw)
    bad = fe.copy(); bad[0] = 1 - bad[0]; bad[1:3] = bad[0]
    with pytest.raises(ValueError, match="disagrees"):
        crossfit_auroc(xtr, gtr, xid, gid, xood, fold_ids=ftr, fold_ids_eval=bad, **kw)

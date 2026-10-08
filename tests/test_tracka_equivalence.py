"""paper_2fold + knn_mean_cosine / mahalanobis_l2 reproduce the paper's Delta_fit definition.

Two checks on a small synthetic fixture (no real data):

* REFERENCE: frozen values produced by the code that generated the paper's tables;
  agreement to 1e-9.
* Independent re-implementation of the paper's scorer formulas (L2 normalisation
  x / (||x|| + 1e-8); LedoitWolf() Mahalanobis with sqrt; mean cosine distance to the
  k = 50 nearest fit samples) and of the 2-fold seen / unseen AUROC contrast.
"""
import numpy as np
import pytest
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from crossfit_ood import crossfit_auroc, get_scorer, make_grouped_features

TOL = 1e-9

REFERENCE = {
    "knn_mean_cosine": {"auroc_seen": 0.884302316702555, "auroc_unseen": 0.811423033653528,
                        "delta": 0.07287928304902702},
    "mahalanobis_l2": {"auroc_seen": 0.9158852695183846, "auroc_unseen": 0.8022358903091482,
                       "delta": 0.11364937920923635},
    "knn_mean_cosine_tracka_seen": {"auroc_seen": 0.8887607145718177, "auroc_unseen": 0.817643194744853,
                                    "delta": 0.07111751982696468, "n_id_eval_orphan": 50},
    "mahalanobis_l2_tracka_seen": {"auroc_seen": 0.926179604261796, "auroc_unseen": 0.7668589281422735,
                                   "delta": 0.1593206761195225, "n_id_eval_orphan": 50},
}
SCORERS = ("knn_mean_cosine", "mahalanobis_l2")


@pytest.fixture(scope="module")
def fixture():
    d = make_grouped_features(0.4, n_groups=12, n_per_group=(16, 40), n_features=16, n_ood_groups=8,
                              random_state=3)
    ftr = (d.groups_train % 2).astype(int)
    gid_tracka = d.groups_id_eval.copy()
    orphan = np.isin(gid_tracka, [0, 1, 2])
    gid_tracka[orphan] += 500  # ID-eval groups absent from train, assigned to a fold below
    fe = np.where(orphan, (gid_tracka - 500) % 2, gid_tracka % 2).astype(int)
    return d, ftr, gid_tracka, fe


def _run(d, name, gid=None, **kw):
    return crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval,
                          d.groups_id_eval if gid is None else gid, d.features_ood, groups_ood=d.groups_ood,
                          scorers=[name], protocol="paper_2fold", uncertainty="jackknife", random_state=0, **kw)


@pytest.mark.parametrize("name", SCORERS)
def test_reference_delta_random_folds(fixture, name):
    d = fixture[0]
    r = _run(d, name).results[name]
    ref = REFERENCE[name]
    assert abs(r.auroc_leaky - ref["auroc_seen"]) < TOL
    assert abs(r.auroc_crossfit - ref["auroc_unseen"]) < TOL
    assert abs(r.delta - ref["delta"]) < TOL


@pytest.mark.parametrize("name", SCORERS)
def test_reference_delta_fixed_folds_tracka_seen(fixture, name):
    d, ftr, gid, fe = fixture
    rep = _run(d, name, gid=gid, fold_ids=ftr, fold_ids_eval=fe)
    r, ref = rep.results[name], REFERENCE[name + "_tracka_seen"]
    assert rep.meta["n_id_eval_orphan"] == ref["n_id_eval_orphan"]
    assert abs(r.auroc_leaky - ref["auroc_seen"]) < TOL
    assert abs(r.auroc_crossfit - ref["auroc_unseen"]) < TOL
    assert abs(r.delta - ref["delta"]) < TOL


# --------------------------------------------------------------------------- independent formulas
def _l2(x):
    x = np.asarray(x, np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _paper_scores(name, x_fit, x):
    if name == "mahalanobis_l2":
        lw = LedoitWolf().fit(_l2(x_fit))
        diff = _l2(x) - lw.location_
        return -np.sqrt(np.maximum(np.sum((diff @ lw.precision_) * diff, axis=1), 0.0))
    nn = NearestNeighbors(n_neighbors=50, metric="cosine").fit(_l2(x_fit))
    return -nn.kneighbors(_l2(x))[0].mean(axis=1)


def _manual_delta(name, x_fit_all, fold_fit, x_id, fold_id, x_ood):
    seen, unseen = [], []
    for f in (0, 1):
        s_id = _paper_scores(name, x_fit_all[fold_fit == f], x_id)
        s_ood = _paper_scores(name, x_fit_all[fold_fit == f], x_ood)
        for mask, acc in ((fold_id == f, seen), ((fold_id >= 0) & (fold_id != f), unseen)):
            y = np.r_[np.ones(mask.sum()), np.zeros(len(s_ood))]
            acc.append(roc_auc_score(y, np.r_[s_id[mask], s_ood]))
    return np.mean(seen), np.mean(unseen)


@pytest.mark.parametrize("name", SCORERS)
@pytest.mark.parametrize("seen", ["strict", "tracka"])
def test_independent_formulas(fixture, name, seen):
    d, ftr, gid, fe = fixture
    if seen == "strict":
        gid, fe_used = d.groups_id_eval, np.where(np.isin(d.groups_id_eval, d.groups_train),
                                                  d.groups_id_eval % 2, -1)
        rep = _run(d, name, fold_ids=ftr)
    else:
        fe_used = fe
        rep = _run(d, name, gid=gid, fold_ids=ftr, fold_ids_eval=fe)
    a_seen, a_unseen = _manual_delta(name, d.features_train, ftr, d.features_id_eval, fe_used, d.features_ood)
    r = rep.results[name]
    assert abs(r.auroc_leaky - a_seen) < TOL
    assert abs(r.auroc_crossfit - a_unseen) < TOL
    assert abs(r.delta - (a_seen - a_unseen)) < TOL


def test_scorer_definitions_match_paper_formulas(fixture):
    d = fixture[0]
    xq = np.r_[d.features_id_eval, d.features_ood]
    for name in SCORERS:
        s = get_scorer(name).fit(d.features_train).score(xq)
        np.testing.assert_allclose(s, _paper_scores(name, d.features_train, xq), rtol=0, atol=TOL)


# --------------------------------------------------------------------------- optional torch path
def _torch_devices():
    torch = pytest.importorskip("torch")
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("name", ["knn_mean_cosine", "knn_kth"])
def test_torch_knn_matches_cpu(fixture, name):
    d = fixture[0]
    xq = np.r_[d.features_id_eval, d.features_ood]
    ref = get_scorer(name).fit(d.features_train).score(xq)
    for dev in _torch_devices():
        s = get_scorer(name, device=dev).fit(d.features_train).score(xq)
        np.testing.assert_allclose(s, ref, rtol=0, atol=TOL)


def test_torch_knn_reproduces_reference_delta(fixture):
    d = fixture[0]
    for dev in _torch_devices():
        rep = crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval, d.groups_id_eval,
                             d.features_ood, groups_ood=d.groups_ood,
                             scorers={"knn_mean_cosine": get_scorer("knn_mean_cosine", device=dev)},
                             protocol="paper_2fold", uncertainty="bootstrap", n_bootstrap=0, random_state=0)
        assert abs(rep.results["knn_mean_cosine"].delta - REFERENCE["knn_mean_cosine"]["delta"]) < TOL

import numpy as np

from crossfit_ood import crossfit_auroc, make_features_from_sizes


def test_sizes_are_respected():
    tr = np.array([5, 1, 3, 0])
    ev = np.array([2, 0, 1, 4])
    oo = np.array([3, 2])
    d = make_features_from_sizes(0.3, tr, ev, oo, n_features=8, random_state=1, population_seed=2)
    assert np.array_equal(np.bincount(d.groups_train, minlength=4), tr)
    assert np.array_equal(np.bincount(d.groups_id_eval, minlength=4), ev)
    assert len(d.features_ood) == 5 and len(np.unique(d.groups_ood)) == 2
    assert d.features_train.shape[1] == 8
    # group labels constant within group
    for g in np.unique(d.groups_train):
        assert len(np.unique(d.labels_train[d.groups_train == g])) == 1


def test_population_seed_fixes_population_only():
    tr, ev, oo = np.full(60, 20), np.full(60, 5), np.full(60, 10)
    a = make_features_from_sizes(0.0, tr, ev, oo, random_state=1, population_seed=9)
    b = make_features_from_sizes(0.0, tr, ev, oo, random_state=2, population_seed=9)
    # OOD mean shift direction is the same population parameter
    da = a.features_ood.mean(0) - a.features_train.mean(0)
    db = b.features_ood.mean(0) - b.features_train.mean(0)
    cos = da @ db / np.linalg.norm(da) / np.linalg.norm(db)
    assert cos > 0.9
    assert not np.allclose(a.features_train, b.features_train)


def test_paper2fold_runs_on_explicit_design():
    rng = np.random.default_rng(0)
    tr = rng.integers(5, 40, size=16)
    ev = np.where(rng.random(16) < 0.8, rng.integers(1, 6, size=16), 0)
    d = make_features_from_sizes(0.5, tr, ev, np.full(6, 15), random_state=0, population_seed=1)
    rep = crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval, d.groups_id_eval,
                         d.features_ood, groups_ood=d.groups_ood, scorers="mahalanobis",
                         protocol="paper_2fold", uncertainty="jackknife")
    r = rep.results["mahalanobis"]
    assert r.delta_ci[0] <= r.delta <= r.delta_ci[1]


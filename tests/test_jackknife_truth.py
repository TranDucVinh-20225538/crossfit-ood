"""The default jackknife CI contains the true Delta on a seeded synthetic case.

True Delta = mean point estimate over many independent datasets drawn from the same
population (fixed ``population_seed``) and design; the CI comes from one seeded dataset.
"""
import numpy as np
import pytest

from crossfit_ood import crossfit_auroc, make_grouped_features

POP, ICC, N_TRUTH = 7, 0.4, 200
SCORERS = ("mahalanobis_l2", "knn_mean_cosine")


def _data(seed):
    return make_grouped_features(ICC, n_groups=16, n_per_group=(10, 20), n_features=8, n_ood_groups=16,
                                 random_state=seed, population_seed=POP)


def _fit(d, **kw):
    return crossfit_auroc(d.features_train, d.groups_train, d.features_id_eval, d.groups_id_eval,
                          d.features_ood, groups_ood=d.groups_ood, scorers=SCORERS, random_state=0, **kw)


@pytest.fixture(scope="module")
def true_delta():
    reps = [_fit(_data(1000 + i), uncertainty="bootstrap", n_bootstrap=0) for i in range(N_TRUTH)]
    return {s: float(np.mean([r.results[s].delta for r in reps])) for s in SCORERS}


def test_jackknife_ci_contains_true_delta(true_delta):
    rep = _fit(_data(0))  # defaults: leave-one-group-out, refit-aware jackknife
    assert rep.uncertainty == "jackknife" and rep.n_splits == 16
    for s in SCORERS:
        r = rep.results[s]
        assert true_delta[s] > 0.0, (s, true_delta[s])
        assert r.delta_ci[0] <= true_delta[s] <= r.delta_ci[1], (s, true_delta[s], r.delta, r.delta_ci)

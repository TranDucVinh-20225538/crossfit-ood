"""crossfit-ood: group-aware evaluation of feature-based OOD detectors.

Public API::

    crossfit_auroc(...)  -> CrossfitReport
    group_overlap(groups_train, groups_id_eval) -> OverlapStats
    make_group_folds(groups_id_eval, n_splits, random_state) -> fold ids
    default_n_splits(n_groups) -> leave-one-group-out if <= 50 groups, else 10
    Scorer protocol, register_scorer, get_scorer, available_scorers, DEFAULT_SCORERS
    KNNMeanCosineScorer, MahalanobisL2Scorer (defaults), KNNScorer, MahalanobisScorer, ViMScorer
    make_grouped_features, make_features_from_sizes (synthetic data with group random effects)
    posthoc_delta_regression (POST-HOC: Delta vs surrogate +/- 1/n_groups)
"""
from .core import (CrossfitReport, OverlapStats, ScoreResult, TPRTransfer, crossfit_auroc,
                   default_n_splits, group_overlap, make_group_folds)
from .posthoc import posthoc_delta_regression
from .scorers import (DEFAULT_SCORERS, KNNMeanCosineScorer, KNNScorer, MahalanobisL2Scorer,
                      MahalanobisScorer, Scorer, ViMScorer, available_scorers, get_scorer,
                      register_scorer)
from .synthetic import GroupedData, make_features_from_sizes, make_grouped_features

__version__ = "0.1.0"

__all__ = [
    "crossfit_auroc", "group_overlap", "make_group_folds", "default_n_splits",
    "CrossfitReport", "OverlapStats", "ScoreResult", "TPRTransfer",
    "Scorer", "register_scorer", "get_scorer", "available_scorers", "DEFAULT_SCORERS",
    "KNNMeanCosineScorer", "MahalanobisL2Scorer", "KNNScorer", "MahalanobisScorer", "ViMScorer",
    "make_grouped_features", "make_features_from_sizes", "GroupedData",
    "posthoc_delta_regression",
]

"""Leaky vs cross-fitted AUROC on synthetic grouped features (runs in well under a minute).

    python examples/make_demo_data.py --out demo.npz
    python examples/quickstart.py demo.npz

Without an argument a dataset is generated in memory.
"""
import sys

import numpy as np

from crossfit_ood import crossfit_auroc, make_grouped_features


def load(path=None):
    if path is None:
        d = make_grouped_features(0.4, n_groups=24, n_per_group=(10, 40), random_state=0)
        return dict(features_train=d.features_train, groups_train=d.groups_train,
                    features_id_eval=d.features_id_eval, groups_id_eval=d.groups_id_eval,
                    features_ood=d.features_ood, groups_ood=d.groups_ood)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    x = load(sys.argv[1] if len(sys.argv) > 1 else None)
    # defaults: mahalanobis_l2 + knn_mean_cosine, per-fit pooling, leave-one-group-out folds
    # (<= 50 groups), refit-aware grouped jackknife intervals
    rep = crossfit_auroc(x["features_train"], x["groups_train"], x["features_id_eval"],
                         x["groups_id_eval"], x["features_ood"], groups_ood=x.get("groups_ood"))
    print(rep.summary())
    print()
    for name, r in rep.results.items():
        t = r.tpr_transfer
        print(f"{name}: Delta = {r.delta:+.3f}, 95% CI [{r.delta_ci[0]:+.3f}, {r.delta_ci[1]:+.3f}]; "
              f"threshold set for {t.target_tpr:.0%} TPR on seen groups keeps {t.tpr_new:.1%} "
              f"of ID samples from new groups")

    # the paper's 2-fold seen-vs-unseen contrast on the same data
    p2 = crossfit_auroc(x["features_train"], x["groups_train"], x["features_id_eval"],
                        x["groups_id_eval"], x["features_ood"], groups_ood=x.get("groups_ood"),
                        protocol="paper_2fold")
    print()
    print(p2.summary())


if __name__ == "__main__":
    main()

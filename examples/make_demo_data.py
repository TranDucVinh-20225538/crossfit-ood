"""Write a synthetic feature file in the crossfit-ood input format.

    python examples/make_demo_data.py --out demo.npz [--icc 0.4] [--n-groups 24]

Each group (e.g. a patient or slide) gets a random effect whose share of the
feature variance is ``icc``. Train and ID-eval are a sample-level split inside
the same groups, which is the leaky setting that crossfit-ood measures.
"""
import argparse

import numpy as np

from crossfit_ood import make_grouped_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="demo.npz")
    ap.add_argument("--icc", type=float, default=0.4, help="share of feature variance between groups")
    ap.add_argument("--n-groups", type=int, default=24)
    ap.add_argument("--n-features", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = make_grouped_features(a.icc, n_groups=a.n_groups, n_per_group=(10, 40), n_features=a.n_features,
                              random_state=a.seed)
    np.savez_compressed(a.out, features_train=d.features_train, groups_train=d.groups_train,
                        labels_train=d.labels_train, features_id_eval=d.features_id_eval,
                        groups_id_eval=d.groups_id_eval, features_ood=d.features_ood,
                        groups_ood=d.groups_ood)
    print(f"wrote {a.out}: {len(d.features_train)} train / {len(d.features_id_eval)} ID-eval / "
          f"{len(d.features_ood)} OOD samples, {a.n_groups} ID groups, ICC = {a.icc}")


if __name__ == "__main__":
    main()

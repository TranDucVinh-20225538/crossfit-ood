"""Command-line interface: ``crossfit-ood run feats.npz`` / ``crossfit-ood demo``.

Also available as ``python -m crossfit_ood ...``.

    crossfit-ood demo --out demo.npz            # write a small synthetic example (seconds)
    crossfit-ood run demo.npz                   # cross-fit, default scorers, jackknife CIs
    crossfit-ood run feats.npz --protocol paper_2fold --scorers knn_mean_cosine \
        --out report.json

The .npz needs the keys features_train, groups_train, features_id_eval, groups_id_eval,
features_ood (optional: labels_train, groups_ood, fold_train, fold_id_eval); rename with
--key logical=stored.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional

import numpy as np

REQUIRED = ("features_train", "groups_train", "features_id_eval", "groups_id_eval", "features_ood")
OPTIONAL = ("labels_train", "groups_ood", "fold_train", "fold_id_eval")
_KNN = ("knn_mean_cosine", "knn_kth")


def _keymap(items: Optional[List[str]]) -> Dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--key expects logical=stored, got {it!r}")
        k, v = it.split("=", 1)
        if k not in REQUIRED + OPTIONAL:
            raise SystemExit(f"unknown logical key {k!r}; expected one of {REQUIRED + OPTIONAL}")
        out[k] = v
    return out


def load_npz(path: str, keymap: Optional[Dict[str, str]] = None, allow_pickle: bool = False):
    """Load the documented arrays from an .npz (missing optional keys -> None)."""
    keymap = keymap or {}
    with np.load(path, allow_pickle=allow_pickle) as z:
        data = {}
        for k in REQUIRED + OPTIONAL:
            stored = keymap.get(k, k)
            if stored in z.files:
                data[k] = np.asarray(z[stored])
            elif k in REQUIRED:
                raise SystemExit(f"{path}: missing required array {k!r} (stored key {stored!r}); "
                                 f"found {sorted(z.files)}")
            else:
                data[k] = None
    return data


def _scorers(a) -> dict:
    from .scorers import get_scorer
    out = {}
    for s in (s.strip() for s in a.scorers.split(",")):
        if not s:
            continue
        out[s] = get_scorer(s, k=a.knn_k, device=a.device) if s in _KNN else get_scorer(s)
    return out


def _cmd_run(a) -> int:
    from .core import crossfit_auroc
    d = load_npz(a.npz, _keymap(a.key), a.allow_pickle)
    fold_ids = fold_ids_eval = None
    if a.protocol == "paper_2fold" and d["fold_train"] is not None:
        fold_ids = d["fold_train"].astype(int)
    if a.seen == "tracka":
        if a.protocol != "paper_2fold" or d["fold_train"] is None or d["fold_id_eval"] is None:
            raise SystemExit("--seen tracka needs --protocol paper_2fold and the keys fold_train and "
                             "fold_id_eval in the input")
        fold_ids_eval = d["fold_id_eval"].astype(int)
    rep = crossfit_auroc(
        d["features_train"], d["groups_train"], d["features_id_eval"], d["groups_id_eval"],
        d["features_ood"], labels_train=None if a.class_agnostic else d["labels_train"],
        groups_ood=d["groups_ood"], scorers=_scorers(a), protocol=a.protocol, n_splits=a.n_splits,
        n_repeats=a.n_repeats, pooling=a.pooling, fold_ids=fold_ids, fold_ids_eval=fold_ids_eval,
        uncertainty=a.uncertainty, n_jackknife_blocks=a.n_jackknife_blocks,
        n_bootstrap=a.n_bootstrap, n_subsamples=a.n_subsamples, ci_level=a.ci_level,
        target_tpr=a.target_tpr, n_jobs=a.n_jobs, random_state=a.seed)
    print(rep.summary())
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rep.to_dict(), fh, indent=1, default=_json_default)
        print(f"report written to {a.out}")
    return 0


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _cmd_demo(a) -> int:
    from .synthetic import make_grouped_features
    g = make_grouped_features(a.icc, n_groups=a.n_groups, n_per_group=(10, 40),
                              random_state=a.seed, population_seed=a.seed)
    np.savez_compressed(a.out, features_train=g.features_train, groups_train=g.groups_train,
                        labels_train=g.labels_train, features_id_eval=g.features_id_eval,
                        groups_id_eval=g.groups_id_eval, features_ood=g.features_ood,
                        groups_ood=g.groups_ood)
    print(f"wrote {a.out}: synthetic features (ICC={a.icc}, {a.n_groups} train groups, "
          f"{len(g.features_train)} train / {len(g.features_id_eval)} ID-eval / "
          f"{len(g.features_ood)} OOD samples)")
    print(f"next: crossfit-ood run {a.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="crossfit-ood", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="leaky vs group-cross-fit AUROC from an .npz of features")
    r.add_argument("npz")
    r.add_argument("--protocol", choices=["default", "paper_2fold"], default="default")
    r.add_argument("--scorers", default="mahalanobis_l2,knn_mean_cosine",
                   help="comma-separated registry names (also: knn_kth, mahalanobis, vim)")
    r.add_argument("--knn-k", type=int, default=50, help="k for knn_mean_cosine / knn_kth")
    r.add_argument("--device", default=None,
                   help="run the kNN scorers with PyTorch on this device, e.g. cuda (needs the [gpu] extra)")
    r.add_argument("--class-agnostic", action="store_true",
                   help="ignore labels_train (class-agnostic Mahalanobis, as in the paper's tables)")
    r.add_argument("--pooling", choices=["per_fit", "naive"], default="per_fit",
                   help="naive is biased and only for comparison (warns)")
    r.add_argument("--seen", choices=["strict", "tracka"], default="strict",
                   help="paper_2fold seen set: strict = ID-eval groups present in training; tracka = also "
                        "groups given a fold by fold_id_eval")
    r.add_argument("--uncertainty", choices=["jackknife", "bootstrap", "subsample"], default="jackknife",
                   help="bootstrap is fast but under-covers when Delta > 0")
    r.add_argument("--n-jackknife-blocks", type=int, default=50)
    r.add_argument("--n-bootstrap", type=int, default=1000)
    r.add_argument("--n-subsamples", type=int, default=200)
    r.add_argument("--n-splits", type=int, default=None,
                   help="K for protocol=default (default: leave-one-group-out if <= 50 ID-eval groups, else 10)")
    r.add_argument("--n-repeats", type=int, default=1)
    r.add_argument("--ci-level", type=float, default=0.95)
    r.add_argument("--target-tpr", type=float, default=0.95)
    r.add_argument("--n-jobs", type=int, default=1)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--key", action="append", metavar="LOGICAL=STORED", help="rename an .npz key")
    r.add_argument("--allow-pickle", action="store_true", help="for object/str group arrays (trusted files only)")
    r.add_argument("--out", help="write the full report as JSON")
    r.set_defaults(func=_cmd_run)
    dm = sub.add_parser("demo", help="write a small synthetic example .npz")
    dm.add_argument("--out", default="crossfit_demo.npz")
    dm.add_argument("--icc", type=float, default=0.3)
    dm.add_argument("--n-groups", type=int, default=24)
    dm.add_argument("--seed", type=int, default=0)
    dm.set_defaults(func=_cmd_demo)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())

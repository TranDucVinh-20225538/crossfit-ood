# crossfit-ood

Group-aware evaluation of feature-based out-of-distribution (OOD) detectors: leaky vs
cross-fitted AUROC, refit-aware grouped jackknife intervals, and threshold transfer to new groups.

## What is group leakage in an OOD evaluation?

Medical and monitoring datasets are grouped: many images come from the same patient, slide,
hospital or camera. Feature-based OOD scorers (Mahalanobis, k-NN, ...) are *fitted* on
in-distribution (ID) training features. If the split into train and ID-evaluation data is made at
the sample level, the ID-evaluation samples come from groups that the scorer has already seen,
while at deployment almost every input comes from a new group. A k-NN scorer, for instance, finds
near-duplicates of the same patient in its fit set, so ID-evaluation samples look "more ID" than
data from new patients would. The reported AUROC (ID vs OOD) is then optimistic, and a threshold
chosen on seen groups flags more new-group ID samples as OOD than intended. `crossfit-ood`
measures this: it refits the scorer so that each ID-evaluation sample is scored by a model that
never saw its group (cross-fitting) and reports the difference to the leaky AUROC.

## Install

```bash
pip install "git+https://github.com/TranDucVinh-20225538/crossfit-ood@v0.1.0"
# optional PyTorch k-NN search (CPU or GPU):
pip install "crossfit-ood[gpu] @ git+https://github.com/TranDucVinh-20225538/crossfit-ood@v0.1.0"
```

Python >= 3.9; dependencies: numpy and scikit-learn (torch only for the optional `[gpu]` extra).

## Usage

Command line:

```bash
crossfit-ood run feats.npz
```

Python:

```python
from crossfit_ood import crossfit_auroc

report = crossfit_auroc(features_train, groups_train, features_id_eval, groups_id_eval, features_ood)
print(report.summary())
```

A synthetic demo: `crossfit-ood demo --out demo.npz && crossfit-ood run demo.npz`, or
`python examples/make_demo_data.py --out demo.npz && python examples/quickstart.py demo.npz`.

## Input format

One `.npz` file with these arrays (rename stored keys with `--key logical=stored`):

| key | shape | meaning |
|---|---|---|
| `features_train` | (n_train, d) | features the scorer is fitted on |
| `groups_train` | (n_train,) | group id (patient, slide, ...) per training sample |
| `features_id_eval` | (n_id, d) | held-out ID features |
| `groups_id_eval` | (n_id,) | group id per ID-evaluation sample |
| `features_ood` | (n_ood, d) | OOD features |
| `labels_train` | (n_train,) | optional: class labels (class-conditional Mahalanobis) |
| `groups_ood` | (n_ood,) | optional: OOD group ids (resampling units for the interval) |
| `fold_train` | (n_train,) | optional, `paper_2fold`: fixed fold (0/1) per training sample |
| `fold_id_eval` | (n_id,) | optional, `--seen tracka`: fold (-1/0/1) per ID-evaluation sample |

Group ids may be integers or strings (string arrays need `--allow-pickle` only if stored as objects).
Features are cast to float64.

## Output

`report.summary()` prints, per scorer:

- **leaky AUROC**: scorer fitted on all training data, as a sample-level split reports it.
- **cross-fitted AUROC**: each ID-evaluation sample scored by a model fitted without its group,
  compared with OOD samples scored by that same model.
- **Δ = leaky − cross-fitted**: the AUROC inflation due to the scorer having seen the groups.
- **interval**: 95% CI for Δ (and for both AUROCs, in `report.results[name]`).
- **overlap rate**: share of ID-evaluation samples (and groups) whose group occurs in training.
- **threshold transfer**: a threshold set for 95% TPR on *seen* groups is applied to *new*-group ID
  samples; `TPR_new` is the realised coverage on new groups and `FA x` the false-alarm rate
  relative to the nominal 5%.
- the number of groups, groups per fit and K, since Δ depends on them.

`report.to_dict()` (or `--out report.json`) holds everything, including per-method diagnostics.

## Defaults and why

| setting | default | why |
|---|---|---|
| scorers | `mahalanobis_l2`, `knn_mean_cosine` | the definitions used in the paper |
| k-NN | mean cosine distance to the k = 50 nearest fit features (L2-normalised); `knn_kth` (k-th neighbour) as an option | |
| Mahalanobis | class-conditional (if labels are given), Ledoit-Wolf, float64, L2-normalised features; class-agnostic without labels | shrinkage keeps the precision matrix stable when d is large |
| ViM | available (`vim`), not in the default list | numerically unstable at subspace dimension D − C (see its docstring) |
| pooling | per-fit (matched) | each ID sample is compared only with OOD scored by the same model; averaging OOD scores across fits (`pooling="naive"`, warns) is not the AUROC of any model and can be biased |
| folds | leave-one-group-out if ≤ 50 ID-eval groups, otherwise K = 10 | Δ depends on the number of groups per fit, so many folds keep each cross-fitted model close to the model being evaluated |
| interval | refit-aware grouped jackknife (Busing weights for unequal blocks) | groups are refitted when deleted; the fixed-score bootstrap (`uncertainty="bootstrap"`, fast, warns) ignores refit variability and under-covers when Δ > 0 |
| protocol | `default` (cross-fit); `paper_2fold` reproduces the paper's Δ_fit | |
| seen definition | strict: an ID sample is seen if its group is in the fit set; the paper's alternative via `fold_ids_eval` / `--seen tracka` | |

### Protocols

- `protocol="default"`: leaky (fit on all training data) vs cross-fitted AUROC on the *same* ID
  samples.
- `protocol="paper_2fold"`: training groups are split into two folds; for each fold a scorer is
  fitted on that fold and the AUROC of seen-group ID samples is compared with that of
  unseen-group ID samples under the *same* model, averaged over the two folds. Each fit sees half
  of the groups, so Δ from the two protocols are not comparable.

## Caveats

- **Scorer channel only.** Only the OOD scorer is refitted; the features are taken as given.
  Cross-fitting does not remove leakage through a backbone that was fine-tuned on the same groups.
- **Δ depends on the number of groups.** Fewer groups per fit give a larger Δ at the same
  within-group correlation. Compare Δ across datasets, scorers or settings only at equal groups
  per fit (reported as `n_groups_fit`).
- The fixed-score bootstrap and the subsampling interval are kept for reference; use the default
  jackknife for reported intervals.

## Adding a scorer

Any object with `fit(features, labels=None) -> self` and `score(features) -> array` (higher = more
ID) works; pass an instance or register it by name:

```python
from crossfit_ood import register_scorer

@register_scorer("my_score")
class MyScorer:
    def fit(self, features, labels=None):
        ...
        return self

    def score(self, features):
        ...
```

## Citation and licence

Please cite the software via [`CITATION.cff`](CITATION.cff); the paper reference will be added
there. Released under the MIT licence, see [`LICENSE`](LICENSE).

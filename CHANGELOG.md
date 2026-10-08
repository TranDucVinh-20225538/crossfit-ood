# Changelog

## 0.1.0 (2026-10-09)

First public release.

- Defaults follow the paper's definitions: scorers `mahalanobis_l2` (Ledoit-Wolf, float64,
  L2-normalised features; class-conditional when labels are given) and `knn_mean_cosine`
  (mean cosine distance to the k = 50 nearest fit features).
- k-th-neighbour k-NN kept as `knn_kth`; raw-feature Mahalanobis as `mahalanobis`; ViM available
  but not in the default list, with a warning about numerical instability at dim = D − C.
- Per-fit (matched) pooling is the only default; naive pooling (OOD scores averaged across fits)
  is available behind `pooling="naive"` with a warning.
- Fold rule: leave-one-group-out for at most 50 ID-evaluation groups, otherwise K = 10.
- Intervals: refit-aware grouped jackknife with Busing weights by default; the fixed-score
  bootstrap is opt-in and warns that it under-covers when Δ > 0.
- Protocols `default` (cross-fit) and `paper_2fold` (reproduces the paper's Δ_fit); strict seen
  definition by default, the paper's alternative via `fold_ids_eval` (CLI `--seen tracka`).
- Optional exact PyTorch k-NN search (`device=`, extra `[gpu]`).
- Reports carry groups per fit (`n_groups_fit`), K and the pooling rule.
- CLI `crossfit-ood run` / `crossfit-ood demo`; synthetic example in `examples/`.

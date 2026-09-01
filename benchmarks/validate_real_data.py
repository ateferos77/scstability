"""Validate scstability against a real dataset.

Everything in ``tests/`` is synthetic, by design: synthetic data is the only
kind whose right answer is known in advance. But synthetic data cannot show
that the measure means anything biologically, so this script runs the package
against a real ``.h5ad`` and applies three checks that a wrong implementation
would fail.

Usage
-----
::

    python benchmarks/validate_real_data.py DATA.h5ad
    python benchmarks/validate_real_data.py DATA.h5ad --truth clusters
    python benchmarks/validate_real_data.py DATA.h5ad --n-pcs 7 --n-neighbors 25

The checks
----------
1. **Negative control** (always runs). The data is compared against a single
   multivariate Gaussian matched to its mean and covariance -- same shape and
   elongation, but exactly one mode, so any cluster found in it is an artefact.
   (A permutation null is the obvious choice and is wrong; see
   ``check_negative_control`` for why.)

   The comparison is made **at matched cluster counts**, which is the only
   fair way to make it. Stability is a measure of how reproducibly a partition
   is redrawn, and a coarse partition is trivially more reproducible than a
   fine one -- in the limit, a single cluster holding every cell scores
   exactly 1.0 under any resample. Comparing headline numbers across different
   cluster counts therefore makes structureless data look *more* stable than
   real data, which says nothing about either. This is not a quirk of the
   implementation; it is the well-known limitation of stability as a criterion
   (von Luxburg 2010), and the reason ``recommend()`` refuses to return a
   resolution that produced a single cluster.

2. **Ground truth** (needs ``--truth``, a column of published cell types).
   For every cluster, compute its Jaccard against the published cell type it
   best matches -- purity *and* completeness. That is an external quantity the
   package never sees. Bootstrap stability should correlate with it.

   Purity alone is deliberately reported alongside as a foil. A cluster that
   is exactly half of one cell type has purity 1.0 while being completely
   irreproducible, because the point at which the split falls moves on every
   resample. If bootstrap stability tracked purity it would be measuring local
   homogeneity; tracking the set Jaccard is what Hennig's measure is for.

3. **Scale**. Wall-clock for the sweep, so the cost is a number rather than a
   guess.
"""

from __future__ import annotations

import argparse
import time
import warnings

import anndata as ad
import numpy as np
import pandas as pd

import scstability as scs

RESOLUTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]


def _representation(adata: ad.AnnData, use_rep: str, n_pcs: int | None) -> ad.AnnData:
    """Build a minimal AnnData holding only the coordinates, optionally truncated."""
    X = np.asarray(adata.obsm[use_rep])
    if n_pcs is not None:
        X = X[:, :n_pcs]
    out = ad.AnnData(np.zeros((adata.n_obs, 1), dtype=np.float32))
    out.obsm["X_pca"] = np.ascontiguousarray(X, dtype=np.float32)
    return out


def _sweep(adata, resolutions, **kwargs):
    """Run a sweep, returning the result and whether recommend() warned."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = scs.stability_sweep(adata, resolutions, progress=False, **kwargs)
        recommendation = result.recommend()
    warned = any("stability" in str(w.message).lower() for w in caught)
    return result, recommendation, warned


def check_negative_control(adata, resolutions, seed=0, **kwargs):
    """Compare against a unimodal null with the same mean and covariance.

    The null draws from a single multivariate Gaussian fitted to the data. That
    preserves the overall shape -- total variance, elongation, the correlation
    between coordinates -- while guaranteeing exactly one mode, so any cluster
    found in it is an artefact of the clustering rather than of the data. It is
    the null behind the gap statistic (Tibshirani 2001) and SigClust (Liu 2008).

    A permutation null was tried first and is **wrong** here, which is worth
    recording because it is the obvious thing to reach for. Shuffling each
    coordinate independently across cells preserves that coordinate's marginal
    distribution *exactly*, multimodality included: in a five-cell-line
    dataset the marginal of PC1 is not a bell curve but a row of clumps. A
    product of multimodal marginals is a lattice of clumps in high dimensions,
    which is more strongly clustered than the real data, not less. Measured on
    sc_10x_5cl, real and column-shuffled data have identical mode counts in
    PC1-PC4 (3, 2, 2, 2) while the Gaussian null has one apiece.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(adata.obsm["X_pca"], dtype=np.float64)

    # method="svd" tolerates the near-singular covariance of a PCA basis.
    null_X = rng.multivariate_normal(
        X.mean(axis=0), np.cov(X, rowvar=False), size=X.shape[0], method="svd"
    )

    null = ad.AnnData(np.zeros((adata.n_obs, 1), dtype=np.float32))
    null.obsm["X_pca"] = np.ascontiguousarray(null_X, dtype=np.float32)
    return _sweep(null, resolutions, random_state=seed, **kwargs)


def truth_jaccard(cluster_mask: np.ndarray, truth: np.ndarray) -> tuple[float, str]:
    """Best Jaccard of this cluster against any published cell type."""
    best, best_label = 0.0, ""
    for label in np.unique(truth):
        in_type = truth == label
        union = np.logical_or(cluster_mask, in_type).sum()
        if union == 0:
            continue
        overlap = np.logical_and(cluster_mask, in_type).sum() / union
        if overlap > best:
            best, best_label = float(overlap), str(label)
    return best, best_label


def main() -> None:
    """Run the checks against the dataset named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument(
        "--truth", default=None, help="obs column of published cell types"
    )
    parser.add_argument("--use-rep", default="X_pca")
    parser.add_argument("--n-pcs", type=int, default=None)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=20)
    parser.add_argument("--resolutions", type=float, nargs="+", default=RESOLUTIONS)
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    full = ad.read_h5ad(args.path)
    print(f"{args.path}\n  {full.n_obs} cells x {full.n_vars} genes")

    adata = _representation(full, args.use_rep, args.n_pcs)
    print(
        f"  representation: {args.use_rep}{adata.obsm['X_pca'].shape}, "
        f"n_neighbors={args.n_neighbors}, n_boot={args.n_boot}\n"
    )

    kwargs = dict(n_boot=args.n_boot, n_neighbors=args.n_neighbors)

    started = time.time()
    result, recommendation, warned = _sweep(
        adata, args.resolutions, random_state=0, **kwargs
    )
    elapsed = time.time() - started

    print("=== real data ===")
    print(result.summary().to_string(index=False))
    print(f"\nrecommend() -> {recommendation}   warned: {warned}")
    print(
        f"sweep: {elapsed:.1f}s for {full.n_obs} cells x "
        f"{len(args.resolutions)} resolutions x {args.n_boot} bootstraps"
    )

    # ---------------------------------------------------------------- 1. null
    null, null_recommendation, null_warned = check_negative_control(
        adata, args.resolutions, **kwargs
    )
    print("\n=== negative control: unimodal Gaussian, matched covariance ===")
    print(null.summary().to_string(index=False))
    print(f"\nrecommend() -> {null_recommendation}   warned: {null_warned}")

    # Compared at MATCHED granularity, which is the only fair comparison.
    # A coarse partition is trivially more reproducible than a fine one -- in
    # the limit, one cluster holding every cell scores exactly 1.0 -- so the
    # headline numbers cannot be compared across different cluster counts.
    # Ignoring this makes structureless data look more stable than real data.
    real_summary = result.summary()
    null_summary = null.summary()
    merged = real_summary.merge(
        null_summary, on="n_clusters", suffixes=("_real", "_null")
    )

    print("\n=== real vs null at matched cluster counts ===")
    if len(merged):
        print(
            merged[
                [
                    "n_clusters",
                    "resolution_real",
                    "min_cluster_stability_real",
                    "resolution_null",
                    "min_cluster_stability_null",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )
        wins = (
            merged["min_cluster_stability_real"] > merged["min_cluster_stability_null"]
        ).sum()
        print(
            f"\n  real beats the null at {wins}/{len(merged)} matched "
            f"cluster counts: {'PASS' if wins > len(merged) / 2 else 'FAIL'}"
        )
    else:
        # No shared cluster count, so the matched comparison is unavailable.
        # The next best fair statement compares each side's best score among
        # non-degenerate partitions: a single cluster is excluded because it
        # scores 1.0 by construction and would flatter the null.
        real_best = real_summary[real_summary["n_clusters"] >= 2][
            "min_cluster_stability"
        ].max()
        null_best = null_summary[null_summary["n_clusters"] >= 2][
            "min_cluster_stability"
        ].max()
        print("  no shared cluster count; comparing best non-degenerate scores")
        print(f"    real {real_best:.3f}   null {null_best:.3f}")
        print(f"    real beats the null: {'PASS' if real_best > null_best else 'FAIL'}")

    print(
        f"\n  unmatched headline numbers (NOT a fair comparison): "
        f"real {real_summary['min_cluster_stability'].max():.3f}, "
        f"null {null_summary['min_cluster_stability'].max():.3f}"
    )
    degenerate = int((null_summary["n_clusters"] < 2).sum())
    if degenerate:
        print(
            f"  {degenerate} null resolution(s) collapsed to a single "
            f"cluster and scored a trivially perfect 1.0"
        )

    # -------------------------------------------------------------- 2. truth
    if args.truth is None:
        print("\n(no --truth column given; skipping the ground-truth check)")
        return

    import scanpy as sc
    from scipy import stats

    truth = full.obs[args.truth].astype(str).to_numpy()
    sc.pp.neighbors(
        adata, n_neighbors=args.n_neighbors, use_rep="X_pca", random_state=0
    )

    rows = []
    for resolution in args.resolutions:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added="_ref",
            flavor="igraph",
            n_iterations=2,
            directed=False,
            random_state=0,
        )
        labels = adata.obs["_ref"].astype(str).to_numpy()
        block = result.cluster_stability
        block = block[block["resolution"] == resolution]
        for _, row in block.iterrows():
            mask = labels == str(int(row["cluster"]))
            if not mask.any():
                continue
            overlap, matched = truth_jaccard(mask, truth)
            counts = pd.Series(truth[mask]).value_counts()
            rows.append(
                dict(
                    resolution=resolution,
                    cluster=int(row["cluster"]),
                    n_cells=int(mask.sum()),
                    # the mean, matching the statistic the bands are defined on
                    boot_jaccard=float(row["jaccard_mean"]),
                    truth_jaccard=overlap,
                    purity=counts.iloc[0] / mask.sum(),
                    matches=matched,
                )
            )

    df = pd.DataFrame(rows).dropna(subset=["boot_jaccard"])
    print(
        f"\n=== ground truth: {len(df)} clusters pooled across "
        f"{len(args.resolutions)} resolutions ==="
    )

    for name, column in (
        ("truth_jaccard (purity AND completeness)", "truth_jaccard"),
        ("purity alone (the foil)", "purity"),
    ):
        rho, p = stats.spearmanr(df[column], df["boot_jaccard"])
        print(f"  Spearman(boot_jaccard, {name:38s}) = {rho:+.3f}  p = {p:.2g}")

    rho, p = stats.spearmanr(df["truth_jaccard"], df["boot_jaccard"])
    print(f"\n  tracks real cell types: {'PASS' if rho > 0 and p < 0.05 else 'FAIL'}")

    df["recovers"] = pd.cut(
        df["truth_jaccard"],
        [0, 0.3, 0.5, 0.7, 1.0],
        labels=[
            "<0.3 carves a continuum",
            "0.3-0.5",
            "0.5-0.7",
            ">0.7 recovers a real type",
        ],
    )
    print("\n=== stability vs how well the cluster recovers a real cell type ===")
    print(
        df.groupby("recovers", observed=True)["boot_jaccard"]
        .agg(n="size", mean="mean", median="median")
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()

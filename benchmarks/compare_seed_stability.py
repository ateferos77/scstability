"""Benchmark against seed stability (ClustAssessPy).

The point of this comparison is not "which tool is better". The two measure
different things, and the whole positioning of ``scstability`` rests on that
difference being real rather than rhetorical:

* **Seed stability** reruns the *same* clustering on the *same* graph with a
  different random seed. It asks whether community detection lands in a
  consistent local optimum. ``ClustAssessPy`` measures it with element-centric
  consistency (ECS); ``scICE`` measures it with an inconsistency coefficient.
* **Sampling stability** reclusters a *resample of the cells*. It asks whether
  the cluster would still be there had you sequenced a different subset of the
  same tissue. This is what ``scstability`` measures.

So this benchmark tests three claims, in increasing order of how much they
could embarrass us:

1. **Convergent validity.** Both measure something real, so they should agree
   more than chance.
2. **Discriminant validity.** They are *not* substitutes, so they should not
   agree almost perfectly. If the correlation were near 1, ``scstability``
   would be an expensive way to compute a number ``ClustAssessPy`` gets ~30x
   cheaper, and the honest conclusion would be to tell users to use that
   instead. The specific prediction is that clusters exist which are
   seed-stable and sampling-unstable -- a partition the algorithm finds
   reliably on this graph, but which does not survive changing the cells.
3. **Criterion validity.** On data with genuine external ground truth, which
   measure better identifies clusters that recover a real biological group?

Fairness of the comparison
--------------------------
The seed-varied clusterings are generated here rather than through
``assess_clustering_stability``, so that both arms use *identical* clustering:
the same kNN graph, the same Leiden implementation, the same flavour and
iteration count, the same resolution. The only difference between the arms is
what is perturbed -- the seed, or the cells. ``ClustAssessPy`` supplies the
statistic (``element_consistency``), which is its actual contribution.

Usage
-----
::

    python benchmarks/compare_seed_stability.py DATA.h5ad --truth ground_truth

Requires ``ClustAssessPy`` (and ``louvain``, which needs a conda-forge build).
It is deliberately not a dependency of this package, and this is not a CI test.
"""

from __future__ import annotations

import argparse
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

import scstability as scs

RESOLUTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]


def leiden_at(adata: ad.AnnData, resolution: float, seed: int) -> np.ndarray:
    """Cluster the prebuilt graph, varying only the seed."""
    key = "_seed_run"
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=key,
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=seed,
    )
    return adata.obs[key].astype(int).to_numpy()


def seed_stability(adata: ad.AnnData, resolution: float, n_seeds: int) -> np.ndarray:
    """Per-cell element-centric consistency across ``n_seeds`` reseedings."""
    from ClustAssessPy import element_consistency

    partitions = [leiden_at(adata, resolution, seed) for seed in range(n_seeds)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(element_consistency(partitions), dtype=float)


def _partial_spearman(x, y, z) -> tuple[float, float]:
    """Spearman of ``x`` and ``y`` with ``z`` partialled out, computed on ranks.

    Threshold-free, which is the point: it asks whether one measure predicts
    the truth once the other is already known, without anyone choosing a cutoff.
    """
    rx, ry, rz = (stats.rankdata(np.asarray(v, dtype=float)) for v in (x, y, z))

    def residual(a, b):
        slope, intercept, *_ = stats.linregress(b, a)
        return a - (slope * b + intercept)

    return stats.pearsonr(residual(rx, rz), residual(ry, rz))


def _rank_r2(target: np.ndarray, predictors: list[np.ndarray]) -> float:
    """Variance in ``target`` explained by a linear model on rank predictors."""
    design = np.column_stack([np.ones(len(target)), *predictors])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(1 - (target - design @ beta).var() / target.var())


def truth_jaccard(mask: np.ndarray, truth: np.ndarray) -> float:
    """Best Jaccard of this cluster against any published group."""
    best = 0.0
    for label in np.unique(truth):
        in_group = truth == label
        union = np.logical_or(mask, in_group).sum()
        if union:
            best = max(best, float(np.logical_and(mask, in_group).sum() / union))
    return best


def main() -> None:
    """Run both arms on the same data and compare them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--truth", default=None)
    parser.add_argument("--n-boot", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--resolutions", type=float, nargs="+", default=RESOLUTIONS)
    args = parser.parse_args()

    pd.set_option("display.width", 210)
    full = ad.read_h5ad(args.path)
    truth = full.obs[args.truth].astype(str).to_numpy() if args.truth else None

    adata = ad.AnnData(np.zeros((full.n_obs, 1), dtype=np.float32))
    adata.obsm["X_pca"] = np.ascontiguousarray(full.obsm["X_pca"], dtype=np.float32)
    print(f"{full.n_obs} cells, {adata.obsm['X_pca'].shape[1]} dims")
    print(f"sampling arm: n_boot={args.n_boot}   seed arm: n_seeds={args.n_seeds}\n")

    # ---- sampling stability (this package) --------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = scs.stability_sweep(
            adata,
            args.resolutions,
            n_boot=args.n_boot,
            n_neighbors=args.n_neighbors,
            random_state=0,
            progress=False,
        )

    # ---- seed stability (ClustAssessPy), on the identical graph -----------
    sc.pp.neighbors(
        adata, n_neighbors=args.n_neighbors, use_rep="X_pca", random_state=0
    )

    rows = []
    per_cell_pairs = []
    for resolution in args.resolutions:
        ecs = seed_stability(adata, resolution, args.n_seeds)
        reference = leiden_at(adata, resolution, 0)
        sampling = result.per_cell[resolution].to_numpy(dtype=float)

        ok = ~np.isnan(sampling)
        per_cell_pairs.append(
            pd.DataFrame(
                {
                    "resolution": resolution,
                    "seed_ecs": ecs[ok],
                    "sampling": sampling[ok],
                    "cluster": reference[ok],
                }
            )
        )

        block = result.cluster_stability
        block = block[block["resolution"] == resolution]
        for _, row in block.iterrows():
            mask = reference == int(row["cluster"])
            if not mask.any():
                continue
            entry = {
                "resolution": resolution,
                "cluster": int(row["cluster"]),
                "n_cells": int(mask.sum()),
                "seed_ecs": float(np.mean(ecs[mask])),
                "sampling_jaccard": float(row["jaccard_mean"]),
            }
            if truth is not None:
                entry["truth_jaccard"] = truth_jaccard(mask, truth)
            rows.append(entry)

    clusters = pd.DataFrame(rows).dropna(subset=["sampling_jaccard"])
    cells = pd.concat(per_cell_pairs, ignore_index=True)

    # ---------------------------------------------------- 1. convergent
    print("=" * 74)
    print("1. CONVERGENT VALIDITY -- do they agree more than chance?")
    print("=" * 74)
    rho_cell, p_cell = stats.spearmanr(cells["seed_ecs"], cells["sampling"])
    rho_clu, p_clu = stats.spearmanr(clusters["seed_ecs"], clusters["sampling_jaccard"])
    print(
        f"  per cell    Spearman = {rho_cell:+.3f}  p = {p_cell:.2g}  n = {len(cells)}"
    )
    print(
        f"  per cluster Spearman = {rho_clu:+.3f}  p = {p_clu:.2g}  n = {len(clusters)}"
    )
    print(f"  -> {'PASS' if rho_clu > 0 and p_clu < 0.05 else 'FAIL'}")

    # ---------------------------------------------------- 2. discriminant
    print()
    print("=" * 74)
    print("2. DISCRIMINANT VALIDITY -- are they measuring different things?")
    print("=" * 74)
    print(f"  shared variance at cluster level: r^2 = {rho_clu**2:.2f}")
    print(
        f"  -> {'PASS (not substitutes)' if abs(rho_clu) < 0.9 else 'FAIL (near-redundant)'}"
    )

    dissociated = clusters[
        (clusters["seed_ecs"] > 0.9) & (clusters["sampling_jaccard"] < 0.75)
    ]
    print(
        "\n  clusters that are SEED-STABLE (ECS > 0.9) but SAMPLING-UNSTABLE (< 0.75):"
    )
    print(
        f"    {len(dissociated)} of {len(clusters)} ({100 * len(dissociated) / len(clusters):.0f}%)"
    )
    if len(dissociated):
        columns = ["resolution", "cluster", "n_cells", "seed_ecs", "sampling_jaccard"]
        if truth is not None:
            columns.append("truth_jaccard")
        print(
            dissociated.sort_values("sampling_jaccard")[columns]
            .round(3)
            .to_string(index=False)
        )
        print("\n  These are the cases the package exists for: the algorithm finds")
        print("  them reliably on this graph, and they do not survive changing")
        print("  which cells were sequenced.")

    reverse = clusters[
        (clusters["seed_ecs"] < 0.75) & (clusters["sampling_jaccard"] > 0.9)
    ]
    print(f"\n  the converse (seed-unstable but sampling-stable): {len(reverse)}")

    # ---------------------------------------------------- 3. criterion
    if truth is None:
        print("\n(no --truth column; skipping the ground-truth comparison)")
        return

    print()
    print("=" * 74)
    print("3. CRITERION VALIDITY -- which better tracks external ground truth?")
    print("=" * 74)
    rho_s, p_s = stats.spearmanr(
        clusters["sampling_jaccard"], clusters["truth_jaccard"]
    )
    rho_e, p_e = stats.spearmanr(clusters["seed_ecs"], clusters["truth_jaccard"])
    print(f"  sampling stability (scstability)   vs truth: {rho_s:+.3f}  p = {p_s:.2g}")
    print(f"  seed stability     (ClustAssessPy) vs truth: {rho_e:+.3f}  p = {p_e:.2g}")
    better = "scstability" if rho_s > rho_e else "ClustAssessPy"
    print(f"  -> {better} tracks ground truth more closely here")

    # ---------------------------------------- 4. threshold-free comparison
    print()
    print("=" * 74)
    print("4. DOES EITHER ADD INFORMATION THE OTHER LACKS? (no threshold)")
    print("=" * 74)
    partial_s, p_partial_s = _partial_spearman(
        clusters["sampling_jaccard"], clusters["truth_jaccard"], clusters["seed_ecs"]
    )
    partial_e, p_partial_e = _partial_spearman(
        clusters["seed_ecs"], clusters["truth_jaccard"], clusters["sampling_jaccard"]
    )
    print(
        f"  sampling ~ truth, seed controlled for : {partial_s:+.3f}  p = {p_partial_s:.2g}"
    )
    print(
        f"  seed ~ truth, sampling controlled for : {partial_e:+.3f}  p = {p_partial_e:.2g}"
    )

    ranks_truth = stats.rankdata(clusters["truth_jaccard"])
    ranks_samp = stats.rankdata(clusters["sampling_jaccard"])
    ranks_seed = stats.rankdata(clusters["seed_ecs"])
    seed_only = _rank_r2(ranks_truth, [ranks_seed])
    samp_only = _rank_r2(ranks_truth, [ranks_samp])
    both = _rank_r2(ranks_truth, [ranks_seed, ranks_samp])
    print(
        f"\n  R^2 predicting truth: seed {seed_only:.3f}, sampling {samp_only:.3f}, "
        f"both {both:.3f}"
    )
    print(f"    sampling adds {both - seed_only:+.3f} over seed alone")
    print(f"    seed adds     {both - samp_only:+.3f} over sampling alone")
    print(
        f"  -> {'sampling subsumes seed here' if (both - samp_only) < 0.02 else 'both contribute'}"
    )

    # ------------------------------------------------ 5. the saturation test
    # The threshold-based dissociation test above is the obvious one and it is
    # the wrong one. ECS is bounded above and reaches its ceiling easily: once
    # the algorithm finds a partition reliably on a fixed graph, reseeding
    # cannot disturb it, whether or not that partition corresponds to anything
    # real. So the two measures correlate strongly overall while seed stability
    # quietly stops discriminating over a large part of the range.
    #
    # The decisive question is therefore conditional: among the clusters that
    # seed stability calls perfect, does sampling stability still separate the
    # real ones from the arbitrary ones?
    print()
    print("=" * 74)
    print("5. THE SATURATION TEST -- where seed stability runs out of range")
    print("=" * 74)
    # Swept rather than hand-picked. A cutoff chosen after looking at the data
    # -- the first version of this used 0.99, having just seen that the median
    # was 0.988 -- proves nothing on its own. If the effect is real it survives
    # the cutoff moving; if it appears at one value only, it is an artefact.
    print("  sensitivity across cutoffs, rather than one hand-picked value:")
    print(f"    {'ECS >=':>7}  {'n':>4}  {'sampling~truth':>17}  {'seed~truth':>10}")
    wins = considered = 0
    for cutoff in (0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995):
        subset = clusters[clusters["seed_ecs"] >= cutoff]
        if len(subset) < 8:
            print(f"    {cutoff:>7.3f}  {len(subset):>4}  {'(too few)':>17}")
            continue
        r_s, pp_s = stats.spearmanr(subset["sampling_jaccard"], subset["truth_jaccard"])
        r_e, _ = stats.spearmanr(subset["seed_ecs"], subset["truth_jaccard"])
        considered += 1
        wins += int(r_s > r_e and pp_s < 0.05)
        print(
            f"    {cutoff:>7.3f}  {len(subset):>4}  {r_s:+.3f} (p={pp_s:>7.1g})  {r_e:+.3f}"
        )
    if considered:
        print(f"  -> sampling stability wins at {wins}/{considered} cutoffs tested")

    saturated = clusters[clusters["seed_ecs"] >= 0.99]
    if len(saturated) < 8:
        print("\n  seed stability is not saturating on this dataset, so it keeps")
        print("  its discriminating power throughout and this section adds little.")
        return
    print(
        f"\n  at ECS >= 0.99: {len(saturated)} of {len(clusters)} clusters "
        f"({100 * len(saturated) / len(clusters):.0f}%)"
    )

    print(
        f"    their true quality still varies widely: truth_jaccard "
        f"{saturated['truth_jaccard'].min():.3f} to "
        f"{saturated['truth_jaccard'].max():.3f}"
    )
    rho_in_s, p_in_s = stats.spearmanr(
        saturated["sampling_jaccard"], saturated["truth_jaccard"]
    )
    rho_in_e, p_in_e = stats.spearmanr(
        saturated["seed_ecs"], saturated["truth_jaccard"]
    )
    print(
        f"    sampling stability vs truth, within that set: "
        f"{rho_in_s:+.3f}  p = {p_in_s:.2g}  n = {len(saturated)}"
    )
    print(
        f"    seed stability     vs truth, within that set: "
        f"{rho_in_e:+.3f}  p = {p_in_e:.2g}"
    )
    adds = rho_in_s > rho_in_e and p_in_s < 0.05
    print(
        f"  -> sampling stability "
        f"{'ADDS INFORMATION seed stability cannot express' if adds else 'adds no signal here'}"
    )

    print("\n  Reported as measured. Neither tool is designed to predict cell-type")
    print("  recovery, so this is a secondary observation, not either one's")
    print("  stated purpose.")


if __name__ == "__main__":
    main()

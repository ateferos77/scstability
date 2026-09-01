# API reference

Complete reference for `scstability` 0.1.x. For the method, the evidence and
the interpretation bands, see the
[README](https://github.com/ateferos77/scstability#readme).

```bash
pip install scstability
```

```python
import scstability as scs
```

---

## Contents

- [At a glance](#at-a-glance)
- [Sweeping](#sweeping): [`stability_sweep`](#stability_sweep)
- [Reading the result](#reading-the-result): [`StabilityResult`](#stabilityresult) · [`.summary`](#stabilityresultsummary) · [`.recommend`](#stabilityresultrecommend) · [`.to_adata`](#stabilityresultto_adata)
- [Plotting](#plotting): [`pl.stability_curve`](#plstability_curve) · [`pl.cluster_stability`](#plcluster_stability) · [`pl.stability_umap`](#plstability_umap)
- [Constants](#constants): [`HENNIG_BANDS`](#hennig_bands)
- [Input requirements](#input-requirements) · [Errors](#errors) · [Recipes](#recipes)

---

## At a glance

```python
import scanpy as sc
import scstability as scs

adata = sc.read_h5ad("my_data.h5ad")  # needs adata.obsm["X_pca"]
result = scs.stability_sweep(adata, [0.2, 0.4, 0.8, 1.2])
result.summary()  # one row per resolution
```

| Symbol | Kind | Use it when |
|---|---|---|
| `stability_sweep` | function | **Start here.** Runs the whole measurement |
| `StabilityResult` | dataclass | What the sweep returns |
| `.summary()` | method | One row per resolution; read `min_cluster_stability` |
| `.recommend()` | method | You want one resolution chosen for you |
| `.to_adata()` | method | You want the scores back in `adata.obs` |
| `pl.stability_curve` | function | Choosing a resolution |
| `pl.cluster_stability` | function | Finding which cluster to distrust |
| `pl.stability_umap` | function | Seeing where the instability sits |
| `HENNIG_BANDS` | tuple | You are labelling scores programmatically |

Everything else in the package is private. `_metrics`, `_cluster` and `_core`
are implementation and may change without notice.

---

## Sweeping

### `stability_sweep`

```python
stability_sweep(
    adata,
    resolutions,
    *,
    n_boot=20,
    frac=0.8,
    use_rep="X_pca",
    n_neighbors=15,
    random_state=0,
    progress=True,
) -> StabilityResult
```

The only function that does work. For each resolution it clusters the full
data once as a reference, then draws `n_boot` subsamples of the cells,
reclusters each from scratch, and records how much of each reference cluster
comes back together.

| Parameter | Default | Meaning |
|---|---|---|
| `adata` | (required) | `AnnData` with a representation already in `obsm`. **No preprocessing is performed**, so bring your own QC, normalisation and PCA |
| `resolutions` | (required) | Leiden resolutions to sweep. Sorted ascending internally; must be positive and unique |
| `n_boot` | `20` | Replicates per resolution. Minimum 2. Use 50-100 for a published figure |
| `frac` | `0.8` | Fraction of cells per replicate, **without replacement**. Must be in `(0, 1]` |
| `use_rep` | `"X_pca"` | Key in `adata.obsm`. Sliced, never recomputed |
| `n_neighbors` | `15` | Neighbours per cell, clamped down automatically for small subsamples |
| `random_state` | `0` | Master seed. Per-replicate seeds derive from it, so `n_boot=50` begins with the same 20 replicates as `n_boot=20` |
| `progress` | `True` | Show a progress bar. Set `False` in scripts |

Returns a [`StabilityResult`](#stabilityresult).

**Cells are sampled without replacement, and this is deliberate.** Sampling
*with* replacement places duplicate cells at distance zero from one another,
which corrupts a kNN graph: every duplicate becomes its own nearest
neighbour. `chooseR` makes the same choice for the same reason.

**The embedding is held fixed.** `use_rep` is sliced, never recomputed per
replicate, so what is measured is the instability of *graph construction and
community detection*, not of PCA or integration.

```python
# ordinary use
result = scs.stability_sweep(adata, [0.2, 0.4, 0.8, 1.2, 1.6])

# an integrated embedding, more replicates, quiet
result = scs.stability_sweep(
    adata, [0.4, 0.6, 0.8], n_boot=100, use_rep="X_scVI", progress=False
)
```

**Cost** is `n_boot × len(resolutions)` reclusterings. Measured on real PBMCs
with 5 resolutions and 20 bootstraps: 10,000 cells in 60 s, 68,000 cells in
10 min at 2.1 GB peak.

---

## Reading the result

### `StabilityResult`

A frozen dataclass. Construct it only by calling `stability_sweep`.

| Attribute | Type | Contents |
|---|---|---|
| `resolutions` | `ndarray` | The grid, ascending |
| `cluster_stability` | `DataFrame` | One row per cluster per resolution: `resolution`, `cluster`, `n_cells`, `jaccard_mean`, `jaccard_median`, `jaccard_q25`, `jaccard_q75` |
| `per_cell` | `DataFrame` | Cells × resolutions. `NaN` where a cell appeared in no replicate |
| `reference_labels` | `DataFrame` | The full-data clustering per resolution, as categoricals |
| `params` | `dict` | Everything the sweep was called with |

**`jaccard_mean` is the number the interpretation bands apply to.**
`fpc::clusterboot` states Hennig's guidance on the mean over resamples, and
bootstrap Jaccards are left-skewed, so the median reads optimistically. The
median and quartiles are reported alongside as distribution *shape*: a median
far above the mean says a cluster fails rarely but catastrophically.

```python
result.cluster_stability.to_csv("cluster_stability.csv", index=False)
result.params["n_boot"]
```

---

### `StabilityResult.summary`

```python
summary(stable_threshold=0.75) -> pd.DataFrame
```

One row per resolution.

| Column | Meaning |
|---|---|
| `resolution` | From the grid |
| `n_clusters` | Clusters in the reference clustering |
| `min_cluster_stability` | **The headline number.** Mean Jaccard of the *weakest* cluster |
| `median_cluster_stability` | Median *across clusters* of that same mean |
| `frac_cells_stable` | Fraction of evidenced cells scoring ≥ `stable_threshold` |

`stable_threshold` affects only `frac_cells_stable`.

**Read `min_cluster_stability`, not the median.** A resolution is only as
trustworthy as its weakest cluster. Real data routinely shows a median of 0.94
beside a minimum of 0.18. The median hides exactly the cluster you needed.

**Read `n_clusters` alongside the score.** A one-cluster partition scores near
1.0 on structureless data, because the resample collapses the same way.

**`min_cluster_stability` is a minimum, so it is noisy.** Varying only
`random_state` on real data moved it by up to 0.33 across ten seeds at a
resolution with marginal clusters. Raising `n_boot` narrows this but does not
remove it. Near a band edge, run two or three seeds.

Clusters unsampled in *every* replicate have `NaN` and are excluded rather than
counted as zero. That means such a cluster does not drag the minimum down, and is only
reachable at very low `n_boot` (probability `(1 - frac) ** n_boot`, so 4% at
`n_boot=2` but 1e-14 at `n_boot=20`).

---

### `StabilityResult.recommend`

```python
recommend(threshold=0.75) -> float
```

Returns the **largest** resolution whose `min_cluster_stability` is at or above
`threshold`, which is the most granularity available while every cluster still holds.

Resolutions producing a single cluster are never returned.

Raises `ValueError` if no resolution has any evidence at all.

**Warns** (and still returns an answer) when:

- no resolution meets `threshold`, in which case the best available is returned, flagged;
- every resolution yields one cluster, so the scores are trivially perfect.

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    chosen = result.recommend()
for w in caught:
    print(w.message)
```

**It cannot find the true number of clusters, and does not claim to.** A
reproducible over-split of a stable cluster is still reproducible. On
genotype-labelled data where the truth is 5, it returns 6.

---

### `StabilityResult.to_adata`

```python
to_adata(adata, key_added="stability") -> None
```

Writes per-cell scores into `adata.obs`, in place.

| Written | Contents |
|---|---|
| `adata.obs[f"{key_added}_res{r:g}"]` | Per-cell stability, one column per resolution. `float`, `NaN` = no evidence |
| `adata.uns[key_added]` | The sweep parameters |

`adata.obs_names` must match the object the sweep ran on. Warns before
overwriting an existing column; the write still happens.

**`NaN` is not `0.0`.** A cell never drawn into any replicate has no evidence,
which is different from being unstable. That distinction is kept throughout.

```python
result.to_adata(adata)
sc.pl.umap(adata, color="stability_res0.8", cmap="Blues", vmin=0, vmax=1)
confident = adata[adata.obs["stability_res0.8"] > 0.75].copy()
```

---

## Plotting

All three return the matplotlib artist, accept an existing `ax`, and never call
`plt.show()`.

### `pl.stability_curve`

```python
pl.stability_curve(result, threshold=0.75, ax=None) -> Axes | Figure
```

Every cluster at every resolution, with the Hennig bands shaded and a black
line following the weakest cluster. **Use it to choose a resolution.**

Returns a two-panel `Figure` (stability above, cluster count below), or the
`Axes` you passed, in which case the count panel is omitted.

The count is a separate panel rather than a second y-axis. Two y-scales let a
reader draw whatever relationship they came for, since the crossing point is an
artefact of the scaling.

---

### `pl.cluster_stability`

```python
pl.cluster_stability(result, resolution, ax=None) -> Axes
```

One bar per cluster at one resolution, weakest first, coloured by band, with
the interquartile range drawn beside it and cluster sizes labelled. **Use it to
find which cluster to distrust.**

`resolution` must be in the grid.

The IQR is drawn at its absolute position, not as an offset from the bar, so a
mean lying below q25, which marks a cluster that usually reassembles and
occasionally shatters, is visible rather than clipped away.

---

### `pl.stability_umap`

```python
pl.stability_umap(adata, result, resolution, ax=None) -> Axes
```

Your UMAP coloured by per-cell stability, with `NaN` cells in grey and their
own legend entry. **Use it to see where the instability sits**, typically at
the boundaries between clusters, with the cores solid.

Requires `adata.obsm["X_umap"]` and matching `obs_names`.

```python
fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
scs.pl.cluster_stability(result, resolution=0.8, ax=left)
scs.pl.stability_umap(adata, result, resolution=0.8, ax=right)
```

---

## Constants

### `HENNIG_BANDS`

```python
(
    (0.85, "highly stable"),
    (0.75, "stable"),
    (0.60, "a real pattern, but uncertain"),
    (0.00, "not trustworthy -- likely dissolved"),
)
```

Descending lower bounds, inclusive. The published interpretation from Hennig
(2007), as data rather than prose.

```python
def band(value):
    for lower, name in scs.HENNIG_BANDS:
        if value >= lower:
            return name
```

`scs.__version__` is also available.

---

## Input requirements

| Requirement | Why | How |
|---|---|---|
| `adata.obsm[use_rep]`, shape `(n_obs, n_dims)` | The coordinates it clusters | `sc.tl.pca(adata)` |
| Enough cells | Each replicate drops `1 - frac` | ≥ ~200 is comfortable |
| `adata.obsm["X_umap"]` | Only for `pl.stability_umap` | `sc.tl.umap(adata)` |

A one-dimensional array assigned to `obsm` is reshaped by AnnData to
`(n_obs, 1)` rather than rejected, and will be clustered as a single column.

---

## Errors

| Condition | Raised |
|---|---|
| `resolutions` empty, non-positive or duplicated | `ValueError` |
| Two resolutions that collide as `res{r:g}` column names | `ValueError`; use a coarser grid |
| `n_boot < 2` | `ValueError` |
| `frac` outside `(0, 1]` | `ValueError` |
| `use_rep` missing from `obsm` | `ValueError`, listing the available keys |
| Representation not `(n_obs, n_dims)` | `ValueError` |
| `to_adata` / `stability_umap` given mismatched `obs_names` | `ValueError` |
| `stability_umap` without `obsm["X_umap"]` | `ValueError`, naming `sc.tl.umap` |
| `recommend()` with no evidence at any resolution | `ValueError` |

Warnings (never silent, never fatal): no resolution meets the threshold; every
resolution is a single cluster; `to_adata` overwriting a column.

---

## Recipes

**Coarse then fine.** Stability is only meaningful relative to other
resolutions, so sweep wide first and deep second.

```python
coarse = scs.stability_sweep(adata, [0.1, 0.3, 0.6, 1.0, 1.5, 2.0], n_boot=20)
fine = scs.stability_sweep(adata, [0.4, 0.5, 0.6, 0.7, 0.8], n_boot=100)
chosen = fine.recommend()
```

**List the clusters you should not trust.**

```python
weak = fine.cluster_stability.query("resolution == @chosen and jaccard_mean < 0.75")
```

**Find where an unstable cluster goes** when it dissolves. Usually it is a
subdivision of a stable parent, not a population that resampling misses.

```python
parents = fine.reference_labels[0.6].astype(str).to_numpy()
mask = (adata.obs["leiden"].astype(str) == "8").to_numpy()
pd.Series(parents[mask]).value_counts(normalize=True)
```

**Match clusters between two runs by overlap, never by label.** Re-running
Leiden reproduces a partition but renumbers its clusters; mapping by index
silently compares unrelated clusters.

**Reproducibility.** With the same `random_state` and the same `obsm`
representation, results are identical. What varies across machines is the
*upstream* PCA: ARPACK is iterative, and a SciPy patch release can move
coordinates enough to shift a score in the third decimal. Pin your embedding,
not just your seed.

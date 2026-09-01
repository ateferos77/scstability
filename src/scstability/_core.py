"""The resolution sweep and its result object.

Holds :func:`stability_sweep` and :class:`StabilityResult`. This is the only
module that joins the two halves of the package: :mod:`._cluster`, which knows
about scanpy and randomness, and :mod:`._metrics`, which knows about neither.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from tqdm.auto import tqdm

from ._cluster import cluster_subsample, derive_seeds, leiden_labels
from ._metrics import stability

__all__ = ["HENNIG_BANDS", "StabilityResult", "stability_sweep"]

#: Hennig's (2007) interpretation bands for the cluster-wise Jaccard, as
#: ``(lower_bound, label)`` in descending order. Surfaced in the plots and the
#: docs so the output is a verdict rather than a number the user has to invent
#: a rule for.
HENNIG_BANDS: tuple[tuple[float, str], ...] = (
    (0.85, "highly stable"),
    (0.75, "stable"),
    (0.60, "a real pattern, but uncertain"),
    (0.00, "not trustworthy -- likely dissolved"),
)

#: Default cutoff for calling a cluster or a cell stable. Hennig's "stable"
#: band, not a number of our own choosing.
STABLE_THRESHOLD = 0.75


def _resolution_key(resolution: float) -> str:
    """Format a resolution for use in an ``obs`` column name."""
    return f"res{resolution:g}"


def _nanmin_quiet(values: NDArray[np.float64]) -> float:
    """``nanmin`` that returns NaN for an all-NaN input without warning."""
    finite = values[~np.isnan(values)]
    return float(finite.min()) if finite.size else float("nan")


def _nanmedian_quiet(values: NDArray[np.float64]) -> float:
    """``nanmedian`` that returns NaN for an all-NaN input without warning."""
    finite = values[~np.isnan(values)]
    return float(np.median(finite)) if finite.size else float("nan")


@dataclass
class StabilityResult:
    """Everything a sweep produced, and the ways of reading it.

    Attributes
    ----------
    resolutions
        The resolution grid, ascending.
    cluster_stability
        One row per (resolution, cluster). Columns: ``resolution``, ``cluster``,
        ``n_cells``, ``jaccard_mean``, ``jaccard_median``, ``jaccard_q25``,
        ``jaccard_q75``. ``jaccard_mean`` is the statistic Hennig's bands are
        defined on and the one every threshold here is compared against; the
        others describe the shape of the same bootstrap distribution.
    per_cell
        Index ``adata.obs_names``, one float column per resolution. Values in
        ``[0, 1]``, or ``NaN`` for a cell no bootstrap ever drew.
    reference_labels
        Index ``adata.obs_names``, one categorical column per resolution: the
        clustering of the full data that the bootstraps were compared against.
    params
        Every argument the sweep was called with, for reproducibility.
    """

    resolutions: NDArray[np.float64]
    cluster_stability: pd.DataFrame
    per_cell: pd.DataFrame
    reference_labels: pd.DataFrame
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self, stable_threshold: float = STABLE_THRESHOLD) -> pd.DataFrame:
        """One row per resolution: how many clusters, and how well they held.

        Parameters
        ----------
        stable_threshold
            Per-cell stability at or above which a cell counts as stable.

        Returns
        -------
        DataFrame
            Columns ``resolution``, ``n_clusters``, ``min_cluster_stability``,
            ``median_cluster_stability``, ``frac_cells_stable``.

            ``min_cluster_stability`` is the headline number: a resolution is
            only as trustworthy as its weakest cluster. ``frac_cells_stable``
            ignores ``NaN`` cells, so its denominator is the number of cells
            with any evidence at all.

        Notes
        -----
        Clusters with no evidence -- unsampled in *every* bootstrap, so their
        Jaccard mean is ``NaN`` -- are excluded from ``min_cluster_stability``
        and ``median_cluster_stability`` rather than counted as zero, on the
        same "absence is not failure" principle used throughout. The
        consequence is worth knowing: such a cluster does not drag the minimum
        down, so a resolution can look better than the evidence supports. This
        is only reachable for very small clusters with very few bootstraps -- a
        one-cell cluster is missed by all replicates with probability
        ``(1 - frac) ** n_boot``, which is 4% at ``frac=0.8, n_boot=2`` but
        ``1e-14`` at ``n_boot=20``. It is one more reason not to run with a
        handful of bootstraps.

        Examples
        --------
        >>> result.summary()  # doctest: +SKIP
           resolution  n_clusters  min_cluster_stability  ...
        0         0.2           3                  0.981  ...
        """
        rows = []
        for resolution in self.resolutions:
            block = self.cluster_stability[
                self.cluster_stability["resolution"] == resolution
            ]
            # Hennig's bands are defined on the MEAN Jaccard over resamples --
            # fpc::clusterboot reports it as `bootmean` -- so the mean is what
            # every threshold in this package is compared against. Banding the
            # median instead reads optimistically, because bootstrap Jaccards
            # are left-skewed: a cluster that usually reassembles and
            # occasionally shatters has a median well above its mean.
            means = block["jaccard_mean"].to_numpy(dtype=float)
            cells = self.per_cell[resolution].to_numpy(dtype=float)
            evidenced = cells[~np.isnan(cells)]

            rows.append(
                {
                    "resolution": float(resolution),
                    "n_clusters": len(block),
                    "min_cluster_stability": _nanmin_quiet(means),
                    "median_cluster_stability": _nanmedian_quiet(means),
                    "frac_cells_stable": (
                        float((evidenced >= stable_threshold).mean())
                        if evidenced.size
                        else float("nan")
                    ),
                }
            )
        return pd.DataFrame(rows)

    def recommend(self, threshold: float = STABLE_THRESHOLD) -> float:
        """Pick the finest resolution whose clusters all hold up.

        Returns the **largest** resolution whose ``min_cluster_stability`` is at
        or above ``threshold`` -- the most granularity you can ask for while
        every cluster is still trustworthy end to end.

        Resolutions that produce a single cluster are never recommended. Such a
        partition scores near 1.0 on structureless data -- when the resample
        also collapses to one cluster the Jaccard is exactly 1.0 -- so without
        this exclusion a dataset with no structure would be reported as the
        most stable of all.

        Parameters
        ----------
        threshold
            Minimum acceptable cluster stability. Defaults to Hennig's
            "stable" band.

        Returns
        -------
        float
            A resolution from the grid.

        Warns
        -----
        UserWarning
            If no resolution meets ``threshold``. The resolution with the
            highest ``min_cluster_stability`` is returned anyway, so callers
            always get an answer, but the caveat is not silent.
        UserWarning
            If *every* resolution yields a single cluster, since the scores are
            then trivially perfect and carry no information.

        Examples
        --------
        >>> result.recommend()  # doctest: +SKIP
        0.8
        """
        summary = self.summary()

        # Checked first because it is the harder failure: no evidence at all is
        # a data or parameter problem, and must not be softened into the
        # warning below just because the partition also happens to be trivial.
        # (Otherwise idxmax raises pandas' opaque "Encountered all NA values".)
        if summary["min_cluster_stability"].isna().all():
            raise ValueError(
                "No resolution has any stability evidence: every cluster was "
                "unsampled in every bootstrap. Raise n_boot or frac, or check "
                "that the input has enough cells to subsample."
            )

        # A one-cluster partition carries almost no information, and on
        # structureless data it scores near 1.0: the single reference cluster
        # is compared against whatever the resample produces, and when the
        # resample also collapses to one cluster the Jaccard is exactly 1.0.
        # (Not *always* 1.0 -- a resample that splits the cluster in half
        # scores 0.5 -- but a dataset with no structure collapses the same way
        # every time, so in practice the score is a near-perfect number earned
        # for the worst possible reason.) Without this guard such a resolution
        # would be recommended. Found by running the package on real data; see
        # benchmarks/validation.ipynb.
        informative = summary[summary["n_clusters"] >= 2]
        if not len(informative):
            warnings.warn(
                "Every resolution in the grid put all cells in a single "
                "cluster, which is trivially stable and tells you nothing. "
                "The data may have no cluster structure at this scale, or the "
                "grid may be too coarse. Try higher resolutions before reading "
                "anything into the stability scores.",
                UserWarning,
                stacklevel=2,
            )
            return float(summary["resolution"].max())

        qualifying = informative[informative["min_cluster_stability"] >= threshold]
        if len(qualifying):
            return float(qualifying["resolution"].max())

        summary = informative

        best = summary.loc[summary["min_cluster_stability"].idxmax()]
        warnings.warn(
            f"No resolution reached a minimum cluster stability of {threshold}. "
            f"The best available is resolution {best['resolution']:g} at "
            f"{best['min_cluster_stability']:.3f}, which is below the threshold. "
            f"Treat every cluster at this resolution as provisional, and consider "
            f"a coarser grid or more cells.",
            UserWarning,
            stacklevel=2,
        )
        return float(best["resolution"])

    def to_adata(self, adata: AnnData, key_added: str = "stability") -> None:
        """Write per-cell stability into ``adata.obs`` and metadata into ``uns``.

        Parameters
        ----------
        adata
            The object the sweep was run on. Its ``obs_names`` must match.
        key_added
            Prefix for the new columns.

        Warns
        -----
        UserWarning
            If any column would be overwritten. The write still happens; it is
            just not silent.

        Examples
        --------
        >>> result.to_adata(adata)  # doctest: +SKIP
        >>> adata.obs["stability_res0.8"]  # doctest: +SKIP
        """
        if not adata.obs_names.equals(self.per_cell.index):
            raise ValueError(
                "adata.obs_names do not match the cells this result was computed "
                "on. Pass the same AnnData that stability_sweep was called with."
            )

        columns = {f"{key_added}_{_resolution_key(r)}": r for r in self.resolutions}
        if len(columns) != len(self.resolutions):
            # `:g` keeps six significant figures, so resolutions differing only
            # beyond that collapse to one name and would silently overwrite each
            # other. Distinct resolutions must get distinct columns.
            raise ValueError(
                f"resolutions {self.resolutions.tolist()} do not produce distinct "
                f"obs column names under the 'res{{value:g}}' format; they differ "
                f"only beyond six significant figures. Use a coarser grid."
            )

        clashes = [name for name in columns if name in adata.obs.columns]
        if clashes:
            warnings.warn(
                f"Overwriting existing obs column(s): {', '.join(clashes)}. "
                f"Pass a different key_added to keep them.",
                UserWarning,
                stacklevel=2,
            )

        for name, resolution in columns.items():
            adata.obs[name] = self.per_cell[resolution].to_numpy(dtype=float)

        adata.uns[key_added] = {
            "params": dict(self.params),
            "summary": self.summary(),
            "cluster_stability": self.cluster_stability,
        }


def _get_representation(adata: AnnData, use_rep: str) -> NDArray[np.floating]:
    """Fetch and validate the embedding the sweep will run on."""
    if use_rep not in adata.obsm:
        available = list(adata.obsm.keys())
        raise ValueError(
            f"adata.obsm has no key {use_rep!r}. Available keys: "
            f"{available if available else '(none)'}. scstability does no "
            f"preprocessing of its own -- compute a representation first, for "
            f"example with `sc.pp.pca(adata)`, which writes 'X_pca'."
        )
    X = np.asarray(adata.obsm[use_rep])
    if X.ndim != 2 or X.shape[0] != adata.n_obs:
        raise ValueError(
            f"adata.obsm[{use_rep!r}] must have shape (n_obs, n_dims); got "
            f"{X.shape} for {adata.n_obs} cells."
        )
    return X


def _validate_sweep_args(
    resolutions: NDArray[np.float64], n_boot: int, frac: float
) -> None:
    """Reject impossible arguments before any expensive clustering starts."""
    if resolutions.size == 0:
        raise ValueError("resolutions must contain at least one value")
    if np.any(resolutions <= 0):
        raise ValueError(
            f"resolutions must all be positive, got {resolutions.tolist()}"
        )
    if np.unique(resolutions).size != resolutions.size:
        raise ValueError(f"resolutions must be unique, got {resolutions.tolist()}")
    if n_boot < 2:
        raise ValueError(
            f"n_boot must be at least 2 to have a distribution to summarise, "
            f"got {n_boot}"
        )
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")


def stability_sweep(
    adata: AnnData,
    resolutions: Any,
    *,
    n_boot: int = 20,
    frac: float = 0.8,
    use_rep: str = "X_pca",
    n_neighbors: int = 15,
    random_state: int | None = 0,
    progress: bool = True,
) -> StabilityResult:
    """Measure how well each clustering survives resampling of the cells.

    For every resolution: cluster the full data once to get a reference, then
    repeatedly draw ``frac`` of the cells without replacement, recluster them,
    and score how well each reference cluster reappears (Hennig 2007).

    Parameters
    ----------
    adata
        Annotated data matrix with a representation already in ``obsm``. No
        preprocessing is performed; bring your own QC, normalisation and PCA.
    resolutions
        Leiden resolutions to sweep. Sorted ascending internally.
    n_boot
        Bootstrap replicates per resolution. At least 2.
    frac
        Fraction of cells drawn per replicate, **without replacement**. With
        replacement would place duplicate cells at distance zero and corrupt
        the kNN graph.
    use_rep
        Key in ``adata.obsm`` holding the embedding. Sliced, never recomputed:
        this isolates instability of graph construction and community detection
        from instability of the embedding itself.
    n_neighbors
        Neighbours per cell, clamped down for small subsamples.
    random_state
        Master seed. Per-bootstrap seeds are derived from it deterministically,
        so the whole sweep is reproducible.
    progress
        Show a progress bar.

    Returns
    -------
    StabilityResult
        Use ``.summary()``, ``.recommend()`` and ``.to_adata()`` to read it.

    See Also
    --------
    StabilityResult.recommend : pick a resolution from the result.

    Notes
    -----
    The embedding is computed once and sliced, never recomputed per replicate.
    See ``recompute_pca`` in the roadmap.

    Examples
    --------
    >>> import scanpy as sc  # doctest: +SKIP
    >>> import scstability as scs  # doctest: +SKIP
    >>> adata = sc.datasets.pbmc3k_processed()  # doctest: +SKIP
    >>> result = scs.stability_sweep(  # doctest: +SKIP
    ...     adata, resolutions=[0.2, 0.4, 0.8], n_boot=20
    ... )
    >>> result.recommend()  # doctest: +SKIP
    0.4
    """
    grid = np.sort(np.asarray(resolutions, dtype=float).ravel())
    _validate_sweep_args(grid, n_boot, frac)
    X = _get_representation(adata, use_rep)

    params = {
        "resolutions": grid.tolist(),
        "n_boot": int(n_boot),
        "frac": float(frac),
        "use_rep": use_rep,
        "n_neighbors": int(n_neighbors),
        "random_state": random_state,
    }

    seeds = derive_seeds(random_state, n_boot)
    obs_names = adata.obs_names

    cluster_rows: list[dict[str, Any]] = []
    per_cell: dict[float, NDArray[np.float64]] = {}
    reference: dict[float, NDArray[np.int64]] = {}

    bar = tqdm(
        total=grid.size * n_boot,
        disable=not progress,
        desc="stability sweep",
        unit="fit",
    )
    try:
        for resolution in grid:
            ref_labels = leiden_labels(
                X, resolution=resolution, n_neighbors=n_neighbors, seed=random_state
            )

            # TODO(v0.2): parallelise over bootstraps -- iterations are
            # independent and their seeds are fixed before the loop starts, so
            # results do not depend on execution order.
            boots = np.empty((n_boot, adata.n_obs), dtype=np.int64)
            for b in range(n_boot):
                boots[b], _ = cluster_subsample(
                    X,
                    frac=frac,
                    resolution=resolution,
                    n_neighbors=n_neighbors,
                    seed=int(seeds[b]),
                )
                bar.update(1)

            arrays = stability(ref_labels, boots)
            reference[float(resolution)] = ref_labels
            per_cell[float(resolution)] = arrays.per_cell
            for i, cluster in enumerate(arrays.cluster_ids):
                cluster_rows.append(
                    {
                        "resolution": float(resolution),
                        "cluster": int(cluster),
                        "n_cells": int(arrays.n_cells[i]),
                        "jaccard_mean": float(arrays.jaccard_mean[i]),
                        "jaccard_median": float(arrays.jaccard_median[i]),
                        "jaccard_q25": float(arrays.jaccard_q25[i]),
                        "jaccard_q75": float(arrays.jaccard_q75[i]),
                    }
                )
    finally:
        bar.close()

    return StabilityResult(
        resolutions=grid,
        cluster_stability=pd.DataFrame(cluster_rows),
        per_cell=pd.DataFrame(per_cell, index=obs_names),
        reference_labels=pd.DataFrame(
            {r: pd.Categorical(v) for r, v in reference.items()}, index=obs_names
        ),
        params=params,
    )

"""Subsample, build the neighbour graph, run Leiden.

The only module permitted to hold subsample-space indices. Everything it
returns is scattered back into original cell space before it leaves, so that
``_metrics`` and ``_core`` see one coordinate system and one only.

Why subsampling and not a classical bootstrap
---------------------------------------------
Cells are drawn **without replacement**. Sampling with replacement would place
duplicate cells at distance zero from one another, which corrupts any
k-nearest-neighbour graph built afterwards: a cell's neighbourhood fills up
with copies of itself. ``chooseR`` subsamples for the same reason. This is a
deliberate departure from the textbook bootstrap, not an oversight.

Why the embedding is sliced rather than recomputed
--------------------------------------------------
The representation is computed once on the full data by the user and then
indexed. This isolates instability arising from graph construction and
community detection from instability in the embedding itself.
"""

from __future__ import annotations

import numpy as np
import scanpy as sc
from anndata import AnnData
from numpy.typing import ArrayLike, NDArray

from ._metrics import NOT_SAMPLED

__all__ = [
    "cluster_subsample",
    "derive_seeds",
    "leiden_labels",
    "subsample_indices",
]

#: Scratch key used inside the throwaway AnnData; never reaches the user's object.
_LEIDEN_KEY = "_scstability_leiden"

#: Upper bound for derived seeds. Kept inside int32 because igraph's random
#: number generator, which Leiden ultimately seeds, is a 32-bit interface.
_MAX_SEED = 2**31 - 1


def derive_seeds(random_state: int | None, n_boot: int) -> NDArray[np.int64]:
    """Deterministic per-bootstrap seeds derived from one master seed.

    Every seed is drawn up front, before any clustering runs. That makes
    bootstrap ``b`` use the same seed regardless of the order the loop happens
    to execute in, which is what keeps results reproducible under a future
    parallel implementation as well as the current serial one.

    Parameters
    ----------
    random_state
        Master seed. ``None`` gives non-reproducible results.
    n_boot
        How many seeds to derive.

    Returns
    -------
    ndarray
        Length ``n_boot``, dtype int64, values in ``[0, 2**31 - 1)``.

    Examples
    --------
    >>> derive_seeds(0, 3) is not None
    True
    >>> bool((derive_seeds(0, 3) == derive_seeds(0, 3)).all())
    True
    >>> bool((derive_seeds(0, 5)[:3] == derive_seeds(0, 3)).all())
    True
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    rng = np.random.default_rng(random_state)
    return rng.integers(0, _MAX_SEED, size=n_boot, dtype=np.int64)


def subsample_indices(n_obs: int, frac: float, seed: int) -> NDArray[np.int64]:
    """Draw ``round(frac * n_obs)`` cell indices without replacement, sorted.

    Parameters
    ----------
    n_obs
        Total number of cells.
    frac
        Fraction to keep, in ``(0, 1]``.
    seed
        Seed for this draw.

    Returns
    -------
    ndarray
        Sorted original-space cell indices.

    Notes
    -----
    At ``frac=1.0`` the draw is a permutation of every index, and sorting turns
    it back into ``arange(n_obs)``. The sliced matrix is therefore byte-identical
    to the full one, and the bootstrap graph identical to the reference graph.
    That is what makes the ``frac=1.0`` invariant test meaningful: any remaining
    difference in the partition comes from Leiden's seed, not from the data.

    Examples
    --------
    >>> subsample_indices(10, 0.5, seed=0).size
    5
    >>> import numpy as np
    >>> bool((subsample_indices(10, 1.0, seed=3) == np.arange(10)).all())
    True
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    n_sub = round(frac * n_obs)
    if n_sub < 2:
        raise ValueError(
            f"frac={frac} on {n_obs} cells leaves {n_sub} cell(s); at least 2 are "
            f"needed to build a neighbour graph. Raise frac or use more cells."
        )
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_obs, size=n_sub, replace=False))


def effective_n_neighbors(n_neighbors: int, n_obs: int) -> int:
    """Neighbours to request, clamped to what the subsample can support.

    Small subsamples can have fewer cells than the requested ``n_neighbors``.
    Left to itself, scanpy silently rewrites the value -- and not to ``n - 1``
    but to a fixed fallback of 5, logging ``"n_obs too small: adjusting to
    n_neighbors = 5"`` as it goes. That has two costs: the graph it builds is
    sparser than the one we asked for (on 8 cells, 42 edges instead of 54), and
    the message fires once per bootstrap, which across a sweep is a warning
    storm. Clamping here keeps the graph ours and the log quiet.

    Parameters
    ----------
    n_neighbors
        Requested neighbours per cell.
    n_obs
        Cells available in this subsample.

    Returns
    -------
    int
        ``min(n_neighbors, n_obs - 1)`` -- a cell cannot be its own neighbour.

    Examples
    --------
    >>> effective_n_neighbors(15, 300)
    15
    >>> effective_n_neighbors(15, 8)
    7
    """
    return min(int(n_neighbors), n_obs - 1)


def leiden_labels(
    X: ArrayLike, *, resolution: float, n_neighbors: int, seed: int | None
) -> NDArray[np.int64]:
    """Build a kNN graph on a coordinate matrix and run Leiden on it.

    The single place scanpy is called. Isolating it here is what lets the
    scatter-back logic in :func:`cluster_subsample` be tested with a stand-in
    clusterer, deterministically and without running Leiden at all.

    Parameters
    ----------
    X
        Coordinates, shape ``(n, n_dims)``. Usually a slice of
        ``adata.obsm[use_rep]``.
    resolution
        Leiden resolution.
    n_neighbors
        Neighbours per cell. Clamped to ``n - 1`` when the matrix has fewer
        rows than that, which happens for small subsamples.
    seed
        Seeds both the neighbour search and Leiden.

    Returns
    -------
    ndarray
        Length ``n`` integer labels, in subsample space. **Callers must not let
        this array escape without scattering it back** -- see
        :func:`cluster_subsample`.

    Notes
    -----
    ``flavor="igraph"`` with ``n_iterations=2`` is scanpy's recommended path;
    the older ``leidenalg`` flavor is deprecated upstream. Both are pinned
    explicitly rather than left to the default, so a change to scanpy's default
    cannot silently change our numbers.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 cells to cluster, got {n}")

    adata = AnnData(X)
    sc.pp.neighbors(
        adata,
        n_neighbors=effective_n_neighbors(n_neighbors, n),
        use_rep="X",
        random_state=seed,
    )
    sc.tl.leiden(
        adata,
        resolution=float(resolution),
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=seed,
        key_added=_LEIDEN_KEY,
    )
    return adata.obs[_LEIDEN_KEY].to_numpy().astype(np.int64)


def cluster_subsample(
    X: ArrayLike,
    *,
    frac: float,
    resolution: float,
    n_neighbors: int,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Subsample cells, recluster them, and return labels in original cell space.

    Parameters
    ----------
    X
        Full coordinate matrix, shape ``(n_obs, n_dims)``.
    frac
        Fraction of cells to draw, without replacement.
    resolution, n_neighbors
        Passed through to :func:`leiden_labels`.
    seed
        Seeds the draw and the clustering.

    Returns
    -------
    labels
        Length ``n_obs``. Position ``i`` is cell ``i``'s bootstrap cluster, or
        ``NOT_SAMPLED`` if the cell was not drawn.
    idx
        The sorted original-space indices that were drawn.

    Notes
    -----
    The two lines that scatter ``sub_labels`` into ``labels`` are the entire
    translation between subsample space and original cell space. Leiden numbers
    its output ``0..n_sub-1`` by *position in the subsample*, so row 3 of a
    subsample may be cell 57. Comparing those numbers against reference labels
    without translating produces scores that are still in ``[0, 1]`` and still
    look plausible, but are meaningless. Keeping the translation to one place
    means there is exactly one line where that bug could live.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(50, 5))
    >>> labels, idx = cluster_subsample(
    ...     X, frac=0.8, resolution=1.0, n_neighbors=10, seed=0
    ... )
    >>> labels.shape
    (50,)
    >>> bool((np.flatnonzero(labels != NOT_SAMPLED) == idx).all())
    True
    """
    X = np.asarray(X)
    n_obs = X.shape[0]

    idx = subsample_indices(n_obs, frac, seed)
    sub_labels = leiden_labels(
        X[idx], resolution=resolution, n_neighbors=n_neighbors, seed=seed
    )

    labels = np.full(n_obs, NOT_SAMPLED, dtype=np.int64)
    labels[idx] = sub_labels
    return labels, idx

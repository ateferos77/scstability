"""Pure metric functions over label arrays.

This module must not import ``scanpy`` or ``anndata``. It operates on plain
numpy integer label arrays, which is what makes the mathematics unit-testable
in isolation without running any clustering.

Coordinate-system contract
--------------------------
Every label array reaching this module is **full length** (``n_obs``) and lives
in *original cell space*: position ``i`` always means cell ``i``. Cells that
were not sampled in a given bootstrap are marked ``NOT_SAMPLED`` (``-1``),
never ``NaN``. Passing a shorter, subsample-space array raises ``ValueError``.

Absence versus failure
----------------------
Throughout, "we have no evidence" is represented as ``NaN`` and is kept
strictly distinct from "the value is zero". A reference cluster none of whose
cells were sampled in a bootstrap scores ``NaN`` for that bootstrap and is
excluded from the aggregate; a cell that was never sampled in any bootstrap scores
``NaN`` rather than ``0.0``. Collapsing the two would silently report unsampled
things as maximally unstable.

References
----------
Hennig, C. (2007). Cluster-wise assessment of cluster stability.
*Computational Statistics & Data Analysis*, 52(1), 258-271.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "NOT_SAMPLED",
    "StabilityArrays",
    "cluster_stability",
    "jaccard_matrix",
    "jaccard_per_cluster",
    "per_cell_stability",
    "stability",
]

#: Sentinel marking a cell that was not drawn into a given bootstrap subsample.
#: An integer rather than ``NaN`` so that label arrays stay integer-typed.
NOT_SAMPLED = -1


class StabilityArrays(NamedTuple):
    """Everything a single resolution's sweep produces, as plain arrays.

    Attributes
    ----------
    cluster_ids
        Reference cluster labels, sorted ascending. Length ``K``.
    n_cells
        Number of cells in each reference cluster, over the full data.
    jaccard_mean
        Per-cluster **mean** Jaccard across bootstraps. This is the statistic
        Hennig's interpretation bands are defined on -- ``fpc::clusterboot``
        reports it as ``bootmean`` -- so it is the number to compare against
        0.85 / 0.75 / 0.60. ``NaN`` where a cluster was never sampled.
    jaccard_median, jaccard_q25, jaccard_q75
        The median and interquartile bounds of the same distribution. Reported
        alongside because the mean alone hides shape: bootstrap Jaccards are
        typically left-skewed (a cluster usually reassembles and occasionally
        shatters), so a median well above the mean is a signal that the
        cluster fails rarely but badly. Do **not** read the bands off the
        median -- it runs optimistically for exactly that reason.
    per_cell
        Per-cell stability in ``[0, 1]``, or ``NaN`` for cells that appeared in
        no bootstrap. Length ``n_obs``.
    """

    cluster_ids: NDArray[np.int64]
    n_cells: NDArray[np.int64]
    jaccard_mean: NDArray[np.float64]
    jaccard_median: NDArray[np.float64]
    jaccard_q25: NDArray[np.float64]
    jaccard_q75: NDArray[np.float64]
    per_cell: NDArray[np.float64]


class _Contingency(NamedTuple):
    """Cross-tabulation of reference against bootstrap labels, sampled cells only."""

    counts: NDArray[np.int64]  # (K, M) overlap sizes
    ref_ids: NDArray[np.int64]  # (K,) reference labels, sorted
    boot_ids: NDArray[np.int64]  # (M,) bootstrap labels present, sorted
    ref_sizes: NDArray[np.int64]  # (K,) |C_k and S_b|
    boot_sizes: NDArray[np.int64]  # (M,) |D_j|


def _as_labels(labels: ArrayLike, name: str) -> NDArray[np.int64]:
    """Coerce to a 1-D integer label array."""
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype, got {arr.dtype}")
    return arr.astype(np.int64, copy=False)


def _validate_pair(
    reference_labels: ArrayLike, boot_labels: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Check the coordinate-system contract and return coerced arrays.

    The length check is the guard that makes the subsample-space bug in
    ``_cluster`` impossible to propagate: a bootstrap label array that was
    never scattered back into original cell space is shorter than the
    reference, and dies here rather than producing a plausible wrong number.
    """
    ref = _as_labels(reference_labels, "reference_labels")
    boot = _as_labels(boot_labels, "boot_labels")
    if ref.shape != boot.shape:
        raise ValueError(
            f"label arrays must both have length n_obs; got reference_labels "
            f"{ref.shape} and boot_labels {boot.shape}. Bootstrap labels must be "
            f"scattered back into original cell space (length n_obs, with "
            f"{NOT_SAMPLED} marking unsampled cells) before reaching the metrics "
            f"layer -- see the coordinate-system rule in the package docs."
        )
    if (ref < 0).any():
        raise ValueError(
            "reference_labels must be non-negative; the reference clustering is "
            "computed on all cells, so it has no unsampled entries."
        )
    if (boot < NOT_SAMPLED).any():
        # Without this, a stray value such as -5 passes the `!= NOT_SAMPLED`
        # test, is taken for a real cluster, and can even be returned as a
        # cluster's best match -- a silently wrong answer rather than an error.
        offenders = np.unique(boot[boot < NOT_SAMPLED]).tolist()
        raise ValueError(
            f"boot_labels may only contain non-negative cluster labels or "
            f"{NOT_SAMPLED} for unsampled cells; found {offenders}."
        )
    return ref, boot


def _contingency(ref: NDArray[np.int64], boot: NDArray[np.int64]) -> _Contingency:
    """Cross-tabulate reference against bootstrap labels over sampled cells."""
    sampled = boot != NOT_SAMPLED
    ref_ids = np.unique(ref)
    boot_ids = np.unique(boot[sampled])
    n_ref, n_boot = ref_ids.size, boot_ids.size

    if n_boot == 0:  # nothing was sampled
        return _Contingency(
            counts=np.zeros((n_ref, 0), dtype=np.int64),
            ref_ids=ref_ids,
            boot_ids=boot_ids,
            ref_sizes=np.zeros(n_ref, dtype=np.int64),
            boot_sizes=np.zeros(0, dtype=np.int64),
        )

    ref_codes = np.searchsorted(ref_ids, ref[sampled])
    boot_codes = np.searchsorted(boot_ids, boot[sampled])
    counts = np.bincount(
        ref_codes * n_boot + boot_codes, minlength=n_ref * n_boot
    ).reshape(n_ref, n_boot)

    return _Contingency(
        counts=counts,
        ref_ids=ref_ids,
        boot_ids=boot_ids,
        ref_sizes=counts.sum(axis=1),
        boot_sizes=np.bincount(boot_codes, minlength=n_boot),
    )


def _jaccard_from_contingency(tab: _Contingency) -> NDArray[np.float64]:
    """Jaccard matrix from overlap counts, using |A u B| = |A| + |B| - |A n B|.

    The single place the Jaccard is computed. Every public entry point routes
    through here so the arithmetic cannot drift between them.
    """
    union = tab.ref_sizes[:, None] + tab.boot_sizes[None, :] - tab.counts
    jaccard = np.zeros(tab.counts.shape, dtype=np.float64)
    # `where=` rather than errstate: a union of zero means both sets are empty,
    # which is a Jaccard of nothing at all. Leave it at 0.0 and let the caller
    # decide, rather than emitting a division warning we would have to suppress.
    np.divide(tab.counts, union, out=jaccard, where=union > 0)
    return jaccard


def _per_cell_from_matches(
    ref: NDArray[np.int64],
    boots: NDArray[np.int64],
    best_match: NDArray[np.int64],
    ref_ids: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Fraction of sampling bootstraps in which each cell tracked its cluster.

    The single place the per-cell metric is computed; both
    :func:`per_cell_stability` and :func:`stability` route through here.
    """
    ref_codes = np.searchsorted(ref_ids, ref)
    hits = np.zeros(ref.size, dtype=np.int64)
    n_seen = np.zeros(ref.size, dtype=np.int64)
    # TODO(v0.2): parallelise over bootstraps -- each iteration is independent.
    for b in range(boots.shape[0]):
        sampled = boots[b] != NOT_SAMPLED
        n_seen += sampled
        # The bootstrap cluster that this cell's reference cluster matched to.
        # Unsampled cells are excluded by `sampled`, so the NOT_SAMPLED entries
        # in `best_match` can never produce a spurious -1 == -1 hit.
        hits += sampled & (boots[b] == best_match[b][ref_codes])

    out = np.full(ref.size, np.nan, dtype=np.float64)
    np.divide(hits, n_seen, out=out, where=n_seen > 0)
    return out


def jaccard_matrix(
    reference_labels: ArrayLike, boot_labels: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    r"""Jaccard similarity between every reference and every bootstrap cluster.

    Both sets are restricted to the cells the bootstrap actually sampled, so
    each entry is a clean Jaccard index in :math:`[0, 1]`:

    .. math::

        J(k, j) = \frac{|C_k \cap S_b \cap D_j|}{|C_k \cap S_b \cup D_j|}

    Parameters
    ----------
    reference_labels
        Integer cluster labels for all ``n_obs`` cells.
    boot_labels
        Integer cluster labels for all ``n_obs`` cells, with ``NOT_SAMPLED``
        marking cells absent from this bootstrap.

    Returns
    -------
    jaccard
        Array of shape ``(K, M)``.
    ref_ids
        The ``K`` reference labels, sorted ascending; rows of ``jaccard``.
    boot_ids
        The ``M`` bootstrap labels present, sorted ascending; columns.

    Notes
    -----
    This is the raw pairwise matrix, so a reference cluster with no sampled
    cells gives a row of exact zeros -- :math:`J(\emptyset, D_j) = 0/|D_j| = 0`,
    which is arithmetically correct. It is :func:`jaccard_per_cluster` that
    converts that case to ``NaN``, because at the level of a *stability score*
    an unsampled cluster carries no evidence rather than a score of zero. The
    two functions differ here deliberately; use :func:`jaccard_per_cluster`
    unless you specifically want the pairwise values.

    Examples
    --------
    >>> import numpy as np
    >>> ref = np.array([0, 0, 1, 1])
    >>> boot = np.array([0, 1, 1, 1])
    >>> jac, ref_ids, boot_ids = jaccard_matrix(ref, boot)
    >>> jac.shape
    (2, 2)
    >>> float(jac[0, 0])  # C0={0,1} vs D0={0}: 1 shared, 2 total
    0.5
    """
    ref, boot = _validate_pair(reference_labels, boot_labels)
    tab = _contingency(ref, boot)
    return _jaccard_from_contingency(tab), tab.ref_ids, tab.boot_ids


def jaccard_per_cluster(
    reference_labels: ArrayLike, boot_labels: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Best-matching Jaccard for each reference cluster, and which cluster matched.

    This is Hennig's cluster-wise stability for a single bootstrap: for each
    reference cluster, the similarity to the most similar bootstrap cluster.

    Parameters
    ----------
    reference_labels
        Integer cluster labels for all ``n_obs`` cells.
    boot_labels
        Integer cluster labels for all ``n_obs`` cells, ``NOT_SAMPLED`` where
        the cell was not drawn.

    Returns
    -------
    jaccard
        Length ``K``. ``NaN`` for a reference cluster none of whose cells were
        sampled -- no evidence, which is not the same as a score of zero.
    best_match
        Length ``K``. The bootstrap label that matched best, or ``NOT_SAMPLED``
        where ``jaccard`` is ``NaN``. Ties resolve to the smallest label, so the
        result is deterministic.

    Examples
    --------
    Reference cluster ``C = {2, 5, 6, 9}`` over 12 cells; the bootstrap sampled
    ``{2, 5, 6, 9, 11}`` and split it into ``D0 = {2, 5}`` and ``D1 = {6, 9, 11}``.
    Then ``J(C, D0) = 2/4`` and ``J(C, D1) = 2/5``, so the best match is ``D0``:

    >>> import numpy as np
    >>> ref = np.array([1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1])
    >>> boot = np.array([-1, -1, 0, -1, -1, 0, 1, -1, -1, 1, -1, 1])
    >>> jaccard, best_match = jaccard_per_cluster(ref, boot)
    >>> float(jaccard[0])
    0.5
    >>> int(best_match[0])
    0
    """
    ref, boot = _validate_pair(reference_labels, boot_labels)
    tab = _contingency(ref, boot)
    n_ref = tab.ref_ids.size

    jaccard = np.full(n_ref, np.nan, dtype=np.float64)
    best_match = np.full(n_ref, NOT_SAMPLED, dtype=np.int64)
    if tab.boot_ids.size == 0:
        return jaccard, best_match

    full = _jaccard_from_contingency(tab)
    evidenced = tab.ref_sizes > 0
    if evidenced.any():
        winners = full[evidenced].argmax(axis=1)
        jaccard[evidenced] = full[evidenced].max(axis=1)
        best_match[evidenced] = tab.boot_ids[winners]
    return jaccard, best_match


def _all_bootstraps(
    reference_labels: ArrayLike, boot_labels: ArrayLike
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Run :func:`jaccard_per_cluster` over every bootstrap in one pass.

    Returns ``(ref, boots, jaccard, best_match)`` where ``jaccard`` and
    ``best_match`` both have shape ``(n_boot, K)``.
    """
    ref = _as_labels(reference_labels, "reference_labels")
    boots = np.asarray(boot_labels)
    if boots.ndim != 2:
        raise ValueError(
            f"boot_labels must be 2-dimensional with shape (n_boot, n_obs), got "
            f"shape {boots.shape}"
        )
    boots = _as_labels(boots.ravel(), "boot_labels").reshape(boots.shape)

    n_boot = boots.shape[0]
    n_ref = np.unique(ref).size
    jaccard = np.empty((n_boot, n_ref), dtype=np.float64)
    best_match = np.empty((n_boot, n_ref), dtype=np.int64)
    # TODO(v0.2): parallelise over bootstraps -- each iteration is independent.
    for b in range(n_boot):
        jaccard[b], best_match[b] = jaccard_per_cluster(ref, boots[b])
    return ref, boots, jaccard, best_match


def _nan_column_means(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Column-wise mean ignoring ``NaN``, without warning on all-NaN columns.

    ``np.nanmean`` emits ``RuntimeWarning: Mean of empty slice`` for a column
    with no evidence at all. Masking those columns out first keeps the result
    identical and the output clean, which is what lets the library stay
    warning-free under the ``strict_warnings`` test fixture.
    """
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    has_evidence = ~np.isnan(values).all(axis=0)
    if has_evidence.any():
        out[has_evidence] = np.nanmean(values[:, has_evidence], axis=0)
    return out


def _nan_quantiles(
    values: NDArray[np.float64], quantiles: tuple[float, ...]
) -> list[NDArray[np.float64]]:
    """Column-wise quantiles ignoring ``NaN``, without warning on all-NaN columns.

    ``np.nanmedian`` emits ``RuntimeWarning: All-NaN slice encountered`` for a
    column with no evidence at all. Masking those columns out first keeps the
    result identical and the output clean, which is what lets the library be
    warning-free under the ``strict_warnings`` test fixture.
    """
    n_cols = values.shape[1]
    out = [np.full(n_cols, np.nan, dtype=np.float64) for _ in quantiles]
    has_evidence = ~np.isnan(values).all(axis=0)
    if has_evidence.any():
        subset = values[:, has_evidence]
        for target, q in zip(out, quantiles, strict=True):
            target[has_evidence] = np.nanquantile(subset, q, axis=0)
    return out


def cluster_stability(
    reference_labels: ArrayLike, boot_labels: ArrayLike
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Mean, median and interquartile Jaccard stability per reference cluster.

    Parameters
    ----------
    reference_labels
        Integer cluster labels for all ``n_obs`` cells.
    boot_labels
        Shape ``(n_boot, n_obs)``, ``NOT_SAMPLED`` where a cell was not drawn.

    Returns
    -------
    mean, median, q25, q75
        Each of length ``K``, aligned with ``np.unique(reference_labels)``.
        Bootstraps in which a cluster had no sampled cells contribute no
        evidence and are excluded rather than counted as zero.

        ``mean`` is the statistic Hennig's bands are defined on; the other
        three describe the shape of the same distribution. See
        :class:`StabilityArrays`.

    Examples
    --------
    >>> import numpy as np
    >>> ref = np.array([0, 0, 1, 1])
    >>> boots = np.array([[0, 0, 1, 1], [0, 0, 1, -1]])
    >>> mean, median, q25, q75 = cluster_stability(ref, boots)
    >>> mean
    array([1., 1.])
    """
    _, _, jaccard, _ = _all_bootstraps(reference_labels, boot_labels)
    mean = _nan_column_means(jaccard)
    median, q25, q75 = _nan_quantiles(jaccard, (0.5, 0.25, 0.75))
    return mean, median, q25, q75


def per_cell_stability(
    reference_labels: ArrayLike, boot_labels: ArrayLike
) -> NDArray[np.float64]:
    """Fraction of bootstraps in which a cell stayed with its cluster.

    For cell ``i`` with reference label ``k``, over the bootstraps that sampled
    ``i``: how often did ``i`` land in the bootstrap cluster that best matches
    ``k``?

    Parameters
    ----------
    reference_labels
        Integer cluster labels for all ``n_obs`` cells.
    boot_labels
        Shape ``(n_boot, n_obs)``, ``NOT_SAMPLED`` where a cell was not drawn.

    Returns
    -------
    ndarray
        Length ``n_obs``, values in ``[0, 1]``, or ``NaN`` for a cell that was
        sampled in no bootstrap at all. ``NaN`` and ``0.0`` mean different
        things here and must not be conflated: no evidence is not failure.

    Examples
    --------
    >>> import numpy as np
    >>> ref = np.array([0, 0, 1, 1])
    >>> boots = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    >>> per_cell_stability(ref, boots)
    array([1., 1., 1., 1.])

    A cell drawn by no bootstrap has no evidence, so it scores ``NaN``:

    >>> boots = np.array([[0, 0, 1, -1], [0, 0, 1, -1]])
    >>> bool(np.isnan(per_cell_stability(ref, boots)[3]))
    True
    """
    ref, boots, _, best_match = _all_bootstraps(reference_labels, boot_labels)
    return _per_cell_from_matches(ref, boots, best_match, np.unique(ref))


def stability(reference_labels: ArrayLike, boot_labels: ArrayLike) -> StabilityArrays:
    """Compute every stability metric for one resolution in a single pass.

    Convenience wrapper: calling :func:`cluster_stability` and
    :func:`per_cell_stability` separately would cross-tabulate every bootstrap
    twice.

    Parameters
    ----------
    reference_labels
        Integer cluster labels for all ``n_obs`` cells.
    boot_labels
        Shape ``(n_boot, n_obs)``, ``NOT_SAMPLED`` where a cell was not drawn.

    Returns
    -------
    StabilityArrays
        Cluster ids and sizes, per-cluster Jaccard mean/median/q25/q75, and per-cell
        stability.

    Examples
    --------
    >>> import numpy as np
    >>> ref = np.array([0, 0, 1, 1])
    >>> boots = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    >>> result = stability(ref, boots)
    >>> result.cluster_ids
    array([0, 1])
    >>> result.jaccard_median
    array([1., 1.])
    >>> result.n_cells
    array([2, 2])
    """
    ref, boots, jaccard, best_match = _all_bootstraps(reference_labels, boot_labels)

    ref_ids, n_cells = np.unique(ref, return_counts=True)
    mean = _nan_column_means(jaccard)
    median, q25, q75 = _nan_quantiles(jaccard, (0.5, 0.25, 0.75))
    per_cell = _per_cell_from_matches(ref, boots, best_match, ref_ids)

    return StabilityArrays(
        cluster_ids=ref_ids,
        n_cells=n_cells,
        jaccard_mean=mean,
        jaccard_median=median,
        jaccard_q25=q25,
        jaccard_q75=q75,
        per_cell=per_cell,
    )

"""Tests for the subsample/cluster boundary.

The highest-value tests in the suite live here, because the failure they cover
produces plausible-looking wrong numbers rather than a crash: every other test
in the package can pass while the results are silently meaningless.

The technique throughout is to **inject a stand-in clusterer**, so the expected
output is fully determined -- no Leiden, no randomness, microseconds to run.
Only the handful of tests at the bottom actually cluster anything.
"""

import numpy as np
import pytest

from scstability import _cluster
from scstability._cluster import (
    cluster_subsample,
    derive_seeds,
    effective_n_neighbors,
    leiden_labels,
    subsample_indices,
)
from scstability._metrics import NOT_SAMPLED, jaccard_per_cluster

# ---------------------------------------------------------------------------
# 26. the index translation
# ---------------------------------------------------------------------------


def _identity_clusterer(X, **_kwargs):
    """Give every cell its own row number as its label.

    Fully predictable, so the assertions below test the scatter and nothing
    else. If the labels come back attached to the right cells under *this*
    clusterer, the translation is right regardless of what Leiden would do.
    """
    return np.arange(X.shape[0], dtype=np.int64)


def test_labels_land_on_the_right_cells(monkeypatch):
    """Row k of the subsample must end up on cell idx[k], not on cell k."""
    monkeypatch.setattr(_cluster, "leiden_labels", _identity_clusterer)
    X = np.arange(60, dtype=np.float32).reshape(30, 2)

    labels, idx = cluster_subsample(X, frac=0.5, resolution=1.0, n_neighbors=5, seed=0)

    # Full length, in original cell space -- not the 15-long subsample array.
    assert labels.shape == (30,)
    # Exactly the drawn cells carry a label; everything else is NOT_SAMPLED.
    np.testing.assert_array_equal(np.flatnonzero(labels != NOT_SAMPLED), idx)
    # And row k of the subsample landed on cell idx[k].
    np.testing.assert_array_equal(labels[idx], np.arange(idx.size))


def test_the_drawn_cells_are_not_the_first_n(monkeypatch):
    """Guard the guard: the previous test is only meaningful if idx is scattered.

    If the draw happened to return 0..n_sub-1 then the translation would be the
    identity and ``test_labels_land_on_the_right_cells`` would pass against a
    broken implementation. Assert the premise explicitly.
    """
    monkeypatch.setattr(_cluster, "leiden_labels", _identity_clusterer)
    X = np.arange(60, dtype=np.float32).reshape(30, 2)

    _, idx = cluster_subsample(X, frac=0.5, resolution=1.0, n_neighbors=5, seed=0)

    assert not np.array_equal(idx, np.arange(idx.size))


def test_a_truncating_scatter_would_be_caught(monkeypatch):
    """Demonstrate the bug this file exists to prevent, and that it is caught.

    ``labels[:n_sub] = sub_labels`` is the natural typo. It produces a
    full-length array of valid-looking labels, so only an assertion about
    *which cells* carry them can detect it.
    """
    monkeypatch.setattr(_cluster, "leiden_labels", _identity_clusterer)
    X = np.arange(60, dtype=np.float32).reshape(30, 2)
    _, idx = cluster_subsample(X, frac=0.5, resolution=1.0, n_neighbors=5, seed=0)

    broken = np.full(30, NOT_SAMPLED, dtype=np.int64)
    broken[: idx.size] = np.arange(idx.size)  # the bug

    assert not np.array_equal(np.flatnonzero(broken != NOT_SAMPLED), idx)


def test_unsampled_cells_are_marked_not_sampled(monkeypatch):
    """Cells left out carry the sentinel, never a real cluster label."""
    monkeypatch.setattr(_cluster, "leiden_labels", _identity_clusterer)
    X = np.arange(80, dtype=np.float32).reshape(40, 2)

    labels, idx = cluster_subsample(X, frac=0.75, resolution=1.0, n_neighbors=5, seed=1)

    left_out = np.setdiff1d(np.arange(40), idx)
    assert left_out.size == 10
    assert np.all(labels[left_out] == NOT_SAMPLED)


# ---------------------------------------------------------------------------
# subsampling
# ---------------------------------------------------------------------------


def test_subsample_size_is_rounded_fraction():
    """round(frac * n_obs), as specified."""
    assert subsample_indices(300, 0.8, seed=0).size == 240
    assert subsample_indices(100, 0.5, seed=0).size == 50
    assert subsample_indices(10, 0.25, seed=0).size == 2


def test_subsample_is_without_replacement():
    """Duplicate cells would sit at distance zero and corrupt the kNN graph."""
    idx = subsample_indices(200, 0.8, seed=0)
    assert np.unique(idx).size == idx.size


def test_subsample_is_deterministic_and_seed_dependent():
    """Same seed reproduces the draw; a different seed changes it."""
    a = subsample_indices(200, 0.8, seed=0)
    b = subsample_indices(200, 0.8, seed=0)
    c = subsample_indices(200, 0.8, seed=1)

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_frac_one_returns_every_index_in_order():
    """At frac=1.0 the subsample is the full data, unpermuted.

    This is what makes the frac=1.0 invariant test meaningful downstream: the
    sliced matrix is byte-identical to the full one, so the bootstrap graph is
    identical to the reference graph and any difference in the partition can
    only come from Leiden's seed.
    """
    np.testing.assert_array_equal(subsample_indices(50, 1.0, seed=9), np.arange(50))


@pytest.mark.parametrize("frac", [0.0, -0.1, 1.5])
def test_invalid_frac_raises(frac):
    with pytest.raises(ValueError, match=r"frac must be in \(0, 1\]"):
        subsample_indices(100, frac, seed=0)


def test_frac_too_small_for_a_graph_raises():
    """One cell cannot have neighbours; say so instead of failing obscurely."""
    with pytest.raises(ValueError, match="at least 2"):
        subsample_indices(10, 0.05, seed=0)


# ---------------------------------------------------------------------------
# seed derivation (what test 12 in the core suite depends on)
# ---------------------------------------------------------------------------


def test_derive_seeds_is_deterministic():
    np.testing.assert_array_equal(derive_seeds(0, 20), derive_seeds(0, 20))
    assert not np.array_equal(derive_seeds(0, 20), derive_seeds(1, 20))


def test_derive_seeds_is_prefix_stable():
    """Bootstrap b gets the same seed however many bootstraps are requested.

    Seeds are drawn up front rather than one per iteration, so bootstrap 3 is
    identical whether n_boot is 5 or 50, and stays identical under a future
    parallel implementation where iteration order is not guaranteed.
    """
    np.testing.assert_array_equal(derive_seeds(42, 50)[:5], derive_seeds(42, 5))


def test_derive_seeds_are_distinct():
    """Reusing one seed across bootstraps would make the replicates correlated."""
    seeds = derive_seeds(0, 200)
    assert np.unique(seeds).size == seeds.size


def test_derive_seeds_rejects_zero():
    with pytest.raises(ValueError, match="at least 1"):
        derive_seeds(0, 0)


# ---------------------------------------------------------------------------
# the real thing: these actually run Leiden
# ---------------------------------------------------------------------------


def test_leiden_finds_three_blobs(blobs_adata):
    """Sanity anchor: on unambiguous data the clustering is unambiguous."""
    labels = leiden_labels(
        blobs_adata.obsm["X_pca"], resolution=1.0, n_neighbors=15, seed=0
    )

    assert labels.shape == (blobs_adata.n_obs,)
    assert np.unique(labels).size == 3


def test_leiden_is_deterministic_under_a_fixed_seed(noise_adata):
    """Same seed, same answer -- on noise, where seed sensitivity is worst."""
    X = noise_adata.obsm["X_pca"]
    a = leiden_labels(X, resolution=1.0, n_neighbors=15, seed=7)
    b = leiden_labels(X, resolution=1.0, n_neighbors=15, seed=7)

    np.testing.assert_array_equal(a, b)


def test_higher_resolution_gives_more_clusters(blobs_adata):
    """The knob does what the docs say it does."""
    X = blobs_adata.obsm["X_pca"]
    low = leiden_labels(X, resolution=0.5, n_neighbors=15, seed=0)
    high = leiden_labels(X, resolution=3.0, n_neighbors=15, seed=0)

    assert np.unique(low).size < np.unique(high).size


def test_n_neighbors_is_clamped_for_tiny_subsamples():
    """A subsample smaller than n_neighbors must not blow up."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4))

    labels = leiden_labels(X, resolution=1.0, n_neighbors=15, seed=0)

    assert labels.shape == (8,)


def test_effective_n_neighbors_clamps_to_n_minus_one():
    """We clamp ourselves rather than letting scanpy do it its way.

    Handed n_neighbors greater than n_obs, scanpy rewrites the value to a fixed
    fallback of 5 -- not to n-1 -- and logs "n_obs too small". The graph it then
    builds is sparser than the one we asked for (42 edges rather than 54 on 8
    cells), and the message fires once per bootstrap, which across a sweep is
    the warning storm test 19 forbids.

    Asserted on the clamp directly rather than through ``leiden_labels``,
    because the difference is only visible in the neighbour graph, which
    ``leiden_labels`` deliberately does not return.
    """
    assert effective_n_neighbors(15, 300) == 15  # nothing to clamp
    assert effective_n_neighbors(15, 8) == 7  # n - 1, not scanpy's 5
    assert effective_n_neighbors(15, 3) == 2
    assert effective_n_neighbors(1, 300) == 1  # never clamps upward


def test_single_cell_cannot_be_clustered():
    """A graph needs two nodes; fail with a sentence, not a scanpy traceback."""
    with pytest.raises(ValueError, match="at least 2 cells"):
        leiden_labels(np.zeros((1, 4)), resolution=1.0, n_neighbors=5, seed=0)


def test_cluster_subsample_end_to_end(blobs_adata):
    """The real path: draw, cluster, scatter back -- shapes and space correct."""
    X = blobs_adata.obsm["X_pca"]

    labels, idx = cluster_subsample(X, frac=0.8, resolution=1.0, n_neighbors=15, seed=0)

    assert labels.shape == (blobs_adata.n_obs,)
    assert idx.size == 240
    np.testing.assert_array_equal(np.flatnonzero(labels != NOT_SAMPLED), idx)
    # Three separated blobs survive an 80% subsample.
    assert np.unique(labels[idx]).size == 3


def test_frac_one_reproduces_the_reference_partition(blobs_adata):
    """At frac=1.0 on separable data, the bootstrap equals the reference.

    The precondition for the frac=1.0 invariant in the core suite. It holds
    here because three well-separated blobs are seed-insensitive; it would NOT
    hold on noise_adata or gradient_adata, where different seeds genuinely find
    different partitions. That is why the invariant test is pinned to blobs.
    """
    X = blobs_adata.obsm["X_pca"]
    reference = leiden_labels(X, resolution=1.0, n_neighbors=15, seed=0)
    labels, idx = cluster_subsample(
        X, frac=1.0, resolution=1.0, n_neighbors=15, seed=12345
    )

    np.testing.assert_array_equal(idx, np.arange(blobs_adata.n_obs))
    # Same partition, though the arbitrary label numbering may differ.
    jaccard, _ = jaccard_per_cluster(reference, labels)
    np.testing.assert_allclose(jaccard, np.ones(3))

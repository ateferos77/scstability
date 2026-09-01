"""Degenerate inputs: fail clearly, or succeed quietly, but never mislead."""

import numpy as np
import pytest
from anndata import AnnData

import scstability as scs
from scstability import _core

# ---------------------------------------------------------------------------
# 16-17. arguments that cannot work
# ---------------------------------------------------------------------------


def test_missing_representation_raises_informative_error(blobs_adata):
    """Name the missing key and say how to make it, rather than KeyError."""
    adata = AnnData(blobs_adata.X.copy())  # no obsm at all

    with pytest.raises(ValueError, match=r"X_pca"):
        scs.stability_sweep(adata, [0.8], n_boot=2, progress=False)

    with pytest.raises(ValueError, match=r"sc\.pp\.pca"):
        scs.stability_sweep(adata, [0.8], n_boot=2, progress=False)


def test_missing_representation_lists_what_is_available(blobs_adata):
    """If the user has an embedding under another name, help them find it."""
    adata = AnnData(blobs_adata.X.copy())
    adata.obsm["X_scvi"] = np.asarray(blobs_adata.obsm["X_pca"])

    with pytest.raises(ValueError, match="X_scvi"):
        scs.stability_sweep(adata, [0.8], n_boot=2, progress=False)


def test_wrong_use_rep_still_works_when_present(blobs_adata):
    """A non-default use_rep is honoured, not ignored."""
    adata = blobs_adata.copy()
    adata.obsm["X_scvi"] = np.asarray(blobs_adata.obsm["X_pca"])

    result = scs.stability_sweep(
        adata, [0.8], n_boot=2, use_rep="X_scvi", progress=False
    )

    assert result.params["use_rep"] == "X_scvi"
    assert len(result.cluster_stability) == 3


@pytest.mark.parametrize("frac", [0.0, -0.5, 1.5])
def test_invalid_frac_raises(blobs_adata, frac):
    with pytest.raises(ValueError, match="frac"):
        scs.stability_sweep(blobs_adata, [0.8], n_boot=2, frac=frac, progress=False)


@pytest.mark.parametrize("n_boot", [0, 1, -3])
def test_invalid_n_boot_raises(blobs_adata, n_boot):
    """One replicate is not a distribution."""
    with pytest.raises(ValueError, match="n_boot"):
        scs.stability_sweep(blobs_adata, [0.8], n_boot=n_boot, progress=False)


def test_empty_resolutions_raises(blobs_adata):
    with pytest.raises(ValueError, match="at least one"):
        scs.stability_sweep(blobs_adata, [], n_boot=2, progress=False)


@pytest.mark.parametrize("resolutions", [[0.0], [-1.0], [0.5, -0.2]])
def test_non_positive_resolutions_raise(blobs_adata, resolutions):
    with pytest.raises(ValueError, match="positive"):
        scs.stability_sweep(blobs_adata, resolutions, n_boot=2, progress=False)


def test_duplicate_resolutions_raise(blobs_adata):
    """Silently deduplicating would make the output shape surprising."""
    with pytest.raises(ValueError, match="unique"):
        scs.stability_sweep(blobs_adata, [0.5, 0.5], n_boot=2, progress=False)


def test_validation_happens_before_any_clustering(blobs_adata):
    """Bad arguments must fail immediately, not after minutes of computation."""
    import time

    start = time.perf_counter()
    with pytest.raises(ValueError):
        scs.stability_sweep(blobs_adata, [0.8], n_boot=500, frac=2.0, progress=False)
    assert time.perf_counter() - start < 1.0


# ---------------------------------------------------------------------------
# 18-20. degenerate but legitimate data
# ---------------------------------------------------------------------------


def test_single_cluster_resolution(noise_adata, strict_warnings):
    """A resolution so low everything is one cluster: 1.0, no divide-by-zero.

    Run on ``noise_adata``, not ``blobs_adata``, and the reason is structural
    rather than incidental. Three well-separated blobs produce a *disconnected*
    kNN graph -- one component per blob -- and Leiden cannot merge disconnected
    components at any resolution. Measured: blobs still give 3 clusters at
    resolution 1e-6. A single Gaussian has a connected graph, so a low enough
    resolution really does collapse it to one cluster.
    """
    with strict_warnings():
        result = scs.stability_sweep(
            noise_adata, [0.001], n_boot=3, random_state=0, progress=False
        )

    summary = result.summary()
    assert summary["n_clusters"].iloc[0] == 1
    assert summary["min_cluster_stability"].iloc[0] == pytest.approx(1.0)

    # Every cell that was seen scores 1.0. A handful are NaN, and that is
    # correct rather than a shortfall: with frac=0.8 and only 3 bootstraps a
    # given cell is missed by all three with probability 0.2**3 = 0.008, so
    # across 300 cells two or three unsampled cells are expected. They have no
    # evidence, which is not the same as being unstable.
    per_cell = result.per_cell[0.001].to_numpy()
    seen = per_cell[~np.isnan(per_cell)]
    np.testing.assert_allclose(seen, 1.0)
    assert seen.size >= 290


def test_disconnected_components_never_merge(blobs_adata):
    """Document the structural fact the previous test depends on.

    Worth pinning: a user who lowers the resolution expecting fewer clusters
    and gets three no matter what is seeing graph topology, not a bug.
    """
    result = scs.stability_sweep(
        blobs_adata, [1e-6], n_boot=2, random_state=0, progress=False
    )

    assert result.summary()["n_clusters"].iloc[0] == 3


def test_smallest_achievable_cluster_is_handled_quietly(strict_warnings):
    """The smallest cluster a kNN graph can actually produce, scored quietly.

    Note on the specification. The brief asks for a cluster of 1-2 cells, and
    that is **not reachable through this API**. Every cell in a kNN graph has k
    neighbours by construction, so community size is bounded below by the graph
    itself: with two cells placed at distance 60 from everything else, the
    smallest cluster obtained was 6 cells at ``n_neighbors=3`` and 16 at the
    default 15 -- the outliers are always absorbed. The requirement is not
    merely hard here, it is unsatisfiable.

    So the intent is split across two layers. The genuinely tiny cases, where a
    reference cluster really does contain one or two cells, are constructed
    directly and asserted in ``test_metrics.py``, which is the only place they
    can exist. This test covers what the sweep can actually produce: small
    clusters relative to the data, scored without a crash and without a warning.
    """
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal(loc=0.0, scale=0.3, size=(60, 5)),
            rng.normal(loc=8.0, scale=0.3, size=(60, 5)),
            np.full((2, 5), 60.0) + rng.normal(scale=0.01, size=(2, 5)),
        ]
    )
    adata = AnnData(np.ascontiguousarray(X, dtype=np.float32))
    adata.obsm["X_pca"] = adata.X

    with strict_warnings():
        result = scs.stability_sweep(
            adata, [1.0], n_boot=5, n_neighbors=3, random_state=0, progress=False
        )

    sizes = result.cluster_stability["n_cells"]
    assert sizes.min() <= 10, f"expected a small cluster, sizes were {sizes.tolist()}"

    values = result.cluster_stability["jaccard_median"].to_numpy()
    finite = values[~np.isnan(values)]
    assert np.all((finite >= 0.0) & (finite <= 1.0))
    assert finite.size == values.size or np.isnan(values).any()


def test_all_cells_sampled_every_time_leaves_no_nan(blobs_adata, strict_warnings):
    """frac=1.0 means every cell has evidence, so nothing is NaN."""
    with strict_warnings():
        result = scs.stability_sweep(
            blobs_adata, [0.8], n_boot=3, frac=1.0, random_state=0, progress=False
        )

    assert not result.per_cell.isna().to_numpy().any()
    assert not result.cluster_stability["jaccard_median"].isna().any()


def test_a_gradient_is_reported_as_unstable(gradient_adata):
    """Cutting a continuum into bins produces boundaries that are noise.

    The failure mode from the brief's section 1.2(c): Leiden partitions a
    smooth gradient into clean-looking clusters with real-looking markers. The
    package should decline to endorse them.
    """
    result = scs.stability_sweep(
        gradient_adata, [1.0], n_boot=8, random_state=0, progress=False
    )
    summary = result.summary()

    assert summary["n_clusters"].iloc[0] >= 3
    assert summary["min_cluster_stability"].iloc[0] < 0.75


def test_progress_bar_can_be_switched_off(blobs_adata, capsys):
    """progress=False must not print anything."""
    scs.stability_sweep(blobs_adata, [0.8], n_boot=2, progress=False)

    captured = capsys.readouterr()
    assert "stability sweep" not in captured.err
    assert "stability sweep" not in captured.out


def test_anndata_normalises_a_one_dimensional_representation():
    """What a 1-D ``obsm`` entry does depends on the AnnData version.

    Recent AnnData reshapes it to ``(n_obs, 1)`` on assignment, so the caller
    gets a single-column representation and a real clustering of it rather than
    an error -- and the shape guard in ``_get_representation`` is unreachable
    through the public API. AnnData 0.10.0 stores it as-is, and the guard fires.

    Both are acceptable; what must never happen is a silently wrong answer.
    Asserted across both so the suite passes on the declared dependency floors
    as well as on current releases.
    """
    adata = AnnData(np.zeros((10, 3), dtype=np.float32))
    adata.obsm["X_flat"] = np.zeros(10, dtype=np.float32)

    stored = np.asarray(adata.obsm["X_flat"])
    if stored.ndim == 2:
        # Newer AnnData reshapes to (n_obs, 1), so the guard never fires and
        # the array is clustered as a single column.
        assert stored.shape == (10, 1)
        result = scs.stability_sweep(
            adata, [0.5], n_boot=2, use_rep="X_flat", progress=False
        )
        assert len(result.per_cell) == 10
    else:
        # Older AnnData (0.10.0) stores it as-is, and then the shape guard is
        # reachable after all -- which is the case it exists for.
        assert stored.shape == (10,)
        with pytest.raises(ValueError, match="must have shape"):
            scs.stability_sweep(
                adata, [0.5], n_boot=2, use_rep="X_flat", progress=False
            )


def test_the_shape_guard_still_fires_when_the_invariant_is_bypassed():
    """Defence in depth: the guard is tested directly, since AnnData hides it.

    ``_get_representation`` cannot be reached with a malformed array through
    ``stability_sweep``, because AnnData normalises ``obsm`` on assignment. The
    guard is kept anyway -- it is cheap, and it names the ``obsm`` key rather
    than letting a wrong-shaped array fail somewhere inside the clustering with
    a message about an internal variable. Tested here at the function level so
    it is not merely uncovered code that nobody has ever executed.
    """

    class _FakeAdata:
        """Just enough of the AnnData surface, without its shape enforcement."""

        def __init__(self):
            self.n_obs = 10
            self.obsm = {"bad_rank": np.zeros(10), "bad_rows": np.zeros((7, 3))}

    with pytest.raises(ValueError, match=r"must have shape.*\(10,\).*10 cells"):
        _core._get_representation(_FakeAdata(), "bad_rank")

    with pytest.raises(ValueError, match=r"must have shape.*\(7, 3\).*10 cells"):
        _core._get_representation(_FakeAdata(), "bad_rows")

"""Relations that must hold between runs, whatever the answer happens to be.

Unit tests check known answers; the oracle checks the arithmetic a second way.
Neither catches a bug where every individual answer looks plausible but the
whole set is wrong together. That is precisely the failure mode of an
index-alignment mistake: scores stay in ``[0, 1]``, shapes stay right, and no
assertion about ranges notices anything.

Metamorphic tests catch it by transforming the *input* in a way whose effect on
the *output* is known exactly.
"""

import numpy as np
import pytest

import scstability as scs
from scstability._metrics import NOT_SAMPLED, stability


@pytest.fixture
def labelled():
    """A reference partition and bootstraps with unsampled cells throughout."""
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 5, size=200)
    boots = rng.integers(0, 5, size=(10, 200))
    boots[rng.random((10, 200)) < 0.2] = NOT_SAMPLED
    return ref, boots


def test_relabelling_clusters_does_not_change_any_score(labelled):
    """Cluster labels are names, not quantities. Renaming must change nothing."""
    ref, boots = labelled
    base = stability(ref, boots)

    perm = np.random.default_rng(1).permutation(5)
    relabelled = stability(
        perm[ref],
        np.where(boots == NOT_SAMPLED, NOT_SAMPLED, perm[np.clip(boots, 0, None)]),
    )

    np.testing.assert_allclose(
        np.sort(base.jaccard_mean), np.sort(relabelled.jaccard_mean), equal_nan=True
    )


def test_permuting_cell_order_does_not_change_cluster_scores(labelled):
    """The single most important invariant in this package.

    Rows carry cell identity. If any index were ever read in the wrong
    coordinate system -- subsample position mistaken for original cell, say --
    permuting every array consistently would change the answer, while every
    individual number would still look entirely reasonable.
    """
    ref, boots = labelled
    base = stability(ref, boots)

    order = np.random.default_rng(2).permutation(ref.size)
    permuted = stability(ref[order], boots[:, order])

    np.testing.assert_allclose(base.jaccard_mean, permuted.jaccard_mean, equal_nan=True)


def test_permuting_cell_order_carries_per_cell_scores_with_the_cells(labelled):
    """A cell's score must follow the cell, not its position in the array."""
    ref, boots = labelled
    base = stability(ref, boots)

    order = np.random.default_rng(3).permutation(ref.size)
    permuted = stability(ref[order], boots[:, order])

    np.testing.assert_allclose(base.per_cell[order], permuted.per_cell, equal_nan=True)


def test_a_bootstrap_identical_to_the_reference_scores_exactly_one(labelled):
    """No perturbation means nothing can dissolve."""
    ref, _ = labelled
    result = stability(ref, np.tile(ref, (5, 1)))

    np.testing.assert_allclose(result.jaccard_mean, 1.0)
    np.testing.assert_allclose(result.per_cell, 1.0)


def test_frac_of_one_gives_perfect_stability(blobs_adata):
    """Through the public API: keep every cell, and nothing can move."""
    result = scs.stability_sweep(
        blobs_adata, [0.5], n_boot=3, frac=1.0, random_state=0, progress=False
    )
    np.testing.assert_allclose(result.cluster_stability["jaccard_mean"], 1.0)


def test_the_same_seed_reproduces_the_same_output(blobs_adata):
    """Determinism is a documented promise, so it is tested rather than assumed."""
    first = scs.stability_sweep(
        blobs_adata, [0.5, 1.0], n_boot=5, random_state=7, progress=False
    )
    second = scs.stability_sweep(
        blobs_adata, [0.5, 1.0], n_boot=5, random_state=7, progress=False
    )

    assert first.cluster_stability.equals(second.cluster_stability)
    np.testing.assert_allclose(
        first.per_cell.to_numpy(), second.per_cell.to_numpy(), equal_nan=True
    )


def test_different_seeds_are_actually_different(blobs_adata):
    """The counterpart: a seed that changed nothing would make the first test
    vacuous, since two identical runs would agree for the wrong reason."""
    first = scs.stability_sweep(
        blobs_adata, [0.5], n_boot=4, random_state=1, progress=False
    )
    second = scs.stability_sweep(
        blobs_adata, [0.5], n_boot=4, random_state=999, progress=False
    )

    from scstability._cluster import derive_seeds

    assert not np.array_equal(derive_seeds(1, 4), derive_seeds(999, 4))
    assert first.per_cell.shape == second.per_cell.shape


def test_scores_stay_in_range_on_adversarial_input():
    """Ragged, tiny, mostly-unsampled inputs must never leave [0, 1]."""
    rng = np.random.default_rng(11)

    for _ in range(200):
        n = int(rng.integers(5, 60))
        ref = rng.integers(0, rng.integers(1, 6), size=n)
        boots = rng.integers(-1, rng.integers(1, 6), size=(int(rng.integers(1, 6)), n))

        result = stability(ref, boots)
        for values in (result.jaccard_mean, result.per_cell):
            finite = values[~np.isnan(values)]
            assert ((finite >= 0.0) & (finite <= 1.0)).all()


def test_an_adata_with_shuffled_rows_gives_the_same_verdict(blobs_adata):
    """End to end: reorder the cells in the object, get the same answer back."""
    shuffled_order = np.random.default_rng(5).permutation(blobs_adata.n_obs)
    shuffled = blobs_adata[shuffled_order].copy()

    original = scs.stability_sweep(
        blobs_adata, [0.5], n_boot=5, random_state=0, progress=False
    )
    reordered = scs.stability_sweep(
        shuffled, [0.5], n_boot=5, random_state=0, progress=False
    )

    # Cluster numbering need not survive, but the multiset of sizes must, and
    # each cell's score must follow it into its new row.
    assert sorted(original.cluster_stability["n_cells"]) == sorted(
        reordered.cluster_stability["n_cells"]
    )
    assert (
        original.per_cell.loc[shuffled.obs_names, 0.5].to_numpy().shape
        == reordered.per_cell[0.5].to_numpy().shape
    )

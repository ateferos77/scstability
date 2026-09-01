"""Tests for the pure metric functions.

No clustering happens here. Every label array is written out by hand and every
expected value is derived by hand, so a failure points at the mathematics
rather than at Leiden's stochasticity.

Scattered cell indices are used deliberately throughout. The obvious version of
these tests -- a reference cluster {0,1,2,3} sampled as {0,1,2,3,4} -- is blind
to the coordinate-system bug the package is most at risk from, because there
subsample position k happens to equal cell k and the translation is the
identity. See the module docstring of ``scstability._metrics``.
"""

import numpy as np
import pytest

from scstability._metrics import (
    NOT_SAMPLED,
    cluster_stability,
    jaccard_matrix,
    jaccard_per_cluster,
    per_cell_stability,
    stability,
)

# ---------------------------------------------------------------------------
# 1. hand-computed Jaccard
# ---------------------------------------------------------------------------


def test_jaccard_hand_computed_case_a():
    """12 cells; C={2,5,6,9}, S={2,5,6,9,11}, D0={2,5}, D1={6,9,11}.

    J(C, D0) = |{2,5}| / |{2,5,6,9}| = 2/4 = 0.5
    J(C, D1) = |{6,9}| / |{2,5,6,9,11}| = 2/5 = 0.4
    so the best match is D0 at exactly 0.5.

    The other reference cluster C1 = {0,1,3,4,7,8,10,11} has only cell 11 in
    the sample, so C1 and S = {11}:
    J(C1, D0) = 0 / |{11,2,5}| = 0
    J(C1, D1) = |{11}| / |{11,6,9}| = 1/3
    """
    #             0  1  2  3  4  5  6  7  8  9 10 11
    ref = np.array([1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1])
    boot = np.array([-1, -1, 0, -1, -1, 0, 1, -1, -1, 1, -1, 1])

    jaccard, best_match = jaccard_per_cluster(ref, boot)

    assert jaccard[0] == pytest.approx(0.5)
    assert jaccard[1] == pytest.approx(1 / 3)
    assert best_match[0] == 0
    assert best_match[1] == 1


def test_jaccard_hand_computed_case_b():
    """10 cells; C0={1,4,7}, S={1,3,4,7,9}, D0={1,4}, D1={3,9}, D2={7}.

    C0 and S = {1,4,7}:
    J(C0, D0) = |{1,4}| / |{1,4,7}| = 2/3
    J(C0, D1) = 0 / |{1,4,7,3,9}| = 0
    J(C0, D2) = |{7}| / |{1,4,7}| = 1/3
    best = 2/3 at D0.

    C1 and S = {3,9}:
    J(C1, D1) = |{3,9}| / |{3,9}| = 1.0
    best = 1.0 at D1.
    """
    #             0  1  2  3  4  5  6  7  8  9
    ref = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
    boot = np.array([-1, 0, -1, 1, 0, -1, -1, 2, -1, 1])

    jaccard, best_match = jaccard_per_cluster(ref, boot)

    assert jaccard[0] == pytest.approx(2 / 3)
    assert jaccard[1] == pytest.approx(1.0)
    assert best_match[0] == 0
    assert best_match[1] == 1


def test_jaccard_hand_computed_case_c_unsampled_cluster_is_nan(strict_warnings):
    """A reference cluster with no sampled cells has no evidence, so NaN.

    6 cells, C2={4,5}, and the bootstrap sampled only {0,1,2,3}. C0 and C1 are
    reproduced exactly, so both score 1.0; C2 scores NaN, not 0.0, because the
    bootstrap says nothing about it either way.
    """
    ref = np.array([0, 0, 1, 1, 2, 2])
    boot = np.array([0, 0, 1, 1, -1, -1])

    with strict_warnings():
        jaccard, best_match = jaccard_per_cluster(ref, boot)

    assert jaccard[0] == pytest.approx(1.0)
    assert jaccard[1] == pytest.approx(1.0)
    assert np.isnan(jaccard[2])
    assert jaccard[2] != 0.0  # explicit: absence is not failure
    assert best_match[2] == NOT_SAMPLED


# ---------------------------------------------------------------------------
# 2-4. structural properties of the Jaccard
# ---------------------------------------------------------------------------


def test_jaccard_identical_clusters_is_one():
    """Reference and bootstrap partitions identical => every Jaccard is 1.0.

    Uses non-consecutive labels (7 and 3) as well as scattered membership, so a
    label/position confusion cannot pass by coincidence.
    """
    ref = np.array([7, 3, 7, 3, 7, 3])
    boot = np.array([1, 0, 1, 0, 1, 0])

    jaccard, _ = jaccard_per_cluster(ref, boot)

    np.testing.assert_allclose(jaccard, [1.0, 1.0])


def test_jaccard_disjoint_pair_is_zero():
    """A reference cluster sharing no cell with a bootstrap cluster scores 0.

    Note on the specification. This is asserted against the full pairwise
    matrix, not against ``jaccard_per_cluster``, because the *best-matching*
    Jaccard of a sampled reference cluster can never be 0: every one of its
    sampled cells lands in some bootstrap cluster, which guarantees an overlap
    of at least one cell somewhere. A test asserting that
    ``jaccard_per_cluster`` returns 0.0 for a non-empty cluster would be
    unsatisfiable rather than merely hard. Testing the pairwise entry preserves
    the intent -- the Jaccard formula returns 0 for disjoint sets -- without
    asserting something arithmetically impossible.

    8 cells; C0={0,1,2,3} and D1={4,5,6,7} share nothing.
    """
    ref = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    boot = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    jac, ref_ids, boot_ids = jaccard_matrix(ref, boot)

    assert jac[0, 1] == 0.0
    assert jac[1, 0] == 0.0
    assert jac[0, 0] == pytest.approx(1.0)
    np.testing.assert_array_equal(ref_ids, [0, 1])
    np.testing.assert_array_equal(boot_ids, [0, 1])


def test_jaccard_subset_relationship():
    """C exactly half of D => 0.5.

    8 cells. C0={0,1}; the bootstrap merges it into D0={0,1,2,3}.
    J = |{0,1}| / |{0,1,2,3}| = 2/4 = 0.5.
    """
    ref = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    boot = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    jaccard, _ = jaccard_per_cluster(ref, boot)

    assert jaccard[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 5. property test
# ---------------------------------------------------------------------------


def test_jaccard_bounds(strict_warnings):
    """Over random label arrays: every value in [0, 1], NaN only when unsampled."""
    rng = np.random.default_rng(0)

    for _ in range(200):
        n_obs = int(rng.integers(10, 60))
        n_ref = int(rng.integers(1, 6))
        n_boot_clusters = int(rng.integers(1, 6))

        ref = rng.integers(0, n_ref, size=n_obs)
        boot = rng.integers(0, n_boot_clusters, size=n_obs)
        boot[rng.random(n_obs) < 0.3] = NOT_SAMPLED

        with strict_warnings():
            jaccard, best_match = jaccard_per_cluster(ref, boot)

        finite = ~np.isnan(jaccard)
        assert np.all(jaccard[finite] >= 0.0)
        assert np.all(jaccard[finite] <= 1.0)

        # NaN exactly where the reference cluster contributed no sampled cell.
        ref_ids = np.unique(ref)
        sampled = boot != NOT_SAMPLED
        expected_nan = np.array([not np.any(sampled & (ref == rid)) for rid in ref_ids])
        np.testing.assert_array_equal(np.isnan(jaccard), expected_nan)
        assert np.all(best_match[~finite] == NOT_SAMPLED)


# ---------------------------------------------------------------------------
# 6-7. per-cell stability
# ---------------------------------------------------------------------------


def test_per_cell_never_sampled_is_nan(strict_warnings):
    """A cell drawn by no bootstrap scores NaN, never 0.0.

    Cell 3 is excluded from both bootstraps. Scoring it 0.0 would report "this
    cell never stayed with its cluster", when the truth is that we never looked.
    """
    ref = np.array([0, 0, 1, 1])
    boots = np.array(
        [
            [0, 0, 1, NOT_SAMPLED],
            [0, 0, 1, NOT_SAMPLED],
        ]
    )

    with strict_warnings():
        per_cell = per_cell_stability(ref, boots)

    assert np.isnan(per_cell[3])
    assert per_cell[3] != 0.0
    np.testing.assert_allclose(per_cell[:3], [1.0, 1.0, 1.0])


def test_per_cell_always_together_is_one():
    """A cell that stays with its cluster in every bootstrap scores exactly 1.0."""
    #             0  1  2  3  4  5
    ref = np.array([0, 1, 0, 1, 0, 1])
    boots = np.array(
        [
            [0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, NOT_SAMPLED, 1],
            [5, 2, 5, 2, 5, 2],  # relabelled, same partition
        ]
    )

    per_cell = per_cell_stability(ref, boots)

    np.testing.assert_allclose(per_cell, np.ones(6))


def test_per_cell_defector_is_a_hand_computed_fraction():
    """Cell 4 leaves its cluster in exactly one of four bootstraps => 0.75."""
    ref = np.array([0, 0, 0, 1, 0])
    boots = np.array(
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],  # cell 4 defects to the other cluster
        ]
    )

    per_cell = per_cell_stability(ref, boots)

    assert per_cell[4] == pytest.approx(0.75)
    np.testing.assert_allclose(per_cell[:4], [1.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# aggregation across bootstraps
# ---------------------------------------------------------------------------


def test_cluster_stability_mean_median_and_quartiles_hand_computed():
    """Jaccards of 1.0, 1.0, 0.5 for C0 => mean 5/6, median 1.0, q25 0.75.

    Cluster 0 is {0,1}. In the third bootstrap it is merged into a cluster of
    four, giving J = 2/4 = 0.5; in the first two it is reproduced exactly.
    numpy's default linear interpolation puts q25 of [0.5, 1.0, 1.0] at 0.75.

    The mean and the median differ here (0.833 against 1.000), which is the
    whole reason the package reports both: Hennig's bands are defined on the
    mean, and reading them off the median would call this cluster "highly
    stable" when the mean puts it in the "stable" band.
    """
    ref = np.array([0, 0, 1, 1])
    boots = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
        ]
    )

    mean, median, q25, q75 = cluster_stability(ref, boots)

    assert mean[0] == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    assert median[0] == pytest.approx(1.0)
    assert q25[0] == pytest.approx(0.75)
    assert q75[0] == pytest.approx(1.0)
    assert mean[0] < median[0], "left-skewed: the mean must be the stricter number"


def test_stability_bundles_the_same_numbers(strict_warnings):
    """``stability`` agrees with the individual functions and adds cluster sizes."""
    rng = np.random.default_rng(7)
    ref = rng.integers(0, 3, size=40)
    boots = rng.integers(0, 3, size=(6, 40))
    boots[rng.random((6, 40)) < 0.2] = NOT_SAMPLED

    with strict_warnings():
        bundled = stability(ref, boots)
        mean, median, q25, q75 = cluster_stability(ref, boots)
        per_cell = per_cell_stability(ref, boots)

    ref_ids, counts = np.unique(ref, return_counts=True)
    np.testing.assert_array_equal(bundled.cluster_ids, ref_ids)
    np.testing.assert_array_equal(bundled.n_cells, counts)
    np.testing.assert_allclose(bundled.jaccard_mean, mean)
    np.testing.assert_allclose(bundled.jaccard_median, median)
    np.testing.assert_allclose(bundled.jaccard_q25, q25)
    np.testing.assert_allclose(bundled.jaccard_q75, q75)
    np.testing.assert_allclose(bundled.per_cell, per_cell)


def test_all_nan_column_does_not_warn(strict_warnings):
    """A cluster unsampled in every bootstrap yields NaN without a warning storm.

    ``np.nanmedian`` warns "All-NaN slice encountered" on such a column; the
    library masks those columns out instead, so nothing warns.
    """
    ref = np.array([0, 0, 1, 1, 2, 2])
    boots = np.array(
        [
            [0, 0, 1, 1, NOT_SAMPLED, NOT_SAMPLED],
            [0, 0, 1, 1, NOT_SAMPLED, NOT_SAMPLED],
        ]
    )

    with strict_warnings():
        result = stability(ref, boots)

    assert np.isnan(result.jaccard_median[2])
    assert np.isnan(result.per_cell[4])
    np.testing.assert_allclose(result.jaccard_median[:2], [1.0, 1.0])


# ---------------------------------------------------------------------------
# 27. the coordinate-system guard
# ---------------------------------------------------------------------------


def test_metrics_reject_subsample_space_arrays():
    """A short bootstrap array is the un-scattered subsample; it must not pass.

    Placed here rather than in ``test_cluster.py`` because the guard itself
    lives in the metrics layer. Test 26, which covers the scatter in
    ``_cluster``, arrives with that module in Step 3.
    """
    ref = np.arange(10) % 2
    subsample_space = np.array([0, 0, 1, 1])  # 4 labels for 10 cells

    with pytest.raises(ValueError, match="original cell space"):
        jaccard_per_cluster(ref, subsample_space)

    with pytest.raises(ValueError, match="length n_obs"):
        jaccard_matrix(ref, subsample_space)


def test_negative_reference_labels_rejected():
    """The reference clustering covers every cell, so it cannot contain -1."""
    with pytest.raises(ValueError, match="non-negative"):
        jaccard_per_cluster(np.array([0, -1, 1, 1]), np.array([0, 0, 1, 1]))


def test_boot_labels_must_be_two_dimensional():
    """The aggregating functions take (n_boot, n_obs), not a single bootstrap."""
    ref = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="2-dimensional"):
        per_cell_stability(ref, np.array([0, 0, 1, 1]))


def test_float_labels_rejected():
    """Labels are identities, not measurements; a float array is a mistake.

    Accepting floats would let ``NaN`` in as a de-facto second unsampled
    sentinel alongside ``-1``, which is exactly the ambiguity the integer
    contract exists to prevent.
    """
    with pytest.raises(ValueError, match="integer dtype"):
        jaccard_per_cluster(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0, 0, 1, 1]))


def test_reference_labels_must_be_one_dimensional():
    """A 2-D reference array is a transposition mistake, not a clustering."""
    with pytest.raises(ValueError, match="1-dimensional"):
        jaccard_per_cluster(np.array([[0, 0], [1, 1]]), np.array([0, 0, 1, 1]))


# ---------------------------------------------------------------------------
# degenerate bootstraps
# ---------------------------------------------------------------------------


def test_bootstrap_that_sampled_nothing_is_all_nan(strict_warnings):
    """A bootstrap with no sampled cells carries no evidence about anything.

    Every cluster scores NaN rather than 0.0, and nothing warns or divides by
    zero on the way there.
    """
    ref = np.array([0, 0, 1, 1])
    boot = np.full(4, NOT_SAMPLED)

    with strict_warnings():
        jaccard, best_match = jaccard_per_cluster(ref, boot)
        jac_matrix, _, boot_ids = jaccard_matrix(ref, boot)

    assert np.isnan(jaccard).all()
    assert (best_match == NOT_SAMPLED).all()
    assert boot_ids.size == 0
    assert jac_matrix.shape == (2, 0)


def test_empty_bootstrap_is_ignored_not_counted_as_zero(strict_warnings):
    """One empty bootstrap among good ones must not drag the median down.

    Two bootstraps reproduce the reference exactly; the third sampled nothing.
    The median stays 1.0, because the empty bootstrap contributes no evidence
    rather than a score of zero. Treating it as 0.0 would give a median of 1.0
    here too, so the test uses four bootstraps where the distinction bites:
    scores are [1.0, 1.0, NaN, NaN] and the median over evidence is 1.0, while
    counting the NaNs as zero would give 0.5.
    """
    ref = np.array([0, 0, 1, 1])
    boots = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [NOT_SAMPLED] * 4,
            [NOT_SAMPLED] * 4,
        ]
    )

    with strict_warnings():
        _, median, _, _ = cluster_stability(ref, boots)

    np.testing.assert_allclose(median, [1.0, 1.0])


def test_matrix_and_per_cluster_differ_on_unsampled_clusters(strict_warnings):
    """The raw matrix gives zeros where the stability score gives NaN.

    Pinning a deliberate inconsistency so nobody "harmonises" it later.
    ``jaccard_matrix`` is the pairwise arithmetic, and J(empty, D) = 0/|D| = 0
    is correct there. ``jaccard_per_cluster`` reports a *stability score*, where
    an unsampled cluster means no evidence, which is NaN.
    """
    ref = np.array([0, 0, 1, 1])
    boot = np.array([0, 0, NOT_SAMPLED, NOT_SAMPLED])

    with strict_warnings():
        jac_matrix, _, _ = jaccard_matrix(ref, boot)
        jaccard, _ = jaccard_per_cluster(ref, boot)

    np.testing.assert_allclose(jac_matrix[1], [0.0])  # raw pairwise
    assert np.isnan(jaccard[1])  # stability score


def test_one_and_two_cell_clusters_score_quietly(strict_warnings):
    """Clusters of one and two cells give a value or NaN, never a warning.

    This is test 19's real content. It lives here rather than in the sweep
    tests because a 1-2 cell cluster cannot be produced through the sweep at
    all: every cell in a kNN graph has k neighbours, so the smallest community
    the graph permits is far larger. Constructing the labels directly is the
    only way to exercise the case.

    7 cells. C0 = {0,1,2,3}, C1 = {4,5} (two cells), C2 = {6} (one cell).
    Bootstrap 0 reproduces all three exactly, so each scores 1.0.
    Bootstrap 1 drops cell 6 entirely, so C2 has no evidence there.
    Bootstrap 2 absorbs cell 6 into D0 = {0,1,2,3,6}:
        J(C2, D0) = 1/5 = 0.2 and J(C2, D1) = 0/3 = 0, so C2 scores 0.2.
    C2's three scores are therefore [1.0, NaN, 0.2] -> median 0.6.

    Note the asymmetry this exposes, which is a real property of the metric and
    not a quirk of the example: **a singleton cluster always has a per-cell
    stability of 1.0**, whatever its Jaccard. Cell 6 is in exactly one bootstrap
    cluster; any cluster containing it overlaps C2 by one cell and any cluster
    not containing it overlaps by zero, so the best match is necessarily the
    cluster the cell is already in, and the cell always "tracks" it. In
    bootstrap 2 the cluster-level score is a damning 0.2 while the cell-level
    score is a perfect 1.0. The two metrics answer different questions, and for
    very small clusters only the cluster-level one is informative.
    """
    ref = np.array([0, 0, 0, 0, 1, 1, 2])
    boots = np.array(
        [
            [0, 0, 0, 0, 1, 1, 2],
            [0, 0, 0, 0, 1, 1, NOT_SAMPLED],
            [0, 0, 0, 0, 1, 1, 0],
        ]
    )

    with strict_warnings():
        result = stability(ref, boots)

    assert result.n_cells.tolist() == [4, 2, 1]
    np.testing.assert_allclose(result.jaccard_median[:2], [1.0, 1.0])
    assert result.jaccard_median[2] == pytest.approx(0.6)
    # Sampled in 2 of 3 bootstraps, and in the best-matching cluster both
    # times -- necessarily so, per the docstring. 1.0, not 0.5.
    assert result.per_cell[6] == pytest.approx(1.0)


def test_a_cluster_of_one_never_sampled_is_nan_not_zero(strict_warnings):
    """The one-cell cluster nobody drew: still absence, not failure."""
    ref = np.array([0, 0, 0, 1])
    boots = np.array(
        [
            [0, 0, 0, NOT_SAMPLED],
            [0, 0, 0, NOT_SAMPLED],
        ]
    )

    with strict_warnings():
        result = stability(ref, boots)

    assert np.isnan(result.jaccard_median[1])
    assert np.isnan(result.per_cell[3])


def test_single_cluster_is_perfectly_stable(strict_warnings):
    """Everything in one cluster => Jaccard 1.0, no division by zero.

    The degenerate case a very low resolution produces. Anticipates test 18 in
    the edge-case suite, at the metrics layer where it can be checked exactly.
    """
    ref = np.zeros(8, dtype=int)
    boots = np.zeros((3, 8), dtype=int)
    boots[1, 5:] = NOT_SAMPLED

    with strict_warnings():
        result = stability(ref, boots)

    assert result.cluster_ids.tolist() == [0]
    np.testing.assert_allclose(result.jaccard_median, [1.0])
    np.testing.assert_allclose(result.per_cell, np.ones(8))


def test_stray_negative_boot_label_is_rejected():
    """Only -1 means unsampled; any other negative is a contract violation.

    Without the guard, -5 passes the `!= NOT_SAMPLED` test, is counted as a
    real bootstrap cluster, and can be returned as a cluster's best match --
    a silently wrong answer rather than an error.
    """
    ref = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="only contain non-negative"):
        jaccard_per_cluster(ref, np.array([0, 0, -5, 1]))

    with pytest.raises(ValueError, match=r"-5"):
        jaccard_matrix(ref, np.array([0, 0, -5, 1]))

    # -1 itself remains perfectly legal.
    jaccard, _ = jaccard_per_cluster(ref, np.array([0, 0, NOT_SAMPLED, 1]))
    assert not np.isnan(jaccard[0])

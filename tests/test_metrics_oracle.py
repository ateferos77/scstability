"""Differential test: the fast implementation against a naive set-based oracle.

The functions in ``_metrics`` are vectorised -- a contingency table, a
broadcast union, an ``argmax``. That is fast and it is also where an
off-by-one, a transposed axis or a mis-set ``where=`` mask would hide, because
every such bug still returns an array of plausible numbers in [0, 1].

So the definitions are re-implemented here in the dumbest possible way: Python
sets, explicit loops, ``len(C & D) / len(C | D)`` written out exactly as the
paper writes it. The oracle is far too slow to ship, but it is transparently a
transcription of the definition, and any disagreement between the two is a bug
in one of them.

Reference
---------
Hennig, C. (2007). Cluster-wise assessment of cluster stability.
*Computational Statistics & Data Analysis*, 52(1), 258-271.
"""

import math
import statistics

import numpy as np
import pytest

from scstability._metrics import (
    NOT_SAMPLED,
    jaccard_per_cluster,
    per_cell_stability,
    stability,
)

# ---------------------------------------------------------------------------
# the oracle: the definition, transcribed literally
# ---------------------------------------------------------------------------


def naive_jaccard_per_cluster(
    ref: np.ndarray, boot: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Hennig's cluster-wise Jaccard, written out with Python sets."""
    n_obs = len(ref)
    sampled = {i for i in range(n_obs) if boot[i] != NOT_SAMPLED}
    ref_ids = sorted(set(ref.tolist()))
    boot_ids = sorted({int(b) for b in boot.tolist() if b != NOT_SAMPLED})

    jaccards, matches = [], []
    for k in ref_ids:
        # C_k intersected with S_b: the reference cluster, restricted to cells
        # this bootstrap actually drew.
        c = {i for i in range(n_obs) if ref[i] == k} & sampled
        if not c:
            jaccards.append(np.nan)
            matches.append(NOT_SAMPLED)
            continue

        best_value, best_label = -1.0, NOT_SAMPLED
        for j in boot_ids:  # ascending, so ties keep the smallest label
            d = {i for i in range(n_obs) if boot[i] == j}
            value = len(c & d) / len(c | d)
            if value > best_value:
                best_value, best_label = value, j
        jaccards.append(best_value)
        matches.append(best_label)

    return np.array(jaccards, dtype=float), np.array(matches, dtype=np.int64)


def naive_per_cell(ref: np.ndarray, boots: np.ndarray) -> np.ndarray:
    """Per-cell stability, written out cell by cell and bootstrap by bootstrap."""
    ref_ids = sorted(set(ref.tolist()))
    matches = [naive_jaccard_per_cluster(ref, b)[1] for b in boots]

    out = np.full(len(ref), np.nan)
    for i in range(len(ref)):
        k_index = ref_ids.index(ref[i])
        seen = hits = 0
        for b_index, boot in enumerate(boots):
            if boot[i] == NOT_SAMPLED:
                continue  # not in B_i: no evidence, not a failure
            seen += 1
            if boot[i] == matches[b_index][k_index]:
                hits += 1
        if seen:
            out[i] = hits / seen
    return out


def naive_mean_median_q25_q75(ref: np.ndarray, boots: np.ndarray) -> np.ndarray:
    """Per-cluster mean/median/q25/q75 over the bootstraps that carried evidence.

    The mean is included because it is the statistic Hennig's bands are defined
    on and therefore the one the package reports as its headline. An oracle
    that checked only the quantiles would leave the number users actually read
    unverified.

    Computed with a plain Python loop and ``statistics.fmean`` rather than
    numpy's ``nanmean``, so that agreement with the vectorised implementation
    is evidence rather than a shared code path.
    """
    per_boot = np.array([naive_jaccard_per_cluster(ref, b)[0] for b in boots])
    n_clusters = per_boot.shape[1]
    out = np.full((4, n_clusters), np.nan)
    for k in range(n_clusters):
        column = per_boot[:, k]
        evidence = [v for v in column.tolist() if not math.isnan(v)]
        if evidence:
            out[0, k] = statistics.fmean(evidence)
            out[1:, k] = np.quantile(np.array(evidence), [0.5, 0.25, 0.75])
    return out


# ---------------------------------------------------------------------------
# random configurations to compare on
# ---------------------------------------------------------------------------


def random_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A random (reference, bootstraps) pair, deliberately including nasty ones.

    The drop rate reaches 1.0, so some bootstraps sample nothing at all, and
    some reference clusters go unsampled -- the NaN paths that a gentler
    generator would never reach.
    """
    n_obs = int(rng.integers(6, 40))
    n_ref = int(rng.integers(1, 6))
    n_boot = int(rng.integers(1, 7))
    n_boot_clusters = int(rng.integers(1, 6))
    drop = float(rng.choice([0.0, 0.2, 0.5, 0.8, 1.0]))

    ref = rng.integers(0, n_ref, size=n_obs)
    boots = rng.integers(0, n_boot_clusters, size=(n_boot, n_obs))
    boots[rng.random((n_boot, n_obs)) < drop] = NOT_SAMPLED
    return ref, boots


def test_jaccard_matches_the_oracle(strict_warnings):
    """Fast and naive cluster-wise Jaccard agree, including which cluster matched."""
    rng = np.random.default_rng(20260831)

    for _ in range(300):
        ref, boots = random_case(rng)
        for boot in boots:
            with strict_warnings():
                fast_j, fast_m = jaccard_per_cluster(ref, boot)
            slow_j, slow_m = naive_jaccard_per_cluster(ref, boot)

            np.testing.assert_allclose(fast_j, slow_j, equal_nan=True)
            np.testing.assert_array_equal(fast_m, slow_m)


def test_per_cell_matches_the_oracle(strict_warnings):
    """Fast and naive per-cell stability agree, NaNs in the same places."""
    rng = np.random.default_rng(11235)

    for _ in range(200):
        ref, boots = random_case(rng)
        with strict_warnings():
            fast = per_cell_stability(ref, boots)
        slow = naive_per_cell(ref, boots)

        np.testing.assert_allclose(fast, slow, equal_nan=True)
        # Same cells lack evidence in both -- the distinction the package rests on.
        np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))


def test_aggregation_matches_the_oracle(strict_warnings):
    """Aggregation across bootstraps agrees, with NaN bootstraps excluded not zeroed.

    Covers the mean as well as the quantiles: the mean is the number the
    package bands and recommends on, so leaving it out of the differential
    test would leave the headline statistic unverified.
    """
    rng = np.random.default_rng(99)

    for _ in range(200):
        ref, boots = random_case(rng)
        with strict_warnings():
            result = stability(ref, boots)
        expected = naive_mean_median_q25_q75(ref, boots)

        np.testing.assert_allclose(result.jaccard_mean, expected[0], equal_nan=True)
        np.testing.assert_allclose(result.jaccard_median, expected[1], equal_nan=True)
        np.testing.assert_allclose(result.jaccard_q25, expected[2], equal_nan=True)
        np.testing.assert_allclose(result.jaccard_q75, expected[3], equal_nan=True)


def test_oracle_reproduces_the_paper_example():
    """Sanity-check the oracle itself against the hand-computed case.

    An oracle that agrees with a buggy implementation because both share the
    same misunderstanding would prove nothing, so the oracle is independently
    pinned to the worked example: C={2,5,6,9}, S={2,5,6,9,11}, D0={2,5},
    D1={6,9,11} gives J = max(2/4, 2/5) = 0.5.
    """
    ref = np.array([1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1])
    boot = np.array([-1, -1, 0, -1, -1, 0, 1, -1, -1, 1, -1, 1])

    jaccard, best_match = naive_jaccard_per_cluster(ref, boot)

    assert jaccard[0] == pytest.approx(0.5)
    assert jaccard[1] == pytest.approx(1 / 3)
    assert best_match.tolist() == [0, 1]

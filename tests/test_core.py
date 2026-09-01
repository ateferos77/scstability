"""Behaviour of the sweep on data whose answer we already know.

These are the tests that decide whether the package does its job. The metric
tests prove the arithmetic; these prove the arithmetic is being applied to the
right thing, and that the verdict it produces matches reality on datasets
constructed so that reality is not in doubt.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import scstability as scs
from scstability import HENNIG_BANDS, _core

# A single sweep shared by the tests that only read it. Module-scoped because
# clustering is the expensive part of this suite and the result is immutable.
SHARED_RESOLUTIONS = [0.2, 0.8, 1.5]


@pytest.fixture(scope="module")
def blobs_result(blobs_adata):
    """One sweep across three resolutions on the three-blob data."""
    return scs.stability_sweep(
        blobs_adata,
        SHARED_RESOLUTIONS,
        n_boot=5,
        random_state=0,
        progress=False,
    )


# ---------------------------------------------------------------------------
# 8-10. does the verdict match reality?
# ---------------------------------------------------------------------------


def test_separated_blobs_are_stable(blobs_result):
    """Three well-separated blobs must come back as near-perfectly stable.

    If this fails the metric is broken: there is no ambiguity in this data.
    """
    summary = blobs_result.summary()
    three = summary[summary["n_clusters"] == 3]
    assert len(three), f"no resolution gave 3 clusters: {summary.to_dict('records')}"

    for resolution in three["resolution"]:
        block = blobs_result.cluster_stability
        medians = block[block["resolution"] == resolution]["jaccard_median"]
        assert (medians > 0.95).all(), f"resolution {resolution}: {medians.tolist()}"


def test_pure_noise_is_unstable(noise_adata):
    """The single most important test in the package.

    ``noise_adata`` is one isotropic Gaussian: there is no cluster structure to
    find. Leiden returns clusters anyway, because that is what it does. The
    package's entire claim is that it reports them as untrustworthy.

    A stability tool that cannot detect its own target failure mode is
    worthless, so this asserts the substantive result -- median cluster
    stability below Hennig's 0.60 "not trustworthy" line -- not merely that the
    code runs.
    """
    result = scs.stability_sweep(
        noise_adata, [1.5], n_boot=10, random_state=0, progress=False
    )
    summary = result.summary()

    assert summary["n_clusters"].iloc[0] >= 5, "expected Leiden to over-partition"
    assert summary["median_cluster_stability"].iloc[0] < 0.60
    assert summary["min_cluster_stability"].iloc[0] < 0.60


def test_stability_falls_with_overclustering(blobs_result):
    """Cutting three real blobs into six pieces must score worse than three."""
    summary = blobs_result.summary()
    coarse = summary[summary["n_clusters"] == 3]["median_cluster_stability"].max()
    fine = summary[summary["n_clusters"] >= 6]["median_cluster_stability"].max()

    assert not np.isnan(coarse) and not np.isnan(fine)
    assert coarse > fine


def test_noise_scores_worse_than_real_structure(blobs_result, noise_adata):
    """The comparison the package exists to make, asserted directly.

    Same code, same resolution, same number of bootstraps: real structure must
    score higher than no structure. This is the claim in one line.
    """
    noise = scs.stability_sweep(
        noise_adata, [0.8], n_boot=5, random_state=0, progress=False
    )
    blobs_median = (
        blobs_result.summary()
        .set_index("resolution")
        .loc[0.8, "median_cluster_stability"]
    )
    noise_median = noise.summary()["median_cluster_stability"].iloc[0]

    assert blobs_median > 0.95
    assert noise_median < 0.75
    assert blobs_median > noise_median


# ---------------------------------------------------------------------------
# 11. the frac=1.0 invariant
# ---------------------------------------------------------------------------


def test_frac_one_gives_perfect_stability(blobs_adata):
    """With frac=1.0 on separable data, every Jaccard is exactly 1.0.

    Pinned to ``blobs_adata`` deliberately, and this is NOT a general invariant
    of the code. At frac=1.0 the subsample is the whole dataset in original
    order, so the bootstrap graph is identical to the reference graph -- but
    each bootstrap still runs Leiden under its own derived seed, and Leiden is
    stochastic. On three well-separated blobs every seed finds the same
    partition, so the assertion holds exactly. On ``noise_adata`` or
    ``gradient_adata`` it would fail, correctly, because there the partition
    really is seed-dependent (measured: 8, 9 and 10 clusters across seeds).

    Do not weaken the == 1.0, and do not generalise this to other fixtures.
    """
    result = scs.stability_sweep(
        blobs_adata, [0.8], n_boot=3, frac=1.0, random_state=0, progress=False
    )

    np.testing.assert_array_equal(
        result.cluster_stability["jaccard_median"].to_numpy(), 1.0
    )
    np.testing.assert_array_equal(
        result.cluster_stability["jaccard_q25"].to_numpy(), 1.0
    )
    np.testing.assert_array_equal(result.per_cell[0.8].to_numpy(), 1.0)


# ---------------------------------------------------------------------------
# 12. reproducibility
# ---------------------------------------------------------------------------


def test_deterministic_with_seed(blobs_adata):
    """The same random_state reproduces the sweep exactly."""
    kwargs = dict(n_boot=4, random_state=7, progress=False)
    first = scs.stability_sweep(blobs_adata, [0.8, 1.5], **kwargs)
    second = scs.stability_sweep(blobs_adata, [0.8, 1.5], **kwargs)

    pd.testing.assert_frame_equal(first.cluster_stability, second.cluster_stability)
    pd.testing.assert_frame_equal(first.per_cell, second.per_cell)


def test_different_seeds_agree_without_being_identical(gradient_adata):
    """Two seeds must disagree in detail but rank cells alike.

    Run on ``gradient_adata`` rather than ``blobs_adata``. On separated blobs
    every seed returns exactly 1.0 everywhere, so "not identical" is false
    there -- the results are identical, correctly. A continuum is where seed
    variation actually shows up, which makes it the only fixture on which this
    test means anything.

    ``n_boot`` matters here and is not padding. A per-cell score with n_boot
    replicates can only take n_boot + 1 distinct values, so with too few
    replicates the rank correlation is capped by measurement granularity rather
    than by real disagreement. Measured on this fixture: rho = 0.31 at n_boot=6,
    0.64 at 10, 0.84 at 20, 0.93 at 40. Twelve gives comfortable margin over the
    0.5 threshold while staying inside the suite's time budget -- and the same
    arithmetic is why n_boot=5 is fine for a coarse verdict but too few to rank
    individual cells.
    """
    a = scs.stability_sweep(
        gradient_adata, [1.0], n_boot=12, random_state=0, progress=False
    )
    b = scs.stability_sweep(
        gradient_adata, [1.0], n_boot=12, random_state=1, progress=False
    )

    left = a.per_cell[1.0].to_numpy()
    right = b.per_cell[1.0].to_numpy()
    assert not np.array_equal(left, right)

    both = ~np.isnan(left) & ~np.isnan(right)
    rho = pd.Series(left[both]).corr(pd.Series(right[both]), method="spearman")
    assert rho > 0.5, f"per-cell rankings should agree across seeds, got {rho:.3f}"


# ---------------------------------------------------------------------------
# 13-15. the result object
# ---------------------------------------------------------------------------


def test_result_shapes_and_ranges(blobs_result, blobs_adata):
    """Shapes line up with the input and every value is in range."""
    n_res = len(SHARED_RESOLUTIONS)

    assert blobs_result.per_cell.shape == (blobs_adata.n_obs, n_res)
    assert blobs_result.reference_labels.shape == (blobs_adata.n_obs, n_res)
    assert blobs_result.per_cell.index.equals(blobs_adata.obs_names)
    assert list(blobs_result.per_cell.columns) == sorted(SHARED_RESOLUTIONS)

    values = blobs_result.per_cell.to_numpy(dtype=float)
    finite = values[~np.isnan(values)]
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0

    # One row per (resolution, cluster) pair.
    expected_rows = sum(
        blobs_result.reference_labels[r].nunique() for r in blobs_result.resolutions
    )
    assert len(blobs_result.cluster_stability) == expected_rows

    jaccard = blobs_result.cluster_stability["jaccard_median"].to_numpy()
    assert np.nanmin(jaccard) >= 0.0
    assert np.nanmax(jaccard) <= 1.0
    # Quartiles bracket the median.
    assert (
        blobs_result.cluster_stability["jaccard_q25"]
        <= blobs_result.cluster_stability["jaccard_median"] + 1e-12
    ).all()
    assert (
        blobs_result.cluster_stability["jaccard_q75"]
        >= blobs_result.cluster_stability["jaccard_median"] - 1e-12
    ).all()


def test_summary_columns_and_recommend(blobs_result):
    """``summary`` has a row per resolution; ``recommend`` returns one of them."""
    summary = blobs_result.summary()

    assert len(summary) == len(SHARED_RESOLUTIONS)
    assert list(summary.columns) == [
        "resolution",
        "n_clusters",
        "min_cluster_stability",
        "median_cluster_stability",
        "frac_cells_stable",
    ]
    assert summary["resolution"].tolist() == sorted(SHARED_RESOLUTIONS)

    chosen = blobs_result.recommend()
    assert chosen in SHARED_RESOLUTIONS


def test_recommend_picks_the_finest_trustworthy_resolution(blobs_result):
    """Not just any passing resolution -- the largest one."""
    summary = blobs_result.summary()
    passing = summary[summary["min_cluster_stability"] >= 0.75]["resolution"]

    assert blobs_result.recommend(0.75) == passing.max()


def test_recommend_warns_when_nothing_qualifies(noise_adata):
    """On noise nothing is trustworthy; say so, but still return an answer."""
    result = scs.stability_sweep(
        noise_adata, [1.0, 1.5], n_boot=5, random_state=0, progress=False
    )

    with pytest.warns(UserWarning, match="No resolution reached"):
        chosen = result.recommend(0.75)

    assert chosen in result.resolutions


def test_to_adata_roundtrip(blobs_result, blobs_adata):
    """Writes the expected obs columns and uns payload."""
    adata = blobs_adata.copy()
    blobs_result.to_adata(adata)

    for resolution in SHARED_RESOLUTIONS:
        column = f"stability_res{resolution:g}"
        assert column in adata.obs
        np.testing.assert_allclose(
            adata.obs[column].to_numpy(dtype=float),
            blobs_result.per_cell[resolution].to_numpy(dtype=float),
        )

    assert "stability" in adata.uns
    assert adata.uns["stability"]["params"]["n_boot"] == 5
    assert adata.uns["stability"]["params"]["frac"] == 0.8
    assert isinstance(adata.uns["stability"]["summary"], pd.DataFrame)


def test_to_adata_warns_rather_than_silently_overwriting(blobs_result, blobs_adata):
    """Clobbering a user's column without a word is not acceptable."""
    adata = blobs_adata.copy()
    adata.obs["stability_res0.8"] = "something the user cared about"

    with pytest.warns(UserWarning, match="Overwriting existing obs column"):
        blobs_result.to_adata(adata)


def test_to_adata_accepts_a_custom_key(blobs_result, blobs_adata):
    adata = blobs_adata.copy()
    blobs_result.to_adata(adata, key_added="jaccard")

    assert "jaccard_res0.8" in adata.obs
    assert "jaccard" in adata.uns


def test_to_adata_rejects_a_mismatched_object(blobs_result, blobs_adata):
    """Writing results onto the wrong cells would be silently meaningless."""
    other = blobs_adata.copy()
    other.obs_names = [f"other_{i}" for i in range(other.n_obs)]

    with pytest.raises(ValueError, match="obs_names do not match"):
        blobs_result.to_adata(other)

    fewer = blobs_adata[:100].copy()
    with pytest.raises(ValueError, match="obs_names do not match"):
        blobs_result.to_adata(fewer)


def test_params_record_every_argument(blobs_result):
    """Reproducibility: the result carries the recipe that made it."""
    params = blobs_result.params

    assert params["resolutions"] == sorted(SHARED_RESOLUTIONS)
    assert params["n_boot"] == 5
    assert params["frac"] == 0.8
    assert params["use_rep"] == "X_pca"
    assert params["n_neighbors"] == 15
    assert params["random_state"] == 0


def test_to_adata_rejects_resolutions_that_collide_in_a_column_name():
    """Distinct resolutions must yield distinct obs columns.

    ``:g`` keeps six significant figures, so 1.0000001 and 1.0000002 both
    format as "res1". Writing both would silently overwrite the first.
    """
    index = pd.Index([f"cell_{i}" for i in range(4)])
    resolutions = np.array([1.0000001, 1.0000002])
    result = scs.StabilityResult(
        resolutions=resolutions,
        cluster_stability=pd.DataFrame(
            [
                {
                    "resolution": r,
                    "cluster": 0,
                    "n_cells": 4,
                    "jaccard_median": 1.0,
                    "jaccard_q25": 1.0,
                    "jaccard_q75": 1.0,
                }
                for r in resolutions
            ]
        ),
        per_cell=pd.DataFrame({r: [1.0] * 4 for r in resolutions}, index=index),
        reference_labels=pd.DataFrame(
            {r: pd.Categorical([0] * 4) for r in resolutions}, index=index
        ),
        params={},
    )
    adata = AnnData(np.zeros((4, 2), dtype=np.float32))
    adata.obs_names = index

    with pytest.raises(ValueError, match="distinct obs column names"):
        result.to_adata(adata)


def test_recommend_raises_when_there_is_no_evidence_at_all():
    """All-NaN must give a sentence, not pandas' 'Encountered all NA values'."""
    index = pd.Index([f"cell_{i}" for i in range(4)])
    result = scs.StabilityResult(
        resolutions=np.array([0.5]),
        cluster_stability=pd.DataFrame(
            [
                {
                    "resolution": 0.5,
                    "cluster": 0,
                    "n_cells": 4,
                    "jaccard_mean": np.nan,
                    "jaccard_median": np.nan,
                    "jaccard_q25": np.nan,
                    "jaccard_q75": np.nan,
                }
            ]
        ),
        per_cell=pd.DataFrame({0.5: [np.nan] * 4}, index=index),
        reference_labels=pd.DataFrame({0.5: pd.Categorical([0] * 4)}, index=index),
        params={},
    )

    with pytest.raises(ValueError, match="any stability evidence"):
        result.recommend()


def test_sweep_wiring_against_hand_computed_values(monkeypatch):
    """The whole sweep, with Leiden replaced by fixed partitions.

    Every other test in this file asserts a *property* of the output, because
    real clustering is stochastic. That leaves the wiring itself only loosely
    pinned: which cell a per-cell score is attached to, which cluster an
    n_cells belongs to, where the threshold sits in frac_cells_stable. Feeding
    the sweep known partitions makes every output exactly computable, so those
    can be asserted element by element.

    6 cells. Reference: C0 = {0,1,2}, C1 = {3,4,5}. Cell 5 is never sampled.

    Bootstrap 0 -- D0 = {0,1,2}, D1 = {3,4}:
        J(C0, D0) = 3/3 = 1        best = D0
        J(C1, D1) = 2/2 = 1        best = D1
        cells 0-4 all sit in their cluster's best match: 5 hits.

    Bootstrap 1 -- D0 = {0,1}, D1 = {2,3,4}; cell 2 has defected:
        J(C0, D0) = 2/3, J(C0, D1) = 1/5   -> best = D0 at 2/3
        J(C1, D1) = 2/3, J(C1, D0) = 0     -> best = D1 at 2/3
        cell 2 is in D1 but C0 matched D0, so cell 2 misses.

    Therefore, exactly:
        jaccard per cluster = median(1, 2/3) = 5/6 for both
        q25 = 2/3 + 0.25/3 = 3/4 ; q75 = 2/3 + 0.75/3 = 11/12
        per_cell = [1, 1, 1/2, 1, 1, NaN]
        frac_cells_stable at 0.75 = 4 of the 5 evidenced cells = 0.8
    """
    reference = np.array([0, 0, 0, 1, 1, 1])
    replicates = iter(
        [
            np.array([0, 0, 0, 1, 1, -1]),
            np.array([0, 0, 1, 1, 1, -1]),
        ]
    )
    monkeypatch.setattr(_core, "leiden_labels", lambda X, **kw: reference.copy())
    monkeypatch.setattr(
        _core, "cluster_subsample", lambda X, **kw: (next(replicates), np.arange(5))
    )

    adata = AnnData(np.zeros((6, 2), dtype=np.float32))
    adata.obsm["X_pca"] = np.zeros((6, 2), dtype=np.float32)
    adata.obs_names = [f"cell_{i}" for i in range(6)]

    result = scs.stability_sweep(adata, [1.0], n_boot=2, progress=False)

    # Cluster level: sizes are of the reference clustering, not of the samples.
    assert result.cluster_stability["cluster"].tolist() == [0, 1]
    assert result.cluster_stability["n_cells"].tolist() == [3, 3]
    np.testing.assert_allclose(
        result.cluster_stability["jaccard_median"].to_numpy(), [5 / 6, 5 / 6]
    )
    np.testing.assert_allclose(
        result.cluster_stability["jaccard_q25"].to_numpy(), [0.75, 0.75]
    )
    np.testing.assert_allclose(
        result.cluster_stability["jaccard_q75"].to_numpy(), [11 / 12, 11 / 12]
    )

    # Cell level: element by element, so any reordering is caught.
    per_cell = result.per_cell[1.0].to_numpy()
    np.testing.assert_allclose(per_cell[:5], [1.0, 1.0, 0.5, 1.0, 1.0])
    assert np.isnan(per_cell[5])
    assert result.per_cell.index.tolist() == [f"cell_{i}" for i in range(6)]

    # Summary, including exactly where the frac_cells_stable threshold sits.
    summary = result.summary()
    assert summary["n_clusters"].iloc[0] == 2
    assert summary["min_cluster_stability"].iloc[0] == pytest.approx(5 / 6)
    assert summary["median_cluster_stability"].iloc[0] == pytest.approx(5 / 6)
    assert summary["frac_cells_stable"].iloc[0] == pytest.approx(0.8)


def test_frac_cells_stable_is_inclusive_at_the_threshold(monkeypatch):
    """A cell scoring exactly the threshold counts as stable (>=, not >)."""
    reference = np.array([0, 0, 0, 0])
    replicates = iter(
        [
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 0]),
            np.array([0, 0, 0, 0]),
            np.array([0, 0, 0, 0]),
        ]
    )
    monkeypatch.setattr(_core, "leiden_labels", lambda X, **kw: reference.copy())
    monkeypatch.setattr(
        _core, "cluster_subsample", lambda X, **kw: (next(replicates), np.arange(4))
    )

    adata = AnnData(np.zeros((4, 2), dtype=np.float32))
    adata.obsm["X_pca"] = np.zeros((4, 2), dtype=np.float32)

    result = scs.stability_sweep(adata, [1.0], n_boot=4, progress=False)

    # Cell 3 is in the best-matching cluster in 3 of 4 replicates: exactly 0.75.
    assert result.per_cell[1.0].to_numpy()[3] == pytest.approx(0.75)
    assert result.summary(stable_threshold=0.75)["frac_cells_stable"].iloc[0] == 1.0
    assert result.summary(stable_threshold=0.76)["frac_cells_stable"].iloc[0] == 0.75


def test_n_cells_matches_the_reference_clustering(blobs_result):
    """Cluster sizes are of the full reference partition, not of a subsample."""
    for resolution in blobs_result.resolutions:
        counts = blobs_result.reference_labels[resolution].value_counts()
        block = blobs_result.cluster_stability
        block = block[block["resolution"] == resolution]
        for _, row in block.iterrows():
            assert row["n_cells"] == counts[row["cluster"]]
        assert block["n_cells"].sum() == len(blobs_result.per_cell)


# ---------------------------------------------------------------------------
# A single cluster is trivially stable -- found by running on real data
# ---------------------------------------------------------------------------


def _fixed_partition_sweep(monkeypatch, partitions, resolutions, n_boot=4):
    """Sweep with the clustering replaced by partitions chosen per resolution.

    ``partitions`` maps a resolution to the label vector every call at that
    resolution returns, reference and bootstrap alike. Fixing the partition
    makes every output hand-computable, which stochastic Leiden never is.
    """
    n_obs = len(next(iter(partitions.values())))

    def fake_leiden(X, *, resolution, **kwargs):
        return partitions[resolution].copy()

    def fake_subsample(X, *, resolution, **kwargs):
        return partitions[resolution].copy(), np.arange(n_obs)

    monkeypatch.setattr(_core, "leiden_labels", fake_leiden)
    monkeypatch.setattr(_core, "cluster_subsample", fake_subsample)

    adata = AnnData(np.zeros((n_obs, 2), dtype=np.float32))
    adata.obsm["X_pca"] = np.zeros((n_obs, 2), dtype=np.float32)
    return scs.stability_sweep(adata, resolutions, n_boot=n_boot, progress=False)


def test_a_single_cluster_scores_a_perfect_but_meaningless_one(monkeypatch):
    """One cluster holding every cell is trivially stable, and says so.

    When the resample also collapses to a single cluster the Jaccard is
    exactly 1.0, so a dataset with no structure earns the best possible score
    for the worst possible reason. (It is not 1.0 under *every* resample -- one
    that splits the cluster in half scores 0.5 -- but structureless data
    collapses the same way each time.) The number is right; treating it as
    evidence is not.
    """
    everything = np.zeros(20, dtype=np.int64)
    result = _fixed_partition_sweep(monkeypatch, {0.1: everything}, [0.1])

    row = result.summary().iloc[0]
    assert row["n_clusters"] == 1
    assert row["min_cluster_stability"] == pytest.approx(1.0)

    with pytest.warns(UserWarning, match="single cluster"):
        result.recommend()


def test_recommend_never_returns_a_degenerate_resolution(monkeypatch):
    """Given the choice, a real partition beats a trivially perfect one.

    Resolution 0.1 collapses to one cluster and scores 1.0; resolution 0.8
    finds two genuine clusters and also scores 1.0. Sorting by score alone
    would be indifferent, and ``max(resolution)`` alone would already prefer
    0.8 -- so the case that matters is the reverse ordering, tested below.
    """
    everything = np.zeros(20, dtype=np.int64)
    split = np.repeat([0, 1], 10).astype(np.int64)

    result = _fixed_partition_sweep(
        monkeypatch, {0.1: everything, 0.8: split}, [0.1, 0.8]
    )

    assert result.recommend() == 0.8


def test_a_degenerate_resolution_is_skipped_even_when_it_is_the_finest(monkeypatch):
    """The guard is not just an artefact of preferring the largest resolution.

    Here the *finest* resolution is the degenerate one, so the usual
    "largest qualifying resolution wins" rule would pick it. It must not.
    """
    split = np.repeat([0, 1], 10).astype(np.int64)
    everything = np.zeros(20, dtype=np.int64)

    result = _fixed_partition_sweep(
        monkeypatch, {0.1: split, 0.8: everything}, [0.1, 0.8]
    )

    assert result.recommend() == 0.1


# ---------------------------------------------------------------------------
# Hennig's bands are defined on the mean, not the median
# ---------------------------------------------------------------------------


def test_bands_are_applied_to_the_mean_not_the_median(monkeypatch):
    """The banded statistic must be the mean, because Hennig's bands are.

    ``fpc::clusterboot`` states its guidance on ``bootmean``. Bootstrap
    Jaccards are left-skewed -- a cluster typically reassembles and
    occasionally shatters -- so the median sits above the mean and reading the
    bands off it reports a cluster as more trustworthy than the published
    calibration says it is.

    Constructed so the two answers fall in different bands: a cluster that is
    reproduced exactly in three resamples out of four and halved in the
    fourth has median 1.0 ("highly stable") and mean 0.875. Sharpened further
    below to straddle the 0.75 decision line.
    """
    reference = np.repeat([0, 1], 4).astype(np.int64)
    intact = np.repeat([0, 1], 4).astype(np.int64)
    # cluster 0 merged into a cluster of 8: J = 4/8 = 0.5
    merged = np.zeros(8, dtype=np.int64)

    replicates = iter([intact.copy(), intact.copy(), intact.copy(), merged.copy()])
    monkeypatch.setattr(_core, "leiden_labels", lambda X, **kw: reference.copy())
    monkeypatch.setattr(
        _core, "cluster_subsample", lambda X, **kw: (next(replicates), np.arange(8))
    )

    adata = AnnData(np.zeros((8, 2), dtype=np.float32))
    adata.obsm["X_pca"] = np.zeros((8, 2), dtype=np.float32)
    result = scs.stability_sweep(adata, [1.0], n_boot=4, progress=False)

    block = result.cluster_stability
    cluster0 = block[block["cluster"] == 0].iloc[0]

    # J = [1.0, 1.0, 1.0, 0.5] -> mean 0.875, median 1.0
    assert cluster0["jaccard_mean"] == pytest.approx(0.875)
    assert cluster0["jaccard_median"] == pytest.approx(1.0)
    assert cluster0["jaccard_mean"] < cluster0["jaccard_median"]

    # summary must report the mean, which is the stricter and correct number
    row = result.summary().iloc[0]
    assert row["min_cluster_stability"] == pytest.approx(0.875)
    assert row["min_cluster_stability"] != pytest.approx(1.0), (
        "banding the median would report a perfect score here"
    )


def test_a_left_skewed_cluster_is_demoted_two_bands_by_using_the_mean(monkeypatch):
    """The distinction changes the verdict, not just the third decimal.

    Reference cluster 0 holds 4 of 20 cells. In five resamples out of eight it
    is reproduced exactly (J = 1.0); in the other three the whole dataset
    collapses into one cluster, so J = 4/20 = 0.2.

    median = 1.0   -> "highly stable"
    mean   = 0.7   -> "a real pattern, but uncertain"

    Two bands apart, from the same numbers. Reading the bands off the median
    would report a cluster as beyond question when the published calibration
    says to seek independent support for it.
    """
    reference = np.repeat([0, 1], [4, 16]).astype(np.int64)
    intact = reference.copy()
    collapsed = np.zeros(20, dtype=np.int64)

    replicates = iter([intact.copy()] * 5 + [collapsed.copy()] * 3)
    monkeypatch.setattr(_core, "leiden_labels", lambda X, **kw: reference.copy())
    monkeypatch.setattr(
        _core, "cluster_subsample", lambda X, **kw: (next(replicates), np.arange(20))
    )

    adata = AnnData(np.zeros((20, 2), dtype=np.float32))
    adata.obsm["X_pca"] = np.zeros((20, 2), dtype=np.float32)
    result = scs.stability_sweep(adata, [1.0], n_boot=8, progress=False)

    block = result.cluster_stability
    cluster0 = block[block["cluster"] == 0].iloc[0]

    # J = [1.0] * 5 + [0.2] * 3
    assert cluster0["jaccard_median"] == pytest.approx(1.0)
    assert cluster0["jaccard_mean"] == pytest.approx((5 * 1.0 + 3 * 0.2) / 8)

    def band(value):
        for lower, name in HENNIG_BANDS:
            if value >= lower:
                return name
        return "unbanded"

    assert band(cluster0["jaccard_median"]) == "highly stable"
    assert band(cluster0["jaccard_mean"]) == "a real pattern, but uncertain"
    assert cluster0["jaccard_mean"] < 0.75, "the mean must demote this cluster"

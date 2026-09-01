"""Real data, end to end, at real scale. Marked slow.

Every other test in this suite runs on synthetic data, because synthetic data
is the only kind whose right answer is known in advance. That proves the
arithmetic, but it cannot show the package survives contact with a real
expression matrix: real sparsity, real cluster-size imbalance, a real kNN
graph that may be disconnected.

This test asserts **properties, not exact numbers**. Exact stability values
depend on the scanpy, igraph and leidenalg versions installed, and on the
upstream PCA -- ARPACK is iterative, so a scipy patch release moves
coordinates enough to shift a score in the third decimal. Pinning a number
here would produce a test that fails on a dependency bump while nothing is
actually wrong, which trains people to ignore it.

Deselected by default. Run with::

    pytest -m slow
"""

import numpy as np
import pytest

import scstability as scs

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def pbmc3k():
    """scanpy's processed pbmc3k, skipped if it cannot be downloaded."""
    import scanpy as sc

    try:
        adata = sc.datasets.pbmc3k_processed()
    except Exception as exc:  # network, or a moved upstream URL
        pytest.skip(f"pbmc3k_processed unavailable: {exc}")

    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=50)
    return adata


def test_pbmc3k_sane(pbmc3k):
    """A full sweep on real data completes and returns coherent output."""
    resolutions = [0.4, 0.8, 1.2]

    result = scs.stability_sweep(
        pbmc3k, resolutions, n_boot=5, random_state=0, progress=False
    )

    # --- shapes line up with the input, not with some internal subsample ----
    assert list(result.resolutions) == resolutions
    assert len(result.per_cell) == pbmc3k.n_obs
    assert list(result.per_cell.columns) == resolutions
    assert (result.per_cell.index == pbmc3k.obs_names).all()

    summary = result.summary()
    assert len(summary) == len(resolutions)

    # --- no NaN anywhere in the summary ------------------------------------
    # A NaN here would mean a whole resolution produced no evidence, which at
    # n_boot=5 on 2700 cells should be unreachable.
    assert not summary.isna().to_numpy().any(), summary

    # --- every score is a Jaccard, so it lives in [0, 1] -------------------
    for column in (
        "min_cluster_stability",
        "median_cluster_stability",
        "frac_cells_stable",
    ):
        values = summary[column].to_numpy(dtype=float)
        assert ((values >= 0.0) & (values <= 1.0)).all(), (column, values)

    block = result.cluster_stability
    for column in ("jaccard_mean", "jaccard_median", "jaccard_q25", "jaccard_q75"):
        values = block[column].to_numpy(dtype=float)
        finite = values[~np.isnan(values)]
        assert ((finite >= 0.0) & (finite <= 1.0)).all(), column

    # --- cluster sizes account for every cell ------------------------------
    for resolution in resolutions:
        at_res = block[block["resolution"] == resolution]
        assert at_res["n_cells"].sum() == pbmc3k.n_obs
        assert (
            len(at_res) == summary.set_index("resolution").loc[resolution, "n_clusters"]
        )

    # --- the big clusters of a real, well-separated dataset should hold ----
    # PBMCs contain genuinely distinct populations (T cells, B cells,
    # monocytes). The largest clusters at a coarse resolution are those
    # populations, and they should survive resampling. Asserted only for
    # clusters of at least 200 cells, since small clusters are legitimately
    # noisy and this test must not fail for a correct reason.
    coarse = block[block["resolution"] == 0.4]
    large = coarse[coarse["n_cells"] >= 200]
    assert len(large) >= 2, "expected at least two large clusters in pbmc3k"
    assert (large["jaccard_mean"] > 0.7).all(), large[
        ["cluster", "n_cells", "jaccard_mean"]
    ]


def test_pbmc3k_recommendation_is_usable(pbmc3k):
    """``recommend`` returns a resolution from the grid, and a non-degenerate one."""
    resolutions = [0.4, 0.8, 1.2]
    result = scs.stability_sweep(
        pbmc3k, resolutions, n_boot=5, random_state=0, progress=False
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        chosen = result.recommend()

    assert chosen in resolutions
    n_clusters = result.summary().set_index("resolution").loc[chosen, "n_clusters"]
    assert n_clusters >= 2, "a single-cluster resolution must never be recommended"

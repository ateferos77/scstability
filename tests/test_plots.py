"""Plot contract tests.

These do not assert that a figure looks good -- no test can. They assert the
contract every plotting function in a library owes its caller: it returns the
artist, it draws where it is told, it never seizes control of the display, and
it fails with a sentence when the input cannot be plotted.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scstability as scs
from scstability.pl._plots import BANDS, band_colour


@pytest.fixture(scope="module")
def result(blobs_adata):
    """A sweep with enough spread across resolutions to exercise every band."""
    return scs.stability_sweep(
        blobs_adata, [0.2, 0.8, 1.5, 3.0], n_boot=5, random_state=0, progress=False
    )


@pytest.fixture
def umap_adata(blobs_adata, result):
    """The blob data with a stand-in 2-D embedding under ``X_umap``."""
    adata = blobs_adata.copy()
    adata.obsm["X_umap"] = np.asarray(adata.obsm["X_pca"])[:, :2].copy()
    return adata


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# 21-22. return values and the ax argument
# ---------------------------------------------------------------------------


def test_plots_return_axes_or_figure(result, umap_adata):
    """Each function hands back the artist it drew on."""
    assert isinstance(scs.pl.stability_curve(result), Figure)
    assert isinstance(scs.pl.cluster_stability(result, resolution=0.8), Axes)
    assert isinstance(scs.pl.stability_umap(umap_adata, result, resolution=0.8), Axes)


def test_plots_accept_ax_and_draw_into_it(result, umap_adata):
    """Given an axes, draw there and return that same axes."""
    for call in (
        lambda ax: scs.pl.stability_curve(result, ax=ax),
        lambda ax: scs.pl.cluster_stability(result, resolution=0.8, ax=ax),
        lambda ax: scs.pl.stability_umap(umap_adata, result, resolution=0.8, ax=ax),
    ):
        fig, ax = plt.subplots()
        returned = call(ax)

        assert returned is ax, "must return the axes it was given, not a new one"
        assert returned.figure is fig, "must not create a figure of its own"
        assert ax.has_data(), "nothing was actually drawn"


def test_stability_curve_with_ax_omits_the_count_panel(result):
    """Handed one axes, it draws the stability panel alone rather than two."""
    fig, ax = plt.subplots()

    returned = scs.pl.stability_curve(result, ax=ax)

    assert returned is ax
    assert len(fig.axes) == 1


def test_stability_curve_uses_panels_not_a_second_y_axis(result):
    """The cluster count gets its own panel on a shared x, never a twin axis.

    Two y-scales on one frame let a reader draw whatever relationship they came
    for, because the crossing point is an artefact of the scaling. Asserted so
    a future "simplification" back to ``twinx`` does not slip through.
    """
    fig = scs.pl.stability_curve(result)

    assert len(fig.axes) == 2, "expected a stability panel and a count panel"
    top, bottom = fig.axes
    # A twinx pair shares a position; stacked panels do not.
    assert top.get_position().y0 > bottom.get_position().y0
    assert top.get_shared_x_axes().joined(top, bottom)


# ---------------------------------------------------------------------------
# 23. never seize the display
# ---------------------------------------------------------------------------


def test_plots_do_not_call_show(result, umap_adata, monkeypatch):
    """A library that calls plt.show() cannot be composed into a figure."""
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))

    scs.pl.stability_curve(result)
    scs.pl.cluster_stability(result, resolution=0.8)
    scs.pl.stability_umap(umap_adata, result, resolution=0.8)

    assert calls == []


# ---------------------------------------------------------------------------
# 24. informative failures
# ---------------------------------------------------------------------------


def test_stability_umap_requires_umap(blobs_adata, result):
    """Missing X_umap names the key and says how to make one."""
    adata = blobs_adata.copy()
    adata.obsm.pop("X_umap", None)

    with pytest.raises(ValueError, match="X_umap"):
        scs.pl.stability_umap(adata, result, resolution=0.8)

    with pytest.raises(ValueError, match=r"sc\.tl\.umap"):
        scs.pl.stability_umap(adata, result, resolution=0.8)


def test_stability_umap_rejects_mismatched_cells(umap_adata, result):
    """Colouring the wrong cells would be silently meaningless."""
    other = umap_adata.copy()
    other.obs_names = [f"other_{i}" for i in range(other.n_obs)]

    with pytest.raises(ValueError, match="obs_names do not match"):
        scs.pl.stability_umap(other, result, resolution=0.8)


@pytest.mark.parametrize("bad", [0.35, 99.0])
def test_unknown_resolution_is_rejected(result, umap_adata, bad):
    """Ask for a resolution not in the grid and be told what is available."""
    with pytest.raises(ValueError, match="not in this result"):
        scs.pl.cluster_stability(result, resolution=bad)

    with pytest.raises(ValueError, match="Available"):
        scs.pl.stability_umap(umap_adata, result, resolution=bad)


# ---------------------------------------------------------------------------
# encoding: colour must mean what it says
# ---------------------------------------------------------------------------


def test_band_colour_follows_hennig_thresholds():
    """The decision line is 0.75, and 0.60 separates uncertain from rejected."""
    assert band_colour(0.99) == band_colour(0.80)  # both trustworthy
    assert band_colour(0.7499) != band_colour(0.7501)  # the decision line
    assert band_colour(0.5999) != band_colour(0.6001)
    assert band_colour(float("nan")) not in {
        band_colour(0.9),
        band_colour(0.7),
        band_colour(0.5),
    }


def test_bands_tile_the_unit_interval_without_gaps():
    """Every value in [0, 1] falls in exactly one band."""
    lowers = [b[0] for b in BANDS]
    uppers = [b[1] for b in BANDS]

    assert lowers[0] == 0.0
    assert uppers[-1] == 1.0
    assert lowers[1:] == uppers[:-1]


def test_no_evidence_cells_are_drawn_but_not_coloured_as_unstable(blobs_adata):
    """NaN cells appear in grey with their own legend entry, never at ramp zero.

    Uses ``n_boot=2`` deliberately so unsampled cells are guaranteed rather
    than hoped for: a cell is missed by both replicates with probability
    ``0.2 ** 2 = 4%``, so roughly a dozen of the 300 cells have no evidence.
    At the suite's usual ``n_boot=5`` the rate is 0.03% and the case would
    almost never arise, leaving the grey path untested.
    """
    sparse = scs.stability_sweep(
        blobs_adata, [0.8], n_boot=2, random_state=0, progress=False
    )
    adata = blobs_adata.copy()
    adata.obsm["X_umap"] = np.asarray(adata.obsm["X_pca"])[:, :2].copy()

    values = sparse.per_cell[0.8].to_numpy(dtype=float)
    assert np.isnan(values).any(), "expected some cells to go unsampled at n_boot=2"

    ax = scs.pl.stability_umap(adata, sparse, resolution=0.8)

    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("no evidence" in text for text in labels)


def test_cluster_stability_sorts_weakest_first(result):
    """The clusters to worry about are read first."""
    ax = scs.pl.cluster_stability(result, resolution=3.0)

    block = result.cluster_stability
    block = block[block["resolution"] == 3.0]
    expected = block.sort_values("jaccard_mean")["cluster"].tolist()
    labels = [t.get_text() for t in ax.get_yticklabels()]

    assert labels == [f"cluster {c}" for c in expected]


def test_cluster_stability_annotates_every_cluster_size(result):
    """Sizes are labelled outside the bar end, so a short bar cannot clip one."""
    ax = scs.pl.cluster_stability(result, resolution=0.8)

    texts = [t.get_text() for t in ax.texts]
    sizes = result.cluster_stability
    sizes = sizes[sizes["resolution"] == 0.8]["n_cells"]

    for size in sizes:
        assert any(f"{size:,} cells" in text for text in texts)


def test_curve_marks_the_recommendation(result):
    """The reader should not have to work out which resolution won."""
    fig = scs.pl.stability_curve(result)
    texts = [t.get_text() for t in fig.axes[0].texts]

    assert any("res =" in text for text in texts)


def test_curve_survives_a_result_with_no_qualifying_resolution(noise_adata):
    """On noise nothing qualifies; the figure must still render, and say so."""
    noisy = scs.stability_sweep(
        noise_adata, [1.0, 1.5], n_boot=5, random_state=0, progress=False
    )

    fig = scs.pl.stability_curve(noisy)

    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("best available" in text for text in texts)


def test_cluster_stability_puts_the_weakest_cluster_at_the_top(result):
    """Ascending sort plus an inverted axis: worst is read first, at the top."""
    ax = scs.pl.cluster_stability(result, resolution=3.0)

    bottom, top = ax.get_ylim()
    assert bottom > top, "y-axis should be inverted so position 0 renders at the top"


def test_cluster_size_labels_clear_the_error_bars(result):
    """Size labels sit beyond the q75 whisker, not on top of it."""
    ax = scs.pl.cluster_stability(result, resolution=3.0)

    block = result.cluster_stability
    block = block[block["resolution"] == 3.0].sort_values("jaccard_mean")
    q75 = block["jaccard_q75"].to_numpy(dtype=float)
    means = block["jaccard_mean"].to_numpy(dtype=float)

    anchors = sorted(a.xy[0] for a in ax.texts if "cells" in a.get_text())
    expected = sorted(np.maximum(np.nan_to_num(means), np.nan_to_num(q75)))

    assert len(anchors) == len(expected)
    np.testing.assert_allclose(anchors, expected)


def test_the_iqr_is_drawn_where_it_actually_is_not_clipped_to_the_bar():
    """A mean below q25 must leave a visible gap, not collapse onto the bar.

    Bootstrap Jaccards can be left-skewed, putting the mean below q25. It is
    uncommon at a magnitude that matters -- about 2% of clusters measured
    across four datasets -- but those are the informative ones: a cluster with
    mean 0.880 and q25 0.993 reassembles almost perfectly most of the time and
    occasionally collapses, which is a different failure from being mediocre
    throughout. Drawing the IQR as a matplotlib error bar cannot show it:
    ``xerr`` may not be negative, so the offset is clipped at zero and the
    lower whisker lands on the bar tip in exactly that case.

    This asserts the rule starts at q25 regardless of where the mean sits.
    """
    resolution = 0.5
    block = pd.DataFrame(
        [
            # mean well below q25: the skewed case that used to be invisible
            {
                "resolution": resolution,
                "cluster": 0,
                "n_cells": 100,
                "jaccard_mean": 0.40,
                "jaccard_median": 0.90,
                "jaccard_q25": 0.85,
                "jaccard_q75": 0.95,
            },
            # an ordinary cluster, mean inside its own IQR
            {
                "resolution": resolution,
                "cluster": 1,
                "n_cells": 100,
                "jaccard_mean": 0.70,
                "jaccard_median": 0.70,
                "jaccard_q25": 0.60,
                "jaccard_q75": 0.80,
            },
        ]
    )
    index = pd.Index([f"cell_{i}" for i in range(4)])
    result = scs.StabilityResult(
        resolutions=np.array([resolution]),
        cluster_stability=block,
        per_cell=pd.DataFrame({resolution: [0.5] * 4}, index=index),
        reference_labels=pd.DataFrame(
            {resolution: pd.Categorical([0, 0, 1, 1])}, index=index
        ),
        params={},
    )

    ax = scs.pl.cluster_stability(result, resolution=resolution)

    segments = [
        seg
        for collection in ax.collections
        for seg in collection.get_segments()
        if len(seg) == 2 and seg[0][1] == seg[1][1]  # horizontal => an IQR rule
    ]
    spans = sorted(
        (round(min(s[0][0], s[1][0]), 6), round(max(s[0][0], s[1][0]), 6))
        for s in segments
    )

    assert (0.85, 0.95) in spans, (
        f"the skewed cluster's IQR must be drawn at 0.85-0.95, got {spans}"
    )
    assert (0.60, 0.80) in spans, spans

    # and the gap is real: the bar ends at 0.40, the rule starts at 0.85
    skewed = next(s for s in spans if s == (0.85, 0.95))
    assert skewed[0] > 0.40, "the mean must be visibly left of its own IQR"

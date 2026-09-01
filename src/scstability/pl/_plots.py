"""Plot implementations.

``stability_curve`` is the headline figure of the package and the top of the
README; ``cluster_stability`` and ``stability_umap`` support it.

Design notes
------------
Colour carries meaning here, never decoration. The Hennig bands are a *status*
encoding -- trust / uncertain / reject -- so they use a reserved status palette
that never doubles as a series colour, and every band is labelled: hue never
carries the meaning on its own.

``stability_curve`` deliberately does **not** use a secondary y-axis for the
cluster count. Two y-scales on one frame is the classic way to make a chart
say whatever the reader wants; the cluster count gets its own short panel under
a shared x-axis instead, which shows the same relationship honestly. When an
``ax`` is supplied the stability panel alone is drawn into it.

No function here calls ``plt.show()``.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .._core import STABLE_THRESHOLD, StabilityResult

__all__ = ["cluster_stability", "stability_curve", "stability_umap"]

# --- chrome ---------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# --- status palette (reserved; never used as a series colour) --------------
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

#: Hennig's bands as ``(lower, upper, label, colour, band_alpha)``, worst first
#: so they paint bottom-up. The two trustworthy bands share a hue and differ in
#: depth: 0.85 refines a judgement that 0.75 has already made, so promoting it
#: to its own hue would overstate it.
BANDS: tuple[tuple[float, float, str, str, float], ...] = (
    (0.00, 0.60, "not trustworthy", CRITICAL, 0.09),
    (0.60, 0.75, "uncertain", WARNING, 0.11),
    (0.75, 0.85, "stable", GOOD, 0.07),
    (0.85, 1.00, "highly stable", GOOD, 0.15),
)

#: Sequential single hue, light to dark, for continuous magnitude.
_BLUE_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
STABILITY_CMAP = LinearSegmentedColormap.from_list("scstability_blue", _BLUE_RAMP)

#: Cells with no evidence. Recessive, and distinct from every ramp step.
NO_EVIDENCE_COLOUR = "#c3c2b7"


def band_colour(value: float) -> str:
    """Status colour for a stability value, by Hennig band.

    Three hues, not four: 0.75 is the decision line, and 0.85 refines a verdict
    already made rather than reversing it.

    Examples
    --------
    >>> band_colour(0.91) == band_colour(0.78)
    True
    >>> band_colour(0.42)
    '#d03b3b'
    """
    if np.isnan(value):
        return NO_EVIDENCE_COLOUR
    if value >= 0.75:
        return GOOD
    if value >= 0.60:
        return WARNING
    return CRITICAL


def _style(ax: Axes, *, grid_axis: str = "y") -> None:
    """Recessive chrome: hairline solid grid, two spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_SECONDARY)


def _check_resolution(result: StabilityResult, resolution: float) -> float:
    """Resolve a requested resolution to one actually present in the sweep."""
    matches = np.flatnonzero(np.isclose(result.resolutions, resolution))
    if matches.size == 0:
        raise ValueError(
            f"resolution {resolution} is not in this result. Available: "
            f"{result.resolutions.tolist()}"
        )
    return float(result.resolutions[matches[0]])


def _paint_bands(ax: Axes, *, horizontal: bool, label: bool = True) -> None:
    """Shade Hennig's interpretation bands behind the data."""
    span = ax.axhspan if horizontal else ax.axvspan
    for lower, upper, text, colour, alpha in BANDS:
        span(lower, upper, color=colour, alpha=alpha, linewidth=0, zorder=0)
        if not label:
            continue
        if horizontal:
            ax.text(
                1.005,
                (lower + upper) / 2,
                text,
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=8,
                color=INK_MUTED,
            )


def stability_curve(
    result: StabilityResult,
    threshold: float = STABLE_THRESHOLD,
    ax: Axes | None = None,
) -> Axes | Figure:
    """Plot cluster stability across the resolution grid. The headline figure.

    Each dot is one cluster, coloured by its Hennig band. The solid line tracks
    the **weakest** cluster at each resolution, which is the number the
    recommendation is made on: a clustering is only as trustworthy as its least
    reproducible group. The lower panel shows how many clusters each resolution
    produced, so granularity can be read against reliability.

    Parameters
    ----------
    result
        A completed sweep.
    threshold
        Decision line drawn across the plot, and the threshold passed to
        :meth:`~scstability.StabilityResult.recommend` for the annotation.
    ax
        Draw the stability panel into an existing axes. When given, the cluster
        count panel is omitted and the axes is returned.

    Returns
    -------
    matplotlib.axes.Axes or matplotlib.figure.Figure
        The axes when ``ax`` was supplied, otherwise the two-panel figure.

    Notes
    -----
    The cluster count is a separate panel rather than a second y-axis on the
    same frame. Two y-scales let a reader draw whatever relationship they came
    for, since the crossing point is an artefact of the scaling choice.

    Examples
    --------
    >>> fig = stability_curve(result)  # doctest: +SKIP
    >>> fig.savefig("stability.png", dpi=200)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    figure: Figure | None = None
    if ax is None:
        figure, (ax, ax_count) = plt.subplots(
            2,
            1,
            figsize=(7.2, 5.6),
            sharex=True,
            height_ratios=(3.2, 1.0),
            constrained_layout=True,
        )
        figure.patch.set_facecolor(SURFACE)
    else:
        ax_count = None

    resolutions = result.resolutions
    positions = np.arange(resolutions.size)
    summary = result.summary(stable_threshold=threshold)

    _paint_bands(ax, horizontal=True)

    # One dot per cluster, jittered so overlapping values stay countable.
    jitter_rng = np.random.default_rng(0)
    for position, resolution in zip(positions, resolutions, strict=True):
        block = result.cluster_stability
        values = block[block["resolution"] == resolution]["jaccard_mean"]
        values = values.to_numpy(dtype=float)
        finite = values[~np.isnan(values)]
        if finite.size == 0:
            continue
        offsets = jitter_rng.uniform(-0.13, 0.13, size=finite.size)
        ax.scatter(
            position + offsets,
            finite,
            s=42,
            c=[band_colour(v) for v in finite],
            edgecolor=SURFACE,
            linewidth=1.1,
            zorder=3,
        )

    minima = summary["min_cluster_stability"].to_numpy(dtype=float)
    ax.plot(
        positions,
        minima,
        color=INK,
        linewidth=2.0,
        marker="o",
        markersize=5,
        markerfacecolor=SURFACE,
        markeredgewidth=1.6,
        zorder=4,
    )
    ax.axhline(threshold, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)))

    # The recommendation, direct-labelled rather than left to the reader.
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            chosen = result.recommend(threshold)
    except ValueError:
        chosen = None

    if chosen is not None:
        index = int(np.flatnonzero(np.isclose(resolutions, chosen))[0])
        meets = bool(minima[index] >= threshold)
        # A vertical rule with the label pinned to the floor, rather than a
        # callout beside the point. The minimum line is at its steepest exactly
        # where the recommendation sits, so anything anchored to that point
        # collides with it; the floor beneath the chosen resolution is reliably
        # clear, because that is the resolution whose weakest cluster is highest.
        ax.axvline(
            index,
            color=GOOD if meets else WARNING,
            linewidth=1.4,
            alpha=0.55,
            zorder=1,
        )
        # Pull the label inside the frame when the chosen resolution is at an
        # end of the grid, which is common: a centred label there would hang
        # off the axes.
        last = max(positions.size - 1, 1)
        if index <= 0.15 * last:
            align, offset = "left", 0.12
        elif index >= 0.85 * last:
            align, offset = "right", -0.12
        else:
            align, offset = "center", 0.0
        ax.text(
            index + offset,
            0.035,
            f"recommended\nres = {chosen:g}"
            if meets
            else f"best available\nres = {chosen:g}",
            ha=align,
            va="bottom",
            fontsize=8.5,
            color=INK_SECONDARY,
            zorder=5,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2.0, "alpha": 0.85},
        )

    # Headroom so markers sitting exactly at 0 or 1 are not sliced by the frame.
    # Perfect stability is the commonest value in this plot, so clipping it
    # would mangle the case the reader most wants to see.
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-0.5, positions.size - 0.5)
    ax.set_ylabel("cluster stability (Jaccard)", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        "Which clusters survive resampling?",
        color=INK,
        fontsize=13,
        loc="left",
        pad=26,
    )
    _style(ax)

    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=7,
                markerfacecolor=INK_MUTED,
                markeredgecolor=SURFACE,
                label="one cluster",
            ),
            Line2D([], [], color=INK, linewidth=2.0, label="weakest cluster"),
            Line2D(
                [],
                [],
                color=INK_SECONDARY,
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                label=f"threshold ({threshold:g})",
            ),
        ],
        # A horizontal strip between the title and the frame. Inside the data
        # area there is nowhere safe: the top fills up whenever clusters are
        # stable, the bottom whenever they are not.
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncols=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_SECONDARY,
        handletextpad=0.6,
        columnspacing=1.8,
        borderpad=0.0,
    )

    if ax_count is None:
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{r:g}" for r in resolutions])
        ax.set_xlabel("Leiden resolution", color=INK_SECONDARY, fontsize=10)
        return ax

    counts = summary["n_clusters"].to_numpy(dtype=float)
    ax_count.plot(
        positions,
        counts,
        color=INK_MUTED,
        linewidth=1.6,
        marker="o",
        markersize=4,
        markerfacecolor=SURFACE,
        markeredgewidth=1.2,
    )
    for position, count in zip(positions, counts, strict=True):
        ax_count.annotate(
            f"{int(count)}",
            xy=(position, count),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK_MUTED,
        )
    ax_count.set_ylim(0, counts.max() * 1.35 if counts.size else 1)
    ax_count.set_ylabel("clusters", color=INK_SECONDARY, fontsize=10)
    ax_count.set_xlabel("Leiden resolution", color=INK_SECONDARY, fontsize=10)
    ax_count.set_xticks(positions)
    ax_count.set_xticklabels([f"{r:g}" for r in resolutions])
    _style(ax_count)

    return figure


def cluster_stability(
    result: StabilityResult,
    resolution: float,
    ax: Axes | None = None,
) -> Axes:
    """Plot each cluster's stability at one resolution, weakest first.

    Bars are sorted so the clusters to worry about are read first. Whiskers
    span the interquartile range across bootstraps: a short bar with a long
    whisker is a cluster whose fate depends on which cells you drew.

    Parameters
    ----------
    result
        A completed sweep.
    resolution
        Which resolution to show. Must be in the grid.
    ax
        Draw into an existing axes.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> ax = cluster_stability(result, resolution=0.8)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    resolution = _check_resolution(result, resolution)
    block = result.cluster_stability
    block = block[block["resolution"] == resolution].sort_values(
        "jaccard_mean", ascending=True, na_position="first"
    )

    if ax is None:
        height = max(2.8, 0.42 * len(block) + 2.0)
        _, ax = plt.subplots(figsize=(7.0, height), constrained_layout=True)
        ax.figure.patch.set_facecolor(SURFACE)

    # The bar is the MEAN, because that is the statistic Hennig's bands are
    # defined on. The rule alongside it is the interquartile range of the same
    # bootstrap distribution, describing spread rather than uncertainty in the
    # mean.
    #
    # The IQR is drawn at its ABSOLUTE position, q25 to q75, rather than as an
    # error bar offset from the bar tip. Matplotlib rejects a negative
    # ``xerr``, so an error bar has to clip the lower offset at zero -- which
    # collapses the whisker onto the bar tip in exactly the case worth seeing,
    # a mean lying below q25.
    #
    # Measured across four datasets and 107 clusters, the mean falls below q25
    # for 14% of them, but 11 of those 15 by less than 0.01, which no plot
    # could show. The materially visible cases are rarer -- about 2% -- and
    # they are the ones that matter: on the genotype-labelled cell lines, one
    # cluster has mean 0.880 against q25 0.993. That cluster reassembles
    # almost perfectly three quarters of the time and occasionally collapses,
    # which is a wholly different failure from being mediocre throughout, and
    # the bar alone cannot distinguish the two. Uncommon, but the most
    # informative thing on the plot when it happens.
    means = block["jaccard_mean"].to_numpy(dtype=float)
    q25 = block["jaccard_q25"].to_numpy(dtype=float)
    q75 = block["jaccard_q75"].to_numpy(dtype=float)
    sizes = block["n_cells"].to_numpy(dtype=int)
    labels = [f"cluster {c}" for c in block["cluster"]]
    positions = np.arange(len(block))

    _paint_bands(ax, horizontal=False, label=False)

    drawn = np.nan_to_num(means, nan=0.0)
    ax.barh(
        positions,
        drawn,
        height=0.66,
        color=[band_colour(v) for v in means],
        edgecolor=SURFACE,
        linewidth=1.4,
        zorder=2,
    )
    has_iqr = ~(np.isnan(q25) | np.isnan(q75))
    if has_iqr.any():
        rows = positions[has_iqr]
        lower = q25[has_iqr]
        upper = q75[has_iqr]
        ax.hlines(
            rows,
            lower,
            upper,
            color=INK_SECONDARY,
            linewidth=1.2,
            zorder=3,
        )
        # End caps, as short ticks, so a narrow IQR is still locatable.
        cap = 0.16
        ax.vlines(
            np.concatenate([lower, upper]),
            np.concatenate([rows, rows]) - cap,
            np.concatenate([rows, rows]) + cap,
            color=INK_SECONDARY,
            linewidth=1.2,
            zorder=3,
        )

    # Sizes go outside the bar end, so a short bar never clips its own label --
    # and outside the *whisker* end, not the bar end, or the label lands on top
    # of the interquartile range it is meant to sit beside.
    label_x = np.maximum(drawn, np.nan_to_num(q75, nan=0.0))
    for position, value, size, anchor in zip(
        positions, means, sizes, label_x, strict=True
    ):
        text = f"{size:,} cells" if not np.isnan(value) else f"{size:,} cells (no data)"
        ax.annotate(
            text,
            xy=(anchor, position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_MUTED,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.7, len(block) - 0.3)
    # Sorted ascending and then flipped, so the weakest cluster is at the top:
    # the ones to worry about should be the ones read first.
    ax.invert_yaxis()
    ax.set_xlabel("cluster stability (Jaccard)", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        f"Cluster stability at resolution {resolution:g}",
        color=INK,
        fontsize=12,
        loc="left",
        pad=34,
    )
    _style(ax, grid_axis="x")

    ax.legend(
        handles=[
            Line2D([], [], color=GOOD, linewidth=7, label="stable (>= 0.75)"),
            Line2D([], [], color=WARNING, linewidth=7, label="uncertain (0.60-0.75)"),
            Line2D(
                [], [], color=CRITICAL, linewidth=7, label="not trustworthy (< 0.60)"
            ),
            Line2D(
                [],
                [],
                color=INK_SECONDARY,
                linewidth=1.2,
                label="IQR across bootstraps",
            ),
        ],
        # Above the frame, for the same reason as the curve plot: there is
        # nowhere safe inside. Bars all start at the left and grow rightward,
        # and the y-axis is inverted so the strongest cluster is the bottom
        # row -- which puts the longest bar exactly where a "lower right"
        # legend would sit, every time the best cluster is a stable one.
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncols=2,
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
        handletextpad=0.6,
        columnspacing=1.6,
    )
    return ax


def stability_umap(
    adata,
    result: StabilityResult,
    resolution: float,
    ax: Axes | None = None,
) -> Axes:
    """Plot the UMAP coloured by per-cell stability.

    Cells with no evidence -- never drawn into any bootstrap -- are grey and
    drawn underneath, so absence never reads as instability.

    Parameters
    ----------
    adata
        The object the sweep was run on. Must have ``obsm["X_umap"]``.
    result
        A completed sweep.
    resolution
        Which resolution to colour by.
    ax
        Draw into an existing axes.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If ``adata.obsm["X_umap"]`` is missing, or the cells do not match.

    Examples
    --------
    >>> ax = stability_umap(adata, result, resolution=0.8)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    if "X_umap" not in adata.obsm:
        available = list(adata.obsm.keys())
        raise ValueError(
            f"adata.obsm has no 'X_umap', so there is no embedding to plot on. "
            f"Available keys: {available if available else '(none)'}. Compute one "
            f"first, for example with `sc.tl.umap(adata)` after `sc.pp.neighbors`."
        )
    if not adata.obs_names.equals(result.per_cell.index):
        raise ValueError(
            "adata.obs_names do not match the cells this result was computed on. "
            "Pass the same AnnData that stability_sweep was called with."
        )

    resolution = _check_resolution(result, resolution)
    coords = np.asarray(adata.obsm["X_umap"])
    values = result.per_cell[resolution].to_numpy(dtype=float)
    missing = np.isnan(values)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
        ax.figure.patch.set_facecolor(SURFACE)

    if missing.any():
        ax.scatter(
            coords[missing, 0],
            coords[missing, 1],
            s=10,
            color=NO_EVIDENCE_COLOUR,
            linewidth=0,
            zorder=1,
            label=f"no evidence ({int(missing.sum())} cells)",
        )
    points = ax.scatter(
        coords[~missing, 0],
        coords[~missing, 1],
        c=values[~missing],
        cmap=STABILITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        s=10,
        linewidth=0,
        zorder=2,
    )

    bar = ax.figure.colorbar(points, ax=ax, fraction=0.045, pad=0.02)
    bar.set_label("per-cell stability", color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)
    bar.outline.set_visible(False)

    if missing.any():
        ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("UMAP 1", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("UMAP 2", color=INK_MUTED, fontsize=9)
    ax.set_title(
        f"Per-cell stability at resolution {resolution:g}",
        color=INK,
        fontsize=12,
        loc="left",
        pad=10,
    )
    return ax

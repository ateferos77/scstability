"""Render the README's opening figure from genotype-labelled ground truth.

Uses ``sc_10x_5cl`` (Tian et al. 2019): five human cell lines mixed and
sequenced together, each cell assigned to its line by SNP genotype. The true
number of groups is 5 and the labels never touch expression, so the figure
shows the measure being right about something it could not have inferred.

Fetch the data first::

    python benchmarks/fetch_ground_truth.py 5cl --out data/
    python docs/make_validation_figure.py data/sc_10x_5cl_ground_truth.h5ad
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

import scstability as scs

RESOLUTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
OUT = pathlib.Path(__file__).parent / "images"
DPI = 200


def main() -> None:
    """Render the validation figure for the given ground-truth dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()

    full = ad.read_h5ad(args.path)
    truth_k = full.obs["ground_truth"].nunique()

    adata = ad.AnnData(np.zeros((full.n_obs, 1), dtype=np.float32))
    adata.obsm["X_pca"] = np.ascontiguousarray(full.obsm["X_pca"], dtype=np.float32)

    result = scs.stability_sweep(
        adata, RESOLUTIONS, n_boot=20, random_state=0, progress=False
    )
    summary = result.summary()
    print(summary.to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)

    figure = scs.pl.stability_curve(result)
    top = figure.axes[0]
    # Mark where the answer is actually known, so the reader can check the
    # claim rather than take it on trust.
    at_truth = summary[summary["n_clusters"] == truth_k]
    if len(at_truth):
        resolution = float(at_truth["resolution"].iloc[0])
        # Ink, not the status green: the plotting module already spends green
        # on "meets the threshold", and a second green line would read as a
        # second verdict rather than as external evidence.
        top.axvline(resolution, color="#0b0b0b", lw=1.1, ls=(0, (5, 3)), zorder=1)
        top.annotate(
            f"true K = {truth_k}\nby SNP genotype",
            xy=(resolution, 0.30),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color="#0b0b0b",
            va="center",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#fcfcfb",
                "edgecolor": "#c3c2b7",
                "linewidth": 0.6,
            },
        )
    figure.savefig(OUT / "validation_5cl_curve.png", dpi=DPI, bbox_inches="tight")
    plt.close(figure)

    print("wrote docs/images/validation_5cl_curve.png")


if __name__ == "__main__":
    main()

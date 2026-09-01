"""Download a benchmark dataset whose labels are genuine ground truth.

Most scRNA-seq "ground truth" is not ground truth. A cell-type column in a
published object was almost always produced by clustering that same expression
matrix and then naming the clusters, so validating a clustering method against
it is partly circular.

The datasets fetched here are different: the labels come from **outside the
expression matrix**.

``5cl``
    ``sc_10x_5cl`` from sc_mixology (Tian et al. 2019, Nature Methods). Five
    human cell lines mixed and sequenced together on 10x; every cell is
    assigned to its line by SNP genotype via demuxlet. Genetics, not
    expression. Doublets and ambiguous calls are dropped, so the retained
    labels are as close to certain as single-cell data gets. Known K = 5.

``3cl``
    ``sc_10x``, the same design with three lines. Smaller and easier.

``rnamix``
    ``RNAmix_celseq2``. Not cells but defined mixtures of RNA from three cell
    lines at known proportions, arranged in a simplex. This is a ground-truth
    *continuum*: the correct answer is that there are no discrete clusters,
    only known positions along a gradient. Useful precisely because it is the
    case where a stability measure should decline to endorse a partition.

Usage
-----
::

    python benchmarks/fetch_ground_truth.py 5cl --out data/
    python benchmarks/fetch_ground_truth.py rnamix --out data/

Writes an ``.h5ad`` with the truth in ``adata.obs["ground_truth"]``, ready for
``validate_real_data.py --truth ground_truth``.

Reference
---------
Tian, L. et al. (2019). Benchmarking single cell RNA-sequencing analysis
pipelines using mixture control experiments. *Nature Methods*, 16, 479-487.
"""

from __future__ import annotations

import argparse
import gzip
import io
import pathlib
import urllib.request

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

BASE = "https://raw.githubusercontent.com/LuyiTian/sc_mixology/master/data/csv"

DATASETS = {
    "5cl": ("sc_10x_5cl", "cell_line_demuxlet", 5),
    "3cl": ("sc_10x", "cell_line_demuxlet", 3),
    "rnamix": ("RNAmix_celseq2", "mix", 8),
}


def _download(name: str) -> bytes:
    """Fetch one gzipped CSV from the sc_mixology repository."""
    url = f"{BASE}/{name}.csv.gz"
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=300) as response:
        return gzip.decompress(response.read())


def build(key: str, out_dir: pathlib.Path) -> pathlib.Path:
    """Download one dataset and write it as an AnnData with ground-truth labels."""
    stem, truth_column, expected_k = DATASETS[key]

    counts = pd.read_csv(io.BytesIO(_download(f"{stem}.count")), index_col=0)
    meta = pd.read_csv(io.BytesIO(_download(f"{stem}.metadata")), index_col=0)

    # counts arrive genes x cells; AnnData wants cells x genes
    counts = counts.T
    meta = meta.loc[counts.index]

    adata = ad.AnnData(
        X=np.ascontiguousarray(counts.to_numpy(), dtype=np.float32),
        obs=meta.copy(),
        var=pd.DataFrame(index=counts.columns),
    )

    # Keep only cells demuxlet called as unambiguous singlets. A doublet has no
    # single true label, so scoring against one would be scoring against noise.
    if "demuxlet_cls" in adata.obs:
        singlet = adata.obs["demuxlet_cls"].astype(str) == "SNG"
        print(
            f"  singlets: {int(singlet.sum())} of {adata.n_obs} "
            f"({adata.n_obs - int(singlet.sum())} doublet/ambiguous dropped)"
        )
        adata = adata[singlet.to_numpy()].copy()

    adata.obs["ground_truth"] = adata.obs[truth_column].astype(str).astype("category")
    print(
        f"  ground truth ({truth_column}): "
        f"{adata.obs['ground_truth'].value_counts().to_dict()}"
    )
    print(f"  expected number of true groups: {expected_k}")

    # Standard preprocessing, so the representation is the one a user would have.
    sc.pp.filter_genes(adata, min_cells=3)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, svd_solver="arpack", random_state=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_ground_truth.h5ad"
    adata.write_h5ad(path)
    print(f"  wrote {path}  ({adata.n_obs} cells x {adata.n_vars} HVGs)")
    return path


def main() -> None:
    """Download the dataset named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--out", default="data", type=pathlib.Path)
    args = parser.parse_args()

    path = build(args.dataset, args.out)
    print("\nnow run:")
    print(f"  python benchmarks/validate_real_data.py {path} --truth ground_truth")


if __name__ == "__main__":
    main()

"""Download and preprocess a real 10x dataset at realistic scale.

Everything else in ``benchmarks/`` runs on a few thousand cells. That is enough
to test correctness and not enough to tell anyone whether the package is usable
on the data they actually have, which for a modern 10x experiment is tens of
thousands of cells.

``68k``
    ``fresh_68k_pbmc_donor_a`` (Zheng et al. 2017, *Nature Communications*) --
    68,579 PBMCs, the dataset that defined the scale of droplet scRNA-seq
    benchmarking. Downloaded as the filtered 10x matrix (~118 MB) and
    preprocessed with the standard scanpy pipeline.

``20k``
    ``20k_PBMC_3p_HT_nextgem_Chromium_X`` -- modern v3 chemistry, for a check
    that nothing depends on the older chemistry's characteristics.

``10k``
    ``pbmc_10k_v3`` -- a smaller, quicker option.

Usage
-----
::

    python benchmarks/fetch_large_pbmc.py 68k --out data/
"""

from __future__ import annotations

import argparse
import pathlib
import tarfile
import tempfile
import urllib.request

import numpy as np
import scanpy as sc

BASE = "https://cf.10xgenomics.com/samples/cell-exp"

DATASETS = {
    "68k": (
        f"{BASE}/1.1.0/fresh_68k_pbmc_donor_a/"
        "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz",
        "tar",
    ),
    "20k": (
        f"{BASE}/6.1.0/20k_PBMC_3p_HT_nextgem_Chromium_X/"
        "20k_PBMC_3p_HT_nextgem_Chromium_X_filtered_feature_bc_matrix.h5",
        "h5",
    ),
    "10k": (
        f"{BASE}/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5",
        "h5",
    ),
}


def _download(url: str, target: pathlib.Path) -> pathlib.Path:
    """Fetch a URL to disk, skipping if it is already there."""
    if target.exists():
        print(f"  reusing {target}")
        return target
    print(f"  downloading {url}")
    # 10x's CDN returns 403 to urllib's default User-Agent, so send a real one.
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with (
        urllib.request.urlopen(request, timeout=1800) as response,
        target.open("wb") as fh,
    ):
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    print(f"  wrote {target} ({target.stat().st_size / 1e6:.0f} MB)")
    return target


def load(key: str, cache: pathlib.Path):
    """Download and read one dataset into an AnnData."""
    url, kind = DATASETS[key]
    cache.mkdir(parents=True, exist_ok=True)

    if kind == "h5":
        path = _download(url, cache / f"pbmc_{key}.h5")
        adata = sc.read_10x_h5(path)
    else:
        path = _download(url, cache / f"pbmc_{key}.tar.gz")
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(path) as archive:
                archive.extractall(tmp, filter="data")
            matrix = next(pathlib.Path(tmp).rglob("matrix.mtx*")).parent
            adata = sc.read_10x_mtx(matrix)

    adata.var_names_make_unique()
    return adata


def preprocess(adata):
    """Run the standard scanpy pipeline, for a realistic representation."""
    print(f"  raw: {adata.n_obs} cells x {adata.n_vars} genes")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, svd_solver="arpack", random_state=0)
    print(f"  after QC and HVG selection: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def main() -> None:
    """Download, preprocess and write the dataset named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--out", default="data", type=pathlib.Path)
    args = parser.parse_args()

    adata = preprocess(load(args.dataset, args.out))

    # Only the representation is needed downstream. AnnData will not let X
    # change shape in place, so build a fresh object around the PCA rather
    # than carrying a 68k x 2000 expression matrix into the cache file.
    import anndata as ad

    slim = ad.AnnData(
        np.zeros((adata.n_obs, 1), dtype=np.float32),
        obs=adata.obs[["n_genes"]].copy() if "n_genes" in adata.obs else None,
    )
    slim.obs_names = adata.obs_names
    slim.obsm["X_pca"] = np.ascontiguousarray(adata.obsm["X_pca"], dtype=np.float32)

    path = args.out / f"pbmc_{args.dataset}_pca.h5ad"
    slim.write_h5ad(path)
    print(f"  wrote {path}  ({slim.n_obs} cells, {slim.obsm['X_pca'].shape[1]} PCs)")


if __name__ == "__main__":
    main()

"""Measure sweep time and memory at realistic scale.

How long does a sweep take, and how much memory, on the data people have?

The correctness work in this repository runs on a few thousand cells. That is
enough to test the arithmetic and not enough to tell anyone whether the package
is usable on the data they have, which for a modern 10x experiment is tens of
thousands of cells.

This measures wall-clock and peak resident memory on subsamples of a real 68k
PBMC dataset, so the README can quote numbers rather than reassurances. Each
size runs in a **separate process**, so one measurement cannot inherit another's
allocator state or warmed caches.

Usage
-----
::

    python benchmarks/fetch_large_pbmc.py 68k --out data/
    python benchmarks/scaling.py data/pbmc_68k_pca.h5ad
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import textwrap

SIZES = [2000, 5000, 10000, 20000, 40000, 68000]

WORKER = """
import json, resource, sys, time, warnings
import anndata as ad, numpy as np
import scstability as scs

path, n_cells, n_boot, n_res = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
full = ad.read_h5ad(path)
rng = np.random.default_rng(0)
n = min(n_cells, full.n_obs)
idx = np.sort(rng.choice(full.n_obs, size=n, replace=False))

adata = ad.AnnData(np.zeros((n, 1), dtype=np.float32))
adata.obsm["X_pca"] = np.ascontiguousarray(
    np.asarray(full.obsm["X_pca"])[idx], dtype=np.float32
)
del full

resolutions = [round(0.2 * (i + 1), 2) for i in range(n_res)]
start = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = scs.stability_sweep(
        adata, resolutions, n_boot=n_boot, random_state=0, progress=False
    )
elapsed = time.perf_counter() - start

peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
summary = result.summary()
print("@@" + json.dumps({
    "n_cells": n,
    "seconds": round(elapsed, 1),
    "peak_mb": round(peak_kb / 1024, 1),
    "n_clusters": [int(v) for v in summary["n_clusters"]],
    "min_stability": round(float(summary["min_cluster_stability"].max()), 3),
}))
"""


def main() -> None:
    """Run one sweep per size, each in its own process, and tabulate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--n-boot", type=int, default=20)
    parser.add_argument("--n-resolutions", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    args = parser.parse_args()

    worker = pathlib.Path(__file__).parent / "_scaling_worker.py"
    worker.write_text(textwrap.dedent(WORKER))

    print(
        f"n_boot={args.n_boot}, resolutions={args.n_resolutions}, "
        f"total reclusterings per size = {args.n_boot * args.n_resolutions}\n"
    )
    header = f"{'cells':>8}  {'seconds':>9}  {'peak RAM':>9}  {'us/cell':>8}  {'clusters':>18}"
    print(header)
    print("-" * len(header))

    rows = []
    for size in args.sizes:
        proc = subprocess.run(
            [
                sys.executable,
                str(worker),
                args.path,
                str(size),
                str(args.n_boot),
                str(args.n_resolutions),
            ],
            capture_output=True,
            text=True,
        )
        line = next(
            (ln for ln in proc.stdout.splitlines() if ln.startswith("@@")), None
        )
        if line is None:
            print(f"{size:>8}  FAILED: {proc.stderr.strip().splitlines()[-1:]}")
            continue
        row = json.loads(line[2:])
        rows.append(row)
        per_cell = row["seconds"] / row["n_cells"] * 1e6
        span = f"{min(row['n_clusters'])}-{max(row['n_clusters'])}"
        print(
            f"{row['n_cells']:>8,}  {row['seconds']:>8.1f}s  "
            f"{row['peak_mb']:>7.0f}MB  {per_cell:>7.0f}  {span:>18}"
        )

    worker.unlink(missing_ok=True)

    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        growth = (last["seconds"] / first["seconds"]) / (
            last["n_cells"] / first["n_cells"]
        )
        print(
            f"\n  {first['n_cells']:,} -> {last['n_cells']:,} cells is "
            f"{last['n_cells'] / first['n_cells']:.0f}x the data and "
            f"{last['seconds'] / first['seconds']:.0f}x the time"
        )
        print(
            f"  -> scaling factor {growth:.2f}x relative to linear "
            f"({'super-linear' if growth > 1.3 else 'roughly linear'})"
        )


if __name__ == "__main__":
    main()

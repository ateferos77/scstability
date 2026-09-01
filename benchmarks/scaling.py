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

Runs on Linux, macOS and Windows. Peak memory comes from ``getrusage`` on the
Unix platforms and from ``GetProcessMemoryInfo`` on Windows; the numbers quoted
in the README were measured on Linux, and memory readings are not exactly
comparable across operating systems.

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
import json, sys, time, warnings
import anndata as ad, numpy as np
import scstability as scs


def peak_memory_mib():
    "Peak resident memory of this process, in MiB, on Linux, macOS or Windows."
    if sys.platform == "win32":
        # Windows has no `resource` module at all. PeakWorkingSetSize is the
        # direct equivalent of ru_maxrss: a high-water mark, in bytes.
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        # The signatures must be declared. Left to ctypes' defaults the process
        # handle is truncated to a C int and the call fails with ERROR_SUCCESS,
        # which looks like a bug in the benchmark rather than in the binding.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize / 1024**2

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KILOBYTES on Linux and BYTES on macOS. Dividing by 1024
    # everywhere, as this script used to, understates macOS by a factor of
    # 1024 -- silently, since the number still looks like a plausible reading.
    return peak / 1024 if sys.platform.startswith("linux") else peak / 1024**2


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

summary = result.summary()
print("@@" + json.dumps({
    "n_cells": n,
    "seconds": round(elapsed, 1),
    "peak_mb": round(peak_memory_mib(), 1),
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

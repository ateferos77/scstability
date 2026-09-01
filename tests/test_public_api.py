"""The public API, exercised the way a user meets it.

Every other test file imports pieces of the package directly -- ``from
scstability._metrics import ...``, ``from scstability.pl._plots import ...``.
That is convenient and it is also a blind spot: those imports pull submodules
into ``sys.modules`` as a side effect, so an entry point that is *only*
reachable because some other import happened to load it still passes.

That is not hypothetical. ``scs.pl`` was unreachable from a bare ``import
scstability`` for the whole of Step 5; the plot tests passed anyway, because
their own ``from scstability.pl._plots import ...`` line had already bound the
submodule. A user following the README would have hit ``AttributeError``.

So the tests here run in a **subprocess** with a clean interpreter. Nothing this
test file does can mask a missing import, because nothing is shared.
"""

import subprocess
import sys
import textwrap


def run_in_clean_interpreter(source: str) -> subprocess.CompletedProcess:
    """Execute ``source`` in a fresh Python, with no test imports in scope."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
    )


def test_every_documented_name_is_reachable_from_a_bare_import():
    """`import scstability as scs` must expose the whole documented surface.

    Cheap: imports only, no clustering. Runs first so that a packaging mistake
    fails fast rather than after a sweep.
    """
    proc = run_in_clean_interpreter(
        """
        import scstability as scs

        assert callable(scs.stability_sweep)
        assert isinstance(scs.StabilityResult, type)
        assert scs.HENNIG_BANDS
        assert scs.__version__

        # Submodules are not attributes until something imports them.
        assert callable(scs.pl.stability_curve)
        assert callable(scs.pl.cluster_stability)
        assert callable(scs.pl.stability_umap)

        assert sorted(scs.__all__) == [
            "HENNIG_BANDS", "StabilityResult", "pl", "stability_sweep"
        ]
        print("ok")
        """
    )

    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_the_quickstart_runs_end_to_end():
    """The README's quick start, executed verbatim in a clean interpreter.

    This is the only test that walks the whole documented path in one go --
    sweep, summary, recommend, to_adata, all three plots -- with exactly the
    imports a user writes. If the package works here, it works for a reader who
    copies the README and nothing else.
    """
    proc = run_in_clean_interpreter(
        """
        import matplotlib
        matplotlib.use("Agg")

        import numpy as np
        from anndata import AnnData
        from sklearn.datasets import make_blobs

        import scstability as scs

        X, _ = make_blobs(
            n_samples=200, centers=3, n_features=8, cluster_std=0.6, random_state=0
        )
        adata = AnnData(np.ascontiguousarray(X, dtype=np.float32))
        adata.obsm["X_pca"] = adata.X
        adata.obsm["X_umap"] = adata.X[:, :2]

        result = scs.stability_sweep(
            adata,
            resolutions=[0.2, 0.8],
            n_boot=4,
            frac=0.8,
            use_rep="X_pca",
            n_neighbors=15,
            random_state=0,
            progress=False,
        )

        summary = result.summary()
        assert len(summary) == 2
        assert result.recommend() in (0.2, 0.8)

        result.to_adata(adata)
        assert "stability_res0.8" in adata.obs
        assert "stability" in adata.uns

        scs.pl.stability_curve(result)
        scs.pl.cluster_stability(result, resolution=0.8)
        scs.pl.stability_umap(adata, result, resolution=0.8)

        print("ok")
        """
    )

    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_importing_the_package_emits_no_warnings():
    """A library that warns on import trains its users to ignore warnings."""
    proc = run_in_clean_interpreter(
        """
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import scstability

        # Match the package's own directory, not the substring "scstability".
        # A virtualenv is very often named after the package it holds, and a
        # substring test then attributes every dependency's import warning to
        # us -- matplotlib's pyparsing deprecations, for instance.
        import os
        package_dir = os.path.dirname(os.path.abspath(scstability.__file__))
        ours = [
            w for w in caught
            if os.path.abspath(str(getattr(w, "filename", ""))).startswith(package_dir)
        ]
        assert not ours, [str(w.message) for w in ours]
        print("ok")
        """
    )

    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_the_readme_workflow_runs_as_written():
    """Every construct the README documents, executed in a clean interpreter.

    A README whose examples do not run is worse than no README: it costs the
    reader their trust as well as their time. This covers the claims that are
    easy to break silently -- the exact ``obs`` column names ``to_adata``
    writes, the columns of ``summary()`` and ``cluster_stability``, the
    ``.query()`` line in the worked workflow, and composing two plots into one
    figure with ``ax=``.
    """
    proc = run_in_clean_interpreter(
        """
        import matplotlib
        matplotlib.use("Agg")

        import matplotlib.pyplot as plt
        import numpy as np
        import scanpy as sc
        from anndata import AnnData
        from sklearn.datasets import make_blobs

        import scstability as scs

        X, _ = make_blobs(
            n_samples=300, centers=4, n_features=10, cluster_std=0.7, random_state=0
        )
        adata = AnnData(np.ascontiguousarray(X, dtype=np.float32))
        adata.obsm["X_pca"] = adata.X.copy()
        adata.obsm["X_umap"] = adata.X[:, :2].copy()

        result = scs.stability_sweep(
            adata, resolutions=[0.2, 0.8], n_boot=4, progress=False
        )

        # the columns the README's tables name
        assert list(result.summary().columns) == [
            "resolution", "n_clusters", "min_cluster_stability",
            "median_cluster_stability", "frac_cells_stable",
        ]
        assert {
            "resolution", "cluster", "n_cells",
            "jaccard_mean", "jaccard_median", "jaccard_q25", "jaccard_q75",
        } <= set(result.cluster_stability.columns)

        # to_adata writes exactly the keys the README shows
        result.to_adata(adata)
        assert "stability_res0.8" in adata.obs
        assert "stability" in adata.uns
        result.to_adata(adata, key_added="stab")
        assert "stab_res0.8" in adata.obs

        # the obs column is usable like any other
        subset = adata[adata.obs["stability_res0.8"] > 0.75]
        assert subset.n_obs >= 0

        # the .query() line from the worked workflow
        chosen = result.recommend()
        weak = result.cluster_stability.query(
            "resolution == @chosen and jaccard_mean < 0.75"
        )
        assert weak is not None

        # composing two plots into one figure
        fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
        scs.pl.cluster_stability(result, resolution=0.8, ax=left)
        scs.pl.stability_umap(adata, result, resolution=0.8, ax=right)

        # passing ax to the curve draws one panel, not two
        fig2, ax2 = plt.subplots()
        assert scs.pl.stability_curve(result, ax=ax2) is ax2
        assert len(fig2.axes) == 1

        assert len(scs.HENNIG_BANDS) == 4
        print("ok")
        """
    )

    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout

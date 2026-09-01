"""Shared fixtures.

Three synthetic datasets with known answers, deliberately small so the default
suite stays under the 60-second budget:

``blobs_adata``
    Three well-separated groups. The answer is unambiguous, so anything that
    fails here is broken rather than borderline.
``noise_adata``
    One isotropic Gaussian. There is *no* cluster structure at all, so any
    clusters found are artifacts and must be reported as unstable.
``gradient_adata``
    A one-dimensional continuum. Structure exists but is not discrete -- the
    case where Leiden invents boundaries that are pure noise.

Plus ``pbmc_adata``, used only by the slow real-data test.

All dataset fixtures are **session-scoped** and deterministic, so building
them costs nothing after the first use. Treat them as read-only: any test
that writes into one (``to_adata``, for instance) must work on ``.copy()``.
"""

import warnings
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from sklearn.datasets import make_blobs

N_CELLS = 300
N_DIMS = 10


def _as_adata(X: np.ndarray) -> AnnData:
    """Wrap coordinates as an AnnData with the representation in ``obsm``.

    Mirrors what a user arrives with: QC, normalisation and PCA already done,
    the embedding sitting in ``obsm["X_pca"]``. The package does no
    preprocessing of its own.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    adata = AnnData(X.copy())
    adata.obsm["X_pca"] = X
    adata.obs_names = [f"cell_{i}" for i in range(X.shape[0])]
    return adata


@pytest.fixture
def strict_warnings() -> Callable[[], AbstractContextManager[None]]:
    """Turn ``RuntimeWarning`` into an error inside the ``with`` block.

    Use this instead of a global ``filterwarnings`` entry in ``pyproject.toml``.
    A global filter cannot distinguish our warnings from scanpy's, and pytest's
    module scoping matches the *filename stem* rather than the dotted import
    path -- which both collides with scanpy's own ``_metrics.py`` and fails
    silently when it does not match. Wrapping the call site is unambiguous:
    the strictness applies to exactly the code under test and nothing else.

    Examples
    --------
    >>> def test_never_sampled_cell_is_nan(strict_warnings):  # doctest: +SKIP
    ...     with strict_warnings():
    ...         result = per_cell_stability(ref, boots)
    ...     assert np.isnan(result[never_sampled])
    """

    @contextmanager
    def _ctx() -> Iterator[None]:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            yield

    return _ctx


@pytest.fixture(scope="session")
def blobs_adata() -> AnnData:
    """Three well-separated Gaussian blobs: data with an unambiguous answer."""
    X, _ = make_blobs(
        n_samples=N_CELLS,
        centers=3,
        n_features=N_DIMS,
        cluster_std=0.5,
        random_state=0,
    )
    return _as_adata(X)


@pytest.fixture(scope="session")
def noise_adata() -> AnnData:
    """A single isotropic Gaussian: no cluster structure exists at all.

    Leiden will still return clusters when asked. Every one of them is an
    artifact of where the algorithm happened to cut, and the package's central
    claim is that it says so.
    """
    rng = np.random.default_rng(0)
    return _as_adata(rng.normal(size=(N_CELLS, N_DIMS)))


@pytest.fixture(scope="session")
def gradient_adata() -> AnnData:
    """Cells along a 1-D continuum: structure that is real but not discrete."""
    rng = np.random.default_rng(1)
    t = np.linspace(0.0, 1.0, N_CELLS)
    direction = rng.normal(size=N_DIMS)
    direction /= np.linalg.norm(direction)
    X = 6.0 * t[:, None] * direction[None, :] + rng.normal(
        scale=0.35, size=(N_CELLS, N_DIMS)
    )
    return _as_adata(X)


@pytest.fixture(scope="session")
def pbmc_adata() -> AnnData:
    """Real PBMC data, downloaded on first use. Slow test only."""
    return sc.datasets.pbmc3k_processed()

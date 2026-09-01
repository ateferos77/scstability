"""Bootstrap cluster stability for single-cell RNA-seq.

Which of your Leiden clusters survive resampling of the cells, and which
dissolve? Implements the cluster-wise Jaccard stability measure of Hennig
(2007) over subsampled reclusterings, AnnData-first and scanpy-compatible.

References
----------
Hennig, C. (2007). Cluster-wise assessment of cluster stability.
*Computational Statistics & Data Analysis*, 52(1), 258-271.
"""

__version__ = "0.1.0"

# _metrics and _cluster stay internal by design: they are the tested core, not
# part of the user-facing surface. `pl` is imported eagerly so that
# `import scstability as scs` makes `scs.pl.stability_curve(...)` work, as the
# documented API promises -- a submodule is not an attribute until imported.
from . import pl
from ._core import HENNIG_BANDS, StabilityResult, stability_sweep

__all__ = ["HENNIG_BANDS", "StabilityResult", "pl", "stability_sweep"]

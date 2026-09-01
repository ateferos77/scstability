"""Plotting functions.

All plots accept an optional ``ax``, return the ``Axes`` (or ``Figure`` for
multi-panel output), and never call ``plt.show()``.
"""

from ._plots import cluster_stability, stability_curve, stability_umap

__all__ = ["cluster_stability", "stability_curve", "stability_umap"]

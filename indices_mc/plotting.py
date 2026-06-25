"""Matplotlib rendering of the probability cone and convergence diagnostic."""

from __future__ import annotations

import numpy as np

from .analysis import ConvergenceCurve, PercentileBands


def plot_cone(bands: PercentileBands, spot: float, ax=None, target: float | None = None):
    """Plot the percentile bands as a shaded probability cone."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.5))

    years = bands.years

    # Shaded bands: P5-P95 (light) and P10-P90 (darker), median line on top.
    if 5 in bands.percentiles and 95 in bands.percentiles:
        ax.fill_between(years, bands.band(5), bands.band(95),
                        alpha=0.20, color="C0", label="P5–P95")
    if 10 in bands.percentiles and 90 in bands.percentiles:
        ax.fill_between(years, bands.band(10), bands.band(90),
                        alpha=0.35, color="C0", label="P10–P90")
    if 50 in bands.percentiles:
        ax.plot(years, bands.band(50), color="C0", lw=2, label="Median (P50)")

    ax.axhline(spot, color="0.4", ls=":", lw=1, label=f"Start ({spot:,.0f})")
    if target is not None:
        ax.axhline(target, color="C3", ls="--", lw=1.2, label=f"Target ({target:,.0f})")

    ax.set_xlabel("Years")
    ax.set_ylabel("Index level")
    ax.set_title("Probability cone — percentile bands by year")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_convergence(curve: ConvergenceCurve, ax=None):
    """Plot the probability estimate stabilizing as iteration count rises."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(curve.counts, curve.estimates, color="C2", lw=1.5)
    ax.axhline(curve.final, color="0.4", ls="--", lw=1,
               label=f"Final estimate = {curve.final:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Iterations (log scale)")
    ax.set_ylabel("Estimated P(target reached)")
    ax.set_title("Convergence diagnostic")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax

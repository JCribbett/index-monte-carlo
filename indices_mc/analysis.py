"""Analysis of a simulated path ensemble.

Produces the three outputs described in readme.txt ("Simulation and outputs"):
percentile bands (the probability cone), the probability of attaining a target,
and a convergence diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import SimulationResult

DEFAULT_PERCENTILES = (5, 10, 50, 90, 95)


@dataclass(frozen=True)
class PercentileBands:
    """Percentile bands of index value sampled once per year (the cone)."""

    years: np.ndarray  # shape (n_years + 1,)
    percentiles: tuple[int, ...]  # e.g. (5, 10, 50, 90, 95)
    values: np.ndarray  # shape (len(percentiles), n_years + 1)

    def band(self, pct: int) -> np.ndarray:
        return self.values[self.percentiles.index(pct)]


def percentile_bands(
    result: SimulationResult,
    percentiles: tuple[int, ...] = DEFAULT_PERCENTILES,
) -> PercentileBands:
    """Compute percentile bands of index value at each year boundary."""
    spy = result.params.steps_per_year
    n_years = int(round(result.params.horizon_years))
    year_cols = [min(y * spy, result.paths.shape[1] - 1) for y in range(n_years + 1)]

    sampled = result.paths[:, year_cols]  # (n_iter, n_years + 1)
    values = np.percentile(sampled, percentiles, axis=0)  # (n_pct, n_years + 1)

    return PercentileBands(
        years=np.arange(n_years + 1, dtype=float),
        percentiles=tuple(percentiles),
        values=values,
    )


def target_probability(
    result: SimulationResult,
    target: float,
    horizon_years: float | None = None,
) -> float:
    """Fraction of paths reaching or exceeding ``target`` by ``horizon_years``.

    If ``horizon_years`` is None, the full simulated horizon is used. The test
    is on the index level at the horizon (not a touch-anytime barrier).
    """
    if horizon_years is None:
        levels = result.terminal
    else:
        levels = result.levels_at(horizon_years)
    return float(np.mean(levels >= target))


@dataclass(frozen=True)
class ConvergenceCurve:
    """Probability estimate as a function of iteration count."""

    counts: np.ndarray
    estimates: np.ndarray
    final: float


def convergence_curve(
    result: SimulationResult,
    target: float,
    horizon_years: float | None = None,
    n_points: int = 40,
) -> ConvergenceCurve:
    """Show the target-probability estimate stabilizing as iterations rise.

    Uses the running mean over the already-simulated paths, so it reflects the
    exact ensemble that produced the headline number (readme.txt: "justifies the
    chosen number of iterations rather than assuming it is sufficient").
    """
    if horizon_years is None:
        levels = result.terminal
    else:
        levels = result.levels_at(horizon_years)

    hits = (levels >= target).astype(float)
    running = np.cumsum(hits) / np.arange(1, hits.size + 1)

    # Sample the running estimate at log-spaced iteration counts for plotting.
    counts = np.unique(
        np.geomspace(1, hits.size, num=n_points).astype(int)
    )
    estimates = running[counts - 1]

    return ConvergenceCurve(
        counts=counts,
        estimates=estimates,
        final=float(running[-1]),
    )

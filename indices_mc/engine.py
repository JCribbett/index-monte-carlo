"""Vectorized geometric Brownian motion simulation engine.

Each path evolves under GBM (readme.txt, "The stochastic process"):

    S(t+dt) = S(t) * exp( (mu - 0.5*sigma**2)*dt + sigma*sqrt(dt)*Z ),  Z ~ N(0,1)

The whole ensemble is generated in a handful of NumPy operations, so runs of
>=10,000 paths complete in well under a second. A fixed seed yields identical
results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import Parameters


@dataclass(frozen=True)
class SimulationResult:
    """Output of a simulation run.

    Attributes
    ----------
    paths : ndarray, shape (n_iterations, n_steps + 1)
        Simulated index levels, including S(0) at column 0.
    time_grid : ndarray, shape (n_steps + 1,)
        Time in years for each column.
    params : Parameters
        The configuration that produced this result.
    """

    paths: np.ndarray
    time_grid: np.ndarray
    params: Parameters

    @property
    def terminal(self) -> np.ndarray:
        """Index level at the simulation horizon, one value per path."""
        return self.paths[:, -1]

    def levels_at(self, years: float) -> np.ndarray:
        """Index levels at the grid point nearest ``years``, one per path."""
        idx = int(np.clip(round(years * self.params.steps_per_year), 0, self.paths.shape[1] - 1))
        return self.paths[:, idx]


def simulate(params: Parameters) -> SimulationResult:
    """Run a vectorized GBM Monte Carlo simulation.

    Drift is sampled once per path (allowing Triangular parameter uncertainty),
    then held constant along that path's timeline.
    """
    n_iter = int(params.n_iterations)
    n_steps = params.n_steps
    dt = params.dt
    sigma = float(params.sigma)

    rng = np.random.default_rng(params.seed)

    # Per-path drift (constant array for a point estimate, Triangular draws otherwise).
    mu = params.drift.sample(n_iter, rng).reshape(n_iter, 1)

    # Standard normal shocks: one per path per step.
    z = rng.standard_normal(size=(n_iter, n_steps))

    # Per-step log increments, then cumulative sum along time.
    drift_term = (mu - 0.5 * sigma**2) * dt
    diffusion_term = sigma * np.sqrt(dt) * z
    log_increments = drift_term + diffusion_term  # broadcasts mu across steps

    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_iter, n_steps + 1), dtype=float)
    paths[:, 0] = params.spot
    paths[:, 1:] = params.spot * np.exp(log_paths)

    time_grid = np.arange(n_steps + 1, dtype=float) * dt

    return SimulationResult(paths=paths, time_grid=time_grid, params=params)

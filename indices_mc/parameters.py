"""Simulation parameters, with optional propagation of parameter uncertainty.

Point estimates understate uncertainty. This module lets the drift input be
treated as a *distribution* (a Triangular low/most-likely/high range) so that
parameter uncertainty is propagated into the output, not just the path-level
randomness (see readme.txt, "Parameter calibration").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DriftSpec:
    """Specification of the annual drift mu.

    If ``low`` and ``high`` are both None, the drift is a fixed point estimate
    (``most_likely``). Otherwise mu is drawn per-path from a Triangular
    distribution, propagating parameter uncertainty into the output.
    """

    most_likely: float
    low: float | None = None
    high: float | None = None

    @property
    def is_stochastic(self) -> bool:
        return self.low is not None and self.high is not None

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``size`` drift values (constant array if a point estimate)."""
        if not self.is_stochastic:
            return np.full(size, self.most_likely, dtype=float)
        lo, hi = float(self.low), float(self.high)
        if not (lo <= self.most_likely <= hi):
            raise ValueError(
                f"Triangular drift requires low <= most_likely <= high; "
                f"got low={lo}, most_likely={self.most_likely}, high={hi}."
            )
        return rng.triangular(lo, self.most_likely, hi, size=size)

    @classmethod
    def fixed(cls, mu: float) -> "DriftSpec":
        return cls(most_likely=mu)

    @classmethod
    def triangular(cls, low: float, most_likely: float, high: float) -> "DriftSpec":
        return cls(most_likely=most_likely, low=low, high=high)


@dataclass(frozen=True)
class Parameters:
    """Full configuration for a simulation run."""

    spot: float  # S(0), the starting index level
    drift: DriftSpec  # annual drift mu (fixed or Triangular)
    sigma: float  # annual volatility
    horizon_years: float = 10.0  # total simulated horizon
    steps_per_year: int = 252  # time discretization (252 = daily)
    n_iterations: int = 10_000  # number of simulated paths
    seed: int = 12345  # RNG seed; fixed seed => reproducible output

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon_years * self.steps_per_year))

    @property
    def dt(self) -> float:
        return 1.0 / self.steps_per_year

    @classmethod
    def from_calibration(
        cls,
        calibration,  # indices_mc.calibration.Calibration
        *,
        horizon_years: float = 10.0,
        drift: DriftSpec | None = None,
        **kwargs,
    ) -> "Parameters":
        """Build Parameters from a Calibration, defaulting drift to the point estimate."""
        return cls(
            spot=calibration.spot,
            drift=drift if drift is not None else DriftSpec.fixed(calibration.mu),
            sigma=calibration.sigma,
            horizon_years=horizon_years,
            **kwargs,
        )

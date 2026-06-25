"""indices_mc — Monte Carlo simulation of stock-index price paths under GBM.

A transparent demonstration of Monte Carlo methodology: calibration,
stochastic simulation, percentile estimation, and convergence diagnostics.
See readme.txt for the full methodology and limitations.

This is an educational tool, not financial advice.
"""

from .calibration import Calibration, calibrate
from .parameters import DriftSpec, Parameters
from .engine import SimulationResult, simulate
from .analysis import (
    convergence_curve,
    percentile_bands,
    target_probability,
)
from . import presets
from .presets import IndexPreset

__all__ = [
    "Calibration",
    "calibrate",
    "DriftSpec",
    "Parameters",
    "SimulationResult",
    "simulate",
    "percentile_bands",
    "target_probability",
    "convergence_curve",
    "presets",
    "IndexPreset",
]

__version__ = "0.1.0"

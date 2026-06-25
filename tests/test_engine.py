"""Sanity tests for the simulation engine and analysis — no network required."""

import numpy as np

from indices_mc.analysis import (
    convergence_curve,
    percentile_bands,
    target_probability,
)
from indices_mc.calibration import calibrate_from_prices
from indices_mc.engine import simulate
from indices_mc.parameters import DriftSpec, Parameters


def _params(**kw):
    base = dict(
        spot=100.0,
        drift=DriftSpec.fixed(0.07),
        sigma=0.18,
        horizon_years=10.0,
        steps_per_year=252,
        n_iterations=20_000,
        seed=42,
    )
    base.update(kw)
    return Parameters(**base)


def test_reproducible_seed():
    a = simulate(_params())
    b = simulate(_params())
    assert np.array_equal(a.paths, b.paths)


def test_starts_at_spot():
    r = simulate(_params())
    assert np.allclose(r.paths[:, 0], 100.0)


def test_terminal_distribution_matches_gbm_theory():
    # E[S_T] = S0 * exp(mu * T) under GBM.
    p = _params(n_iterations=200_000)
    r = simulate(p)
    expected = p.spot * np.exp(p.drift.most_likely * p.horizon_years)
    assert abs(r.terminal.mean() - expected) / expected < 0.02


def test_target_probability_bounds():
    r = simulate(_params())
    prob = target_probability(r, target=150.0, horizon_years=10.0)
    assert 0.0 <= prob <= 1.0
    # A target below spot at a long horizon with positive drift is very likely.
    assert target_probability(r, target=50.0, horizon_years=10.0) > 0.9


def test_percentile_bands_ordered():
    r = simulate(_params())
    b = percentile_bands(r)
    # Bands must be monotonically increasing across percentiles at every year.
    for col in range(b.values.shape[1]):
        assert np.all(np.diff(b.values[:, col]) >= 0)


def test_convergence_settles_near_final():
    r = simulate(_params())
    c = convergence_curve(r, target=150.0, horizon_years=10.0)
    direct = target_probability(r, target=150.0, horizon_years=10.0)
    assert abs(c.final - direct) < 1e-9
    # Late estimates should be close to the final value.
    assert abs(c.estimates[-1] - c.final) < 0.02


def test_triangular_drift_propagates():
    # Stochastic drift must widen the terminal spread vs a point estimate.
    fixed = simulate(_params(drift=DriftSpec.fixed(0.07)))
    tri = simulate(_params(drift=DriftSpec.triangular(0.0, 0.07, 0.14)))
    assert tri.terminal.std() > fixed.terminal.std()


def test_calibrate_from_prices():
    rng = np.random.default_rng(0)
    # Synthetic GBM price series with known-ish parameters.
    rets = rng.normal(0.0003, 0.011, size=2520)
    closes = 100.0 * np.exp(np.cumsum(rets))
    dates = [f"2015-01-{i:02d}" for i in range(1, 3)] * 1260
    cal = calibrate_from_prices("TEST", closes, dates[: len(closes)], lookback_years=10)
    assert cal.spot == closes[-1]
    assert 0.0 < cal.sigma < 0.5

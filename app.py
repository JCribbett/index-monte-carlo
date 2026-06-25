"""Streamlit web frontend for indices_mc.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from indices_mc import presets
from indices_mc.analysis import convergence_curve, percentile_bands, target_probability
from indices_mc.calibration import calibrate
from indices_mc.engine import simulate
from indices_mc.parameters import DriftSpec, Parameters
from indices_mc.plotting import plot_cone, plot_convergence

st.set_page_config(page_title="Index Monte Carlo", layout="wide")


@st.cache_data(show_spinner=False)
def _calibrate(ticker: str, lookback: float):
    cal = calibrate(ticker, lookback_years=lookback)
    # Return a plain dict so Streamlit can cache it across reruns.
    return dict(
        ticker=cal.ticker, mu=cal.mu, sigma=cal.sigma, spot=cal.spot,
        lookback_years=cal.lookback_years, n_observations=cal.n_observations,
        start=cal.start, end=cal.end, summary=cal.summary(),
    )


st.title("Stock-Index Monte Carlo Simulator")
st.caption(
    "Geometric Brownian motion calibrated to historical returns. "
    "A demonstration of Monte Carlo methodology — **not** a forecast and "
    "**not** financial advice."
)

with st.sidebar:
    st.header("Inputs")

    _all = presets.all_presets()
    _labels = {f"{p.name} — {p.country}": p.ticker for p in _all}
    _CUSTOM = "Custom ticker…"
    options = list(_labels) + [_CUSTOM]
    default_idx = next(i for i, p in enumerate(_all) if p.key == "sp500")
    choice = st.selectbox("National index", options, index=default_idx,
                          help="Headline benchmark indices from major markets.")
    if choice == _CUSTOM:
        ticker = st.text_input("Custom yfinance symbol", value="^GSPC",
                               help="Any yfinance index symbol, e.g. ^OMXC25.")
    else:
        ticker = _labels[choice]
        st.caption(f"Ticker: `{ticker}`")
    lookback = st.slider("Calibration lookback (years)", 1.0, 20.0, 10.0, 0.5)
    horizon = st.slider("Simulation horizon (years)", 1, 30, 10)
    iterations = st.select_slider(
        "Iterations", options=[1_000, 5_000, 10_000, 20_000, 50_000, 100_000],
        value=20_000,
    )
    seed = st.number_input("Random seed", value=12345, step=1)

    st.subheader("Drift uncertainty")
    propagate = st.checkbox(
        "Treat drift as a Triangular distribution",
        value=False,
        help="Propagate parameter uncertainty, not just path-level randomness.",
    )

run = st.sidebar.button("Run simulation", type="primary")

if run:
    with st.spinner(f"Calibrating {ticker} ..."):
        try:
            cal = _calibrate(ticker, lookback)
        except Exception as exc:  # noqa: BLE001 - surface any data error to the user
            st.error(f"Calibration failed: {exc}")
            st.stop()

    st.success(cal["summary"])
    spot = cal["spot"]
    mu = cal["mu"]

    if propagate:
        # Default a symmetric +/- band around the calibrated drift; user-adjustable.
        col_a, col_b = st.columns(2)
        lo = col_a.number_input("Drift low", value=round(mu - 0.04, 4), format="%.4f")
        hi = col_b.number_input("Drift high", value=round(mu + 0.04, 4), format="%.4f")
        drift = DriftSpec.triangular(lo, mu, hi)
    else:
        drift = DriftSpec.fixed(mu)

    default_target = round(spot * 1.2, -2)
    target = st.number_input(
        "Target index level", value=float(default_target), step=100.0,
        help="Probability of the index reaching or exceeding this by the horizon.",
    )

    params = Parameters(
        spot=spot, drift=drift, sigma=cal["sigma"],
        horizon_years=float(horizon), n_iterations=int(iterations), seed=int(seed),
    )
    with st.spinner(f"Simulating {iterations:,} paths ..."):
        result = simulate(params)

    bands = percentile_bands(result)
    prob = target_probability(result, target, float(horizon))
    curve = convergence_curve(result, target, float(horizon))

    m1, m2, m3 = st.columns(3)
    m1.metric(f"P(level ≥ {target:,.0f} by yr {horizon})", f"{prob * 100:.1f}%")
    m2.metric("Median at horizon", f"{bands.band(50)[-1]:,.0f}")
    m3.metric("P5 – P95 at horizon",
              f"{bands.band(5)[-1]:,.0f} – {bands.band(95)[-1]:,.0f}")

    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        plot_cone(bands, spot, ax=ax1, target=target)
        st.pyplot(fig1)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        plot_convergence(curve, ax=ax2)
        st.pyplot(fig2)

    with st.expander("Percentile table"):
        import numpy as np
        table = {"year": bands.years.astype(int)}
        for p in bands.percentiles:
            table[f"P{p}"] = np.round(bands.band(p)).astype(int)
        st.dataframe(table, use_container_width=True)

    st.info(
        "**Limitations** — This is conditional illustration, not prediction. "
        "Every output is contingent on the calibrated μ and σ continuing to hold. "
        "Constant-parameter GBM does not capture regime shifts, fat tails, or "
        "volatility clustering. Calibration is backward-looking. Educational use "
        "only — not financial advice. See readme.txt."
    )
else:
    st.write("Set inputs in the sidebar and click **Run simulation**.")

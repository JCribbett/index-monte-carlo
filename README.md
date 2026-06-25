# Stock-Index Monte Carlo Simulator

A transparent demonstration of Monte Carlo methodology — calibration, stochastic
simulation, percentile estimation, and convergence diagnostics — applied to a
major stock index under geometric Brownian motion (GBM).

**This is an educational tool, not a market forecast and not financial advice.**
See [`readme.txt`](readme.txt) for the full methodology and limitations.

## What it does

1. **Calibrates** annual drift μ and volatility σ from historical index returns
   (live data via `yfinance`) over a stated lookback window.
2. **Simulates** a configurable number of GBM paths (default ≥10,000), fully
   vectorized in NumPy.
3. **Reports** percentile bands (the "probability cone"), the probability of
   reaching a target by a horizon, and a convergence diagnostic.

All runs are reproducible: a fixed seed yields identical results.

## Methodology

Each simulated path evolves under geometric Brownian motion (GBM), the standard
model for the multiplicative, non-negative behaviour of asset prices:

```
S(t+Δt) = S(t) · exp( (μ − ½σ²)·Δt + σ·√Δt · Z ),   Z ~ N(0, 1)
```

where **μ** is the annual drift (expected log-return), **σ** is annual
volatility, and **Z** is a standard normal shock drawn independently each step.
μ and σ are calibrated from the realized log-returns of the chosen index over a
stated lookback window — grounding the simulation in observed behaviour rather
than arbitrary assumptions. Drift can optionally be treated as a *distribution*
(a Triangular low/most-likely/high range) so parameter uncertainty is propagated
into the output, not just the path-level randomness.

The full methodology write-up lives in [`readme.txt`](readme.txt).

## Limitations — read these; they define what the number means

- **This is conditional illustration, not prediction.** Every output is
  contingent on the calibrated μ and σ continuing to hold. Constant-parameter
  GBM does not capture regime shifts, volatility clustering, or crashes.
- **GBM is a deliberate simplification.** Real returns exhibit fat tails,
  autocorrelation, and time-varying volatility.
- **Calibration is backward-looking.** A bull-heavy or crisis-heavy lookback
  window biases μ and σ accordingly; the window is stated explicitly in every run.
- **Garbage in, garbage out.** Monte Carlo propagates the input assumptions into
  an output distribution — it does not manufacture information.

**Not financial advice. For educational and methodological demonstration only.**

## Install

```bash
pip install -r requirements.txt
```

## Three frontends

### Command line

```bash
# Pick a national index by preset key:
python -m indices_mc.cli --index sp500 --lookback 10 --horizon 10 \
    --target 9000 --iterations 20000 --save run

# List every available national-index preset:
python -m indices_mc.cli --list-indices

# Or use any other yfinance symbol directly:
python -m indices_mc.cli --ticker ^OMXC25 --target 3000
```

Add `--show` to display plots, or `--drift-low/--drift-high` to treat drift as a
Triangular distribution (propagating parameter uncertainty).

### National index presets

~35 headline benchmark indices across the major markets are built in
([`indices_mc/presets.py`](indices_mc/presets.py)) — e.g. `sp500`, `dow`,
`nasdaq`, `tsx`, `bovespa`, `ftse100`, `dax`, `cac40`, `ibex35`, `ftsemib`,
`smi`, `eurostoxx50`, `nikkei225`, `hangseng`, `csi300`, `nifty50`, `sensex`,
`kospi`, `taiex`, `asx200`, `sti`, and more. Run `--list-indices` for the full
table, or pick one from the dropdown in the Streamlit app.

### Streamlit web app

```bash
streamlit run app.py
```

Interactive sliders for ticker, lookback, horizon, iterations, and target.

### Jupyter notebook

```bash
jupyter notebook demo.ipynb
```

A narrative walkthrough interleaving the methodology, code, and inline charts.

## Package layout

| Module | Responsibility |
|---|---|
| `indices_mc/calibration.py` | Estimate annualized μ, σ from historical closes |
| `indices_mc/parameters.py`  | Run config; fixed or Triangular drift |
| `indices_mc/engine.py`      | Vectorized GBM simulation |
| `indices_mc/analysis.py`    | Percentile bands, target probability, convergence |
| `indices_mc/plotting.py`    | Probability cone + convergence charts |
| `indices_mc/cli.py`         | Command-line frontend |

## Tests

```bash
pytest tests/
```

The test suite is network-free (uses synthetic price series) and checks
reproducibility, GBM theoretical moments, percentile ordering, convergence, and
that Triangular drift widens the terminal spread.

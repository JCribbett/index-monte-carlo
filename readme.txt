Methodology
What this model does
This tool estimates the distribution of possible future paths for a major stock index using Monte Carlo simulation, and from that distribution derives the probability of reaching a specified target by a given horizon. It is a demonstration of Monte Carlo methodology — parameter calibration, stochastic simulation, percentile estimation, and convergence diagnostics — not a market forecast. See Limitations below; this distinction is the whole point.
The stochastic process
Each simulated path evolves under geometric Brownian motion (GBM), the standard model for the multiplicative, non-negative behaviour of asset prices:
S(t+Δt) = S(t) · exp( (μ − ½σ²)·Δt + σ·√Δt · Z ),   Z ~ N(0, 1)
where μ is the annual drift (expected log-return), σ is annual volatility, and Z is a standard normal shock drawn independently each step. GBM is chosen because it keeps prices positive, compounds multiplicatively (matching how index returns actually behave), and has well-understood properties — so the model's assumptions are transparent rather than hidden inside a black box.
Parameter calibration
The drift and volatility inputs are calibrated from historical index returns rather than assumed arbitrarily: μ and σ are estimated from the realized log-returns of the chosen index over a stated lookback window. This grounds the simulation in observed behaviour. Because point estimates understate uncertainty, the model also supports treating the inputs themselves as distributions (e.g. a Triangular low/most-likely/high range on drift) so that parameter uncertainty is propagated into the output, not just the path-level randomness.
Simulation and outputs
The engine runs a configurable number of iterations (default ≥10,000), fully vectorized in NumPy so large run counts complete in well under a second. From the resulting ensemble of paths it produces:

Percentile bands (P5 / P10 / P50 / P90 / P95) of index value by year — the "probability cone."
Probability of attaining a target — the fraction of paths reaching or exceeding a specified level by a specified horizon.
A convergence diagnostic showing the probability estimate stabilizing as iteration count rises, which justifies the chosen number of iterations rather than assuming it is sufficient.

All outputs are reproducible: a fixed random seed yields identical results.
Limitations (read these — they define what the number means)

This is conditional illustration, not prediction. Every output is contingent on the calibrated μ and σ continuing to hold. Real markets undergo regime shifts — volatility clustering, crashes, structural breaks — that a constant-parameter GBM does not capture. The model quantifies modeled uncertainty, not the unknown-unknowns that dominate real tail events.
GBM assumes log-normal returns with constant drift and volatility. Real index returns exhibit fat tails, autocorrelation, and time-varying volatility. The model is a deliberate simplification; more elaborate processes (jump-diffusion, stochastic volatility, GARCH) would capture more of this at the cost of transparency.
Calibration is backward-looking. Parameters estimated from a historical window embed that window's regime. A bull-heavy or crisis-heavy lookback biases the inputs accordingly; the lookback choice is itself an assumption, stated explicitly in every run.
The output is only as good as the input assumptions. Monte Carlo does not manufacture information — it propagates the assumed input distributions into an output distribution. Garbage in, garbage out applies with full force.

This model is intended as a transparent demonstration of Monte Carlo methodology and for educational use. It is not financial advice and should not be used as a basis for investment decisions.
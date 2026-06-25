"""Parameter calibration from historical index returns.

Drift (mu) and volatility (sigma) are estimated from the realized log-returns
of a chosen index over a stated lookback window, then annualized. The lookback
choice is itself an assumption and is carried through on the result so every
run states it explicitly (see readme.txt, "Parameter calibration").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Trading days per calendar year. Used to annualize statistics estimated from
# daily log-returns.
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Calibration:
    """Annualized parameters estimated from a historical window."""

    ticker: str
    mu: float  # annual drift (expected log-return)
    sigma: float  # annual volatility
    spot: float  # most recent close, used as the simulation start S(0)
    lookback_years: float
    n_observations: int
    start: str  # ISO date of first observation
    end: str  # ISO date of last observation

    def summary(self) -> str:
        return (
            f"{self.ticker}: mu={self.mu:.4f}/yr, sigma={self.sigma:.4f}/yr, "
            f"spot={self.spot:,.2f} "
            f"(calibrated on {self.n_observations} daily returns, "
            f"{self.start}..{self.end}, ~{self.lookback_years:.1f}y lookback)"
        )


def calibrate_from_prices(
    ticker: str,
    closes: np.ndarray,
    dates: list[str],
    lookback_years: float,
) -> Calibration:
    """Estimate annualized mu and sigma from a series of closing prices.

    Parameters
    ----------
    closes : 1-D array of closing prices, chronological order.
    dates  : ISO date strings aligned with ``closes`` (first and last reported).
    """
    closes = np.asarray(closes, dtype=float)
    closes = closes[np.isfinite(closes) & (closes > 0)]
    if closes.size < 30:
        raise ValueError(
            f"Need at least 30 valid closing prices to calibrate; got {closes.size}."
        )

    log_returns = np.diff(np.log(closes))
    daily_mu = float(np.mean(log_returns))
    daily_sigma = float(np.std(log_returns, ddof=1))

    mu = daily_mu * TRADING_DAYS_PER_YEAR
    sigma = daily_sigma * np.sqrt(TRADING_DAYS_PER_YEAR)

    return Calibration(
        ticker=ticker,
        mu=mu,
        sigma=sigma,
        spot=float(closes[-1]),
        lookback_years=lookback_years,
        n_observations=int(log_returns.size),
        start=dates[0],
        end=dates[-1],
    )


def calibrate(ticker: str = "^GSPC", lookback_years: float = 10.0) -> Calibration:
    """Download history via yfinance and calibrate annualized mu and sigma.

    ``ticker`` defaults to ^GSPC (the S&P 500 index). The lookback window is
    recorded on the returned :class:`Calibration` so it is stated in every run.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "yfinance is required for live calibration. Install it with "
            "`pip install yfinance`, or supply closes via calibrate_from_prices()."
        ) from exc

    period = f"{max(1, int(round(lookback_years)))}y"
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if data is None or data.empty:
        raise ValueError(
            f"yfinance returned no data for {ticker!r}. Check the ticker symbol "
            "and network connectivity."
        )

    close = data["Close"]
    # yfinance may return a single-column DataFrame; squeeze to a Series.
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close = close.dropna()

    dates = [d.strftime("%Y-%m-%d") for d in close.index]
    return calibrate_from_prices(
        ticker=ticker,
        closes=close.to_numpy(dtype=float),
        dates=dates,
        lookback_years=lookback_years,
    )

"""Presets for the main national stock indices.

Each preset maps a short key to its yfinance ticker, a display name, and the
country/region. yfinance is the calibration data source, so the tickers use
Yahoo Finance's index symbology (the leading ``^`` denotes an index).

These are headline benchmark indices — one or two per major market. Pass any
other yfinance symbol directly via the ticker option if you need one not listed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndexPreset:
    key: str  # short lookup key, e.g. "sp500"
    ticker: str  # yfinance symbol, e.g. "^GSPC"
    name: str  # display name
    country: str  # country / region


# Ordered roughly by region. Keys are lower-case and stable.
_PRESETS: tuple[IndexPreset, ...] = (
    # --- North America ---
    IndexPreset("sp500", "^GSPC", "S&P 500", "United States"),
    IndexPreset("dow", "^DJI", "Dow Jones Industrial Average", "United States"),
    IndexPreset("nasdaq", "^IXIC", "NASDAQ Composite", "United States"),
    IndexPreset("russell2000", "^RUT", "Russell 2000", "United States"),
    IndexPreset("tsx", "^GSPTSE", "S&P/TSX Composite", "Canada"),
    IndexPreset("ipc", "^MXX", "IPC (Bolsa Mexicana)", "Mexico"),
    # --- South America ---
    IndexPreset("bovespa", "^BVSP", "Bovespa (Ibovespa)", "Brazil"),
    IndexPreset("merval", "^MERV", "S&P MERVAL", "Argentina"),
    # --- Europe ---
    IndexPreset("ftse100", "^FTSE", "FTSE 100", "United Kingdom"),
    IndexPreset("dax", "^GDAXI", "DAX", "Germany"),
    IndexPreset("cac40", "^FCHI", "CAC 40", "France"),
    IndexPreset("ibex35", "^IBEX", "IBEX 35", "Spain"),
    IndexPreset("ftsemib", "FTSEMIB.MI", "FTSE MIB", "Italy"),
    IndexPreset("aex", "^AEX", "AEX", "Netherlands"),
    IndexPreset("smi", "^SSMI", "SMI", "Switzerland"),
    IndexPreset("omxs30", "^OMX", "OMX Stockholm 30", "Sweden"),
    IndexPreset("bel20", "^BFX", "BEL 20", "Belgium"),
    IndexPreset("psi20", "PSI20.LS", "PSI 20", "Portugal"),
    IndexPreset("atx", "^ATX", "ATX", "Austria"),
    IndexPreset("eurostoxx50", "^STOXX50E", "EURO STOXX 50", "Eurozone"),
    # --- Middle East & Africa ---
    IndexPreset("tasi", "^TASI.SR", "Tadawul All Share (TASI)", "Saudi Arabia"),
    IndexPreset("ta35", "^TA125.TA", "TA-125", "Israel"),
    IndexPreset("jse40", "^J203.JO", "FTSE/JSE All Share", "South Africa"),
    # --- Asia-Pacific ---
    IndexPreset("nikkei225", "^N225", "Nikkei 225", "Japan"),
    IndexPreset("hangseng", "^HSI", "Hang Seng", "Hong Kong"),
    IndexPreset("sse", "000001.SS", "SSE Composite", "China"),
    IndexPreset("csi300", "000300.SS", "CSI 300", "China"),
    IndexPreset("szse", "399001.SZ", "SZSE Component", "China"),
    IndexPreset("nifty50", "^NSEI", "NIFTY 50", "India"),
    IndexPreset("sensex", "^BSESN", "BSE SENSEX", "India"),
    IndexPreset("kospi", "^KS11", "KOSPI", "South Korea"),
    IndexPreset("taiex", "^TWII", "TAIEX", "Taiwan"),
    IndexPreset("asx200", "^AXJO", "S&P/ASX 200", "Australia"),
    IndexPreset("nzx50", "^NZ50", "S&P/NZX 50", "New Zealand"),
    IndexPreset("sti", "^STI", "Straits Times Index", "Singapore"),
    IndexPreset("klci", "^KLSE", "FTSE Bursa Malaysia KLCI", "Malaysia"),
    IndexPreset("set", "^SET.BK", "SET Index", "Thailand"),
    IndexPreset("jci", "^JKSE", "Jakarta Composite", "Indonesia"),
    IndexPreset("psei", "PSEI.PS", "PSEi Composite", "Philippines"),
)

# Key -> preset, for O(1) lookup.
PRESETS: dict[str, IndexPreset] = {p.key: p for p in _PRESETS}

# Ticker -> preset, so a known ticker resolves back to its display name.
_BY_TICKER: dict[str, IndexPreset] = {p.ticker: p for p in _PRESETS}


def all_presets() -> tuple[IndexPreset, ...]:
    """Return every preset in display order."""
    return _PRESETS


def get(key: str) -> IndexPreset:
    """Look up a preset by key (case-insensitive). Raises KeyError if unknown."""
    norm = key.strip().lower()
    if norm not in PRESETS:
        raise KeyError(
            f"Unknown index preset {key!r}. "
            f"Known keys: {', '.join(PRESETS)}."
        )
    return PRESETS[norm]


def by_ticker(ticker: str) -> IndexPreset | None:
    """Return the preset for a yfinance ticker, or None if not a preset."""
    return _BY_TICKER.get(ticker)


def resolve_ticker(index_or_ticker: str) -> str:
    """Resolve a preset key to its ticker; pass through anything else unchanged."""
    norm = index_or_ticker.strip().lower()
    if norm in PRESETS:
        return PRESETS[norm].ticker
    return index_or_ticker

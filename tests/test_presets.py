"""Tests for the national-index presets — no network required."""

import pytest

from indices_mc import presets


def test_keys_unique_and_lowercase():
    keys = [p.key for p in presets.all_presets()]
    assert len(keys) == len(set(keys)), "duplicate preset keys"
    assert all(k == k.lower() for k in keys)


def test_tickers_unique():
    tickers = [p.ticker for p in presets.all_presets()]
    assert len(tickers) == len(set(tickers)), "duplicate tickers"


def test_get_is_case_insensitive():
    assert presets.get("SP500").ticker == "^GSPC"
    assert presets.get(" Dax ").ticker == "^GDAXI"


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        presets.get("not_an_index")


def test_resolve_ticker_passthrough():
    assert presets.resolve_ticker("nikkei225") == "^N225"
    # An arbitrary symbol is returned unchanged.
    assert presets.resolve_ticker("^OMXC25") == "^OMXC25"


def test_by_ticker_roundtrip():
    for p in presets.all_presets():
        assert presets.by_ticker(p.ticker) is p
    assert presets.by_ticker("^NOTREAL") is None


def test_major_markets_present():
    keys = {p.key for p in presets.all_presets()}
    for expected in ("sp500", "dax", "ftse100", "nikkei225", "hangseng",
                     "nifty50", "asx200", "tsx", "bovespa", "kospi"):
        assert expected in keys

"""
Nifty 50 — Top-20 weighted constituents.

Reconstructed module. The original upload's `nifty_weights.py` actually
contained the Bulkowski engine (now correctly placed at
core/bulkowski_engine.py); the real weights module was missing, so
app.py's `nw.TOP_20_NIFTY`, `nw.angel_key_for`, etc. would have failed
to import. This restores them.

------------------------------------------------------------------------
IMPORTANT — these weights are STATIC and must be updated by hand.

NSE rebalances Nifty 50 semi-annually (cut-offs Jan 31 / Jul 31) and the
daily weight drifts with prices. The numbers below are representative
values around mid-2026 and are good enough for a *breadth* read (the
score is dominated by direction, not by 0.2% weight wobble) — but they
are NOT live. Refresh them each quarter:

  1. Get the current constituent weights from
     https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-50
  2. Update WEIGHTS below (top 20 by weight).
  3. Bump WEIGHTS_LAST_UPDATED.
  4. If you change which 20 symbols are used, also update
     ANGEL_TOKENS in core/angel_client.py so tokens stay in sync.
------------------------------------------------------------------------
"""

WEIGHTS_LAST_UPDATED = "2026-05 (manual; refresh quarterly)"

# (symbol, index_weight_pct, sector)
# symbol MUST exist in core/angel_client.ANGEL_TOKENS.
TOP_20_NIFTY = [
    ("HDFCBANK",   8.10, "Financials"),
    ("RELIANCE",   8.00, "Energy"),
    ("ICICIBANK",  6.40, "Financials"),
    ("BHARTIARTL", 4.60, "Telecom"),
    ("INFY",       4.20, "IT"),
    ("SBIN",       3.30, "Financials"),
    ("LT",         3.90, "Industrials"),
    ("ITC",        3.70, "FMCG"),
    ("TCS",        3.40, "IT"),
    ("AXISBANK",   3.00, "Financials"),
    ("KOTAKBANK",  2.80, "Financials"),
    ("HINDUNILVR", 2.30, "FMCG"),
    ("BAJFINANCE", 2.20, "Financials"),
    ("MM",         2.10, "Auto"),
    ("MARUTI",     1.90, "Auto"),
    ("SUNPHARMA",  1.80, "Pharma"),
    ("HCLTECH",    1.70, "IT"),
    ("NTPC",       1.60, "Power"),
    ("TITAN",      1.50, "Consumer"),
    ("ULTRACEMCO", 1.40, "Cement"),
]

# Display alias → Angel One symbol key (kept identical so the token map
# in angel_client.py resolves). M&M trades under "MM" in that map.
_ANGEL_KEY_ALIASES = {
    "M&M": "MM",
}


def angel_key_for(symbol: str) -> str:
    """Map a display symbol to the key used in
    core/angel_client.ANGEL_TOKENS. Identity for all current symbols;
    the alias table exists only for future-proofing."""
    return _ANGEL_KEY_ALIASES.get(symbol, symbol)


def total_weight_top_n(n: int = 20) -> float:
    """Sum of index weight covered by the top-n constituents."""
    return round(sum(w for _, w, _ in TOP_20_NIFTY[:n]), 1)

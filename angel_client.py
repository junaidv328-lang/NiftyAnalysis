"""
Angel One SmartAPI client — stateless, direct-HTTP.

Mirrors the working approach from BulkowskiApp._angel_fetch_data
(bypasses the SmartAPI library for candle fetching, uses pyotp +
generateSession only for the login handshake).

Public API:
    login(api_key, client_id, password, totp_key) -> dict
        Returns {"ok": True, "jwt": str, "feed": str, "refresh": str}
        Or     {"ok": False, "error": str}

    fetch_ohlc(jwt, api_key, symbol, interval, days_back) -> pd.DataFrame | None
        Returns a DataFrame with Date/Open/High/Low/Close/Volume,
        or None on failure (error message is logged via the
        `last_error` field on a returned sentinel — see code).
"""

import datetime as _dt
import json
import os
import ssl
import urllib.error
import urllib.request
from typing import Optional

import pandas as pd

# Symbol → (exchange, token). Lifted verbatim from BulkowskiApp.ANGEL_TOKENS
# for the top-20 weighted Nifty 50 stocks. Keep in sync with the original
# analyzer if you add/remove stocks.
ANGEL_TOKENS = {
    "NIFTY 50":    ("NSE", "99926000"),
    "HDFCBANK":    ("NSE", "1333"),
    "RELIANCE":    ("NSE", "2885"),
    "ICICIBANK":   ("NSE", "4963"),
    "INFY":        ("NSE", "1594"),
    "BHARTIARTL":  ("NSE", "10604"),
    "TCS":         ("NSE", "11536"),
    "LT":          ("NSE", "11483"),
    "ITC":         ("NSE", "1660"),
    "AXISBANK":    ("NSE", "5900"),
    "KOTAKBANK":   ("NSE", "1922"),
    "SBIN":        ("NSE", "3045"),
    "HINDUNILVR":  ("NSE", "1394"),
    "BAJFINANCE":  ("NSE", "317"),
    "MM":          ("NSE", "2031"),     # M&M
    "MARUTI":      ("NSE", "10999"),
    "SUNPHARMA":   ("NSE", "3351"),
    "HCLTECH":     ("NSE", "7229"),
    "NTPC":        ("NSE", "11630"),
    "TITAN":       ("NSE", "3506"),
    "ULTRACEMCO":  ("NSE", "11532"),
}

ANGEL_INTERVALS = {
    "1 Hour":  "ONE_HOUR",
    "1 Day":   "ONE_DAY",
    "1 Week":  "ONE_WEEK",
    "1 Month": "ONE_MONTH",
}

# Days of history to pull for each timeframe so we have enough bars
# for Bulkowski detection (which needs at least ~20 bars but performs
# much better with 80–200).
DEFAULT_DAYS_BACK = {
    "1 Hour":   60,    # ~60d × 6h = 360 hourly bars (4h resample → ~90 bars)
    "1 Day":    400,   # ~280 trading days
    "1 Week":  1825,   # ~5 years
    "1 Month": 3650,   # ~10 years
}


def _ssl_ctx():
    """
    Strict TLS verification by default — the request carries your
    SmartAPI JWT, so an unverified connection is a real MITM risk.

    Only disable verification if you explicitly set the env var
    ANGEL_INSECURE_SSL=1 (e.g. a local Windows box behind a
    corporate proxy with a broken cert store). On Streamlit Cloud
    leave it unset so production stays strict.
    """
    ctx = ssl.create_default_context()
    if os.environ.get("ANGEL_INSECURE_SSL") == "1":
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _base_headers(api_key: str, jwt: str = "") -> dict:
    h = {
        "Content-Type":     "application/json",
        "Accept":           "application/json",
        "X-UserType":       "USER",
        "X-SourceID":       "WEB",
        "X-ClientLocalIP":  "127.0.0.1",
        "X-ClientPublicIP": "127.0.0.1",
        "X-MACAddress":     "00:00:00:00:00:00",
        "X-PrivateKey":     api_key,
    }
    if jwt:
        h["Authorization"] = f"Bearer {jwt}"
    return h


def login(api_key: str, client_id: str, password: str, totp_key: str) -> dict:
    """
    Authenticate against Angel One SmartAPI using TOTP.
    Returns {"ok": True, "jwt": ..., "feed": ..., "refresh": ...}
    or     {"ok": False, "error": "..."}.
    """
    try:
        from SmartApi import SmartConnect
        import pyotp
    except ImportError as e:
        return {"ok": False,
                "error": f"Missing package: {e}. "
                         f"Run: pip install smartapi-python pyotp"}

    try:
        obj = SmartConnect(api_key=api_key)
        totp_code = pyotp.TOTP(totp_key.strip()).now()
        data = obj.generateSession(client_id, password, totp_code)

        if not (data and data.get("status")):
            msg = (data or {}).get("message", "Unknown error")
            return {"ok": False, "error": f"Login failed: {msg}"}

        sd = data.get("data", {})
        jwt = sd.get("jwtToken", "")
        ref = sd.get("refreshToken", "")
        feed = sd.get("feedToken", "")

        # Strip 'Bearer ' prefix if Angel One includes it
        if jwt.startswith("Bearer "):
            jwt = jwt[7:]
        if ref.startswith("Bearer "):
            ref = ref[7:]

        return {"ok": True, "jwt": jwt, "feed": feed, "refresh": ref}

    except Exception as e:
        return {"ok": False, "error": f"Connection error: {e}"}


def fetch_ohlc(
    jwt: str,
    api_key: str,
    symbol: str,
    interval_label: str,
    days_back: Optional[int] = None,
) -> "pd.DataFrame | tuple[None, str]":
    """
    Fetch OHLC for a symbol using Angel One's getCandleData endpoint.

    Returns a DataFrame on success, or (None, error_message) on failure.
    """
    symbol_upper = symbol.strip().upper()
    interval = ANGEL_INTERVALS.get(interval_label, "ONE_DAY")
    if days_back is None:
        days_back = DEFAULT_DAYS_BACK.get(interval_label, 400)

    to_date = _dt.date.today()
    from_date = to_date - _dt.timedelta(days=days_back)
    # Angel One requires 09:15 for the from_time on all timeframes
    # (NSE opens at 09:15 — sending 09:00 returns HTTP 400)
    from_time = "09:15"

    preset = ANGEL_TOKENS.get(symbol_upper)
    if not preset:
        return None, f"Symbol '{symbol_upper}' not in ANGEL_TOKENS preset map."
    exchange, token = preset

    body = json.dumps({
        "exchange":    exchange,
        "symboltoken": str(token),
        "interval":    interval,
        "fromdate":    f"{from_date.strftime('%Y-%m-%d')} {from_time}",
        "todate":      f"{to_date.strftime('%Y-%m-%d')} 15:30",
    }).encode()

    url = ("https://apiconnect.angelone.in/rest/secure/"
           "angelbroking/historical/v1/getCandleData")

    try:
        req = urllib.request.Request(
            url, data=body,
            headers=_base_headers(api_key, jwt), method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=20, context=_ssl_ctx())
        raw = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # Angel One returns the real error message in the response body
        try:
            body_bytes = e.read()
            err_body = json.loads(body_bytes.decode("utf-8"))
            msg = err_body.get("message") or err_body.get("errorcode") or str(err_body)
        except Exception:
            msg = f"HTTP {e.code}"
        return None, f"{symbol_upper}: API rejected request — {msg}"
    except Exception as e:
        return None, f"HTTP error for {symbol_upper}: {e}"

    if not raw.get("status"):
        return None, f"API error for {symbol_upper}: {raw.get('message', '?')}"

    raw_data = raw.get("data", [])
    if not raw_data:
        return None, f"No data returned for {symbol_upper}."

    rows = []
    for bar in raw_data:
        try:
            rows.append({
                "Date":   str(bar[0])[:19],
                "Open":   float(bar[1]),
                "High":   float(bar[2]),
                "Low":    float(bar[3]),
                "Close":  float(bar[4]),
                "Volume": int(float(bar[5])) if len(bar) > 5 else 0,
            })
        except Exception:
            continue

    if not rows:
        return None, f"Could not parse any bars for {symbol_upper}."

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    SmartAPI has no native 4-hour interval; resample from hourly.
    Assumes df_1h has a Date column with hourly timestamps and OHLCV columns.
    """
    if df_1h is None or df_1h.empty:
        return df_1h
    df = df_1h.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    agg = df.resample("4h", origin="start_day").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna(subset=["Open", "Close"])
    agg.reset_index(inplace=True)
    return agg


def resample_daily_to(df_daily: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Resample a daily OHLCV DataFrame to weekly ('W') or monthly ('ME').

    Why we do this: Angel One's getCandleData endpoint inconsistently
    supports ONE_WEEK / ONE_MONTH (some accounts get HTTP 400). Daily
    data is rock-solid, so we fetch daily and aggregate in Python.

    target: 'W' (weekly, Mon-Sun) or 'ME' (month-end)
    """
    if df_daily is None or df_daily.empty:
        return df_daily
    if target not in ("W", "ME"):
        raise ValueError(f"Unsupported resample target: {target}")

    df = df_daily.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    agg = df.resample(target).agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna(subset=["Open", "Close"])
    agg.reset_index(inplace=True)
    return agg

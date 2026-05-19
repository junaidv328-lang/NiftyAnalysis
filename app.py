"""
Nifty 50 Breadth Dashboard
==========================
Streamlit app that pulls OHLC for the top-20 weighted Nifty 50 stocks
via Angel One SmartAPI, runs Bulkowski pattern detection on each,
and aggregates the results into a weighted directional breadth score.

DESIGN PHILOSOPHY (important — read before "trusting" the output):
  This is a BREADTH INDICATOR, not a Nifty forecast.
  - Bulkowski's pattern stats are calibrated for daily charts.
    On 4h/weekly/monthly the *direction* is still informative but
    the probability numbers are approximations.
  - Index price is driven by FII flows, dealer gamma, expiry
    mechanics, and macro — things that don't show up in constituent
    chart patterns. Use this dashboard as one input among many.

Author : For Junaid — VAR Fisheries / FishyBiz Research Tools
"""

import os
import time
import hmac
from typing import Optional


def _consteq(a: str, b: str) -> bool:
    """Constant-time string compare. Avoids a timing side-channel on the
    password gate if this URL is ever shared more widely than intended."""
    if not a or not b:
        return False
    return hmac.compare_digest(str(a), str(b))

import pandas as pd
import streamlit as st

# Local modules
from core import bulkowski_engine as be
from core import angel_client as ac
from core import nifty_weights as nw
from core import cred_store as cs
from core.aggregator import score_stock, aggregate_breadth, implied_nifty_range


# ─────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (mobile-friendly: centered single column)
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty Breadth — Bulkowski",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────
#  CSS — mobile-first tweaks
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tighter padding on mobile */
.block-container { padding-top: 1.2rem; padding-bottom: 4rem; max-width: 720px; }

/* Big tap-friendly buttons */
.stButton > button {
    width: 100%;
    padding: 0.75rem 1rem;
    font-size: 1.05rem;
    font-weight: 600;
    border-radius: 10px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
    color: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.metric-card .label { font-size: 0.85rem; opacity: 0.75; }
.metric-card .value { font-size: 1.8rem; font-weight: 700; margin-top: 0.2rem; }
.metric-card .sub   { font-size: 0.8rem; opacity: 0.7; margin-top: 0.3rem; }

.verdict-bull { color: #10b981; }
.verdict-bear { color: #ef4444; }
.verdict-neutral { color: #94a3b8; }

/* Stock row */
.stock-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.7rem 0.9rem;
    border-radius: 10px;
    background: #f8fafc;
    margin: 0.3rem 0;
    border-left: 4px solid #cbd5e1;
}
.stock-row.bull { border-left-color: #10b981; background: #ecfdf5; }
.stock-row.bear { border-left-color: #ef4444; background: #fef2f2; }
.stock-row .sym { font-weight: 700; font-size: 1rem; }
.stock-row .meta { font-size: 0.8rem; color: #64748b; }
.stock-row .score { font-weight: 700; font-size: 1rem; }

/* Small caveat text */
.caveat {
    font-size: 0.8rem;
    color: #64748b;
    padding: 0.6rem 0.8rem;
    background: #f1f5f9;
    border-radius: 8px;
    border-left: 3px solid #94a3b8;
    margin: 0.6rem 0;
}

/* Hide Streamlit chrome on mobile */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  PASSWORD GATE
# ─────────────────────────────────────────────────────────────────────
def password_gate() -> bool:
    """
    Returns True if the user is authenticated, False otherwise.
    Password lives in st.secrets['app_password'].
    """
    if st.session_state.get("authed"):
        return True

    st.title("🔒 Nifty Breadth Dashboard")
    st.markdown("This app is private. Enter the access password to continue.")

    pw = st.text_input("Password", type="password", key="pw_input")
    if st.button("Unlock"):
        expected = st.secrets.get("app_password", "")
        if not expected:
            st.error("Server is missing `app_password` in secrets. "
                     "Add it via Streamlit Cloud → Settings → Secrets.")
            return False
        if _consteq(pw, expected):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False


if not password_gate():
    st.stop()


# ─────────────────────────────────────────────────────────────────────
#  SESSION STATE  (initialise once)
# ─────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "jwt":          None,
        "api_key":      "",
        "client_id":    "",
        "ohlc_data":    {},      # {symbol: DataFrame}
        "analysis":     None,    # last analysis result dict
        "current_tf":   "1 Day",
        "fetch_errors": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

_init_state()


# ─────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────
st.title("📊 Nifty 50 Breadth")
st.caption(
    f"Top {len(nw.TOP_20_NIFTY)} weighted constituents "
    f"(≈{nw.total_weight_top_n(20)}% of index) · "
    f"weights as of {nw.WEIGHTS_LAST_UPDATED}"
)


# ─────────────────────────────────────────────────────────────────────
#  SECTION 1 — ANGEL ONE LOGIN
# ─────────────────────────────────────────────────────────────────────
with st.expander("🔐 Angel One SmartAPI Login",
                 expanded=(st.session_state.jwt is None)):

    if st.session_state.jwt:
        st.success(f"Logged in as `{st.session_state.client_id}`")
        if st.button("Log out", key="logout_btn"):
            st.session_state.jwt = None
            st.session_state.ohlc_data = {}
            st.session_state.analysis = None
            st.rerun()
    else:
        # ── Determine pre-fill source: saved keyring > secrets.toml > blank
        saved_creds = cs.load_credentials()
        kr_available = cs.is_available()

        if saved_creds:
            default_api  = saved_creds["api_key"]
            default_cid  = saved_creds["client_id"]
            default_pwd  = saved_creds["password"]
            default_totp = saved_creds["totp_key"]
            st.info("🔐 Credentials loaded from Windows Credential Manager.")
        else:
            # Fall back to secrets.toml convenience pre-fill (safe lookup —
            # secrets.toml may be missing on local installs)
            def _safe_secret(key: str) -> str:
                try:
                    return st.secrets.get(key, "")
                except Exception:
                    return ""
            default_api  = _safe_secret("angel_api_key")
            default_cid  = _safe_secret("angel_client_id")
            default_pwd  = _safe_secret("angel_password")
            default_totp = _safe_secret("angel_totp_key")

        api_key = st.text_input("API Key", value=default_api, type="password")
        client_id = st.text_input("Client ID", value=default_cid)
        password = st.text_input("Password (PIN)", value=default_pwd, type="password")
        totp_key = st.text_input(
            "TOTP Secret Key", value=default_totp, type="password",
            help="From your Angel One Google-Authenticator QR setup. "
                 "Not the 6-digit code — the underlying secret string."
        )

        # Remember-me checkbox (only shown if a keyring backend exists)
        if kr_available:
            remember = st.checkbox(
                "💾 Remember me on this computer",
                value=bool(saved_creds),
                help="Saves your credentials encrypted in Windows Credential "
                     "Manager. Only accessible by your Windows user account.",
            )
        else:
            remember = False
            st.caption("⚠️ No OS keyring available — 'Remember me' is disabled "
                       "on this system. Use secrets.toml for pre-fill instead.")

        col_login, col_forget = st.columns([2, 1])

        with col_login:
            if st.button("🔑 Connect to SmartAPI"):
                if not all([api_key, client_id, password, totp_key]):
                    st.error("Fill all four fields.")
                else:
                    with st.spinner("Authenticating with Angel One..."):
                        result = ac.login(api_key, client_id, password, totp_key)
                    if result["ok"]:
                        st.session_state.jwt = result["jwt"]
                        st.session_state.api_key = api_key
                        st.session_state.client_id = client_id

                        # Save or clear credentials based on checkbox
                        if remember:
                            ok, msg = cs.save_credentials(
                                api_key, client_id, password, totp_key)
                            if not ok:
                                st.warning(f"Login OK, but: {msg}")
                        else:
                            # User unticked: if any were previously saved, clear them
                            if saved_creds:
                                cs.clear_credentials()

                        st.success("Connected!")
                        st.rerun()
                    else:
                        st.error(result["error"])

        with col_forget:
            if saved_creds and st.button("🗑️ Forget me"):
                ok, msg = cs.clear_credentials()
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)


# Gate the rest of the app behind login
if not st.session_state.jwt:
    st.info("👆 Log in to Angel One to fetch market data.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────
#  SECTION 2 — FETCH DATA
# ─────────────────────────────────────────────────────────────────────
st.subheader("1️⃣ Fetch market data")

tf_options = ["4 Hour", "1 Day", "1 Week", "1 Month"]
timeframe = st.selectbox(
    "Timeframe",
    tf_options,
    index=tf_options.index(st.session_state.current_tf)
          if st.session_state.current_tf in tf_options else 1,
)
st.session_state.current_tf = timeframe

# Honest caveat for non-daily timeframes
if timeframe != "1 Day":
    st.markdown(
        f'<div class="caveat">⚠️ Bulkowski success rates are calibrated for '
        f'<b>daily</b> charts. On {timeframe.lower()} bars the pattern '
        f'<i>direction</i> is still useful, but the probability numbers '
        f'are approximations. Take them as directional, not precise.</div>',
        unsafe_allow_html=True
    )

if st.button("⬇️ Fetch OHLC for top-20 stocks", key="fetch_btn"):
    st.session_state.ohlc_data = {}
    st.session_state.fetch_errors = []
    st.session_state.analysis = None

    progress = st.progress(0.0)
    status = st.empty()

    stocks = nw.TOP_20_NIFTY + [("NIFTY 50", 0.0, "Index")]  # +Nifty itself
    total = len(stocks)

    for i, (symbol, weight, sector) in enumerate(stocks):
        angel_key = nw.angel_key_for(symbol)
        status.text(f"Fetching {symbol}  ({i+1}/{total})")

        # 4 Hour = fetch hourly then resample
        if timeframe == "4 Hour":
            res = ac.fetch_ohlc(
                st.session_state.jwt, st.session_state.api_key,
                angel_key, "1 Hour",
            )
            if isinstance(res, tuple):
                df, err = res
                st.session_state.fetch_errors.append(f"{symbol}: {err}")
            else:
                df = ac.resample_to_4h(res)
                st.session_state.ohlc_data[symbol] = df

        # 1 Week / 1 Month — fetch daily, resample.
        # Angel One getCandleData has inconsistent support for ONE_WEEK
        # and ONE_MONTH (returns HTTP 400 on some accounts), so we
        # always fetch ONE_DAY and aggregate in Python.
        elif timeframe in ("1 Week", "1 Month"):
            # Pull enough daily history to build the higher timeframe.
            # 1 Week → ~5 yrs of daily ≈ 1825 days
            # 1 Month → ~10 yrs ≈ 3650 days
            days_back = 1825 if timeframe == "1 Week" else 3650
            res = ac.fetch_ohlc(
                st.session_state.jwt, st.session_state.api_key,
                angel_key, "1 Day", days_back=days_back,
            )
            if isinstance(res, tuple):
                df, err = res
                st.session_state.fetch_errors.append(f"{symbol}: {err}")
            else:
                target = "W" if timeframe == "1 Week" else "ME"
                df = ac.resample_daily_to(res, target)
                st.session_state.ohlc_data[symbol] = df

        # 1 Day — native fetch
        else:
            res = ac.fetch_ohlc(
                st.session_state.jwt, st.session_state.api_key,
                angel_key, timeframe,
            )
            if isinstance(res, tuple):
                df, err = res
                st.session_state.fetch_errors.append(f"{symbol}: {err}")
            else:
                st.session_state.ohlc_data[symbol] = res

        progress.progress((i + 1) / total)
        # Angel One has rate limits — small sleep keeps us under them
        time.sleep(0.35)

    status.text(f"Done — {len(st.session_state.ohlc_data)}/{total} fetched.")
    progress.empty()

    if st.session_state.fetch_errors:
        with st.expander(f"⚠️ {len(st.session_state.fetch_errors)} fetch error(s)"):
            for e in st.session_state.fetch_errors:
                st.text(e)


# Show fetched-data summary
if st.session_state.ohlc_data:
    n_fetched = len(st.session_state.ohlc_data)
    st.caption(f"✅ {n_fetched} symbols loaded · timeframe: **{st.session_state.current_tf}**")


# ─────────────────────────────────────────────────────────────────────
#  SECTION 3 — RUN ANALYSIS
# ─────────────────────────────────────────────────────────────────────
if st.session_state.ohlc_data:
    st.subheader("2️⃣ Run Bulkowski analysis")

    if st.button("🔍 Analyse patterns", key="analyse_btn"):
        results = []
        progress = st.progress(0.0)
        status = st.empty()

        for i, (symbol, weight, sector) in enumerate(nw.TOP_20_NIFTY):
            status.text(f"Analysing {symbol}  ({i+1}/{len(nw.TOP_20_NIFTY)})")
            df = st.session_state.ohlc_data.get(symbol)
            if df is None or df.empty:
                progress.progress((i + 1) / len(nw.TOP_20_NIFTY))
                continue

            try:
                # Detect market context for this stock from its 200-bar trend
                close = df["Close"].values
                if len(close) >= 200:
                    sma200 = sum(close[-200:]) / 200
                    mkt_ctx = "bull" if close[-1] > sma200 else "bear"
                else:
                    sma200 = sum(close) / len(close)
                    mkt_ctx = "bull" if close[-1] > sma200 else "bear"

                patterns = be.detect_patterns(df)
                forecasts = [
                    be.compute_pattern_forecast(p, df, market_context=mkt_ctx)
                    for p in patterns
                ]
                stock_score = score_stock(patterns, forecasts)

                # Compute implied % move to T1 target (signed).
                # We want a move whose SIGN agrees with the stock
                # verdict (bullish stock -> positive move still ahead).
                # A pattern can have already blown through its measure-
                # rule target (move_to_t1_pct goes negative while still
                # bullish) — using that would feed a contradictory number
                # into the weighted range and get silently discarded.
                implied_pct = None
                if forecasts:
                    verdict = stock_score["verdict"]
                    target_pat = (stock_score["top_bull_pat"]
                                  if verdict == "BULLISH"
                                  else stock_score["top_bear_pat"])
                    want_sign = (1 if verdict == "BULLISH"
                                 else -1 if verdict == "BEARISH" else 0)

                    named_move = None     # move from the top pattern
                    any_agreeing = None   # any same-direction forecast

                    for p, fc in zip(patterns, forecasts):
                        if not fc:
                            continue
                        # FIX: engine key is 'move_to_t1_pct'
                        mt1 = fc.get("move_to_t1_pct")
                        if mt1 is None:
                            continue
                        mt1 = float(mt1)
                        if p["name"] == target_pat:
                            named_move = mt1
                        if want_sign and (mt1 * want_sign) > 0:
                            # remaining move in the verdict's direction
                            if (any_agreeing is None
                                    or abs(mt1) < abs(any_agreeing)):
                                any_agreeing = mt1

                    # Prefer the top pattern's move only if it still
                    # points the verdict's way; else use the nearest
                    # same-direction target; else leave None (target
                    # already met — no projectable move).
                    if (named_move is not None and want_sign
                            and named_move * want_sign > 0):
                        implied_pct = named_move
                    elif any_agreeing is not None:
                        implied_pct = any_agreeing

                current_price = float(close[-1])
                results.append({
                    "symbol":   symbol,
                    "weight":   weight,
                    "sector":   sector,
                    "price":    round(current_price, 2),
                    "market":   mkt_ctx,
                    "implied_pct_move": implied_pct,
                    **stock_score,
                })
            except Exception as e:
                results.append({
                    "symbol": symbol, "weight": weight, "sector": sector,
                    "price": None, "market": "?",
                    "net_score": 0.0, "bull_strength": 0.0, "bear_strength": 0.0,
                    "verdict": "ERROR", "top_bull_pat": None, "top_bear_pat": None,
                    "n_patterns": 0, "implied_pct_move": None,
                    "error": str(e),
                })

            progress.progress((i + 1) / len(nw.TOP_20_NIFTY))

        # Aggregated breadth
        valid_rows = [r for r in results if r["verdict"] != "ERROR"]
        breadth = aggregate_breadth(valid_rows)

        # Current Nifty level (for implied range)
        nifty_df = st.session_state.ohlc_data.get("NIFTY 50")
        current_nifty = (float(nifty_df["Close"].iloc[-1])
                         if nifty_df is not None and not nifty_df.empty
                         else None)
        nifty_range = implied_nifty_range(valid_rows, current_nifty)

        st.session_state.analysis = {
            "rows":           results,
            "breadth":        breadth,
            "nifty_range":    nifty_range,
            "current_nifty":  current_nifty,
            "timeframe":      st.session_state.current_tf,
        }
        status.text("Done.")
        progress.empty()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────
#  SECTION 4 — RESULTS
# ─────────────────────────────────────────────────────────────────────
if st.session_state.analysis:
    res = st.session_state.analysis
    breadth = res["breadth"]
    rows = res["rows"]
    rng = res["nifty_range"]

    st.subheader("3️⃣ Aggregated breadth")

    # Coverage honesty: a breadth score over 60% of the index is a
    # different object than one over 100%. Make a shortfall loud.
    _full_w = nw.total_weight_top_n(20)
    _cov = breadth["coverage_pct"]
    if _full_w > 0 and _cov < _full_w * 0.9:
        st.markdown(
            f'<div class="caveat" style="border-left-color:#ef4444; '
            f'background:#fef2f2; color:#991b1b;">⚠️ <b>Partial coverage.</b> '
            f'This score reflects only {_cov:.1f}% of {_full_w:.1f}% '
            f'expected index weight — {len(res["rows"]) - len([r for r in rows if r["verdict"] != "ERROR"])} '
            f'stock(s) failed to fetch/analyse. Treat the reading as '
            f'indicative only until coverage is restored.</div>',
            unsafe_allow_html=True,
        )

    # Verdict card
    verdict = breadth["verdict"]
    pct = breadth["directional_pct"]
    v_class = ("verdict-bull" if verdict == "BULLISH"
               else "verdict-bear" if verdict == "BEARISH"
               else "verdict-neutral")
    arrow = "▲" if verdict == "BULLISH" else "▼" if verdict == "BEARISH" else "■"

    st.markdown(f'''
    <div class="metric-card">
      <div class="label">Weighted directional score · {res["timeframe"]}</div>
      <div class="value {v_class}">{arrow} {verdict}</div>
      <div class="sub">Score: {pct:+.1f} / 100 · covers {breadth["coverage_pct"]}% of Nifty 50</div>
    </div>
    ''', unsafe_allow_html=True)

    # Weight-distribution row
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Bullish weight", f"{breadth['bull_weight_pct']:.1f}%")
    c2.metric("⚪ Neutral",       f"{breadth['neutral_weight_pct']:.1f}%")
    c3.metric("🔴 Bearish weight", f"{breadth['bear_weight_pct']:.1f}%")

    # Implied Nifty range
    if rng.get("implied_high") and res["current_nifty"]:
        st.markdown("##### Implied Nifty range (Bulkowski targets, weighted)")
        cur = res["current_nifty"]
        st.markdown(f'''
        <div class="metric-card" style="background: linear-gradient(135deg, #1e3a8a, #1e293b);">
          <div class="label">Current Nifty 50</div>
          <div class="value">{cur:,.2f}</div>
          <div class="sub">
            Implied range: <b>{rng["implied_low"]:,.0f}</b> ↔ <b>{rng["implied_high"]:,.0f}</b><br>
            Bull avg move: {rng["bull_avg_pct"]:+.2f}% · Bear avg move: {rng["bear_avg_pct"]:+.2f}%
          </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown(
            '<div class="caveat">📌 The implied range is a <b>mechanical projection</b> '
            'assuming every pattern hits its T1 target — it is not a forecast. '
            'Real hit rates are well below 100%.</div>',
            unsafe_allow_html=True
        )

    # Sector breakdown
    st.markdown("##### Sector breakdown")
    sec_rows = sorted(
        breadth["sector_scores"].items(),
        key=lambda kv: -kv[1]["weight"],
    )
    for sec, d in sec_rows:
        klass = ("bull" if d["verdict"] == "BULLISH"
                 else "bear" if d["verdict"] == "BEARISH" else "")
        st.markdown(f'''
        <div class="stock-row {klass}">
          <div>
            <div class="sym">{sec}</div>
            <div class="meta">{d["weight"]:.1f}% of analysed index weight</div>
          </div>
          <div class="score">{d["net_score"]:+.2f}</div>
        </div>
        ''', unsafe_allow_html=True)

    # Per-stock details
    with st.expander("📋 Per-stock details", expanded=False):
        for r in rows:
            if r["verdict"] == "ERROR":
                st.markdown(f'''
                <div class="stock-row">
                  <div>
                    <div class="sym">{r["symbol"]}</div>
                    <div class="meta">⚠️ {r.get("error", "error")}</div>
                  </div>
                  <div class="score">—</div>
                </div>
                ''', unsafe_allow_html=True)
                continue

            klass = ("bull" if r["verdict"] == "BULLISH"
                     else "bear" if r["verdict"] == "BEARISH" else "")
            pat = (r["top_bull_pat"] or r["top_bear_pat"]
                   or f'{r["n_patterns"]} patterns')
            price = f'₹{r["price"]:,.2f}' if r["price"] else "—"
            implied = (f' · target {r["implied_pct_move"]:+.1f}%'
                       if r["implied_pct_move"] else "")
            st.markdown(f'''
            <div class="stock-row {klass}">
              <div>
                <div class="sym">{r["symbol"]} <span class="meta">· {r["weight"]:.1f}%</span></div>
                <div class="meta">{price} · {pat}{implied}</div>
              </div>
              <div class="score">{r["net_score"]:+.2f}</div>
            </div>
            ''', unsafe_allow_html=True)

    # Methodology note (collapsed)
    with st.expander("ℹ️ How this score is built"):
        st.markdown("""
- For each of the top-20 weighted Nifty stocks, we run **`detect_patterns`** (your Bulkowski engine) on the chosen timeframe.
- For every detected pattern, **`compute_pattern_forecast`** produces a `completion_prob` that blends the pattern's confidence with Bulkowski's historical failure rate from `PATTERNS_DB`.
- Per-stock score = strongest bullish reliability − strongest bearish reliability, in **[−1, +1]**.
- Index-level score = **Σ(index_weight × stock_score) / Σ(index_weight)** — i.e. weighted by both how much the stock matters to Nifty AND how reliable its pattern is.
- The **implied range** is a measure-rule projection: for each stock, take the Bulkowski T1 target's % move from current price, weight it, apply to Nifty's current level. Mechanical, not predictive.

**What this is NOT:**  An intraday timing tool. A standalone trade signal. A replacement for your gamma / GEX / max-pain workflow — those are causally upstream of Nifty's intraday moves; this is a slower-timeframe internals indicator.
        """)

st.markdown("---")
st.caption("Built for Junaid · VAR Fisheries / FishyBiz Research Tools")

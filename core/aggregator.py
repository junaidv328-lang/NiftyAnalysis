"""
Aggregation logic for the Nifty Breadth Dashboard.

Per the design choice: signals are weighted by BOTH the index weight
AND the Bulkowski pattern's historical success rate.

The "directional score" is a normalised number in [-100, +100]:
  +100 = every stock, weighted, screaming bullish at full conviction
  -100 = same but bearish
     0 = balanced or no signal

This is a BREADTH INDICATOR, not a Nifty forecast. It tells you what
the index constituents are doing now, weighted by how much they matter
and how reliable the underlying pattern is historically.

------------------------------------------------------------------------
CONTRACT (verified against core/bulkowski_engine.py):

detect_patterns(df) -> list of dicts, each with at least:
    "name"        : str   (pattern name)
    "direction"   : str   (contains "BULL" or "BEAR")
    "confidence"  : float (0-100)

compute_pattern_forecast(pat, df, market_context) -> dict with:
    "completion_prob" : float (0-100, Bulkowski-failure-aware)
    "move_to_t1_pct"  : float (signed % move to T1)   <-- NOTE the _pct
    ...

The previous app.py read fc.get("move_to_t1") which never existed —
the correct key is "move_to_t1_pct". That bug silently disabled the
entire implied-range feature. Fixed here + in app.py.
------------------------------------------------------------------------
"""

from typing import Iterable


# Keys we depend on. Used by assert_engine_contract() so a future engine
# refactor fails loudly instead of silently scoring everything 0.
_REQUIRED_PATTERN_KEYS = ("name", "direction", "confidence")
_REQUIRED_FORECAST_KEYS = ("completion_prob", "move_to_t1_pct")


class EngineContractError(RuntimeError):
    """Raised when the Bulkowski engine's output shape drifts from what
    the aggregator expects, so bugs surface immediately."""


def assert_engine_contract(patterns: list[dict],
                           forecasts: list[dict | None]) -> None:
    """Cheap sanity check run once per stock. If the engine ever changes
    its dict keys, this raises instead of producing a silent zero score.

    Empty pattern lists are fine (a stock can legitimately have no
    patterns) — we only validate the shape of what *is* present.
    """
    for p in patterns:
        missing = [k for k in _REQUIRED_PATTERN_KEYS if k not in p]
        if missing:
            raise EngineContractError(
                f"detect_patterns dict missing {missing}. "
                f"Got keys: {sorted(p.keys())}"
            )
    for fc in forecasts:
        if fc is None:
            continue
        missing = [k for k in _REQUIRED_FORECAST_KEYS if k not in fc]
        if missing:
            raise EngineContractError(
                f"compute_pattern_forecast dict missing {missing}. "
                f"Got keys: {sorted(fc.keys())}"
            )


def _pattern_direction_score(pattern: dict,
                             forecast: dict | None) -> tuple[float, float]:
    """For one detected pattern, return (signed_score, reliability).

      signed_score : +1 if BULLISH, -1 if BEARISH, 0 otherwise
      reliability  : 0-1, from the forecast's completion_prob (already
                     blends pattern confidence with Bulkowski failure
                     rate) — falling back to raw pattern confidence.
    """
    direction = (pattern.get("direction") or "").upper()
    if "BULL" in direction:
        sign = +1.0
    elif "BEAR" in direction:
        sign = -1.0
    else:
        return 0.0, 0.0

    if forecast and forecast.get("completion_prob") is not None:
        reliability = float(forecast["completion_prob"]) / 100.0
    else:
        reliability = float(pattern.get("confidence", 50)) / 100.0

    return sign, max(0.0, min(1.0, reliability))


def score_stock(patterns: list[dict],
                forecasts: list[dict | None]) -> dict:
    """Combine all detected patterns for a single stock into one score.

    Take the highest-reliability bullish signal and the highest-
    reliability bearish signal; net them. Avoids double-counting
    overlapping same-direction patterns.
    """
    assert_engine_contract(patterns, forecasts)

    best_bull = 0.0
    best_bear = 0.0
    bull_pat, bear_pat = None, None

    for pat, fc in zip(patterns, forecasts):
        sign, rel = _pattern_direction_score(pat, fc)
        if sign > 0 and rel > best_bull:
            best_bull = rel
            bull_pat = pat["name"]
        elif sign < 0 and rel > best_bear:
            best_bear = rel
            bear_pat = pat["name"]

    net = best_bull - best_bear   # in [-1, +1]
    if net > 0.05:
        verdict = "BULLISH"
    elif net < -0.05:
        verdict = "BEARISH"
    else:
        verdict = "NEUTRAL"

    return {
        "net_score":      round(net, 3),
        "bull_strength":  round(best_bull, 3),
        "bear_strength":  round(best_bear, 3),
        "verdict":        verdict,
        "top_bull_pat":   bull_pat,
        "top_bear_pat":   bear_pat,
        "n_patterns":     len(patterns),
    }


def aggregate_breadth(stock_rows: list[dict]) -> dict:
    """Aggregate per-stock scores into an index-level breadth reading.

    Each row needs: symbol, weight, sector, net_score, verdict.
    """
    if not stock_rows:
        return {
            "weighted_score": 0.0, "directional_pct": 0.0,
            "verdict": "NEUTRAL", "bull_weight_pct": 0.0,
            "bear_weight_pct": 0.0, "neutral_weight_pct": 0.0,
            "sector_scores": {}, "coverage_pct": 0.0,
        }

    total_w = sum(r["weight"] for r in stock_rows)
    weighted_score = (
        sum(r["weight"] * r["net_score"] for r in stock_rows) / total_w
    ) if total_w else 0.0

    bull_w = sum(r["weight"] for r in stock_rows if r["verdict"] == "BULLISH")
    bear_w = sum(r["weight"] for r in stock_rows if r["verdict"] == "BEARISH")
    neutral_w = sum(r["weight"] for r in stock_rows
                    if r["verdict"] == "NEUTRAL")

    sectors: dict[str, dict] = {}
    for r in stock_rows:
        s = r["sector"]
        if s not in sectors:
            sectors[s] = {"weight": 0.0, "weighted_sum": 0.0}
        sectors[s]["weight"] += r["weight"]
        sectors[s]["weighted_sum"] += r["weight"] * r["net_score"]

    sector_scores = {}
    for s, d in sectors.items():
        sec_score = d["weighted_sum"] / d["weight"] if d["weight"] else 0.0
        sector_scores[s] = {
            "weight": round(d["weight"], 2),
            "net_score": round(sec_score, 3),
            "verdict": ("BULLISH" if sec_score > 0.05
                        else "BEARISH" if sec_score < -0.05
                        else "NEUTRAL"),
        }

    if weighted_score > 0.05:
        verdict = "BULLISH"
    elif weighted_score < -0.05:
        verdict = "BEARISH"
    else:
        verdict = "NEUTRAL"

    return {
        "weighted_score":    round(weighted_score, 3),
        "directional_pct":   round(weighted_score * 100, 1),
        "verdict":           verdict,
        "bull_weight_pct":   round(bull_w, 2),
        "bear_weight_pct":   round(bear_w, 2),
        "neutral_weight_pct": round(neutral_w, 2),
        "sector_scores":     sector_scores,
        "coverage_pct":      round(total_w, 2),
    }


def implied_nifty_range(stock_rows: list[dict],
                        current_nifty: float | None) -> dict:
    """Arithmetic projection (NOT a forecast).

    For each stock with a Bulkowski T1 target, weight its signed %
    move by index weight and apply to the current Nifty level.
    Mechanical projection if every pattern fully plays out — real
    hit rate is far below 100%.
    """
    if not stock_rows or current_nifty is None or current_nifty <= 0:
        return {"implied_low": None, "implied_high": None,
                "implied_mid": None, "bull_avg_pct": 0.0,
                "bear_avg_pct": 0.0}

    bull_pct_moves = []
    bear_pct_moves = []
    total_bull_w = 0.0
    total_bear_w = 0.0

    for r in stock_rows:
        w = r["weight"]
        pct = r.get("implied_pct_move")     # signed % move to T1
        if pct is None:
            continue
        if r["verdict"] == "BULLISH" and pct > 0:
            bull_pct_moves.append(w * pct)
            total_bull_w += w
        elif r["verdict"] == "BEARISH" and pct < 0:
            bear_pct_moves.append(w * pct)
            total_bear_w += w

    bull_avg = sum(bull_pct_moves) / total_bull_w if total_bull_w else 0.0
    bear_avg = sum(bear_pct_moves) / total_bear_w if total_bear_w else 0.0

    implied_high = (current_nifty * (1 + bull_avg / 100)
                    if bull_avg > 0 else current_nifty)
    implied_low = (current_nifty * (1 + bear_avg / 100)
                   if bear_avg < 0 else current_nifty)
    implied_mid = (implied_high + implied_low) / 2

    return {
        "implied_low":  round(implied_low,  2),
        "implied_high": round(implied_high, 2),
        "implied_mid":  round(implied_mid,  2),
        "bull_avg_pct": round(bull_avg, 2),
        "bear_avg_pct": round(bear_avg, 2),
    }

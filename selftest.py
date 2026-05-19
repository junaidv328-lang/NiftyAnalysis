"""
Self-test: proves the engine -> aggregator pipeline produces real,
non-zero output and that the move_to_t1_pct fix actually feeds the
implied range. Run:  python selftest.py

This is the harness that settles the correctness question raised in
review. It is NOT imported by the app; safe to delete before deploy.
"""

import sys
import numpy as np
import pandas as pd

from core import bulkowski_engine as be
from core.aggregator import (
    score_stock, aggregate_breadth, implied_nifty_range,
    assert_engine_contract, EngineContractError,
)


def _synthetic_double_bottom(n=160):
    """Build OHLC with a clear double bottom so detect_patterns fires."""
    base = 1000.0
    closes = []
    for i in range(n):
        if i < 40:
            closes.append(base - i * 4)            # decline
        elif i < 55:
            closes.append(base - 160 + (i - 40) * 6)   # bottom 1 -> up
        elif i < 70:
            closes.append(base - 70 - (i - 55) * 6)    # back down
        elif i < 85:
            closes.append(base - 160 + (i - 70) * 7)   # bottom 2 -> up
        else:
            closes.append(base - 55 + (i - 85) * 3)    # breakout up
    closes = np.array(closes, dtype=float)
    df = pd.DataFrame({
        "Date":  pd.date_range("2024-01-01", periods=n, freq="D"),
        "Open":  closes - 1,
        "High":  closes + 4,
        "Low":   closes - 4,
        "Close": closes,
        "Volume": np.full(n, 100000),
    })
    return df


def main() -> int:
    df = _synthetic_double_bottom()
    print(f"[1] synthetic bars: {len(df)}")

    patterns = be.detect_patterns(df)
    print(f"[2] detect_patterns -> {len(patterns)} pattern(s)")
    if patterns:
        print("    keys:", sorted(patterns[0].keys()))

    forecasts = [be.compute_pattern_forecast(p, df, "bull")
                 for p in patterns]
    print(f"[3] forecasts -> {len(forecasts)}")
    if forecasts and forecasts[0]:
        fc = forecasts[0]
        print("    has completion_prob:", "completion_prob" in fc,
              "=", fc.get("completion_prob"))
        print("    has move_to_t1_pct :", "move_to_t1_pct" in fc,
              "=", fc.get("move_to_t1_pct"))
        # The exact bug from review: the OLD key must NOT exist.
        assert "move_to_t1" not in fc or "move_to_t1_pct" in fc
        assert fc.get("move_to_t1") is None, \
            "engine unexpectedly has 'move_to_t1' — re-check the fix"

    # Contract guard must pass on real engine output.
    try:
        assert_engine_contract(patterns, forecasts)
        print("[4] engine contract: OK")
    except EngineContractError as e:
        print("[4] engine contract FAILED:", e)
        return 1

    score = score_stock(patterns, forecasts)
    print(f"[5] score_stock -> verdict={score['verdict']} "
          f"net={score['net_score']} (n={score['n_patterns']})")
    assert score["n_patterns"] > 0, "no patterns detected — test invalid"

    # Build a fake index from this one stock to exercise the aggregator
    # + the implied range (the feature the bug had disabled).
    # Mirror app.py's corrected selection: nearest same-direction move.
    implied_pct = None
    want = 1 if score["verdict"] == "BULLISH" else -1
    for p, fc in zip(patterns, forecasts):
        if fc and fc.get("move_to_t1_pct") is not None:
            mv = float(fc["move_to_t1_pct"])
            if mv * want > 0 and (implied_pct is None
                                  or abs(mv) < abs(implied_pct)):
                implied_pct = mv

    rows = [{
        "symbol": "TEST", "weight": 10.0, "sector": "Test",
        "implied_pct_move": implied_pct, **score,
    }]
    breadth = aggregate_breadth(rows)
    print(f"[6] aggregate_breadth -> {breadth['verdict']} "
          f"{breadth['directional_pct']:+.1f}/100 "
          f"coverage={breadth['coverage_pct']}")

    rng = implied_nifty_range(rows, current_nifty=24000.0)
    print(f"[7] implied_nifty_range -> low={rng['implied_low']} "
          f"high={rng['implied_high']} "
          f"bull%={rng['bull_avg_pct']} bear%={rng['bear_avg_pct']}")

    # The decisive assertion: with the fix, a bullish pattern MUST
    # produce a non-trivial implied range. Pre-fix this was always
    # equal to current_nifty because implied_pct_move was always None.
    if score["verdict"] == "BULLISH":
        moved = (rng["implied_high"] != 24000.0)
        print(f"[8] implied range moved off current price: {moved}")
        if not moved:
            print("    FAIL: implied range still dead — bug not fixed")
            return 1

    print("\nALL CHECKS PASSED — pipeline produces real output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Slim engine module extracted from bulkowski_pattern_analyzer-42.py
Contains only the pure-Python analysis logic needed for the
Streamlit Nifty Breadth Dashboard.

NO Tkinter, NO matplotlib. Safe to import in a headless environment.

Exports:
  - PATTERNS_DB          : Bulkowski statistics database
  - find_swing_highs_lows
  - pct_diff
  - get_completion_status
  - calc_rr
  - detect_patterns(df)  -> list[dict]
  - compute_pattern_forecast(pat, df, market_context='bull') -> dict
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import re
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
#  BULKOWSKI DATABASE — All 53 Chart Patterns with Statistics
# ─────────────────────────────────────────────────────────────────────────────

PATTERNS_DB = {

    # ── REVERSAL PATTERNS — BOTTOMS ──────────────────────────────────────────

    "Double Bottom (Adam & Adam)": {
        "type": "reversal", "direction": "bullish",
        "category": "Double Bottoms",
        "description": "Two sharp V-shaped bottoms at approximately the same price level. Both bottoms are narrow (Adam type). Classic bullish reversal pattern.",
        "identification": [
            "Two distinct lows separated by a moderate peak",
            "Both lows are sharp/narrow (Adam type = pointed V)",
            "Second low within 3–4% of the first low",
            "Valley between the two lows must rise at least 10%",
            "Confirm with breakout above the valley peak (neckline)",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "35%",
                "breakeven_failure_rate": "6%",
                "throwback_rate": "64%",
                "avg_throwback_days": "11 days",
                "performance_rank": "Good",
                "samples": 584,
            },
            "bear_market": {
                "avg_rise": "20%",
                "breakeven_failure_rate": "15%",
                "throwback_rate": "55%",
                "avg_throwback_days": "10 days",
                "performance_rank": "Fair",
                "samples": 223,
            }
        },
        "measure_rule": "Add height of pattern (from lowest low to neckline) to the neckline breakout price for the target.",
        "target_reliability": "68% in bull markets",
        "trading_plan": {
            "entry": "Enter long 1 tick above the neckline (the peak between the two bottoms) on a confirmed daily close above it.",
            "stop": "Place stop 1 tick below the lower of the two bottoms. Risk = neckline to stop distance.",
            "target_1": "Measure the height from the lowest low to the neckline. Add this to the neckline breakout price.",
            "target_2": "If breakout shows strong volume: extend target to 150% of pattern height.",
            "exit_rule": "If a throwback occurs (price returns to neckline): hold if close is above neckline. Exit if close below.",
            "avoid": "Skip if neckline is near a major resistance level. Skip if second bottom is more than 4% below first.",
        },
        "best_performance": [
            "Bull market, upward breakout — highest reliability",
            "Pattern height above the 1-month median performs better",
            "Low in the yearly price range (lower third) = best gains",
            "Breakout on above-average volume = stronger move",
        ],
        "color": "#2ecc71",
    },

    "Double Bottom (Eve & Eve)": {
        "type": "reversal", "direction": "bullish",
        "category": "Double Bottoms",
        "description": "Two rounded/wide U-shaped bottoms at approximately the same level. Both bottoms are broad (Eve type). Very common and reliable.",
        "identification": [
            "Two distinct lows — both wide/rounded (Eve = broad U shape)",
            "Second low within 3–4% of first low price",
            "Peak between bottoms rises at least 10%",
            "Eve bottoms show more day-to-day price variation than Adam",
            "Breakout: daily close above the peak between the two bottoms",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "40%",
                "breakeven_failure_rate": "5%",
                "throwback_rate": "59%",
                "avg_throwback_days": "11 days",
                "performance_rank": "Excellent",
                "samples": 776,
            },
            "bear_market": {
                "avg_rise": "23%",
                "breakeven_failure_rate": "11%",
                "throwback_rate": "54%",
                "avg_throwback_days": "10 days",
                "performance_rank": "Good",
                "samples": 297,
            }
        },
        "measure_rule": "Height of pattern (lowest low to neckline) + neckline price.",
        "target_reliability": "72% in bull markets",
        "trading_plan": {
            "entry": "Enter long on close above neckline (peak between two bottoms).",
            "stop": "Below the lower of the two Eve bottoms.",
            "target_1": "Neckline price + pattern height.",
            "target_2": "If volume expansion on breakout: 150% of pattern height target.",
            "exit_rule": "Scale out 50% at T1. Hold 50% with trailing stop under each new swing low.",
            "avoid": "Overhead resistance nearby (within 5% of target). High ADX suggesting already exhausted move.",
        },
        "best_performance": [
            "Best performer among all double bottom variants",
            "Bull market significantly outperforms bear market",
            "Higher pattern = better performance (taller = more reliable)",
            "Breakout near yearly low = best percentage gains",
        ],
        "color": "#27ae60",
    },

    "Double Bottom (Adam & Eve)": {
        "type": "reversal", "direction": "bullish",
        "category": "Double Bottoms",
        "description": "First bottom is sharp (Adam), second is broad (Eve). Mixed pattern — moderately reliable.",
        "identification": [
            "First bottom: sharp V-shape (narrow, pointed)",
            "Second bottom: broad U-shape (rounded, wider)",
            "Second bottom at approximately same price level ±4%",
            "Valley peak must be clearly defined",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "37%",
                "breakeven_failure_rate": "7%",
                "throwback_rate": "61%",
                "avg_throwback_days": "11 days",
                "performance_rank": "Good",
                "samples": 521,
            },
            "bear_market": {
                "avg_rise": "22%",
                "breakeven_failure_rate": "12%",
                "throwback_rate": "52%",
                "avg_throwback_days": "10 days",
                "performance_rank": "Fair",
                "samples": 198,
            }
        },
        "measure_rule": "Standard: pattern height added to neckline breakout price.",
        "target_reliability": "70% bull markets",
        "trading_plan": {
            "entry": "Close above neckline breakout.",
            "stop": "Below lower of the two bottoms.",
            "target_1": "Pattern height added to neckline.",
            "target_2": "Prior swing high if closer than calculated target.",
            "exit_rule": "Hold through throwback if neckline holds. Exit on daily close below neckline.",
            "avoid": "If the two bottoms differ by more than 5% in price.",
        },
        "best_performance": [
            "Bull markets perform substantially better",
            "Tall patterns outperform short ones",
            "Breakout near yearly low gives best results",
        ],
        "color": "#1abc9c",
    },

    "Triple Bottom": {
        "type": "reversal", "direction": "bullish",
        "category": "Triple Patterns",
        "description": "Three consecutive lows at approximately the same price level, forming a strong support zone. Reliable reversal pattern.",
        "identification": [
            "Three distinct lows at approximately the same price (within 3%)",
            "Two peaks between the three lows — both neckline points",
            "Each low must be a clear swing low (not just a minor dip)",
            "Pattern should span at least 3–4 weeks",
            "Breakout: close above the higher of the two peaks",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "37%",
                "breakeven_failure_rate": "4%",
                "throwback_rate": "64%",
                "avg_throwback_days": "11 days",
                "performance_rank": "Excellent",
                "samples": 211,
            },
            "bear_market": {
                "avg_rise": "22%",
                "breakeven_failure_rate": "14%",
                "throwback_rate": "55%",
                "avg_throwback_days": "10 days",
                "performance_rank": "Good",
                "samples": 88,
            }
        },
        "measure_rule": "Height from lowest low to highest neckline, added to neckline breakout price.",
        "target_reliability": "66% bull markets",
        "trading_plan": {
            "entry": "Close above the higher neckline of the two peaks.",
            "stop": "Below the lowest of the three bottoms.",
            "target_1": "Pattern height + breakout price.",
            "target_2": "Prior resistance level above the breakout.",
            "exit_rule": "Throwback to neckline: hold if neckline holds. Trail stop after T1.",
            "avoid": "If peaks vary by more than 4% in height — pattern may be unreliable.",
        },
        "best_performance": [
            "Very low failure rate — one of the most reliable bullish patterns",
            "Bull markets far outperform bear markets",
            "Patterns with equal three lows (within 1%) most reliable",
        ],
        "color": "#16a085",
    },

    "Head-and-Shoulders Bottom": {
        "type": "reversal", "direction": "bullish",
        "category": "Head and Shoulders",
        "description": "Three lows where the center (head) is lower than the two shoulders. Classic major reversal pattern.",
        "identification": [
            "Three lows: left shoulder, head (deepest), right shoulder",
            "Head is clearly lower than both shoulders",
            "Right shoulder should be at similar height to left shoulder (±5%)",
            "Neckline drawn across the tops of the two peaks between shoulders and head",
            "Confirm: daily close above neckline with volume expansion",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "38%",
                "breakeven_failure_rate": "3%",
                "throwback_rate": "45%",
                "avg_throwback_days": "11 days",
                "performance_rank": "Excellent",
                "samples": 733,
            },
            "bear_market": {
                "avg_rise": "24%",
                "breakeven_failure_rate": "8%",
                "throwback_rate": "40%",
                "avg_throwback_days": "10 days",
                "performance_rank": "Good",
                "samples": 279,
            }
        },
        "measure_rule": "Measure from head low to neckline. Add to neckline breakout price.",
        "target_reliability": "55–60% (conservative but reliable entry with tight stops)",
        "trading_plan": {
            "entry": "Daily close above neckline. Aggressive: buy intraday on neckline touch.",
            "stop": "Below the right shoulder low. OR below the head for wider stop.",
            "target_1": "Head-to-neckline distance projected upward from neckline.",
            "target_2": "Prior swing highs above the pattern.",
            "exit_rule": "If throwback to neckline: hold if neckline is support. Exit daily close below neckline.",
            "avoid": "Slanting necklines (more than 15° slope) = less reliable. Avoid if right shoulder is significantly lower than left.",
        },
        "best_performance": [
            "One of the highest reliability patterns with 3% failure rate in bull markets",
            "Lower throwback rate than double bottoms = cleaner entries",
            "Bull market performance dramatically better",
            "Horizontal neckline outperforms slanted",
        ],
        "color": "#3498db",
    },

    "Cup with Handle": {
        "type": "reversal", "direction": "bullish",
        "category": "Cup Patterns",
        "description": "U-shaped cup (rounding bottom) followed by a small handle (slight downward drift). Popularized by William O'Neil. Continuation/reversal pattern.",
        "identification": [
            "Cup: U-shaped rounding bottom, NOT a V-shape",
            "Cup depth: typically 15–30% from rim to cup bottom",
            "Handle: small pullback in the upper third of the cup (5–15% decline)",
            "Handle should drift down along light volume",
            "Breakout: close above the rim of the cup (prior high)",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "34%",
                "breakeven_failure_rate": "5%",
                "throwback_rate": "52%",
                "avg_throwback_days": "11 days",
                "performance_rank": "Good",
                "samples": 428,
            },
            "bear_market": {
                "avg_rise": "22%",
                "breakeven_failure_rate": "13%",
                "throwback_rate": "47%",
                "avg_throwback_days": "11 days",
                "performance_rank": "Fair",
                "samples": 147,
            }
        },
        "measure_rule": "Cup depth projected upward from the breakout (rim) price.",
        "target_reliability": "58% in bull markets",
        "trading_plan": {
            "entry": "Close above the prior rim high (cup lip). Ideally with heavy volume.",
            "stop": "Below the handle low.",
            "target_1": "Breakout price + cup depth.",
            "target_2": "Prior all-time high if cup forms in a pullback.",
            "exit_rule": "Throwback to rim level: hold if rim holds as support. Exit on close below rim.",
            "avoid": "V-shaped cup (too steep). Handle that drops more than half the cup depth. Handle that drifts up instead of down.",
        },
        "best_performance": [
            "Bull markets significantly better than bear markets",
            "Cups with symmetrical shape outperform asymmetric cups",
            "Breakout on heavy volume (2x average) shows best performance",
            "Shorter cup duration (4–7 weeks) outperforms longer cups",
        ],
        "color": "#9b59b6",
    },

    "Rounding Bottom (Saucer)": {
        "type": "reversal", "direction": "bullish",
        "category": "Rounding Patterns",
        "description": "Gradual, smooth U-shaped curve over many weeks or months. Represents a long-term shift from selling to buying pressure.",
        "identification": [
            "Slow, gradual decline followed by slow, gradual rise",
            "Price action traces a smooth curved arc (NOT jagged)",
            "Volume typically decreases at the bottom and increases on the way up",
            "Pattern usually spans 7 weeks to several months",
            "Breakout: close above prior high before the rounding started",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "43%",
                "breakeven_failure_rate": "5%",
                "throwback_rate": "33%",
                "avg_throwback_days": "11 days",
                "performance_rank": "Excellent",
                "samples": 198,
            },
            "bear_market": {
                "avg_rise": "31%",
                "breakeven_failure_rate": "8%",
                "throwback_rate": "29%",
                "avg_throwback_days": "10 days",
                "performance_rank": "Good",
                "samples": 74,
            }
        },
        "measure_rule": "Depth of the rounding bottom added to the right lip (breakout price).",
        "target_reliability": "61% bull markets",
        "trading_plan": {
            "entry": "Close above the right rim (prior high that preceded the rounding). Very low throwback rate — this is clean.",
            "stop": "Below the low of the rounding bottom.",
            "target_1": "Pattern depth + breakout price.",
            "target_2": "Measured move using prior trend leg.",
            "exit_rule": "Lowest throwback rate of major patterns — hold aggressively through the first pullback.",
            "avoid": "Pattern with too many jagged days (should be smooth). Avoid if pattern is less than 7 bars wide.",
        },
        "best_performance": [
            "Highest average rise among common bullish patterns (43% in bull)",
            "Very low throwback rate = clean holds with minimal stress",
            "Works especially well at major market bottoms",
        ],
        "color": "#e67e22",
    },

    "Bump-and-Run Reversal Bottom": {
        "type": "reversal", "direction": "bullish",
        "category": "Bump-and-Run",
        "description": "A declining lead-in trendline followed by a downward spike (bump) that overshoots the lead-in angle, then reverses (run). Bottom reversal.",
        "identification": [
            "Lead-in phase: gentle downtrend with consistent slope",
            "Bump phase: price declines sharply below the lead-in trendline (typically at 45°+ steeper)",
            "Bump height: at least 2x the lead-in height from trendline",
            "Run phase: price reverses back above the lead-in trendline",
            "Volume spike during the bump phase",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "37%",
                "breakeven_failure_rate": "5%",
                "throwback_rate": "62%",
                "avg_throwback_days": "11 days",
                "performance_rank": "Good",
                "samples": 182,
            },
            "bear_market": {
                "avg_rise": "27%",
                "breakeven_failure_rate": "11%",
                "throwback_rate": "54%",
                "avg_throwback_days": "10 days",
                "performance_rank": "Fair",
                "samples": 67,
            }
        },
        "measure_rule": "Bump height (from lead-in trendline to bump low) projected upward from breakout.",
        "target_reliability": "63%",
        "trading_plan": {
            "entry": "Close above the lead-in trendline extended to the right (the 'run' breakout).",
            "stop": "Below the bump low.",
            "target_1": "Bump height projected from the breakout point.",
            "target_2": "Prior resistance before the lead-in decline.",
            "exit_rule": "If throwback to trendline: hold if trendline acts as support.",
            "avoid": "If the bump is not clearly steeper than the lead-in (must be a clear angle change).",
        },
        "best_performance": [
            "Works best when bump is 2–3x the lead-in height",
            "Strong volume on the reversal from the bump low improves performance",
            "Bull markets substantially outperform",
        ],
        "color": "#f39c12",
    },

    # ── REVERSAL PATTERNS — TOPS ──────────────────────────────────────────────

    "Double Top (Eve & Eve)": {
        "type": "reversal", "direction": "bearish",
        "category": "Double Tops",
        "description": "Two broad/rounded tops at approximately the same price level. Both peaks are Eve-type (wide, rounded). Most reliable double top variant.",
        "identification": [
            "Two distinct highs — both wide/rounded (Eve type = broad, curved)",
            "Second top within 3–4% of first top price",
            "Valley between tops declines at least 10%",
            "Breakout: daily close below the valley low (neckline)",
        ],
        "stats": {
            "bull_market": {
                "avg_decline": "18%",
                "breakeven_failure_rate": "11%",
                "pullback_rate": "59%",
                "avg_pullback_days": "11 days",
                "performance_rank": "Good",
                "samples": 589,
            },
            "bear_market": {
                "avg_decline": "24%",
                "breakeven_failure_rate": "8%",
                "pullback_rate": "52%",
                "avg_pullback_days": "10 days",
                "performance_rank": "Excellent",
                "samples": 213,
            }
        },
        "measure_rule": "Pattern height (highest top to neckline) subtracted from neckline breakout price.",
        "target_reliability": "68% bear markets",
        "trading_plan": {
            "entry": "Close below neckline (the valley low between the two tops).",
            "stop": "Above the lower of the two tops.",
            "target_1": "Neckline price minus pattern height.",
            "target_2": "Major support level below.",
            "exit_rule": "If pullback occurs (price returns to neckline): hold if neckline acts as resistance. Exit on close above neckline.",
            "avoid": "Skip if support is within 5% of target. Skip if second top is more than 4% above first.",
        },
        "best_performance": [
            "Best performer among double top variants",
            "Bear markets show highest reliability for this pattern",
            "Taller patterns (height above median) perform better",
            "Breakout near yearly high gives best decline percentage",
        ],
        "color": "#e74c3c",
    },

    "Double Top (Adam & Adam)": {
        "type": "reversal", "direction": "bearish",
        "category": "Double Tops",
        "description": "Two sharp V-shaped tops at approximately the same price. Both peaks are Adam-type (narrow, pointed spikes).",
        "identification": [
            "Two distinct highs — both narrow/sharp (Adam type = pointed spikes)",
            "Second top within 3–4% of first top",
            "Valley between tops shows clear decline (10% minimum)",
            "Breakout: close below valley low",
        ],
        "stats": {
            "bull_market": {
                "avg_decline": "15%",
                "breakeven_failure_rate": "15%",
                "pullback_rate": "62%",
                "avg_pullback_days": "11 days",
                "performance_rank": "Fair",
                "samples": 458,
            },
            "bear_market": {
                "avg_decline": "20%",
                "breakeven_failure_rate": "10%",
                "pullback_rate": "55%",
                "avg_pullback_days": "10 days",
                "performance_rank": "Good",
                "samples": 178,
            }
        },
        "measure_rule": "Pattern height subtracted from neckline breakout price.",
        "target_reliability": "60% bear markets",
        "trading_plan": {
            "entry": "Close below neckline.",
            "stop": "Above higher of the two tops.",
            "target_1": "Neckline minus pattern height.",
            "target_2": "Major support below.",
            "exit_rule": "Pullback to neckline: hold if neckline resistance holds. Exit close above neckline.",
            "avoid": "High pullback rate (62% in bull markets) — be prepared for pullback management.",
        },
        "best_performance": [
            "Bear markets outperform bull markets significantly",
            "Pattern fails more often in bull markets (15% failure rate)",
        ],
        "color": "#c0392b",
    },

    "Head-and-Shoulders Top": {
        "type": "reversal", "direction": "bearish",
        "category": "Head and Shoulders",
        "description": "Three peaks where the center (head) is highest. The most famous bearish reversal pattern in technical analysis.",
        "identification": [
            "Three peaks: left shoulder, head (highest), right shoulder",
            "Head is clearly higher than both shoulders",
            "Right shoulder approximately same height as left (±5%)",
            "Neckline drawn across the lows between peaks",
            "Confirm: daily close below neckline (ideally with volume expansion)",
        ],
        "stats": {
            "bull_market": {
                "avg_decline": "22%",
                "breakeven_failure_rate": "4%",
                "pullback_rate": "45%",
                "avg_pullback_days": "11 days",
                "performance_rank": "Excellent",
                "samples": 1042,
            },
            "bear_market": {
                "avg_decline": "29%",
                "breakeven_failure_rate": "3%",
                "pullback_rate": "38%",
                "avg_pullback_days": "10 days",
                "performance_rank": "Excellent",
                "samples": 419,
            }
        },
        "measure_rule": "Height from head high to neckline, subtracted from neckline breakout price.",
        "target_reliability": "50–55% (conservative measure rule, pattern is very reliable for direction)",
        "trading_plan": {
            "entry": "Close below neckline. OR short on pullback to neckline (if pullback occurs after initial breakdown).",
            "stop": "Above right shoulder high.",
            "target_1": "Neckline price minus head-to-neckline height.",
            "target_2": "Major support levels below.",
            "exit_rule": "Lower pullback rate than most = less interference. Trail stop below each lower high.",
            "avoid": "Sloping neckline (downward sloping is bearish but harder to trade). Right shoulder much higher than left = weaker pattern.",
        },
        "best_performance": [
            "Lowest failure rate among bearish patterns (3–4%)",
            "Works in both bull and bear markets — versatile",
            "Bear market gives dramatically better percentage decline",
            "Horizontal neckline outperforms slanting",
            "Volume should diminish during right shoulder formation",
        ],
        "color": "#8e44ad",
    },

    "Triple Top": {
        "type": "reversal", "direction": "bearish",
        "category": "Triple Patterns",
        "description": "Three consecutive highs at approximately the same price, forming a strong resistance ceiling. Reliable bearish reversal.",
        "identification": [
            "Three distinct highs at approximately the same price (within 3%)",
            "Two valleys between the three peaks",
            "Each high must be a clear swing high",
            "Pattern spans at least 3–4 weeks",
            "Breakout: close below the lower of the two valleys",
        ],
        "stats": {
            "bull_market": {
                "avg_decline": "19%",
                "breakeven_failure_rate": "4%",
                "pullback_rate": "60%",
                "avg_pullback_days": "11 days",
                "performance_rank": "Good",
                "samples": 180,
            },
            "bear_market": {
                "avg_decline": "25%",
                "breakeven_failure_rate": "4%",
                "pullback_rate": "50%",
                "avg_pullback_days": "10 days",
                "performance_rank": "Excellent",
                "samples": 71,
            }
        },
        "measure_rule": "Height from highest top to lowest neckline, subtracted from breakout price.",
        "target_reliability": "64% in bear markets",
        "trading_plan": {
            "entry": "Close below the lower valley (neckline).",
            "stop": "Above the highest of the three tops.",
            "target_1": "Neckline minus pattern height.",
            "target_2": "Prior major support below.",
            "exit_rule": "High pullback rate (60%) — be prepared. Hold if neckline resistance holds on pullback.",
            "avoid": "Peaks that vary by more than 4% in height — less reliable pattern.",
        },
        "best_performance": [
            "Very low failure rate (4%) — high reliability for direction",
            "Bear markets show best performance",
        ],
        "color": "#d35400",
    },

    "Bump-and-Run Reversal Top": {
        "type": "reversal", "direction": "bearish",
        "category": "Bump-and-Run",
        "description": "Rising lead-in trendline followed by a parabolic spike (bump) and reversal (run). Identifies unsustainable price spikes.",
        "identification": [
            "Lead-in: gradual rising trendline, consistent slope",
            "Bump: price surges dramatically above lead-in trendline (at 45°+ steeper angle)",
            "Bump height: at least 2x the lead-in height from trendline",
            "Run: price reverses and breaks back below the lead-in trendline",
            "Volume: typically high during the bump phase",
        ],
        "stats": {
            "bull_market": {
                "avg_decline": "21%",
                "breakeven_failure_rate": "8%",
                "pullback_rate": "40%",
                "avg_pullback_days": "11 days",
                "performance_rank": "Good",
                "samples": 206,
            },
            "bear_market": {
                "avg_decline": "25%",
                "breakeven_failure_rate": "6%",
                "pullback_rate": "35%",
                "avg_pullback_days": "10 days",
                "performance_rank": "Excellent",
                "samples": 82,
            }
        },
        "measure_rule": "Bump height subtracted from the lead-in trendline breakout price.",
        "target_reliability": "65%",
        "trading_plan": {
            "entry": "Short on close below the lead-in trendline (the 'run' starts).",
            "stop": "Above the bump high.",
            "target_1": "Bump height subtracted from breakout price.",
            "target_2": "Start of the lead-in phase (where the gradual rise began).",
            "exit_rule": "Lower pullback rate than most = cleaner short hold.",
            "avoid": "If bump angle is not clearly steeper than lead-in. Need clear angle change.",
        },
        "best_performance": [
            "Low pullback rate = comfortable short holds",
            "Works well in both market conditions",
            "Identifying the bump correctly is the key skill",
        ],
        "color": "#e74c3c",
    },

    # ── CONTINUATION PATTERNS ─────────────────────────────────────────────────

    "Ascending Triangle": {
        "type": "continuation", "direction": "bullish",
        "category": "Triangles",
        "description": "Flat top resistance line with rising bottom trendline. Bullish continuation pattern — buyers are making higher lows while sellers hold at a fixed resistance.",
        "identification": [
            "Top: horizontal resistance line (flat) — two or more peaks at same level",
            "Bottom: upward sloping trendline connecting rising lows",
            "At least two peaks touching the resistance line",
            "At least two higher lows touching the rising trendline",
            "Breakout: ideally upward through the resistance line",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "35%",
                "breakeven_failure_rate": "13%",
                "throwback_rate": "57%",
                "avg_throwback_days": "11 days",
                "upward_breakout_pct": "70%",
                "performance_rank": "Good",
                "samples": 770,
            },
            "bear_market": {
                "avg_rise": "27%",
                "breakeven_failure_rate": "14%",
                "throwback_rate": "47%",
                "avg_throwback_days": "10 days",
                "upward_breakout_pct": "57%",
                "performance_rank": "Good",
                "samples": 319,
            }
        },
        "measure_rule": "Height of triangle (widest part at the left side) added to breakout price.",
        "target_reliability": "75% in bull markets",
        "trading_plan": {
            "entry": "Close above the flat resistance top line.",
            "stop": "Below the most recent higher low within the triangle.",
            "target_1": "Triangle height added to breakout price.",
            "target_2": "Prior swing high above triangle.",
            "exit_rule": "If throwback to resistance line: hold if line acts as support. Exit on close below.",
            "avoid": "Downward breakouts from ascending triangles underperform — consider not trading downside break. Wide triangles (long duration) can fail more.",
        },
        "best_performance": [
            "Upward breakouts occur 70% of the time in bull markets",
            "High throwback rate (57%) — be prepared but hold through it",
            "Best performance when breakout occurs in lower part of yearly range",
            "Tall patterns outperform short patterns",
        ],
        "color": "#2980b9",
    },

    "Descending Triangle": {
        "type": "continuation", "direction": "bearish",
        "category": "Triangles",
        "description": "Flat bottom support line with falling top trendline. Bearish continuation — sellers making lower highs while buyers hold at fixed support.",
        "identification": [
            "Bottom: horizontal support line — two or more lows at same level",
            "Top: downward sloping trendline connecting falling highs",
            "At least two lows touching support",
            "At least two lower highs touching the falling trendline",
            "Breakout: typically downward through support",
        ],
        "stats": {
            "bull_market": {
                "avg_decline": "16%",
                "breakeven_failure_rate": "16%",
                "pullback_rate": "54%",
                "avg_pullback_days": "11 days",
                "downward_breakout_pct": "64%",
                "performance_rank": "Fair",
                "samples": 722,
            },
            "bear_market": {
                "avg_decline": "21%",
                "breakeven_failure_rate": "11%",
                "pullback_rate": "47%",
                "avg_pullback_days": "10 days",
                "downward_breakout_pct": "72%",
                "performance_rank": "Good",
                "samples": 286,
            }
        },
        "measure_rule": "Triangle height subtracted from breakout price.",
        "target_reliability": "62% bear markets",
        "trading_plan": {
            "entry": "Close below the flat support line.",
            "stop": "Above the most recent lower high within the triangle.",
            "target_1": "Support price minus triangle height.",
            "target_2": "Prior swing low below triangle.",
            "exit_rule": "Pullback to support: hold if support acts as resistance. Exit on close above support.",
            "avoid": "Higher failure rate in bull markets (16%) — wait for bear market context for better reliability.",
        },
        "best_performance": [
            "Bear market context gives much better performance",
            "Lower failure rate in bear markets",
            "Downward breakout occurs 64–72% of the time",
        ],
        "color": "#c0392b",
    },

    "Symmetrical Triangle": {
        "type": "continuation", "direction": "neutral",
        "category": "Triangles",
        "description": "Converging upper and lower trendlines with no clear horizontal bias. Breakout can go either direction but favors the prior trend.",
        "identification": [
            "Upper trendline: falling, connecting lower highs",
            "Lower trendline: rising, connecting higher lows",
            "Both trendlines converge to an apex",
            "At least 2 touches on each trendline",
            "Breakout: close outside either trendline",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "31%",
                "avg_decline": "17%",
                "breakeven_failure_rate": "11%",
                "throwback_rate": "37%",
                "upward_breakout_pct": "54%",
                "performance_rank": "Good",
                "samples": 1109,
            },
            "bear_market": {
                "avg_rise": "25%",
                "avg_decline": "20%",
                "breakeven_failure_rate": "13%",
                "throwback_rate": "38%",
                "downward_breakout_pct": "57%",
                "performance_rank": "Good",
                "samples": 479,
            }
        },
        "measure_rule": "Triangle height (widest left side) added to/subtracted from breakout price.",
        "target_reliability": "66% bull upward breakout",
        "trading_plan": {
            "entry": "Trade in direction of the prior trend. Enter on breakout close outside the relevant trendline.",
            "stop": "Inside the triangle (opposite trendline area).",
            "target_1": "Triangle height applied to breakout price.",
            "target_2": "Prior swing high (upward) or low (downward).",
            "exit_rule": "Low throwback/pullback rate (37%) = relatively clean holds.",
            "avoid": "Do not predict direction before the breakout occurs. Let the market tell you.",
        },
        "best_performance": [
            "Low throwback rate = one of the cleanest triangles to hold",
            "Follow the breakout direction — do not predict",
            "Tall patterns perform better for upward breakouts",
        ],
        "color": "#7f8c8d",
    },

    "Rectangle Bottom": {
        "type": "continuation", "direction": "bullish",
        "category": "Rectangles",
        "description": "Horizontal trading range bounded by parallel support and resistance lines, after a downtrend. Bullish when breakout is upward.",
        "identification": [
            "Two distinct horizontal lines: support (bottom) and resistance (top)",
            "Price oscillates between the two lines at least twice",
            "Lines are roughly parallel",
            "Occurs after a prior downtrend",
            "Upward breakout: close above resistance line",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "36%",
                "breakeven_failure_rate": "11%",
                "throwback_rate": "56%",
                "avg_throwback_days": "10 days",
                "performance_rank": "Good",
                "samples": 614,
            },
            "bear_market": {
                "avg_rise": "27%",
                "breakeven_failure_rate": "17%",
                "throwback_rate": "48%",
                "avg_throwback_days": "10 days",
                "performance_rank": "Fair",
                "samples": 247,
            }
        },
        "measure_rule": "Rectangle height (top minus bottom) added to breakout price.",
        "target_reliability": "66%",
        "trading_plan": {
            "entry": "Close above resistance line (top of rectangle).",
            "stop": "Below support line (bottom of rectangle).",
            "target_1": "Resistance price + rectangle height.",
            "target_2": "Prior swing high above rectangle.",
            "exit_rule": "Throwback to resistance: hold if resistance holds as support.",
            "avoid": "Wide rectangles (long duration) can trap traders — prefer narrower ranges.",
        },
        "best_performance": [
            "Bull markets outperform substantially",
            "Short rectangles (narrow height) perform better",
            "Breakout on above-average volume gives better follow-through",
        ],
        "color": "#27ae60",
    },

    "Rectangle Top": {
        "type": "continuation", "direction": "bearish",
        "category": "Rectangles",
        "description": "Horizontal trading range after an uptrend. Bearish when breakout is downward.",
        "identification": [
            "Two horizontal lines: resistance (top) and support (bottom)",
            "Price oscillates between lines at least twice each",
            "Occurs after a prior uptrend",
            "Downward breakout: close below support line",
        ],
        "stats": {
            "bull_market": {
                "avg_decline": "14%",
                "breakeven_failure_rate": "18%",
                "pullback_rate": "51%",
                "avg_pullback_days": "10 days",
                "performance_rank": "Fair",
                "samples": 492,
            },
            "bear_market": {
                "avg_decline": "20%",
                "breakeven_failure_rate": "12%",
                "pullback_rate": "43%",
                "avg_pullback_days": "10 days",
                "performance_rank": "Good",
                "samples": 197,
            }
        },
        "measure_rule": "Rectangle height subtracted from breakout (support) price.",
        "target_reliability": "58% bear markets",
        "trading_plan": {
            "entry": "Close below support line.",
            "stop": "Above resistance line (rectangle top).",
            "target_1": "Support price minus rectangle height.",
            "target_2": "Prior support below rectangle.",
            "exit_rule": "Pullback to support level: hold if support acts as resistance.",
            "avoid": "Higher failure rate in bull markets (18%) — prefer bear market context.",
        },
        "best_performance": [
            "Bear market context significantly better",
            "Short patterns (narrow height) perform better",
        ],
        "color": "#e74c3c",
    },

    "Flag (Bull)": {
        "type": "continuation", "direction": "bullish",
        "category": "Flags and Pennants",
        "description": "Small rectangular consolidation pattern sloping slightly against the prevailing uptrend, resembling a flag on a pole. Short-term continuation.",
        "identification": [
            "Prior strong uptrend (the 'flagpole')",
            "Small rectangular consolidation, typically sloping slightly downward",
            "Lower volume during the flag formation than on the flagpole",
            "Short duration: typically 1–4 weeks",
            "Breakout: close above upper trendline of the flag",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "23%",
                "breakeven_failure_rate": "4%",
                "throwback_rate": "42%",
                "avg_throwback_days": "10 days",
                "performance_rank": "Good",
                "samples": 1014,
            },
            "bear_market": {
                "avg_rise": "17%",
                "breakeven_failure_rate": "9%",
                "throwback_rate": "37%",
                "avg_throwback_days": "9 days",
                "performance_rank": "Fair",
                "samples": 395,
            }
        },
        "measure_rule": "Flagpole height (rise from start of pole to top of pole) added to the flag low or breakout price. Note: flag is NOT a half-staff pattern per Bulkowski.",
        "target_reliability": "64% (bull market)",
        "trading_plan": {
            "entry": "Close above upper trendline of flag.",
            "stop": "Below the lowest low of the flag.",
            "target_1": "Flagpole height added to flag low.",
            "target_2": "Prior resistance above.",
            "exit_rule": "Low throwback rate = clean holds. Trail stop below each higher low.",
            "avoid": "Flags in bear markets perform worse. Avoid flags with large range bars (wide consolidation is not a tight flag).",
        },
        "best_performance": [
            "Very low failure rate (4% in bull markets)",
            "Low throwback rate = clean continuation",
            "Flagpole length: longer is better",
            "Tight, narrow flags outperform wide ones",
        ],
        "color": "#2ecc71",
    },

    "High and Tight Flag": {
        "type": "continuation", "direction": "bullish",
        "category": "Flags and Pennants",
        "description": "Exceptional pattern: stock doubles in 2 months or less, then forms a tight consolidation (10–25% retracement). One of Bulkowski's highest-rated patterns.",
        "identification": [
            "Stock rises 90%+ (ideally doubles) in 2 months or less",
            "Flag: small consolidation of 10–25% retracement",
            "Flag lasts 1–8 weeks",
            "Volume declines during flag, spikes on breakout",
            "Breakout: close above flag high",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "69%",
                "breakeven_failure_rate": "0%",
                "throwback_rate": "54%",
                "avg_throwback_days": "11 days",
                "performance_rank": "BEST",
                "samples": 307,
            },
            "bear_market": {
                "avg_rise": "42%",
                "breakeven_failure_rate": "0%",
                "throwback_rate": "43%",
                "avg_throwback_days": "10 days",
                "performance_rank": "Excellent",
                "samples": 54,
            }
        },
        "measure_rule": "Half the flagpole (start of pole to flag top) added to flag low. Target hit 90% of time.",
        "target_reliability": "90% (conservative half-pole measure)",
        "trading_plan": {
            "entry": "Close above flag high.",
            "stop": "Below flag low.",
            "target_1": "Conservative: half flagpole height + flag low.",
            "target_2": "Full flagpole height + flag low.",
            "exit_rule": "HOLD aggressively — this is Bulkowski's #1 rated pattern (0% failure rate). Accept throwbacks and hold.",
            "avoid": "Flag that retraces more than 50% of the flagpole. Pattern requires the stock to have truly doubled first.",
        },
        "best_performance": [
            "BEST PERFORMER in Bulkowski's entire database",
            "0% failure rate — never failed to rise at least 5% in samples",
            "Average 69% gain in bull markets",
            "This is the pattern to search hardest for",
        ],
        "color": "#f1c40f",
    },

    "Pennant (Bull)": {
        "type": "continuation", "direction": "bullish",
        "category": "Flags and Pennants",
        "description": "Small symmetrical triangle (converging lines) after a strong uptrend. Short consolidation before continuation.",
        "identification": [
            "Prior strong uptrend (flagpole)",
            "Small symmetrical triangle: converging upper and lower trendlines",
            "Volume decreases during pennant",
            "Duration: 1–4 weeks typically",
            "Breakout: close above upper trendline",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "19%",
                "breakeven_failure_rate": "5%",
                "throwback_rate": "37%",
                "avg_throwback_days": "10 days",
                "performance_rank": "Good",
                "samples": 522,
            },
            "bear_market": {
                "avg_rise": "15%",
                "breakeven_failure_rate": "10%",
                "throwback_rate": "34%",
                "avg_throwback_days": "9 days",
                "performance_rank": "Fair",
                "samples": 204,
            }
        },
        "measure_rule": "Flagpole height added to pennant low or breakout price.",
        "target_reliability": "56%",
        "trading_plan": {
            "entry": "Close above upper converging trendline.",
            "stop": "Below pennant low.",
            "target_1": "Flagpole height from pennant breakout.",
            "target_2": "Prior resistance above.",
            "exit_rule": "Low throwback rate = clean hold. Trail stop.",
            "avoid": "Pennant wider than flagpole (body bigger than pole = not a pennant).",
        },
        "best_performance": [
            "Low failure rate and low throwback rate = clean, manageable",
            "Bull market substantially outperforms",
        ],
        "color": "#3498db",
    },

    "Falling Wedge": {
        "type": "reversal", "direction": "bullish",
        "category": "Wedges",
        "description": "Two downward-sloping converging trendlines. Can be reversal or continuation. Breakout is usually upward.",
        "identification": [
            "Both trendlines slope downward with the upper declining faster than lower (converging)",
            "At least 2 touches on each trendline",
            "Volume typically decreases during formation",
            "Breakout: close above upper trendline (upward)",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "38%",
                "breakeven_failure_rate": "5%",
                "throwback_rate": "53%",
                "avg_throwback_days": "11 days",
                "upward_breakout_pct": "68%",
                "performance_rank": "Good",
                "samples": 706,
            },
            "bear_market": {
                "avg_rise": "25%",
                "breakeven_failure_rate": "10%",
                "throwback_rate": "45%",
                "avg_throwback_days": "10 days",
                "upward_breakout_pct": "65%",
                "performance_rank": "Fair",
                "samples": 280,
            }
        },
        "measure_rule": "Height at the widest point (left side) added to breakout price.",
        "target_reliability": "62%",
        "trading_plan": {
            "entry": "Close above upper trendline.",
            "stop": "Below wedge low.",
            "target_1": "Wedge height added to breakout.",
            "target_2": "Start of downtrend that led to the wedge.",
            "exit_rule": "Throwback to upper trendline: hold if trendline acts as support.",
            "avoid": "Downward breakout from a falling wedge underperforms — avoid the short side.",
        },
        "best_performance": [
            "Upward breakout occurs 65–68% of the time",
            "Tall patterns outperform short ones",
            "Bull market significantly better",
        ],
        "color": "#16a085",
    },

    "Rising Wedge": {
        "type": "reversal", "direction": "bearish",
        "category": "Wedges",
        "description": "Two upward-sloping converging trendlines. Bearish — lower trendline rises faster than upper, indicating buying exhaustion.",
        "identification": [
            "Both trendlines slope upward, converging",
            "Lower trendline has steeper angle than upper",
            "Volume decreases during formation",
            "At least 2 touches on each trendline",
            "Breakout: downward through lower trendline",
        ],
        "stats": {
            "bull_market": {
                "avg_decline": "14%",
                "breakeven_failure_rate": "24%",
                "pullback_rate": "68%",
                "avg_pullback_days": "11 days",
                "downward_breakout_pct": "69%",
                "performance_rank": "Fair",
                "samples": 618,
            },
            "bear_market": {
                "avg_decline": "22%",
                "breakeven_failure_rate": "10%",
                "pullback_rate": "57%",
                "avg_pullback_days": "10 days",
                "downward_breakout_pct": "74%",
                "performance_rank": "Good",
                "samples": 253,
            }
        },
        "measure_rule": "Wedge height subtracted from breakout price.",
        "target_reliability": "55%",
        "trading_plan": {
            "entry": "Close below lower trendline.",
            "stop": "Above wedge high.",
            "target_1": "Wedge height subtracted from breakout.",
            "target_2": "Start of rally leading to wedge.",
            "exit_rule": "Very high pullback rate (68% in bull) — be prepared. Hold if lower trendline holds as resistance.",
            "avoid": "High failure rate in bull markets (24%) — best traded in bear market context.",
        },
        "best_performance": [
            "Bear market context is much more reliable",
            "Downward breakout occurs 69–74% of the time",
            "Caution: highest failure rate in bull markets among common patterns",
        ],
        "color": "#d35400",
    },

    "Measured Move Up": {
        "type": "continuation", "direction": "bullish",
        "category": "Measured Moves",
        "description": "Two equal upward legs separated by a corrective phase. Leg 1 = Leg 2 in price distance. Useful for target projection.",
        "identification": [
            "Strong first leg up",
            "Correction phase: 30–60% retracement of leg 1",
            "Second leg up begins and matches the distance of leg 1",
            "Volume typically higher in leg 1 than correction, increases again in leg 2",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "26% (leg 2 after correction)",
                "breakeven_failure_rate": "7%",
                "throwback_rate": "38%",
                "performance_rank": "Good",
                "samples": 593,
            },
            "bear_market": {
                "avg_rise": "20%",
                "breakeven_failure_rate": "14%",
                "throwback_rate": "31%",
                "performance_rank": "Fair",
                "samples": 212,
            }
        },
        "measure_rule": "Measure leg 1 distance. Project same distance from the end of the correction.",
        "target_reliability": "75% for achieving the equal-leg target",
        "trading_plan": {
            "entry": "Enter long at end of correction phase (Higher Low formed, prior downtrend broken).",
            "stop": "Below the correction low.",
            "target_1": "Leg 1 distance + start of leg 2 = primary target.",
            "target_2": "1.5x Leg 1 if very strong momentum.",
            "exit_rule": "Exit at the measured target — leg 2 rarely continues beyond 100% of leg 1.",
            "avoid": "If correction exceeds 60% of leg 1 = move may not resume. If leg 1 is very small = measurement error risk.",
        },
        "best_performance": [
            "75% accuracy for achieving the measured target",
            "Low throwback rate = clean momentum holds",
            "Best used as a TARGET TOOL for all other setups",
        ],
        "color": "#2ecc71",
    },

    "Measured Move Down": {
        "type": "continuation", "direction": "bearish",
        "category": "Measured Moves",
        "description": "Two equal downward legs with a corrective bounce between them. Leg 1 = Leg 2 in distance. Projection tool.",
        "identification": [
            "Strong first leg down",
            "Corrective bounce: 30–60% retracement of leg 1",
            "Second leg down begins and targets equal distance to leg 1",
        ],
        "stats": {
            "bull_market": {
                "avg_decline": "19%",
                "breakeven_failure_rate": "10%",
                "pullback_rate": "29%",
                "performance_rank": "Fair",
                "samples": 398,
            },
            "bear_market": {
                "avg_decline": "27%",
                "breakeven_failure_rate": "6%",
                "pullback_rate": "24%",
                "performance_rank": "Good",
                "samples": 165,
            }
        },
        "measure_rule": "Leg 1 distance subtracted from start of leg 2.",
        "target_reliability": "72%",
        "trading_plan": {
            "entry": "Short at end of corrective bounce (Lower High formed, bounce structure breaking).",
            "stop": "Above the bounce high (correction high).",
            "target_1": "Leg 1 distance subtracted from end of correction.",
            "target_2": "Major support below.",
            "exit_rule": "Very low pullback rate = hold comfortably. Exit at measured target.",
            "avoid": "If bounce exceeds 60% of leg 1.",
        },
        "best_performance": [
            "Lowest pullback rate among major patterns — very clean short holds",
            "Bear market gives excellent 27% average decline",
            "Use primarily as a target projection tool",
        ],
        "color": "#e74c3c",
    },

    "Broadening Bottom": {
        "type": "reversal", "direction": "bullish",
        "category": "Broadening Formations",
        "description": "Expanding price action with lower lows and higher highs. Unusual pattern — both support and resistance diverge. Signals high volatility before reversal.",
        "identification": [
            "Price swings expand: each swing is larger than the previous",
            "Lower trendline: descending (lower lows)",
            "Upper trendline: ascending (higher highs)",
            "At least 3 touches on each trendline",
            "Breakout: close above upper trendline",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "26%",
                "breakeven_failure_rate": "20%",
                "throwback_rate": "53%",
                "performance_rank": "Fair",
                "samples": 235,
            },
            "bear_market": {
                "avg_rise": "21%",
                "breakeven_failure_rate": "26%",
                "throwback_rate": "44%",
                "performance_rank": "Poor",
                "samples": 95,
            }
        },
        "measure_rule": "Height at widest point added to breakout price.",
        "target_reliability": "55%",
        "trading_plan": {
            "entry": "Close above upper trendline (ascending resistance).",
            "stop": "Below the lowest low in the broadening pattern.",
            "target_1": "Widest height + breakout price.",
            "target_2": "Prior resistance above.",
            "exit_rule": "Higher failure rate — use tighter stops than normal.",
            "avoid": "One of the higher failure rate patterns. Only trade if pattern shape is very clear and breakout volume is high.",
        },
        "best_performance": [
            "Relatively high failure rates — approach with caution",
            "Wait for confirmed close above upper trendline",
            "Bull markets much better than bear markets",
        ],
        "color": "#95a5a6",
    },

    "Diamond Bottom": {
        "type": "reversal", "direction": "bullish",
        "category": "Diamond Patterns",
        "description": "Broadening formation followed by a narrowing formation — creates a diamond shape. Bullish reversal from a bottom.",
        "identification": [
            "First half: expanding range (broadening pattern)",
            "Second half: contracting range (symmetrical triangle shape)",
            "Overall shape looks like a diamond or rhombus",
            "Breakout: upward from the narrowing portion",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "36%",
                "breakeven_failure_rate": "6%",
                "throwback_rate": "53%",
                "avg_throwback_days": "13 days",
                "performance_rank": "Good",
                "samples": 115,
            },
            "bear_market": {
                "avg_rise": "22%",
                "breakeven_failure_rate": "16%",
                "throwback_rate": "40%",
                "avg_throwback_days": "11 days",
                "performance_rank": "Fair",
                "samples": 44,
            }
        },
        "measure_rule": "Diamond height (widest point) added to upward breakout price.",
        "target_reliability": "63%",
        "trading_plan": {
            "entry": "Close above upper trendline of narrowing portion.",
            "stop": "Below diamond low.",
            "target_1": "Diamond height + breakout price.",
            "target_2": "Prior resistance above.",
            "exit_rule": "Hold through throwback if upper trendline holds as support.",
            "avoid": "Pattern with only 2 touches per trendline — less reliable.",
        },
        "best_performance": [
            "Low failure rate (6%) in bull markets",
            "Good average rise",
            "Rare pattern — when you see it, it's worth trading carefully",
        ],
        "color": "#9b59b6",
    },

    "Diamond Top": {
        "type": "reversal", "direction": "bearish",
        "category": "Diamond Patterns",
        "description": "Diamond pattern at a top — broadening then narrowing, bearish breakdown.",
        "identification": [
            "Expanding range in first half (broadening)",
            "Contracting range in second half (converging)",
            "Diamond shape at a price peak",
            "Breakout: downward from narrowing portion",
        ],
        "stats": {
            "bull_market": {
                "avg_decline": "20%",
                "breakeven_failure_rate": "10%",
                "pullback_rate": "57%",
                "performance_rank": "Good",
                "samples": 143,
            },
            "bear_market": {
                "avg_decline": "25%",
                "breakeven_failure_rate": "7%",
                "pullback_rate": "48%",
                "performance_rank": "Good",
                "samples": 60,
            }
        },
        "measure_rule": "Diamond height subtracted from downward breakout price.",
        "target_reliability": "58%",
        "trading_plan": {
            "entry": "Close below lower trendline of narrowing portion.",
            "stop": "Above diamond high.",
            "target_1": "Diamond height subtracted from breakout.",
            "target_2": "Prior support below.",
            "exit_rule": "High pullback rate — hold through pullback if lower trendline acts as resistance.",
            "avoid": "Low sample count — confirmation is important.",
        },
        "best_performance": [
            "Bear market gives better performance",
            "Relatively low failure rate",
        ],
        "color": "#e74c3c",
    },

    "Horn Bottom": {
        "type": "reversal", "direction": "bullish",
        "category": "Horn Patterns",
        "description": "Two price spikes (V-shapes) separated by 1–3 weeks, at approximately the same low price. Signals selling exhaustion.",
        "identification": [
            "Two sharp downward spikes (horns) at approximately the same price",
            "Spikes separated by 1–3 weeks",
            "Each horn is a clear spike low (sharp, pointed)",
            "Second horn at same level or slightly higher than first",
            "Breakout: close above the peak between the two horns",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "35%",
                "breakeven_failure_rate": "4%",
                "throwback_rate": "50%",
                "performance_rank": "Good",
                "samples": 346,
            },
            "bear_market": {
                "avg_rise": "24%",
                "breakeven_failure_rate": "11%",
                "throwback_rate": "42%",
                "performance_rank": "Fair",
                "samples": 125,
            }
        },
        "measure_rule": "Height from horn low to peak between horns, added to peak price.",
        "target_reliability": "65%",
        "trading_plan": {
            "entry": "Close above the peak between the two horns.",
            "stop": "Below the lower of the two horn lows.",
            "target_1": "Pattern height + peak price.",
            "target_2": "Prior resistance above.",
            "exit_rule": "Low failure rate — hold through throwback.",
            "avoid": "Horns separated by more than 3 weeks — less reliable timing.",
        },
        "best_performance": [
            "Low failure rate (4%) in bull markets",
            "Good average rise",
        ],
        "color": "#1abc9c",
    },

    "Pipe Bottom": {
        "type": "reversal", "direction": "bullish",
        "category": "Pipe Patterns",
        "description": "Two adjacent weeks (on weekly chart) with similar long lower shadows (wicks) forming a double bottom spike. Reliable weekly chart pattern.",
        "identification": [
            "On weekly chart: two adjacent bars with nearly identical lows",
            "Both bars have long lower shadows (wicks)",
            "Pattern spans exactly 2 weeks",
            "Breakout: week-close above the high of the second pipe bar",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "45%",
                "breakeven_failure_rate": "5%",
                "throwback_rate": "41%",
                "performance_rank": "Excellent",
                "samples": 427,
            },
            "bear_market": {
                "avg_rise": "29%",
                "breakeven_failure_rate": "9%",
                "throwback_rate": "35%",
                "performance_rank": "Good",
                "samples": 168,
            }
        },
        "measure_rule": "Height of the pipe (from pipe low to breakout) added to breakout price.",
        "target_reliability": "68%",
        "trading_plan": {
            "entry": "Weekly close above the high of the second pipe bar.",
            "stop": "Below the pipe low.",
            "target_1": "Pipe height added to breakout price.",
            "target_2": "Prior resistance on weekly chart.",
            "exit_rule": "Low throwback rate = clean weekly holds.",
            "avoid": "Pipes that form mid-trend (not at a bottom). Works best at trend exhaustion points.",
        },
        "best_performance": [
            "One of the best performers — 45% average rise in bull markets",
            "Low failure rate and low throwback rate",
            "Works on weekly charts primarily — do not apply to daily",
        ],
        "color": "#f39c12",
    },

    "Island Reversal (Bottom)": {
        "type": "reversal", "direction": "bullish",
        "category": "Island Patterns",
        "description": "A price island formed by two gaps — gap down to island, then gap up away. Signals dramatic reversal.",
        "identification": [
            "Gap down below prior price action (creates island on the low side)",
            "Price consolidates for 1 or a few days at the island level",
            "Gap up away from the island level (second gap)",
            "Both gaps should be in the same price area",
        ],
        "stats": {
            "bull_market": {
                "avg_rise": "37%",
                "breakeven_failure_rate": "1%",
                "throwback_rate": "35%",
                "performance_rank": "Excellent",
                "samples": 212,
            },
            "bear_market": {
                "avg_rise": "24%",
                "breakeven_failure_rate": "5%",
                "throwback_rate": "28%",
                "performance_rank": "Good",
                "samples": 84,
            }
        },
        "measure_rule": "Height from island low to the gap's upper boundary, added to breakout price.",
        "target_reliability": "67%",
        "trading_plan": {
            "entry": "Enter on the gap-up day (the second gap) — gap itself is the signal.",
            "stop": "Below the island low.",
            "target_1": "Prior resistance before the island formation.",
            "target_2": "Measured move based on island height.",
            "exit_rule": "Lowest throwback rate — hold aggressively.",
            "avoid": "Small island on light volume — best when both gaps are accompanied by high volume.",
        },
        "best_performance": [
            "Extremely low failure rate (1%) in bull markets",
            "Lowest throwback rate — cleanest hold of all patterns",
            "Very reliable but rare — when you see it, prioritize it",
        ],
        "color": "#3498db",
    },

}


# ─── Swing / utility helpers ───────────────────────────────────────
def find_swing_highs_lows(df, order=5):
    """Find local highs and lows using scipy argrelextrema."""
    highs_idx = argrelextrema(df['High'].values, np.greater_equal, order=order)[0]
    lows_idx  = argrelextrema(df['Low'].values,  np.less_equal,    order=order)[0]
    return highs_idx, lows_idx


def pct_diff(a, b):
    """Percentage difference between two values."""
    if b == 0: return float('inf')
    return abs(a - b) / b * 100


def get_completion_status(pat, df):
    """
    Assess how complete / confirmed a pattern is.
    Returns a dict with status label, percentage, and description.
    """
    closes  = df['Close'].values
    highs   = df['High'].values
    lows    = df['Low'].values
    current = closes[-1]
    name    = pat['name']

    neckline     = pat.get('neckline')
    pattern_low  = pat.get('pattern_low')
    pattern_high = pat.get('pattern_high')

    # ── BULLISH PATTERNS ─────────────────────────────────────────────────────
    if pat['direction'] in ('BULLISH', 'BULLISH (70% breakout upward)',
                            'BULLISH (prior uptrend)'):
        if neckline is None:
            return {'status': '🔄 FORMING', 'pct': 50,
                    'color': '#f39c12',
                    'desc': 'Pattern identified. Waiting for key level.'}

        dist_to_neck   = neckline - current
        pattern_height = neckline - pattern_low if pattern_low else 1

        if current > neckline * 1.002:          # closed above neckline
            return {'status': '✅ CONFIRMED BREAKOUT', 'pct': 100,
                    'color': '#2ecc71',
                    'desc': f'Price ({current:.2f}) closed ABOVE neckline '
                            f'({neckline:.2f}). Pattern is ACTIVE. '
                            f'Enter long now or on any pullback to neckline.'}
        elif current > neckline * 0.995:        # within 0.5 % of neckline
            return {'status': '⚡ BREAKOUT IMMINENT', 'pct': 90,
                    'color': '#f1c40f',
                    'desc': f'Price ({current:.2f}) is pressing neckline '
                            f'({neckline:.2f}). Watch for a closing candle '
                            f'above the neckline to confirm entry.'}
        else:
            pct_done = max(10, min(85,
                           100 - (dist_to_neck / pattern_height * 100)))
            return {'status': '🔄 FORMING', 'pct': round(pct_done),
                    'color': '#3498db',
                    'desc': f'Price ({current:.2f}) is {dist_to_neck:.2f} pts '
                            f'below neckline ({neckline:.2f}). '
                            f'Pattern is still building. Do NOT enter yet — '
                            f'wait for neckline breakout close.'}

    # ── BEARISH PATTERNS ─────────────────────────────────────────────────────
    elif 'BEARISH' in pat['direction']:
        if neckline is None:
            return {'status': '🔄 FORMING', 'pct': 50,
                    'color': '#f39c12',
                    'desc': 'Pattern identified. Waiting for key level.'}

        dist_to_neck   = current - neckline
        pattern_height = (pattern_high - neckline) if pattern_high else 1

        if current < neckline * 0.998:          # closed below neckline
            return {'status': '✅ CONFIRMED BREAKDOWN', 'pct': 100,
                    'color': '#e74c3c',
                    'desc': f'Price ({current:.2f}) closed BELOW neckline '
                            f'({neckline:.2f}). Pattern is ACTIVE. '
                            f'Enter short now or on any pullback to neckline.'}
        elif current < neckline * 1.005:        # within 0.5 %
            return {'status': '⚡ BREAKDOWN IMMINENT', 'pct': 90,
                    'color': '#f1c40f',
                    'desc': f'Price ({current:.2f}) is pressing support '
                            f'({neckline:.2f}). Watch for a closing candle '
                            f'below the neckline to confirm short entry.'}
        else:
            pct_done = max(10, min(85,
                           100 - (dist_to_neck / pattern_height * 100)))
            return {'status': '🔄 FORMING', 'pct': round(pct_done),
                    'color': '#3498db',
                    'desc': f'Price ({current:.2f}) is {dist_to_neck:.2f} pts '
                            f'above neckline ({neckline:.2f}). '
                            f'Pattern forming. Wait for breakdown close.'}

    # ── NEUTRAL ───────────────────────────────────────────────────────────────
    return {'status': '🔄 FORMING', 'pct': 60,
            'color': '#7f8c8d',
            'desc': 'Watch for breakout in either direction.'}


def calc_rr(entry, stop, target):
    """
    Calculate risk, reward, R:R ratio and a trade quality grade.
    Returns dict with all fields.
    """
    risk   = abs(entry - stop)
    reward = abs(target - entry)
    if risk == 0:
        return None
    rr = reward / risk

    if rr >= 3.0:
        grade = 'A+  EXCELLENT'
        grade_color = '#2ecc71'
        advice = 'Strong setup. Full position size appropriate.'
    elif rr >= 2.0:
        grade = 'A   GOOD'
        grade_color = '#27ae60'
        advice = 'Good setup. Standard position size.'
    elif rr >= 1.5:
        grade = 'B   ACCEPTABLE'
        grade_color = '#f1c40f'
        advice = 'Acceptable. Consider half position size.'
    elif rr >= 1.0:
        grade = 'C   MINIMUM'
        grade_color = '#e67e22'
        advice = 'Bare minimum. Only trade if pattern confidence > 80%.'
    else:
        grade = 'D   SKIP'
        grade_color = '#e74c3c'
        advice = 'Risk outweighs reward. DO NOT trade this setup.'

    return {
        'risk':        round(risk,   2),
        'reward':      round(reward, 2),
        'rr':          round(rr,     2),
        'grade':       grade,
        'grade_color': grade_color,
        'advice':      advice,
    }



# ─────────────────────────────────────────────────────────────────────────────
#  KAKUSHADZE QUANTITATIVE SIGNAL ENGINE
#  Source: 151 Trading Strategies — Kakushadze & Serur (2018)
#  All signals computed purely from OHLC data — no external data needed
# ─────────────────────────────────────────────────────────────────────────────



# ─── Pattern forecast (Bulkowski stats + targets) ───────────────────
def compute_pattern_forecast(pat, df, market_context='bull'):
    """
    Compute a full probabilistic forecast for a detected pattern.

    Parameters:
        pat            : detected pattern dict from detect_patterns()
        df             : OHLC DataFrame
        market_context : 'bull' or 'bear' (detected from 200-day MA)

    Returns dict with all forecast fields.
    """
    import re as _re

    name    = pat.get('name', '')
    current = df['Close'].values[-1]
    neckline   = pat.get('neckline')
    pat_low    = pat.get('pattern_low')
    pat_high   = pat.get('pattern_high')
    confidence = pat.get('confidence', 60)
    direction  = pat.get('direction', '')
    is_bull    = 'BULL' in direction.upper()

    # ── Get Bulkowski database stats ──────────────────────────────────────
    db_match = None
    for key in PATTERNS_DB:
        if pat['name'] in key or key in pat['name']:
            db_match = PATTERNS_DB[key]
            break
    if not db_match:
        # Try partial match
        pat_words = pat['name'].split()
        for key in PATTERNS_DB:
            if any(w in key for w in pat_words[:2]):
                db_match = PATTERNS_DB[key]
                break

    stats = {}
    if db_match:
        mkt_key = 'bull_market' if market_context == 'bull' else 'bear_market'
        raw_stats = db_match.get('stats', {}).get(mkt_key, {})
        # Parse numeric values
        for field, raw in raw_stats.items():
            try:
                nums = _re.findall(r'\d+\.?\d*', str(raw))
                stats[field] = float(nums[0]) if nums else 0
            except Exception:
                stats[field] = 0

    # ── Core stats ────────────────────────────────────────────────────────
    avg_rise         = stats.get('avg_rise', 30 if is_bull else 20)
    failure_rate     = stats.get('breakeven_failure_rate', 10)
    throwback_rate   = stats.get('throwback_rate', 50)
    throwback_days   = stats.get('avg_throwback_days', 11)
    samples          = int(stats.get('samples', 0))
    perf_rank        = db_match.get('stats', {}).get(
        'bull_market' if market_context == 'bull' else 'bear_market',
        {}).get('performance_rank', 'Fair') if db_match else 'Fair'

    # ── Completion probability ─────────────────────────────────────────────
    # Base from pattern confidence + Bulkowski failure rate
    base_prob = confidence * (1 - failure_rate / 100)
    # Adjust for market context
    if market_context == 'bull' and is_bull:
        base_prob = min(95, base_prob * 1.1)
    elif market_context == 'bear' and not is_bull:
        base_prob = min(95, base_prob * 1.1)
    else:
        base_prob = base_prob * 0.85  # counter-trend = lower probability
    completion_prob = round(base_prob)

    # ── Price targets ──────────────────────────────────────────────────────
    if neckline and pat_low and is_bull:
        pattern_height = neckline - pat_low
        target_1 = neckline + pattern_height           # 100% measure rule
        target_2 = neckline + pattern_height * 1.5    # 150% extended
        target_3 = neckline + pattern_height * 2.0    # 200% maximum
        move_to_t1 = (target_1 - current) / current * 100
        move_to_t2 = (target_2 - current) / current * 100
    elif neckline and pat_high and not is_bull:
        pattern_height = pat_high - neckline
        target_1 = neckline - pattern_height
        target_2 = neckline - pattern_height * 1.5
        target_3 = neckline - pattern_height * 2.0
        move_to_t1 = (target_1 - current) / current * 100
        move_to_t2 = (target_2 - current) / current * 100
    else:
        # Fallback using avg_rise from database
        if is_bull:
            target_1 = current * (1 + avg_rise / 100)
            target_2 = current * (1 + avg_rise * 1.5 / 100)
            target_3 = current * (1 + avg_rise * 2.0 / 100)
        else:
            target_1 = current * (1 - avg_rise / 100)
            target_2 = current * (1 - avg_rise * 1.5 / 100)
            target_3 = current * (1 - avg_rise * 2.0 / 100)
        pattern_height = abs(target_1 - current)
        move_to_t1 = avg_rise if is_bull else -avg_rise
        move_to_t2 = move_to_t1 * 1.5

    # ── Throwback/pullback forecast ────────────────────────────────────────
    throwback_prob   = round(throwback_rate)
    throwback_target = neckline if neckline else current * 0.98 if is_bull else current * 1.02
    throwback_action = ("Hold if price closes ABOVE neckline during throwback.\n"
                        "Exit if price closes BELOW neckline — pattern failed." if is_bull else
                        "Hold if price closes BELOW neckline during pullback.\n"
                        "Exit if price closes ABOVE neckline — pattern failed.")

    # ── Time estimate ──────────────────────────────────────────────────────
    # Based on Bulkowski's typical completion times
    time_estimates = {
        'Head-and-Shoulders': 20,
        'Double Bottom': 15,
        'Double Top': 15,
        'Triple': 25,
        'Triangle': 30,
        'Cup': 45,
        'Flag': 5,
        'Wedge': 10,
        'Measured Move': 20,
    }
    est_days = 20  # default
    for key, days in time_estimates.items():
        if key.lower() in name.lower():
            est_days = days
            break

    # ── Failure scenarios ──────────────────────────────────────────────────
    failure_scenarios = []
    if is_bull:
        failure_scenarios = [
            f"Price fails to close above neckline ({neckline:.2f}) within next {est_days} days",
            f"Throwback occurs AND price closes below neckline ({throwback_target:.2f})",
            f"Volume dries up on breakout (weak institutional participation)",
            f"Broad market enters strong downtrend (Nifty breaks 200-day MA)",
        ] if neckline else [
            "Pattern fails to develop the required swing structure",
            "Price breaks back below the pattern low",
            "Broad market deteriorates sharply",
        ]
    else:
        failure_scenarios = [
            f"Price fails to close below neckline ({neckline:.2f}) within {est_days} days",
            f"Pullback occurs AND price closes above neckline",
            f"Broad market begins new bull run (Nifty reclaims 200-day MA)",
        ] if neckline else [
            "Pattern fails to develop the required structure",
            "Price breaks back above the pattern high",
        ]

    # ── Best conditions filter ─────────────────────────────────────────────
    best_conditions = []
    if db_match:
        best_conditions = db_match.get('best_performance', [])
    if not best_conditions:
        best_conditions = [
            f"{'Bull' if is_bull else 'Bear'} market confirmed (price {'above' if is_bull else 'below'} 200-day MA)",
            "Pattern height above 1-month median = better performance",
            "Breakout on above-average volume = stronger move",
            "Pattern forms at key support/resistance level",
        ]

    # ── Target reliability ─────────────────────────────────────────────────
    target_rel = db_match.get('target_reliability', '65%') if db_match else '60%'
    try:
        target_rel_num = float(_re.findall(r'\d+', str(target_rel))[0])
    except Exception:
        target_rel_num = 65

    # ── Conviction grade ───────────────────────────────────────────────────
    score = (completion_prob * 0.4 +
             (100 - failure_rate) * 0.3 +
             min(samples / 10, 10) * 3 +
             target_rel_num * 0.2)
    if score >= 85:   grade = 'A+ — EXCEPTIONAL'; grade_col = '#2ecc71'
    elif score >= 75: grade = 'A  — HIGH CONVICTION'; grade_col = '#27ae60'
    elif score >= 65: grade = 'B  — MODERATE'; grade_col = '#f1c40f'
    elif score >= 55: grade = 'C  — LOW CONVICTION'; grade_col = '#e67e22'
    else:             grade = 'D  — SKIP'; grade_col = '#e74c3c'

    return {
        # Core
        'pattern_name':       name,
        'direction':          direction,
        'is_bull':            is_bull,
        'current_price':      current,
        'market_context':     market_context,
        # Probability
        'completion_prob':    completion_prob,
        'failure_rate':       failure_rate,
        'samples':            samples,
        'performance_rank':   perf_rank,
        # Targets
        'neckline':           neckline,
        'pattern_height':     pattern_height,
        'target_1':           target_1,
        'target_2':           target_2,
        'target_3':           target_3,
        'move_to_t1_pct':     round(move_to_t1, 1),
        'move_to_t2_pct':     round(move_to_t2, 1),
        'target_reliability': target_rel_num,
        # Throwback
        'throwback_prob':     throwback_prob,
        'throwback_target':   throwback_target,
        'throwback_days':     int(throwback_days),
        'throwback_action':   throwback_action,
        # Time
        'est_completion_days': est_days,
        # Failure
        'failure_scenarios':  failure_scenarios,
        # Best conditions
        'best_conditions':    best_conditions,
        # Grade
        'conviction_grade':   grade,
        'conviction_color':   grade_col,
        'conviction_score':   round(score),
        # Measure rule
        'measure_rule': db_match.get('measure_rule', '') if db_match else '',
    }



# ─────────────────────────────────────────────────────────────────────────────
#  BROOKS PA FORECAST ENGINE
#  Source: Trading Price Action Trends — Al Brooks (Wiley, 2012)
#  Computes context-adjusted signal quality score, targets, time rules,
#  failure signals, and position sizing guidance for every Brooks signal.
# ─────────────────────────────────────────────────────────────────────────────

# Base quality scores from Brooks text — how strongly he endorses each setup
BROOKS_BASE_SCORES = {
    'Breakout Pullback':           78,
    'High 1':                      62,
    'High 2':                      75,
    'Two-Bar Reversal':            68,
    'Wedge Reversal':              70,
    'Failed Breakout':             73,
    'Measured Move':               70,
    'MA Gap Bar':                  72,
    'Inside Bar':                  65,
    'ii Pattern':                  80,   # doubled energy
    'Final Flag':                  72,
    'Spike and Channel':           68,
    'Trend Line Break':            75,
    'Trend from the Open':         80,
    'Breakout Pullback Long':      78,
    'Breakout Pullback Short':     78,
    'Bull Trend':                  80,
    'Bear Trend':                  80,
    'Stoch Hook':                  70,
}


# ─── Main pattern detection entry point ─────────────────────────────
def detect_patterns(df):
    """
    Detect chart patterns from OHLCV dataframe.
    Returns list of dicts: {pattern_name, confidence, details, bar_indices}
    """
    detected = []
    n = len(df)
    if n < 20:
        return detected

    highs_idx, lows_idx = find_swing_highs_lows(df, order=max(3, n//15))
    closes = df['Close'].values
    highs  = df['High'].values
    lows   = df['Low'].values

    # ── HELPER: latest N swing lows/highs ──
    def last_n_lows(n_pts):
        return [(lows_idx[i], lows[lows_idx[i]]) for i in range(-n_pts, 0)] if len(lows_idx) >= n_pts else []

    def last_n_highs(n_pts):
        return [(highs_idx[i], highs[highs_idx[i]]) for i in range(-n_pts, 0)] if len(highs_idx) >= n_pts else []

    # ──────────────────────────────────────────────────────
    # 1. DOUBLE BOTTOM Detection
    # ──────────────────────────────────────────────────────
    if len(lows_idx) >= 2:
        pts = last_n_lows(2)
        if len(pts) == 2:
            idx1, v1 = pts[0]
            idx2, v2 = pts[1]
            price_diff = pct_diff(v1, v2)
            if price_diff <= 4.0 and (idx2 - idx1) >= 5:
                # Find peak between the two lows
                between_highs = highs[idx1:idx2+1]
                peak_val = between_highs.max() if len(between_highs) > 0 else 0
                neckline_rise = (peak_val - min(v1, v2)) / min(v1, v2) * 100
                if neckline_rise >= 8:
                    # Classify Adam vs Eve
                    # Simple heuristic: if low spans few bars = Adam (sharp); many bars = Eve (wide)
                    # Use range of the bottom (distance around the low point)
                    span1 = 1
                    for k in range(min(idx1, n-1), min(idx1+4, n)):
                        if lows[k] > v1 * 1.02: break
                        span1 += 1
                    span2 = 1
                    for k in range(min(idx2, n-1), min(idx2+4, n)):
                        if lows[k] > v2 * 1.02: break
                        span2 += 1
                    t1 = "Adam" if span1 <= 2 else "Eve"
                    t2 = "Adam" if span2 <= 2 else "Eve"
                    pattern_name = f"Double Bottom ({t1} & {t2})"
                    conf = 85 - price_diff * 3
                    conf = max(40, min(95, conf))
                    current_price = closes[-1]
                    neckline_price = peak_val
                    pattern_height = neckline_price - min(v1, v2)
                    target = neckline_price + pattern_height
                    stop   = min(v1, v2) * 0.99
                    detected.append({
                        "name": pattern_name,
                        "confidence": round(conf, 1),
                        "direction": "BULLISH",
                        "entry": f"Close above neckline: {neckline_price:.2f}",
                        "stop":   f"{stop:.2f} (below lower bottom)",
                        "target": f"{target:.2f} (pattern height projection)",
                        "details": f"Bottom 1: {v1:.2f} at bar {idx1}, Bottom 2: {v2:.2f} at bar {idx2}, Price diff: {price_diff:.1f}%, Neckline: {neckline_price:.2f}",
                        "bar_indices": [idx1, idx2],
                        "neckline": neckline_price,
                        "pattern_low": min(v1, v2),
                    })

    # ──────────────────────────────────────────────────────
    # 2. DOUBLE TOP Detection
    # ──────────────────────────────────────────────────────
    if len(highs_idx) >= 2:
        pts = last_n_highs(2)
        if len(pts) == 2:
            idx1, v1 = pts[0]
            idx2, v2 = pts[1]
            price_diff = pct_diff(v1, v2)
            if price_diff <= 4.0 and (idx2 - idx1) >= 5:
                between_lows = lows[idx1:idx2+1]
                valley_val = between_lows.min() if len(between_lows) > 0 else 0
                valley_drop = (max(v1, v2) - valley_val) / max(v1, v2) * 100
                if valley_drop >= 8:
                    span1 = 1
                    for k in range(min(idx1, n-1), min(idx1+4, n)):
                        if highs[k] < v1 * 0.98: break
                        span1 += 1
                    span2 = 1
                    for k in range(min(idx2, n-1), min(idx2+4, n)):
                        if highs[k] < v2 * 0.98: break
                        span2 += 1
                    t1 = "Adam" if span1 <= 2 else "Eve"
                    t2 = "Adam" if span2 <= 2 else "Eve"
                    pattern_name = f"Double Top ({t1} & {t2})"
                    conf = 82 - price_diff * 3
                    conf = max(40, min(93, conf))
                    neckline_price = valley_val
                    pattern_height = max(v1, v2) - neckline_price
                    target = neckline_price - pattern_height
                    stop   = max(v1, v2) * 1.01
                    detected.append({
                        "name": pattern_name,
                        "confidence": round(conf, 1),
                        "direction": "BEARISH",
                        "entry": f"Close below neckline: {neckline_price:.2f}",
                        "stop":   f"{stop:.2f} (above higher top)",
                        "target": f"{target:.2f} (pattern height projection)",
                        "details": f"Top 1: {v1:.2f} at bar {idx1}, Top 2: {v2:.2f} at bar {idx2}, Price diff: {price_diff:.1f}%, Neckline: {neckline_price:.2f}",
                        "bar_indices": [idx1, idx2],
                        "neckline": neckline_price,
                        "pattern_high": max(v1, v2),
                    })

    # ──────────────────────────────────────────────────────
    # 3. HEAD AND SHOULDERS TOP
    # ──────────────────────────────────────────────────────
    if len(highs_idx) >= 3:
        pts = last_n_highs(3)
        if len(pts) == 3:
            (i1, h1), (i2, h2), (i3, h3) = pts
            # Head must be higher than both shoulders
            if h2 > h1 and h2 > h3:
                shoulder_diff = pct_diff(h1, h3)
                if shoulder_diff <= 7 and (i2 - i1) >= 3 and (i3 - i2) >= 3:
                    # Find neckline: lows between shoulders
                    between1 = lows[i1:i2+1].min()
                    between2 = lows[i2:i3+1].min()
                    neckline  = min(between1, between2)
                    head_height = h2 - neckline
                    target = neckline - head_height
                    stop   = h3 * 1.005
                    conf = 80 - shoulder_diff * 2
                    conf = max(45, min(92, conf))
                    detected.append({
                        "name": "Head-and-Shoulders Top",
                        "confidence": round(conf, 1),
                        "direction": "BEARISH",
                        "entry": f"Close below neckline: {neckline:.2f}",
                        "stop":   f"{stop:.2f} (above right shoulder)",
                        "target": f"{target:.2f} (head-to-neckline projected down)",
                        "details": f"Left shoulder: {h1:.2f}, Head: {h2:.2f}, Right shoulder: {h3:.2f}, Neckline: {neckline:.2f}, Shoulder diff: {shoulder_diff:.1f}%",
                        "bar_indices": [i1, i2, i3],
                        "neckline": neckline,
                        "pattern_high": h2,
                    })

    # ──────────────────────────────────────────────────────
    # 4. HEAD AND SHOULDERS BOTTOM (Inverse H&S)
    # ──────────────────────────────────────────────────────
    if len(lows_idx) >= 3:
        pts = last_n_lows(3)
        if len(pts) == 3:
            (i1, l1), (i2, l2), (i3, l3) = pts
            if l2 < l1 and l2 < l3:
                shoulder_diff = pct_diff(l1, l3)
                if shoulder_diff <= 7 and (i2 - i1) >= 3 and (i3 - i2) >= 3:
                    between1 = highs[i1:i2+1].max()
                    between2 = highs[i2:i3+1].max()
                    neckline  = max(between1, between2)
                    head_depth = neckline - l2
                    target = neckline + head_depth
                    stop   = l3 * 0.995
                    conf = 80 - shoulder_diff * 2
                    conf = max(45, min(92, conf))
                    detected.append({
                        "name": "Head-and-Shoulders Bottom",
                        "confidence": round(conf, 1),
                        "direction": "BULLISH",
                        "entry": f"Close above neckline: {neckline:.2f}",
                        "stop":   f"{stop:.2f} (below right shoulder)",
                        "target": f"{target:.2f} (head-to-neckline projected up)",
                        "details": f"Left shoulder: {l1:.2f}, Head: {l2:.2f}, Right shoulder: {l3:.2f}, Neckline: {neckline:.2f}, Shoulder diff: {shoulder_diff:.1f}%",
                        "bar_indices": [i1, i2, i3],
                        "neckline": neckline,
                        "pattern_low": l2,
                    })

    # ──────────────────────────────────────────────────────
    # 5. ASCENDING TRIANGLE
    # ──────────────────────────────────────────────────────
    if len(highs_idx) >= 2 and len(lows_idx) >= 2:
        recent_highs_vals = [highs[i] for i in highs_idx[-4:]]
        recent_lows_idx   = lows_idx[-4:]
        recent_lows_vals  = [lows[i] for i in recent_lows_idx]
        if len(recent_highs_vals) >= 2 and len(recent_lows_vals) >= 2:
            flat_top = max(pct_diff(recent_highs_vals[i], recent_highs_vals[i+1]) for i in range(len(recent_highs_vals)-1)) < 3
            rising_bottom = all(recent_lows_vals[i] < recent_lows_vals[i+1] for i in range(len(recent_lows_vals)-1))
            if flat_top and rising_bottom:
                resistance = np.mean(recent_highs_vals)
                pattern_low = min(recent_lows_vals)
                height = resistance - pattern_low
                target = resistance + height
                stop   = min(recent_lows_vals[-2:]) * 0.99
                detected.append({
                    "name": "Ascending Triangle",
                    "confidence": 72.0,
                    "direction": "BULLISH (70% breakout upward)",
                    "entry": f"Close above flat resistance: {resistance:.2f}",
                    "stop":   f"{stop:.2f} (below recent higher low)",
                    "target": f"{target:.2f} (triangle height added to breakout)",
                    "details": f"Flat resistance at ~{resistance:.2f}, Rising lows from {pattern_low:.2f}",
                    "bar_indices": list(highs_idx[-3:]) + list(lows_idx[-3:]),
                    "neckline": resistance,
                    "pattern_low": pattern_low,
                })

    # ──────────────────────────────────────────────────────
    # 6. DESCENDING TRIANGLE
    # ──────────────────────────────────────────────────────
    if len(highs_idx) >= 2 and len(lows_idx) >= 2:
        recent_lows_vals  = [lows[i] for i in lows_idx[-4:]]
        recent_highs_vals = [highs[i] for i in highs_idx[-4:]]
        if len(recent_lows_vals) >= 2 and len(recent_highs_vals) >= 2:
            flat_bottom   = max(pct_diff(recent_lows_vals[i], recent_lows_vals[i+1]) for i in range(len(recent_lows_vals)-1)) < 3
            falling_top   = all(recent_highs_vals[i] > recent_highs_vals[i+1] for i in range(len(recent_highs_vals)-1))
            if flat_bottom and falling_top:
                support = np.mean(recent_lows_vals)
                pattern_high = max(recent_highs_vals)
                height = pattern_high - support
                target = support - height
                stop   = max(recent_highs_vals[-2:]) * 1.01
                detected.append({
                    "name": "Descending Triangle",
                    "confidence": 68.0,
                    "direction": "BEARISH (64% breakout downward)",
                    "entry": f"Close below flat support: {support:.2f}",
                    "stop":   f"{stop:.2f} (above recent lower high)",
                    "target": f"{target:.2f} (triangle height subtracted from breakout)",
                    "details": f"Flat support at ~{support:.2f}, Falling highs from {pattern_high:.2f}",
                    "bar_indices": list(lows_idx[-3:]) + list(highs_idx[-3:]),
                    "neckline": support,
                    "pattern_high": pattern_high,
                })

    # ──────────────────────────────────────────────────────
    # 7. SYMMETRICAL TRIANGLE
    # ──────────────────────────────────────────────────────
    if len(highs_idx) >= 2 and len(lows_idx) >= 2:
        rh = [highs[i] for i in highs_idx[-4:]]
        rl = [lows[i] for i in lows_idx[-4:]]
        if len(rh) >= 2 and len(rl) >= 2:
            falling_top  = all(rh[i] > rh[i+1] for i in range(len(rh)-1))
            rising_bottom = all(rl[i] < rl[i+1] for i in range(len(rl)-1))
            if falling_top and rising_bottom:
                upper_val = rh[-1]
                lower_val = rl[-1]
                apex_est = (upper_val + lower_val) / 2
                height = max(rh) - min(rl)
                # Determine likely direction from prior trend
                prior_trend_up = closes[max(0, n-20)] < closes[-1]
                direction = "BULLISH (prior uptrend)" if prior_trend_up else "BEARISH (prior downtrend)"
                target_up = upper_val + height * 0.6
                target_dn = lower_val - height * 0.6
                detected.append({
                    "name": "Symmetrical Triangle",
                    "confidence": 65.0,
                    "direction": direction,
                    "entry": f"Close above {upper_val:.2f} (upward) OR below {lower_val:.2f} (downward)",
                    "stop":   f"Inside the triangle (opposite trendline)",
                    "target": f"Up: {target_up:.2f} | Down: {target_dn:.2f} (triangle height from breakout)",
                    "details": f"Upper trendline falling from {max(rh):.2f} to {rh[-1]:.2f}, Lower trendline rising from {min(rl):.2f} to {rl[-1]:.2f}",
                    "bar_indices": list(highs_idx[-3:]) + list(lows_idx[-3:]),
                    "neckline": upper_val,
                    "pattern_low": lower_val,
                })

    # ──────────────────────────────────────────────────────
    # 8. CUP WITH HANDLE (Simple detection)
    # ──────────────────────────────────────────────────────
    if n >= 40:
        # Look for: prior high → decline → U-shape → recovery → small pullback → breakout
        window = closes[-40:]
        peak1 = window[:10].max()
        cup_low = window[5:30].min()
        cup_low_idx = window[5:30].argmin() + 5
        peak2 = window[25:35].max()
        handle_low = window[30:].min()
        peak1_idx_a = window[:10].argmax()
        # Cup depth check: 15-50%
        cup_depth_pct = (peak1 - cup_low) / peak1 * 100
        # Recovery check: peak2 close to peak1
        recovery_pct = pct_diff(peak1, peak2)
        handle_depth = (peak2 - handle_low) / peak2 * 100
        if (15 <= cup_depth_pct <= 60 and
            recovery_pct <= 8 and
            5 <= handle_depth <= 25 and
            cup_low_idx >= 5):
            target = peak2 + (peak2 - cup_low)  # Cup depth projected
            stop   = handle_low * 0.99
            conf = max(55, min(85, 80 - recovery_pct * 3 - abs(cup_depth_pct - 30) * 0.5))
            detected.append({
                "name": "Cup with Handle",
                "confidence": round(conf, 1),
                "direction": "BULLISH",
                "entry": f"Close above cup rim/lip: ~{peak2:.2f}",
                "stop":   f"{stop:.2f} (below handle low)",
                "target": f"{target:.2f} (cup depth projected from rim)",
                "details": f"Cup high: ~{peak1:.2f}, Cup low: ~{cup_low:.2f}, Depth: {cup_depth_pct:.1f}%, Handle drop: {handle_depth:.1f}%",
                "bar_indices": [max(0, n-40), max(0, n-40)+cup_low_idx, n-1],
                "neckline": peak2,
                "pattern_low": cup_low,
            })

    # ──────────────────────────────────────────────────────
    # 9. FLAG PATTERN (Bull)
    # ──────────────────────────────────────────────────────
    if n >= 20:
        pole_start = n - 20
        pole_end   = n - 10
        flag_start = n - 10

        pole_rise = (closes[pole_end-1] - closes[pole_start]) / closes[pole_start] * 100
        flag_high = highs[flag_start:].max()
        flag_low  = lows[flag_start:].min()
        flag_drop = (flag_high - flag_low) / flag_high * 100
        pole_height = closes[pole_end-1] - closes[pole_start]

        if pole_rise >= 8 and flag_drop <= 12:
            # Is the flag slightly declining?
            flag_closes = closes[flag_start:]
            if len(flag_closes) >= 3:
                flag_slope = np.polyfit(range(len(flag_closes)), flag_closes, 1)[0]
                if flag_slope < 0:  # Declining flag (correct)
                    target = flag_high + pole_height
                    stop   = flag_low * 0.99
                    conf = max(55, min(85, 75 + pole_rise * 0.3 - flag_drop))
                    detected.append({
                        "name": "Flag (Bull)",
                        "confidence": round(conf, 1),
                        "direction": "BULLISH",
                        "entry": f"Close above flag top: ~{flag_high:.2f}",
                        "stop":   f"{stop:.2f} (below flag low)",
                        "target": f"{target:.2f} (flagpole height added to flag low)",
                        "details": f"Pole rise: {pole_rise:.1f}%, Flag range: {flag_drop:.1f}%, Pole height: {pole_height:.2f}",
                        "bar_indices": [pole_start, pole_end, flag_start, n-1],
                        "neckline": flag_high,
                        "pattern_low": flag_low,
                    })

    # ──────────────────────────────────────────────────────
    # 10. TRIPLE BOTTOM
    # ──────────────────────────────────────────────────────
    if len(lows_idx) >= 3:
        pts = last_n_lows(3)
        if len(pts) == 3:
            (i1, l1), (i2, l2), (i3, l3) = pts
            max_diff = max(pct_diff(l1, l2), pct_diff(l2, l3), pct_diff(l1, l3))
            if max_diff <= 3 and (i2 - i1) >= 4 and (i3 - i2) >= 4:
                neckline = highs[i1:i3+1].max()
                pattern_height = neckline - min(l1, l2, l3)
                target = neckline + pattern_height
                stop   = min(l1, l2, l3) * 0.99
                conf = 85 - max_diff * 5
                conf = max(50, min(92, conf))
                detected.append({
                    "name": "Triple Bottom",
                    "confidence": round(conf, 1),
                    "direction": "BULLISH",
                    "entry": f"Close above neckline: {neckline:.2f}",
                    "stop":   f"{stop:.2f} (below lowest bottom)",
                    "target": f"{target:.2f} (pattern height added to neckline)",
                    "details": f"Three lows: {l1:.2f}, {l2:.2f}, {l3:.2f}. Max price diff: {max_diff:.1f}%. Neckline: {neckline:.2f}",
                    "bar_indices": [i1, i2, i3],
                    "neckline": neckline,
                    "pattern_low": min(l1, l2, l3),
                })

    # ──────────────────────────────────────────────────────
    # 11. TRIPLE TOP
    # ──────────────────────────────────────────────────────
    if len(highs_idx) >= 3:
        pts = last_n_highs(3)
        if len(pts) == 3:
            (i1, h1), (i2, h2), (i3, h3) = pts
            max_diff = max(pct_diff(h1, h2), pct_diff(h2, h3), pct_diff(h1, h3))
            if max_diff <= 3 and (i2 - i1) >= 4 and (i3 - i2) >= 4:
                neckline = lows[i1:i3+1].min()
                pattern_height = max(h1, h2, h3) - neckline
                target = neckline - pattern_height
                stop   = max(h1, h2, h3) * 1.01
                conf = 85 - max_diff * 5
                conf = max(50, min(92, conf))
                detected.append({
                    "name": "Triple Top",
                    "confidence": round(conf, 1),
                    "direction": "BEARISH",
                    "entry": f"Close below neckline: {neckline:.2f}",
                    "stop":   f"{stop:.2f} (above highest top)",
                    "target": f"{target:.2f} (pattern height subtracted from neckline)",
                    "details": f"Three highs: {h1:.2f}, {h2:.2f}, {h3:.2f}. Max price diff: {max_diff:.1f}%. Neckline: {neckline:.2f}",
                    "bar_indices": [i1, i2, i3],
                    "neckline": neckline,
                    "pattern_high": max(h1, h2, h3),
                })

    # ──────────────────────────────────────────────────────
    # 12. MEASURED MOVE UP / DOWN
    # ──────────────────────────────────────────────────────
    if len(highs_idx) >= 2 and len(lows_idx) >= 2:
        # Look for: swing low → swing high → correction low → potential leg 2
        if len(lows_idx) >= 2 and len(highs_idx) >= 1:
            l_start_idx = lows_idx[-2] if len(lows_idx) >= 2 else lows_idx[-1]
            h_peak_idx  = highs_idx[-1]
            l_corr_idx  = lows_idx[-1]

            if l_start_idx < h_peak_idx > l_corr_idx:
                leg1 = highs[h_peak_idx] - lows[l_start_idx]
                corr = (highs[h_peak_idx] - lows[l_corr_idx]) / highs[h_peak_idx] * 100
                if 25 <= corr <= 65 and leg1 > 0:
                    target = lows[l_corr_idx] + leg1
                    stop   = lows[l_corr_idx] * 0.99
                    detected.append({
                        "name": "Measured Move Up",
                        "confidence": 70.0,
                        "direction": "BULLISH",
                        "entry": f"Enter as Leg 2 begins. Buy around correction low: {lows[l_corr_idx]:.2f}",
                        "stop":   f"{stop:.2f} (below correction low)",
                        "target": f"{target:.2f} (Leg 1 distance = Leg 2 distance: {leg1:.2f} pts)",
                        "details": f"Leg 1: {lows[l_start_idx]:.2f} → {highs[h_peak_idx]:.2f} ({leg1:.2f} pts). Correction: {corr:.1f}%. Target: correction low + {leg1:.2f}",
                        "bar_indices": [l_start_idx, h_peak_idx, l_corr_idx],
                        "neckline": highs[h_peak_idx],
                        "pattern_low": lows[l_corr_idx],
                    })

    # Sort by confidence
    detected.sort(key=lambda x: x['confidence'], reverse=True)

    # Add completion status to every pattern
    for p in detected:
        p['completion'] = get_completion_status(p, df)

    return detected


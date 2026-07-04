"""
metrics.py — expected-value metrics for the one-free-throw rule.

Turns the parsed FT fact table into rule-impact metrics at any grain. The engine
(aggregate_ev) is grain-agnostic: pass group keys (player, team, home/away, ...) and
it returns the same EV columns for that grain. Player level is the headline; other
grains come free once the dim tables land (Phase B).

Modeling notes (aligned with Conner):
- Rates are PER TRIP TYPE: EV2 uses the first-shot rate within 2-shot trips, EV3 within
  3-shot trips. FT1Pct_Blended (shot 1 pooled across both) is stored for downstream /
  low-volume use, never silently substituted.
- Rates and EV are NON-ADDITIVE, so we sum additive measures (makes, attempts, trips)
  at the grain first, then derive rates and EV. That is what makes any-grain slicing valid.
- EV deltas are written explicitly as (New - Current), not the reduced FT1-FT2 form.
- Nothing is imputed or dropped silently: missing trip types stay NaN in the rate columns
  and contribute 0 to point totals; low-volume groups are flagged, not removed.
- Built-in check: CurrentTotalPts (expected) must equal ActualCurrentPts (observed makes),
  because rates are estimated from the same trips we scale over.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

from parse import load_pbp, parse_free_throws   # parse.py is in the same dir when run as a script

PROCESSED_DIR = Path("data/processed")
DEFAULT_MIN_TRIPS = 25   # tunable: below this a group's rates are noisy -> LowVolume flag

FINAL_COLS_TAIL = [
    "Trips2Shot", "Trips3Shot", "TotalTrips", "VolumePctile", "LowVolume",
    "FT1Pct_2Shots", "FT2Pct_2Shots", "FT1Pct_3Shots", "FT2Pct_3Shots", "FT3Pct_3Shots",
    "FT1Pct_Blended", "FT1Pct_TripTypeGap",
    "CurrentEV2", "NewEV2", "DeltaEV2", "CurrentEV3", "NewEV3", "DeltaEV3",
    "ActualCurrentPts", "CurrentTotalPts", "NewTotalPts", "DeltaTotalPts",
    "NewTotalPts_Blended", "DeltaTotalPts_Blended",
]


def _convertible(fact: pd.DataFrame) -> pd.DataFrame:
    """The rule-convertible FT set: standard, complete 2/3-shot trips outside the clutch window."""
    keep = (fact["IsStandard"] & fact["TripLen"].isin([2, 3])
            & fact["RuleApplies"] & fact["TripComplete"])
    return fact[keep]


def _shot_rate(conv: pd.DataFrame, keys, triplen: int, shotnum: int, name: str) -> pd.DataFrame:
    """make rate (+ attempt count) for one shot position within one trip type."""
    sub = conv[(conv["TripLen"] == triplen) & (conv["ShotNum"] == shotnum)]
    return (sub.groupby(keys, dropna=False)["IsMade"]
               .agg([(name, "mean"), (f"{name}_n", "size")]))


def aggregate_ev(fact: pd.DataFrame, group_keys, min_trips: int = DEFAULT_MIN_TRIPS) -> pd.DataFrame:
    """EV metrics for the rule change at an arbitrary grain.

    group_keys: column(s) in fact to aggregate by, e.g. ["PersonId", "PlayerName"] or ["TeamId"].
    Returns one row per group with per-type rates, per-trip EV, scaled point totals, volume flags.
    """
    if isinstance(group_keys, str):
        group_keys = [group_keys]
    conv = _convertible(fact)

    # Per-trip-type shot rates. The attempt count of shot 1 doubles as the trip count.
    parts = [
        _shot_rate(conv, group_keys, 2, 1, "FT1Pct_2Shots"),
        _shot_rate(conv, group_keys, 2, 2, "FT2Pct_2Shots"),
        _shot_rate(conv, group_keys, 3, 1, "FT1Pct_3Shots"),
        _shot_rate(conv, group_keys, 3, 2, "FT2Pct_3Shots"),
        _shot_rate(conv, group_keys, 3, 3, "FT3Pct_3Shots"),
    ]
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, how="outer")

    # Blended first-shot rate: shot 1 pooled across both trip types (stored, not auto-used).
    out["FT1Pct_Blended"] = conv[conv["ShotNum"] == 1].groupby(group_keys, dropna=False)["IsMade"].mean()

    # Observed current-rule points = every made FT (1 pt each) over convertible trips.
    out["ActualCurrentPts"] = conv.groupby(group_keys, dropna=False)["IsMade"].sum().astype(float)

    out = out.reset_index()

    # Trip counts fall out of the shot-1 attempt counts.
    out["Trips2Shot"] = out["FT1Pct_2Shots_n"].fillna(0).astype(int)
    out["Trips3Shot"] = out["FT1Pct_3Shots_n"].fillna(0).astype(int)
    out["TotalTrips"] = out["Trips2Shot"] + out["Trips3Shot"]

    # Per-trip EV, explicit (New - Current). NaN where a group lacks that trip type.
    out["CurrentEV2"] = out["FT1Pct_2Shots"] + out["FT2Pct_2Shots"]
    out["NewEV2"] = 2 * out["FT1Pct_2Shots"]
    out["DeltaEV2"] = out["NewEV2"] - out["CurrentEV2"]

    out["CurrentEV3"] = out["FT1Pct_3Shots"] + out["FT2Pct_3Shots"] + out["FT3Pct_3Shots"]
    out["NewEV3"] = 3 * out["FT1Pct_3Shots"]
    out["DeltaEV3"] = out["NewEV3"] - out["CurrentEV3"]

    out["FT1Pct_TripTypeGap"] = out["FT1Pct_2Shots"] - out["FT1Pct_3Shots"]

    # Scale by trip volume; a missing trip type contributes 0, never NaN.
    def scaled(trips_col, ev):
        return (out[trips_col] * ev).fillna(0.0)

    out["CurrentTotalPts"] = scaled("Trips2Shot", out["CurrentEV2"]) + scaled("Trips3Shot", out["CurrentEV3"])
    out["NewTotalPts"] = scaled("Trips2Shot", out["NewEV2"]) + scaled("Trips3Shot", out["NewEV3"])
    out["DeltaTotalPts"] = out["NewTotalPts"] - out["CurrentTotalPts"]

    # Blended-rate variant of the new rule (for the low-volume / who-to-foul angle).
    b = out["FT1Pct_Blended"]
    out["NewTotalPts_Blended"] = scaled("Trips2Shot", 2 * b) + scaled("Trips3Shot", 3 * b)
    out["DeltaTotalPts_Blended"] = out["NewTotalPts_Blended"] - out["CurrentTotalPts"]

    # Volume: rank within the pool, flag the thin-sample groups (labeled, not dropped).
    out["VolumePctile"] = out["TotalTrips"].rank(pct=True)
    out["LowVolume"] = out["TotalTrips"] < min_trips

    # Validation: expected current points must equal observed made FTs.
    assert np.allclose(out["CurrentTotalPts"], out["ActualCurrentPts"], atol=1e-6), \
        "CurrentTotalPts must equal ActualCurrentPts (rate math is off)"

    return out[group_keys + FINAL_COLS_TAIL].reset_index(drop=True)


if __name__ == "__main__":
    # Windows console defaults to cp1252, which can't encode names like Dončić/Jokić.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    fact = parse_free_throws(load_pbp())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fact.to_parquet(PROCESSED_DIR / "fact_ft.parquet", index=False)

    players = aggregate_ev(fact, ["PersonId", "PlayerName"])
    players.to_parquet(PROCESSED_DIR / "player_ft_metrics.parquet", index=False)

    print(f"fact_ft rows: {len(fact)} | players: {len(players)}")
    print(f"LowVolume players (< {DEFAULT_MIN_TRIPS} trips): {int(players['LowVolume'].sum())} / {len(players)}")

    # In a 20-game sample everyone is low-volume; fall back to the full set so the demo shows rows.
    sig = players[~players["LowVolume"]]
    if sig.empty:
        sig = players
    cols = ["PlayerName", "Trips2Shot", "FT1Pct_2Shots", "FT2Pct_2Shots", "DeltaEV2", "DeltaTotalPts"]
    print("\nMost HURT by the rule (lowest DeltaTotalPts):")
    print(sig.nsmallest(8, "DeltaTotalPts")[cols].to_string(index=False))
    print("\nMost HELPED by the rule (highest DeltaTotalPts):")
    print(sig.nlargest(8, "DeltaTotalPts")[cols].to_string(index=False))

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
from scrape import get_player_shooting, get_player_rim_shooting   # cached FG% pulls (API)

PROCESSED_DIR = Path("data/processed")
DEFAULT_MIN_TRIPS = 25   # tunable: below this a group's rates are noisy -> LowVolume flag
PTS_PER_WIN = 30.5       # points per win, Oliver (2004) Pythagorean expectation

# A convertible trip is uniquely keyed by game + player + period + dead-ball clock + trip length.
TRIP_KEYS = ["GameId", "PersonId", "Period", "ClockSeconds", "TripLen"]

# Per-trip outcome columns add_trip_outcomes() writes onto each trip's first-shot row.
OUTCOME_COLS = ["TripValue", "FT1Made", "NewPts", "CurrentPts",
                "ValueLost", "ValueGained", "TripDelta"]

FINAL_COLS_TAIL = [
    "Trips2Shot", "Trips3Shot", "TotalTrips", "VolumePctile", "LowVolume",
    "FT1Pct_2Shots", "FT2Pct_2Shots", "FT1Pct_3Shots", "FT2Pct_3Shots", "FT3Pct_3Shots",
    "FT1Pct_Blended", "FT1Pct_TripTypeGap",
    "CurrentEV2", "NewEV2", "DeltaEV2", "CurrentEV3", "NewEV3", "DeltaEV3",
    "ActualCurrentPts", "CurrentTotalPts", "NewTotalPts", "DeltaTotalPts",
    "NewTotalPts_Blended", "DeltaTotalPts_Blended",
]

OUTCOME_FINAL_TAIL = [
    "Trips2Shot", "Trips3Shot", "TotalTrips",
    "TotalValueLost", "TotalValueGained", "TotalPtsSalvaged",
    "ActualCurrentPts", "TrueNet", "TrueNetRate", "Winpact",
]

# FT split rates joined from aggregate_ev onto player_outcomes (everything in one place).
FT_SPLIT_RATE_COLS = ["FT1Pct_2Shots", "FT2Pct_2Shots", "FT1Pct_3Shots",
                      "FT2Pct_3Shots", "FT3Pct_3Shots", "FT1Pct_Blended"]


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


def add_trip_outcomes(fact: pd.DataFrame) -> pd.DataFrame:
    """Attach per-trip rule-outcome columns to the fact table.

    One convertible trip = one first-shot row, so every trip-level outcome is carried on that
    ShotNum==1 row and left 0/False on the trip's other shot rows (and on all non-convertible
    rows). That placement is deliberate: the columns then sum correctly at ANY grain without
    double-counting a trip's 2-3 shot rows.

    Per convertible trip:
      TripValue    = TripLen (2 or 3): points a single made FT is worth under the new rule.
      FT1Made      = was the first free throw made.
      NewPts       = TripValue if FT1 made else 0 (new-rule points).
      CurrentPts   = points actually scored in the trip = makes at 1 pt each.
      ValueLost    = TripValue if FT1 MISSED else 0 (points forfeited by the miss).
      ValueGained  = TripValue if FT1 made else 0 (== NewPts, kept for readability).
      TripDelta    = NewPts - CurrentPts (per-trip gain/loss from the rule change).
    ValueLost and ValueGained are mutually exclusive: exactly one is nonzero per trip.
    """
    out = fact.copy()
    for c in OUTCOME_COLS:
        out[c] = 0.0
    out["FT1Made"] = False

    conv = (out["IsStandard"] & out["TripLen"].isin([2, 3])
            & out["RuleApplies"] & out["TripComplete"])

    # Trip's current points (1 pt per make) broadcast across its shot rows, then read on shot 1.
    out["_TripCurr"] = 0.0
    out.loc[conv, "_TripCurr"] = (out.loc[conv]
                                  .groupby(TRIP_KEYS, dropna=False)["IsMade"].transform("sum"))

    s1 = conv & (out["ShotNum"] == 1)
    tv = out.loc[s1, "TripLen"].astype(float)
    made1 = out.loc[s1, "IsMade"].astype(bool)

    out.loc[s1, "TripValue"] = tv
    out.loc[s1, "FT1Made"] = made1
    out.loc[s1, "CurrentPts"] = out.loc[s1, "_TripCurr"].astype(float)
    out.loc[s1, "NewPts"] = tv * made1.astype(float)
    out.loc[s1, "ValueGained"] = out.loc[s1, "NewPts"]
    out.loc[s1, "ValueLost"] = tv * (~made1).astype(float)
    out.loc[s1, "TripDelta"] = out.loc[s1, "NewPts"] - out.loc[s1, "CurrentPts"]
    out.drop(columns="_TripCurr", inplace=True)

    # Guards: exactly one of ValueLost/ValueGained is nonzero per trip, and ValueGained == NewPts.
    tr = out.loc[s1]
    assert (tr["ValueLost"] * tr["ValueGained"] == 0).all(), \
        "ValueLost and ValueGained must be mutually exclusive per trip"
    assert np.allclose(tr["ValueGained"], tr["NewPts"]), "ValueGained must equal NewPts"
    return out


def aggregate_outcomes(fact: pd.DataFrame, group_keys) -> pd.DataFrame:
    """Outcome-based rule metrics at an arbitrary grain (same group_keys pattern as aggregate_ev).

    Sums the per-trip outcome columns (from add_trip_outcomes) over each group, working off the
    first-shot rows where those columns live.

      TotalValueLost   = points at stake from all FT1 misses (sum ValueLost).
      TotalValueGained = new-rule points (sum ValueGained = sum NewPts).
      TotalPtsSalvaged = points scored on FT2/FT3 on trips where FT1 was MISSED — the second-
                         chance points the current rule allows that vanish under the new rule.
                         On an FT1-miss trip every made FT is a later shot, so this is CurrentPts
                         summed over FT1-missed trips.
      ActualCurrentPts = points actually scored (sum CurrentPts).
      TrueNet          = TotalValueGained - ActualCurrentPts (actual new-vs-current difference).
                         Equals aggregate_ev's DeltaTotalPts by construction (independent path).
      Winpact          = TrueNet / PTS_PER_WIN (wins translation, Oliver 2004 Pythagorean).
    """
    if isinstance(group_keys, str):
        group_keys = [group_keys]
    enriched = fact if "TripDelta" in fact.columns else add_trip_outcomes(fact)

    conv = (enriched["IsStandard"] & enriched["TripLen"].isin([2, 3])
            & enriched["RuleApplies"] & enriched["TripComplete"])
    trips = enriched[conv & (enriched["ShotNum"] == 1)]

    g = trips.groupby(group_keys, dropna=False)
    out = g[["ValueLost", "ValueGained", "CurrentPts"]].sum()
    out.columns = ["TotalValueLost", "TotalValueGained", "ActualCurrentPts"]

    out["Trips2Shot"] = trips[trips["TripLen"] == 2].groupby(group_keys, dropna=False).size()
    out["Trips3Shot"] = trips[trips["TripLen"] == 3].groupby(group_keys, dropna=False).size()
    out[["Trips2Shot", "Trips3Shot"]] = out[["Trips2Shot", "Trips3Shot"]].fillna(0).astype(int)
    out["TotalTrips"] = out["Trips2Shot"] + out["Trips3Shot"]

    # Salvaged = current-rule points on trips where FT1 was missed (all from later shots).
    salv = trips[~trips["FT1Made"]].groupby(group_keys, dropna=False)["CurrentPts"].sum()
    out["TotalPtsSalvaged"] = salv
    out["TotalPtsSalvaged"] = out["TotalPtsSalvaged"].fillna(0.0)

    out["TrueNet"] = out["TotalValueGained"] - out["ActualCurrentPts"]
    # Per-trip TrueNet (RATE metric): floor by volume only when SORTING by it, never to define it.
    out["TrueNetRate"] = out["TrueNet"] / out["TotalTrips"].replace(0, np.nan)
    out["Winpact"] = out["TrueNet"] / PTS_PER_WIN

    out = out.reset_index()
    return out[group_keys + OUTCOME_FINAL_TAIL].reset_index(drop=True)


def player_name_map(pbp: pd.DataFrame) -> pd.Series:
    """PersonId -> playerNameI ("S. Gilgeous-Alexander") from raw play-by-play.

    V3's playerName is a bare surname that collides across players (multiple Johnsons /
    Jacksons); playerNameI adds the first initial to disambiguate. It already lives in the
    cached pbp, so no extra API call is needed. Returned indexed by PersonId (int).
    """
    m = pbp.loc[pbp["playerNameI"].notna() & (pbp["playerNameI"].astype(str) != ""),
                ["personId", "playerNameI"]].drop_duplicates("personId")
    return m.set_index("personId")["playerNameI"]


def attach_display_name(df: pd.DataFrame, namei: pd.Series) -> pd.DataFrame:
    """Insert a PlayerNameI display column right after PlayerName (falls back to surname)."""
    df = df.copy()
    disp = df["PersonId"].map(namei).fillna(df["PlayerName"])
    df.insert(df.columns.get_loc("PlayerName") + 1, "PlayerNameI", disp)
    return df


def _hack_category(old_shaq: pd.Series, new_shaq: pd.Series, suffix: str = "") -> pd.Series:
    """Classify each player by how the rule change moves them across the hack-a-Shaq threshold.

    Old (current FT rule) and new (one-FT rule) ShaqScore signs give four buckets; NoLonger
    (old target, new safe) can only happen when FT1 > FT2. Computed WITHIN one scope (2FG or
    Rim) — never mix scopes. suffix tags the scope in the label (e.g. '_Rim'); missing inputs
    -> 'Unknown'.
    """
    cat = pd.Series(pd.NA, index=old_shaq.index, dtype="object")
    valid = old_shaq.notna() & new_shaq.notna()
    cat[valid & (old_shaq > 0) & (new_shaq > 0)] = "AlwaysHackable" + suffix
    cat[valid & (old_shaq <= 0) & (new_shaq > 0)] = "NewlyHackable" + suffix
    cat[valid & (old_shaq > 0) & (new_shaq <= 0)] = "NoLongerHackable" + suffix
    cat[valid & (old_shaq <= 0) & (new_shaq <= 0)] = "NeverHackable" + suffix
    cat[~valid] = "Unknown"
    return cat


if __name__ == "__main__":
    # Windows console defaults to cp1252, which can't encode names like Dončić/Jokić.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    raw = load_pbp()
    fact = add_trip_outcomes(parse_free_throws(raw))

    # Disambiguating display name (playerNameI) carried onto every persisted output.
    namei = player_name_map(raw)
    fact["PlayerNameI"] = fact["PersonId"].map(namei).fillna(fact["PlayerName"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fact.to_parquet(PROCESSED_DIR / "fact_ft.parquet", index=False)

    players = attach_display_name(aggregate_ev(fact, ["PersonId", "PlayerName"]), namei)
    players.to_parquet(PROCESSED_DIR / "player_ft_metrics.parquet", index=False)

    outcomes = attach_display_name(aggregate_outcomes(fact, ["PersonId", "PlayerName"]), namei)
    # Join the FT split rates from player_ft_metrics so player_outcomes is one-stop.
    outcomes = outcomes.merge(players[["PersonId"] + FT_SPLIT_RATE_COLS], on="PersonId", how="left")
    # Join 2PT/3PT FG% (one cached API call) for the hack-a-Shaq analysis.
    shooting = get_player_shooting("2025-26")
    outcomes = outcomes.merge(shooting[["PersonId", "TwoPT_FGPct", "ThreePT_FGPct"]],
                              on="PersonId", how="left")
    # ShaqScore = 2FG_EV - NewFT_EV = 2*(2PT% - FT1Pct_2Shots): points the defense gains by
    # fouling under the one-FT rule. Positive -> foul them; sort desc for the biggest targets.
    outcomes["ShaqScore"] = 2 * outcomes["TwoPT_FGPct"] - 2 * outcomes["FT1Pct_2Shots"]
    # OldShaqScore_2FG = 2FG_EV - CurrentFT_EV: whether fouling beat a two under the CURRENT rule.
    outcomes["OldShaqScore_2FG"] = (2 * outcomes["TwoPT_FGPct"]
                                    - (outcomes["FT1Pct_2Shots"] + outcomes["FT2Pct_2Shots"]))
    outcomes["HackCategory_2FG"] = _hack_category(outcomes["OldShaqScore_2FG"], outcomes["ShaqScore"])

    # Rim (Restricted Area) variant: a center's real alternative to a foul is a dunk/layup, so
    # RA FG% is a truer "let them shoot" value than blended 2PT%. Keep both for comparison.
    rim = get_player_rim_shooting("2025-26")
    outcomes = outcomes.merge(rim[["PersonId", "RimFGPct", "RimFGA", "TotalFGA"]],
                              on="PersonId", how="left")
    outcomes["RimFG_EV"] = 2 * outcomes["RimFGPct"]
    outcomes["ShaqScore_2FG"] = outcomes["ShaqScore"]                       # 2PT%-based (alias)
    outcomes["ShaqScore_Rim"] = outcomes["RimFG_EV"] - 2 * outcomes["FT1Pct_2Shots"]
    outcomes["OldShaqScore_Rim"] = (2 * outcomes["RimFGPct"]
                                    - (outcomes["FT1Pct_2Shots"] + outcomes["FT2Pct_2Shots"]))
    outcomes["HackCategory_Rim"] = _hack_category(outcomes["OldShaqScore_Rim"],
                                                  outcomes["ShaqScore_Rim"], suffix="_Rim")
    # Rim-attempt share gates ShaqScore_Rim to rim-dependent players (a center's alternative to
    # a foul really is a rim shot); perimeter finishers with high RimFGPct but few rim attempts
    # are excluded at display time (RimFGA_Share >= 0.40).
    outcomes["RimFGA_Share"] = outcomes["RimFGA"] / outcomes["TotalFGA"].where(outcomes["TotalFGA"] > 0)
    outcomes = outcomes.drop(columns=["RimFGA", "TotalFGA"])
    outcomes.to_parquet(PROCESSED_DIR / "player_outcomes.parquet", index=False)

    print(f"fact_ft rows: {len(fact)} | players: {len(players)} | outcomes: {len(outcomes)}")
    print(f"LowVolume players (< {DEFAULT_MIN_TRIPS} trips): {int(players['LowVolume'].sum())} / {len(players)}")

    # Cross-check: outcome-based TrueNet must match the rate-based DeltaTotalPts (independent paths).
    chk = players.merge(outcomes[["PersonId", "TrueNet"]], on="PersonId")
    r = chk["TrueNet"].corr(chk["DeltaTotalPts"])
    max_abs_diff = (chk["TrueNet"] - chk["DeltaTotalPts"]).abs().max()
    print(f"corr(TrueNet, DeltaTotalPts) = {r:.6f} | max |diff| = {max_abs_diff:.6g}")

    # Outcome metrics are observed facts -> no volume floor (show all players).
    print("\nMost HURT by the rule (lowest TrueNet, all players):")
    ocol = ["PlayerNameI", "Trips2Shot", "Trips3Shot", "TotalPtsSalvaged", "TrueNet"]
    print(outcomes.nsmallest(8, "TrueNet")[ocol].to_string(index=False))
    print("\nMost HELPED by the rule (highest TrueNet, all players):")
    print(outcomes.nlargest(8, "TrueNet")[ocol].to_string(index=False))

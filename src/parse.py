"""
parse.py — bucket and filter free-throw events from cached PlayByPlayV3.

Turns raw play-by-play into one tidy row per free-throw attempt, with shot number,
trip length, make/miss, and the flags metrics.py needs. No make-rate or EV math here
(that lives in metrics.py) — this layer only buckets and filters.

Data quirks this file handles (all confirmed against the 2025-26 cache):
- FT events are actionType == "Free Throw".
- Shot number & trip length come from subType ("Free Throw 1 of 2"), NOT the description.
- shotResult is blank for FTs, so make/miss comes from the "MISS" prefix in description.
- subType labels the exclusions explicitly: "Free Throw Technical", "Free Throw Flagrant
  X of Y", and-1/single "Free Throw 1 of 1".
- Standard convertible FTs match exactly "Free Throw N of M" — we whitelist that pattern so
  technicals, flagrants, and unseen exotics (clear-path, away-from-play) all drop out cleanly.
"""

from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/pbp")

# subType is EXACTLY "Free Throw N of M" for standard shooting/bonus FTs. Anything with an
# extra descriptor word (Technical, Flagrant, Clear Path, ...) fails this and is excluded.
STANDARD_SUBTYPE = re.compile(r"^Free Throw (\d+) of (\d+)$")

# V3 clock is an ISO-8601 duration: "PT03M26.00S" = 3 min 26.00 s remaining in the period.
CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")

CLUTCH_SECONDS = 120   # last 2:00 of Q4 keeps the old multi-shot rule
Q4 = 4                 # period numbering: 1-4 regulation, 5+ overtime


def load_pbp(raw_dir: str | Path = RAW_DIR) -> pd.DataFrame:
    """Concat every cached game's play-by-play into one DataFrame."""
    files = sorted(glob.glob(str(Path(raw_dir) / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No cached play-by-play found in {raw_dir}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def _clock_seconds(clock: str) -> float:
    """Parse a V3 clock string ("PT03M26.00S") to seconds remaining in the period."""
    m = CLOCK_RE.match(str(clock))
    if not m:
        return np.nan
    return int(m.group(1)) * 60 + float(m.group(2))


def parse_free_throws(pbp: pd.DataFrame) -> pd.DataFrame:
    """One tidy row per free-throw attempt, with shot/trip/make-miss and rule flags.

    Keeps every FT event (nothing dropped) but labels each so the exclusions are
    auditable. Use analysis_set() to get the convertible subset for metrics.py.
    """
    ft = pbp[pbp["actionType"] == "Free Throw"].copy()

    # Standard "Free Throw N of M" only; ShotNum/TripLen stay NaN for tech/flagrant/etc.
    parts = ft["subType"].str.extract(STANDARD_SUBTYPE)
    ft["IsStandard"] = parts[0].notna()
    ft["ShotNum"] = pd.to_numeric(parts[0], errors="coerce").astype("Int64")
    ft["TripLen"] = pd.to_numeric(parts[1], errors="coerce").astype("Int64")

    # Make/miss: shotResult is blank for FTs, so read the "MISS" prefix. Fail loud if the
    # make/miss split ever stops accounting for every row (e.g. a new description format).
    ft["IsMade"] = ~ft["description"].astype(str).str.startswith("MISS")
    made = int(ft["IsMade"].sum())
    missed = int((~ft["IsMade"]).sum())
    assert made + missed == len(ft), f"make/miss split lost rows: {made}+{missed} != {len(ft)}"

    ft["ClockSeconds"] = ft["clock"].map(_clock_seconds)

    # The rule change does NOT apply in the last 2:00 of Q4 or in OT (old multi-shot stays).
    in_clutch = (ft["period"] == Q4) & (ft["ClockSeconds"] <= CLUTCH_SECONDS)
    in_ot = ft["period"] >= Q4 + 1
    ft["RuleApplies"] = ~(in_clutch | in_ot)

    # Trip-completeness: a trip is (game, player, period, dead-ball clock, trip length); both
    # FTs of a trip share a stopped clock, so they group cleanly. A trip is complete only if it
    # holds every shot 1..TripLen. Injury substitutions (two personIds split one trip) and raw
    # PBP gaps leave orphan shots that can't be valued — the current-rule EV needs all shots of
    # a trip — so we flag and later drop them (~0.1% of trips).
    ft["TripComplete"] = False
    std = ft["IsStandard"]
    trip_keys = ["gameId", "personId", "period", "ClockSeconds", "TripLen"]
    n_present = ft.loc[std].groupby(trip_keys, dropna=False)["ShotNum"].transform("nunique")
    ft.loc[std, "TripComplete"] = (n_present == ft.loc[std, "TripLen"])

    out = ft.rename(columns={
        "gameId": "GameId",
        "personId": "PersonId",
        "playerName": "PlayerName",
        "teamId": "TeamId",
        "teamTricode": "TeamTricode",
        "period": "Period",
        "subType": "SubType",
        "description": "Description",
    })
    cols = ["GameId", "PersonId", "PlayerName", "TeamId", "TeamTricode", "Period", "ClockSeconds",
            "ShotNum", "TripLen", "IsMade", "IsStandard", "RuleApplies", "TripComplete",
            "SubType", "Description"]
    return out[cols].reset_index(drop=True)


def analysis_set(ft: pd.DataFrame) -> pd.DataFrame:
    """The convertible FT set metrics.py operates on.

    Standard 2- and 3-shot trips only, outside the clutch window. Drops and-1 "1 of 1",
    technicals, flagrants, last-2:00-Q4 / OT trips, and positionally incomplete trips.
    """
    keep = (ft["IsStandard"] & ft["TripLen"].isin([2, 3])
            & ft["RuleApplies"] & ft["TripComplete"])
    return ft[keep].reset_index(drop=True)


if __name__ == "__main__":
    pbp = load_pbp()
    ft = parse_free_throws(pbp)

    print(f"Total FT rows: {len(ft)}")
    print(f"  makes/misses: {int(ft['IsMade'].sum())} / {int((~ft['IsMade']).sum())}")
    print("\nsubType breakdown:")
    print(ft["SubType"].value_counts().to_string())

    tech_flag = (~ft["IsStandard"]).sum()
    and1 = (ft["IsStandard"] & (ft["TripLen"] == 1)).sum()
    clutch = (ft["IsStandard"] & ft["TripLen"].isin([2, 3]) & ~ft["RuleApplies"]).sum()
    incomplete = (ft["IsStandard"] & ft["TripLen"].isin([2, 3]) & ft["RuleApplies"]
                  & ~ft["TripComplete"]).sum()
    a = analysis_set(ft)
    print(f"\nExcluded - technical/flagrant: {tech_flag} | and-1 (1 of 1): {and1} | "
          f"clutch Q4/OT: {clutch} | incomplete trips: {incomplete}")
    print(f"Analysis set: {len(a)} rows")
    print("  TripLen breakdown:", a["TripLen"].value_counts().to_dict())

    # Sanity: every 2-shot trip has exactly one "1 of 2" and one "2 of 2".
    two = a[a["TripLen"] == 2]
    s1 = int((two["ShotNum"] == 1).sum())
    s2 = int((two["ShotNum"] == 2).sum())
    print(f"  2-shot consistency: 1-of-2 count={s1}, 2-of-2 count={s2}, match={s1 == s2}")

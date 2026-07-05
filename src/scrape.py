"""
scrape.py — pull and cache NBA play-by-play for the FT rule analysis.

stats.nba.com throttles hard, so we cache every game to disk and hit the
API at most once per game, ever. An interrupted run resumes from the cache.
Free-throw events are actionType == "Free Throw"; the shot number and trip type
("Free Throw 1 of 2") live in the subType column and get parsed downstream in
parse.py. (PlayByPlayV2 was retired mid-2025 and now returns empty JSON, so we
use PlayByPlayV3, which has a different, richer schema.)
"""

from pathlib import Path
import time
import pandas as pd
from nba_api.stats.endpoints import (leaguegamelog, playbyplayv3, leaguedashplayerstats,
                                     leaguedashplayershotlocations)

RAW_DIR = Path("data/raw/pbp")
GAME_LIST_DIR = Path("data/raw")
REQUEST_SLEEP = 0.6      # seconds between real API calls
API_TIMEOUT = 60         # stats.nba.com hangs/cold-starts slowly without a generous timeout
MAX_RETRIES = 3          # attempts per game before giving up
BACKOFF_BASE = 2         # seconds; exponential backoff = BACKOFF_BASE * 2**(attempt-1) → 2s, 4s


def _game_list_path(season: str) -> Path:
    """Season-key the game-list cache so seasons don't collide on one filename."""
    return GAME_LIST_DIR / f"game_ids_{season}.parquet"


def get_game_ids(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Return unique game IDs for a season, cached to disk (keyed by season)."""
    cache = _game_list_path(season)
    if cache.exists():
        return pd.read_parquet(cache)

    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        timeout=API_TIMEOUT,
    ).get_data_frames()[0]

    # One row per team per game → dedup to one row per GAME_ID.
    games = (log[["GAME_ID", "GAME_DATE"]]
             .drop_duplicates("GAME_ID")
             .sort_values("GAME_DATE")
             .reset_index(drop=True))

    cache.parent.mkdir(parents=True, exist_ok=True)
    games.to_parquet(cache, index=False)
    return games


def get_player_shooting(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Return PersonId -> 2PT and 3PT FG% for a season, cached to disk (one API call).

    leaguedashplayerstats pulls every player at once (Base measure). The 2PT split is derived
    (FG2 = FG - FG3) since the endpoint only exposes total and 3PT lines. Players with no 2PT
    attempts get NaN (undefined rate). Feeds the hack-a-Shaq flip analysis in metrics.py.
    """
    cache = GAME_LIST_DIR / f"player_shooting_{season}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, season_type_all_star=season_type,
                per_mode_detailed="Totals", measure_type_detailed_defense="Base",
                timeout=API_TIMEOUT,
            ).get_data_frames()[0]
            fg2m = df["FGM"] - df["FG3M"]
            fg2a = df["FGA"] - df["FG3A"]
            out = pd.DataFrame({
                "PersonId": df["PLAYER_ID"].astype(int),
                "TwoPT_FGM": fg2m.astype(int),
                "TwoPT_FGA": fg2a.astype(int),
                "TwoPT_FGPct": fg2m / fg2a.where(fg2a > 0),   # NaN where no 2PT attempts
                "ThreePT_FGPct": df["FG3_PCT"].astype(float),
            })
            cache.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(cache, index=False)
            return out
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"  retry {attempt}/{MAX_RETRIES} for leaguedashplayerstats "
                  f"after {type(e).__name__}: {e}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"unreachable: exhausted retries for player shooting ({season})")


def get_player_rim_shooting(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Return PersonId -> Restricted Area (rim) FG% for a season, cached to disk (one API call).

    leaguedashplayershotlocations with distance_range='By Zone' returns FGM/FGA per shot zone
    for every player at once. The Restricted Area (<4 ft) is the rim proxy — a center's real
    alternative to being fouled is a dunk/layup, so this is a truer "let them shoot" value than
    blended 2PT%. Players with no RA attempts get NaN. Columns are a (zone, stat) MultiIndex.
    """
    cache = GAME_LIST_DIR / f"player_rim_{season}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
                season=season, season_type_all_star=season_type,
                per_mode_detailed="Totals", distance_range="By Zone", timeout=API_TIMEOUT,
            ).get_data_frames()[0]
            ra_fgm = df[("Restricted Area", "FGM")]
            ra_fga = df[("Restricted Area", "FGA")]
            # Total FGA = sum of the DISJOINT zone FGA. 'Corner 3' is a Left+Right aggregate,
            # so it must be excluded or corner threes double-count.
            fga_zones = [c for c in df.columns if c[1] == "FGA" and c[0] != "Corner 3"]
            total_fga = df[fga_zones].sum(axis=1)
            out = pd.DataFrame({
                "PersonId": df[("", "PLAYER_ID")].astype(int),
                "RimFGM": ra_fgm.astype(int),
                "RimFGA": ra_fga.astype(int),
                "RimFGPct": ra_fgm / ra_fga.where(ra_fga > 0),   # NaN where no rim attempts
                "TotalFGA": total_fga.astype(int),
            })
            cache.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(cache, index=False)
            return out
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"  retry {attempt}/{MAX_RETRIES} for leaguedashplayershotlocations "
                  f"after {type(e).__name__}: {e}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"unreachable: exhausted retries for rim shooting ({season})")


def _fetch_pbp_with_retry(game_id: str) -> pd.DataFrame:
    """Hit PlayByPlayV3 with exponential backoff on throttle/timeout.

    stats.nba.com cold-starts slowly (first hit often read-times-out then works) and
    throttles under load (429), so a lone failure is not a dead endpoint. Retry up to
    MAX_RETRIES times, waiting 2s then 4s, before letting the error propagate.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return playbyplayv3.PlayByPlayV3(
                game_id=game_id, timeout=API_TIMEOUT
            ).get_data_frames()[0]
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE * (2 ** (attempt - 1))   # 2s, 4s
            print(f"  retry {attempt}/{MAX_RETRIES} for {game_id} "
                  f"after {type(e).__name__}: {e}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"unreachable: exhausted retries for {game_id}")  # loop returns or raises


def scrape_game(game_id: str) -> pd.DataFrame:
    """Pull one game's play-by-play, using the disk cache if present."""
    cache = RAW_DIR / f"{game_id}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    pbp = _fetch_pbp_with_retry(game_id)

    cache.parent.mkdir(parents=True, exist_ok=True)
    pbp.to_parquet(cache, index=False)
    time.sleep(REQUEST_SLEEP)   # sleep only on a genuine API hit
    return pbp


def load_or_scrape(season: str, season_type: str = "Regular Season",
                   max_games: int | None = None) -> pd.DataFrame:
    """Scrape or load all play-by-play for a season. Use max_games to test."""
    games = get_game_ids(season, season_type)
    if max_games is not None:
        games = games.head(max_games)

    frames, total = [], len(games)
    for i, game_id in enumerate(games["GAME_ID"], start=1):
        try:
            frames.append(scrape_game(game_id))
        except Exception as e:
            print(f"[{i}/{total}] FAILED {game_id}: {e}")
        if i % 50 == 0 or i == total:
            print(f"[{i}/{total}] processed through {game_id}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    df = load_or_scrape("2025-26", max_games=20)   # small validation run
    print(df.shape)
    print(df["actionType"].value_counts())
    print("Free Throw rows:", (df["actionType"] == "Free Throw").sum())
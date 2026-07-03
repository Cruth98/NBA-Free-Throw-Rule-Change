"""
scrape.py — pull and cache NBA play-by-play for the FT rule analysis.

stats.nba.com throttles hard, so we cache every game to disk and hit the
API at most once per game, ever. An interrupted run resumes from the cache.
Free-throw events are EVENTMSGTYPE == 3; the shot text ("Free Throw 1 of 2")
lives in the description columns and gets parsed downstream in parse.py.
"""

from pathlib import Path
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, playbyplayv2

RAW_DIR = Path("data/raw/pbp")
GAME_LIST_PATH = Path("data/raw/game_ids.parquet")
REQUEST_SLEEP = 0.6      # seconds between real API calls
API_TIMEOUT = 30         # stats.nba.com hangs without an explicit timeout


def get_game_ids(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Return unique game IDs for a season, cached to disk."""
    if GAME_LIST_PATH.exists():
        return pd.read_parquet(GAME_LIST_PATH)

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

    GAME_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    games.to_parquet(GAME_LIST_PATH, index=False)
    return games


def scrape_game(game_id: str) -> pd.DataFrame:
    """Pull one game's play-by-play, using the disk cache if present."""
    cache = RAW_DIR / f"{game_id}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    pbp = playbyplayv2.PlayByPlayV2(
        game_id=game_id, timeout=API_TIMEOUT
    ).get_data_frames()[0]

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
    print(df["EVENTMSGTYPE"].value_counts())
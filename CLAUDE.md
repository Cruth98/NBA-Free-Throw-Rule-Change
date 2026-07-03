# NBA Free Throw Rule Change — Project Memory

## Project Overview
Data science project analyzing the NBA's proposed "one free-throw rule" (one shot
worth 1/2/3 pts, replacing 1/2/3 separate shots; standard multi-shot FTs apply in the last 2 min of Q4 and throughout OT.
Goal: identify which players the rule helps or hurts, for a public sports-analytics
portfolio piece. Scope: 2025-26 NBA regular season.

## Working Style (Conner)
- Analyst + MBA student. Strong in SQL, Python, Power BI, Streamlit, Excel, data modeling.
- Explain in plain English BEFORE code. No over-engineering. MVP-first with clear tradeoffs.
- NEVER blindly agree — push back with evidence. Accuracy over agreement. Cite sources.
- Direct, WSJ-style. Bold key terms, bullets, tables. No oxford comma. Suggest next steps.

## Core Analytical Framework (locked)
- FT1Pct / FT2Pct / FT3Pct = make rate by shot number within a foul trip.
- CurrentEV2 = FT1Pct + FT2Pct ; NewEV2 = 2*FT1Pct ; DeltaEV2 = NewEV2 - CurrentEV2.
- CurrentEV3 = FT1Pct+FT2Pct+FT3Pct ; NewEV3 = 3*FT1Pct ; DeltaEV3 = NewEV3 - CurrentEV3.
- Scale by trip counts: CurrentTotalPts, NewTotalPts, DeltaTotalPts.
- Define EV deltas explicitly as (New - Current) in code, not the reduced algebraic form.

## Data Conventions (invariants)
- Basketball-facing PascalCase column names (FT1Pct, Trips2Shot, NewTotalPts). Never p1/p2.
- Source is PlayByPlayV3 (V2 was retired mid-2025, returns empty JSON). FT events are
  actionType == "Free Throw"; shot number & trip type come from subType ("Free Throw X of Y");
  make/miss from the "MISS" prefix in description (shotResult is BLANK for FTs — V3 quirk;
  makes carry a "(X PTS)" suffix, misses a "MISS" prefix); player via personId/playerName.
- MUST exclude technical FTs, flagrant FTs, and and-1 "1 of 1" shots from the 2/3-shot analysis.
- Cache scrapes to disk as Parquet; read from cache downstream; hit stats.nba.com at most once/game.

## Known Pitfalls
- GAME_ID has leading zeros (e.g. 0022500001) — keep as string. CSV strips them; use Parquet.
- stats.nba.com throttles — set request timeouts, sleep between real API calls, raise sleep if blocked.
- stats.nba.com cold-starts slowly — first hit often read-times-out then works. Use a generous
  timeout (60s) and retry; don't treat a lone timeout as a dead endpoint.
- Use PlayByPlayV3, NOT V2. V2 endpoints return empty JSON and nba_api raises KeyError('resultSet').
- VS Code Python auto-activation and Rich shell integration corrupt Git Bash PATH — keep python.terminal.activateEnvironment 
and shellIntegration.enabled false; activate venv manually

## Repo Layout
- src/scrape.py (pull+cache pbp), src/parse.py (bucket/filter FT events), src/metrics.py (EV cols).
- notebooks/analysis.ipynb = shareable narrative layer.
- data/raw = cached scrapes (gitignored). data/processed = final CSV for Power BI.

## Commands
- Activate venv: source .venv/Scripts/activate
- Test scrape:  python src/scrape.py

## Maintenance
- If anything we decide in a session conflicts with or changes what's written in this
  file (framework, conventions, scope, pitfalls), flag it to me and add a backlog note
  to update this file. Doesn't need to be immediate, but never let the file silently drift.

## Backlog — do NOT build unless asked
G League before/after validation; college 1-and-1 comparison; modern-era (since 1996-97) all-time list.

### scrape.py hardening (before full-season run, not the sample)
- Season-key the game-list cache: GAME_LIST_PATH is a fixed constant, so get_game_ids()
  returns the wrong season if called for a second season. Put {season} in the filename.
- Adaptive backoff/retry on throttle: a 429/timeout currently prints and skips the game
  (no retry). Add retry-with-backoff before the full ~1,230-game season run ("raise sleep if blocked").

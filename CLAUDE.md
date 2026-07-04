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
- Rates are PER TRIP TYPE (make rate by shot number WITHIN a trip type), stored separately:
  FT1Pct_2Shots, FT2Pct_2Shots (2-shot trips); FT1Pct_3Shots, FT2Pct_3Shots, FT3Pct_3Shots
  (3-shot trips); plus FT1Pct_Blended (shot 1 pooled across both) stored for downstream /
  low-volume use only — never silently substituted into the primary EV.
- CurrentEV2 = FT1Pct_2Shots + FT2Pct_2Shots ; NewEV2 = 2*FT1Pct_2Shots ; DeltaEV2 = NewEV2 - CurrentEV2.
- CurrentEV3 = FT1Pct_3Shots+FT2Pct_3Shots+FT3Pct_3Shots ; NewEV3 = 3*FT1Pct_3Shots ; DeltaEV3 = NewEV3 - CurrentEV3.
- Scale by trip counts: CurrentTotalPts, NewTotalPts, DeltaTotalPts. Keep ActualCurrentPts
  (observed made FTs) visible; it must equal CurrentTotalPts (built-in assertion).
- Define EV deltas explicitly as (New - Current) in code, not the reduced algebraic form.
- Rates/EV are NON-ADDITIVE: metrics.aggregate_ev(fact, group_keys) sums additive measures
  at the grain FIRST, then derives rates/EV — so any grain (player/team/home-away) is valid.
- Nothing imputed or dropped silently: missing trip types stay NaN and contribute 0 to totals;
  low-volume groups get a LowVolume flag (tunable min_trips), not removal.
- TripComplete invariant: a trip (keyed by game+player+period+dead-ball clock+trip length) must
  hold every shot 1..TripLen. Positionally incomplete trips (injury subs split across two
  personIds, raw PBP gaps) can't be valued on the current-rule side and are flagged + excluded
  (~0.1% of trips) via TripComplete — this is what makes the CurrentTotalPts==ActualCurrentPts
  assertion hold per grain (shot-1 count, the trip count, then equals every later shot's count).

## Data Conventions (invariants)
- Basketball-facing PascalCase column names (FT1Pct, Trips2Shot, NewTotalPts). Never p1/p2.
- Source is PlayByPlayV3 (V2 was retired mid-2025, returns empty JSON). FT events are
  actionType == "Free Throw"; shot number & trip type come from subType ("Free Throw X of Y");
  make/miss from the "MISS" prefix in description (shotResult is BLANK for FTs — V3 quirk;
  makes carry a "(X PTS)" suffix, misses a "MISS" prefix); player via personId/playerName.
- MUST exclude technical FTs, flagrant FTs, and-1 "1 of 1" shots, and positionally incomplete
  trips (see TripComplete invariant above) from the 2/3-shot analysis.
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
- src/scrape.py (pull+cache pbp), src/parse.py (bucket/filter FT -> fact table),
  src/metrics.py (grain-agnostic aggregate_ev engine + EV cols).
- notebooks/analysis.ipynb = shareable narrative layer (Streamlit page a likely companion).
- data/raw = cached scrapes (gitignored). data/processed = Parquet outputs (fact_ft,
  player_ft_metrics; dim tables later) consumed by the notebook/Streamlit viz layer and
  pushed to GitHub as the portfolio deliverable. Parquet everywhere (any table with GameId
  MUST be Parquet — CSV strips its leading zeros). Power BI is out of scope for this piece.

## Commands
- Activate venv: source .venv/Scripts/activate
- Test scrape:  python src/scrape.py
- Build parse:  python src/parse.py    (FT fact smoke test)
- Build metrics: python src/metrics.py (writes data/processed/*.parquet)

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

### Phase B — dimensional model (star schema, after Phase A metrics)
- dim_game: enrich get_game_ids() to keep TEAM_ID + MATCHUP (currently dropped) and re-pull
  the game log once, to derive HomeTeamId/AwayTeamId/opponent -> enables home/away + opponent slicing.
- dim_team: static 30-team reference (TeamId -> Conference/Division), stored as Parquet.
- dim_player enrichment: layer in Basketball-Reference player stats for richer slicing.

### Analysis extensions (after MVP)
- "Who to foul" / actual-vs-expected: deeper use of NewTotalPts_Blended vs ActualCurrentPts
  to flag players whose realized FT output trails their blended new-rule expectation.
- Power BI dashboard as a possible future add-back if a BI-facing deliverable is wanted.

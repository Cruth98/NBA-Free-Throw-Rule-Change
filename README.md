# NBA One-Free-Throw Rule: Who Wins, Who Loses

An expected-value analysis of the NBA's proposed **one-free-throw rule** — where a trip to the line becomes a single shot worth 1, 2, or 3 points instead of 1, 2, or 3 separate attempts (standard multi-shot free throws still apply in the final 2:00 of Q4 and all of overtime). Using play-by-play from all **1,230 games of the 2025-26 regular season**, this project rebuilds every free-throw trip, values it under both the current and proposed rules, and quantifies the per-player point swing. The headline: the rule quietly taxes players who rely on second-shot "warm-up" makes and revives the hack-a-Shaq foul as a rational defensive play.

## Key Findings

- **The warm-up gap is real and systematic.** League-wide, players hit **75.8%** of the *first* free throw in a two-shot trip but **80.7%** of the *second* — a **~5-percentage-point** bump from settling in. The one-FT rule collapses every trip to a single cold first shot, so it structurally penalizes the observed warm-up rather than being point-neutral.
- **TrueNet leaders and laggards.** Across the season, **Giannis Antetokounmpo (+12 pts)**, **Isaiah Stewart (+9)** and **Dyson Daniels (+9)** gain the most — strong first-shot shooters who draw heavy contact. The biggest losers are high-volume, streaky shooters: **James Harden (−26)**, **Jalen Johnson (−26)**, **Luka Dončić (−25)** and **Rudy Gobert (−21)**, who forfeit the second-chance points the current rule hands them.
- **ShaqScore revives hack-a-Shaq.** Under one FT, a missed first shot ends the trip — no salvage. That makes intentionally fouling a poor free-throw shooter mathematically sound again. **Mitchell Robinson** and **Rudy Gobert** top the target list: fouling them yields fewer expected points than letting them finish at the rim, flipping a foul from a mistake into a strategy.

## Methodology

The core unit is the **trip** — one visit to the line — keyed by game, player, period, dead-ball clock, and trip length, with an invariant that a valid trip holds every shot 1..N. For each trip type the model stores make rates **per shot position** (first shot vs. later shots, kept separate because they differ), then derives expected value two independent ways:

- **Rate-based (`aggregate_ev`)** — sums additive measures at the chosen grain first, then computes non-additive rates and EV, so any slice (player, team, home/away) is valid without re-deriving.
- **Outcome-based (`aggregate_outcomes`)** — sums observed per-trip results (points salvaged, value lost/gained) directly.

The two paths agree by construction (correlation = 1.0), which serves as a built-in correctness check. Technical, flagrant, and-1, clutch-window, and positionally incomplete trips are excluded; low-volume players are flagged, never silently dropped or imputed.

## Original Metrics

| Metric | Definition |
|---|---|
| **TrueNet** | Observed points gained minus lost under the new rule (new-rule points − actual current points). The bottom-line per-player swing. |
| **TrueNetRate** | TrueNet per trip — normalizes for volume so low- and high-usage players compare fairly. |
| **Winpact** | TrueNet translated to wins via Oliver's Pythagorean expectation (÷ 30.5 pts/win). |
| **PtsSalvaged** | Second-chance points scored *after* a missed first shot — exactly the value the one-FT rule deletes. |
| **ShaqScore** | Expected points a defense gains by fouling instead of letting a player shoot (rim/2PT FG EV − new-rule FT EV). Positive = foul them. |

## Data Sources

- **[nba_api](https://github.com/swar/nba_api)** — `PlayByPlayV3` for free-throw events, plus player shooting and rim-finishing splits for ShaqScore.
- **Scope:** 2025-26 NBA regular season (1,230 games). Scrapes are cached to Parquet; `stats.nba.com` is hit at most once per game.

## Repo Structure

```
src/
  scrape.py    Pull + cache play-by-play and shooting splits (Parquet)
  parse.py     Bucket and filter FT events into the trip-level fact table
  metrics.py   Grain-agnostic EV + outcome engine; writes processed metrics
notebooks/
  analysis.ipynb   Narrative + visualization layer
data/
  raw/         Cached scrapes (gitignored)
  processed/   fact_ft, player_ft_metrics, player_outcomes (Parquet)
visuals/       Rendered charts and scorecards (PNG/HTML)
```

## Built With

- **Python**
- **pandas**
- **nba_api**
- **matplotlib**

## Read the Full Write-Up

📄 **Substack article:** _[link coming soon]_

# World Cup 2026 — Live Win Probability Model
 
A two-stage neural network pipeline that predicts pre-match outcome probabilities (Home Win / Draw / Away Win) for World Cup 2026 fixtures, then adjusts those probabilities live during matches using real-time stats from ESPN's API. Served via FastAPI and visualized in a single-page HTML/Chart.js dashboard.

---

## 1. Project Structure

```
.
├── notebook.ipynb              # Data prep, feature engineering, model training, baseline generation
├── app.py                      # FastAPI server
├── engine.py                   # ESPN data fetching + live probability adjustment logic
├── index.html                  # Frontend dashboard
├── all_matches.csv             # Historical international match results
├── world_cup_teams.csv         # Custom team profile data (market value, FIFA ranking, etc.)
├── world_cup_26_games.csv      # WC26 fixture list (input)
├── global_teams_profiles.csv   # Generated: rolling team form stats (2021+)
└── world_cup_26_baselines.csv  # Generated: pre-match probabilities for every WC26 fixture
```

---

## 2. How the Model Works

### 2.1 Data Sources & Preparation (Cell 1-3)

- **`all_matches.csv`** — full historical international match log, filtered to matches from 2000 onward (`df_modern`).
- Each match is labeled `H` / `D` / `A` based on score, plus a binary `target_binary` (1 = home win, 0 = away win, NaN for draws) and `target_draw` (1 = draw, 0 = decisive).
- **Match weighting (`match_wt`)**: matches are weighted by competitive importance —
  - `4` = World Cup (non-qualifier)
  - `3` = continental championships (Euro, Copa América, AFCON)
  - `2` = qualifiers / Nations League
  - `1` = everything else (friendlies)
- **Elo ratings**: a custom Elo system (`compute_elo`) is run over the full historical dataset (K=20, starting rating 1500) to produce `home_elo`, `away_elo`, and `elo_diff` for every match. The final `elo_dict` is also reused later for WC26 inference.

### 2.2 Live Form Stats (Cell 2)

- Pulls a fresh historical results file from a public GitHub mirror (`martj42/international_results`) and filters to matches since Jan 1, 2021.
- Aggregates per-team rolling stats: matches played, goals for/against, win/draw/loss counts, goals scored/conceded per game, and win ratio.
- Falls back to `df_modern` (2021+) if the live fetch fails.
- Saved to `global_teams_profiles.csv`.

### 2.3 Team Profiles & Feature Assembly (Cell 3)

- `world_cup_teams.csv` provides squad-level metadata: market value, average market value, FIFA ranking, FIFA points, average age, squad size, historical participation, host country.
- Market value strings (e.g. `"€1.2bn"`, `"450k"`) are parsed into raw floats via `convertor`.
- These profiles are merged with the rolling form stats from `global_teams_profiles.csv` to create `df_master_profiles`.
- Historical matches involving any WC26 team are isolated (`df_wc_matches`) and joined with home/away team profiles to produce `df_final_features`.
- **Host indicators**: `home_host_indicator` / `away_host_indicator` flag whether a team is playing in its home country in a non-neutral match.
- **Missing data fallback**: `fill_baselines` provides median/default values (e.g. median FIFA ranking, 26.5 average age, $15M market value) for any team without a full profile — this matters for smaller or newly-promoted nations.

### 2.4 Feature Engineering (Cell 5)

Final features used by the model:
 
| Feature | Description |
|---|---|
| `match_wt` | Competition importance weight |
| `neutral_flag` | 1 if match is on neutral ground |
| `elo_diff` | Home Elo − Away Elo |
| `elo_balance` | `exp(-|elo_diff| / 300)` — how evenly matched teams are |
| `win_ratio_delta` | Home win ratio − Away win ratio (2021+ form) |
| `home_host_indicator` / `away_host_indicator` | Host nation flags |
| `elo_tightness` | `exp(-|elo_diff| / 350)` — second balance metric with different decay |
| `strength_asymmetry` | `|win_ratio_delta| × |elo_diff|` |
| `market_gap` | `|log1p(home_market_value) − log1p(away_market_value)|` |
 
Other engineered fields (`home_power_score`, `market_value_ratio`, `scoring_delta`, `defensive_delta`, `goal_diff_abs`) are computed but **not** included in the final `feature_columns` list used for training.
 
- **Train/test split**: chronological — train on matches before 2021-01-01, test on matches from 2021-01-01 onward (avoids lookahead bias).
- All features are scaled with `StandardScaler` (separate scalers for each stage).

### 2.5 Two-Stage Mode Architecture (Cell 6)


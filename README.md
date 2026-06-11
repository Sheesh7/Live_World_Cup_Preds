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

Rather than a single 3-class classifier, the model is split into two independent binary networks:
 
**Stage 1 — Strength Model (`model_s1`)**
- Trained **only on decisive matches** (draws excluded).
- Predicts `P(Home Win | match is decisive)`.
- Architecture: `Dense(96, relu) → Dropout(0.3) → Dense(48, relu) → Dropout(0.25) → Dense(24, relu) → Dense(1, sigmoid)`
- Loss: Binary cross-entropy with label smoothing (0.05)
- Class-balanced via `compute_class_weight`
- Early stopping on `val_loss` (patience=5)
**Stage 2 — Draw Model (`model_s2`)**
- Trained on **all matches** (draw vs. non-draw).
- Predicts `P(Draw)`.
- Same architecture, loss, and training setup as Stage 1.
**Combining the two stages (Cell 7, 9):**

```
p_draw_final  = P(draw) × draw_sensitivity        # draw_sensitivity = 0.79
p_home_final  = (1 - p_draw_final) × P(home | decisive)
p_away_final  = (1 - p_draw_final) × (1 - P(home | decisive))
```
 
The `draw_sensitivity = 0.79` constant **deliberately discounts** Stage 2's raw draw probability. Neural nets trained on draw classification tend to over-predict draws (since draws are a large, somewhat noisy class); scaling by 0.79 was tuned empirically to better match real-world draw frequency (historically ~24-26% in international football) and improve overall 3-class accuracy.

### 2.6 Backtest Evaluation (Cell 7-8)

- Evaluated on the 2021+ test set.
- Reports overall 3-class accuracy, confusion matrix, classification report (precision/recall/F1 per class), and a comparison of predicted vs. actual class distributions (to check for systematic bias toward over/under-predicting any outcome).
- Loss curves for both stages are plotted to check for overfitting (train vs. validation loss divergence).

### 2.7 Generating WC26 Baselines (Cell 9)

For each fixture in `world_cup_26_games.csv`:
 
1. **Two-pass symmetric inference**: the model is run once with teams in their listed home/away order, and again with home/away **swapped**. The two results are averaged:
```
   avg_home_win = (pass1_home + pass2_away) / 2
   avg_draw     = (pass1_draw + pass2_draw) / 2
   avg_away_win = (pass1_away + pass2_home) / 2
```
   This cancels out any artificial "home advantage" bias from the model, since all WC26 matches are technically on neutral ground (`neutral_flag = 1`, `match_wt = 4`, host indicators = 0) except true host-nation games.
2. **Elo lookup**: each team's Elo rating is pulled from the `elo_dict` computed in Cell 1 (defaults to 1500 if a team is unseen).
3. **Renormalization**: the averaged probabilities are renormalized to sum to 100%.
4. **Placeholder fixtures**: knockout-stage fixtures with TBD teams (containing "Winner", "Loser", "Place", "Third" in the team name) are assigned a fixed 37.5% / 25% / 37.5% split, since no real team data exists yet.
5. Output saved to **`world_cup_26_baselines.csv`** — this is the file `app.py` reads at startup and is the source of the `priors` returned by `/match/{game_id}`.

### 2.8 Live Evaluation Cell (Cell 10)

- `eval_live_preds()` compares stored baseline predictions against actual completed match outcomes.
- Computes **accuracy** (highest-probability outcome), **Brier score** (target < 0.55), and **log loss**.
- Intended to be re-run throughout the tournament as real results come in, by updating the `results_test` DataFrame with actual `match_id` → `actual_outcome` pairs.

---

## 3. Live Match Adjustment (`engine.py`)

### 3.1 `fetch_espn_data(game_id)`

- Queries ESPN's public soccer API (`fifa.world` endpoint, falling back to `intl.friendly` if the first returns no competition data).
- Extracts: match status, current minute (handles halftime/full-time edge cases), scores, and — for live/finished matches — possession, shots, shots on target, yellow cards, and red cards from the boxscore.
- Returns `None` on any failure (timeout, malformed response, etc.), which `app.py` surfaces as `{"error": "Could not fetch match data"}`.

### 3.2 `compute_live_probabilities(stats, priors)`

Takes the pre-match `priors` (from `world_cup_26_baselines.csv`) and adjusts them based on real-time match state:
 
1. **Card penalties**: red cards reduce a team's win probability by 7% each; yellow cards by 1.5% each.
2. **Momentum modifier**: combines possession differential (×0.10) and a "shot threat" differential (`shots + 2×shots_on_target`, ×0.015) to nudge probabilities toward the team controlling the game.
3. **Time-decay / score-state logic**:
   - **Leading team**: probability converges toward certainty as time runs out, scaled by goal difference (`holding_power = 1 - (time_remaining_ratio)^(1.2 × goal_diff)`).
   - **Trailing team**: probability decays toward zero following the same exponential curve.
   - **Drawn match**: draw probability grows toward 1 as time remaining decreases (`draw_gravity = 1 - time_remaining_ratio`).
4. Final probabilities are renormalized to sum to 100%.
5. If the match hasn't started (`is_live_or_dead = False`), the raw pre-match priors are returned unchanged.

---

## 4. Backend (`app.py`)

FastAPI app with CORS fully open (`allow_origins=["*"]`).
 
| Endpoint | Description |
|---|---|
| `GET /matches` | Returns `home_team`, `away_team`, `ESPN_ID` for every fixture in `world_cup_26_games.csv` |
| `GET /match/{game_id}` | Looks up pre-match priors from `world_cup_26_baselines.csv` by `ESPN_ID`, fetches live ESPN data, runs `compute_live_probabilities`, and returns combined stats + probabilities + priors |
 
- If a `game_id` isn't found in the baselines file, default priors of `[50.0, 30.0, 20.0]` are used.
- `ESPN_ID` is stored as a stripped string for matching consistency.

---

## 5. Frontend (`index.html`)

A single-file dashboard (vanilla JS + Chart.js) that:
 
- Fetches the fixture list from `/matches` and populates a dropdown selector.
- On load (and every 10 seconds via auto-refresh, toggleable) calls `/match/{ESPN_ID}` and renders:
  - Live status badge, match clock, score
  - Home/Draw/Away win probabilities with a segmented bar
  - Pre-match prior probabilities for comparison
  - A live-updating line chart of probability evolution by minute (deduplicated per minute)
  - Possession, shots, shots on target, yellow/red cards
- API base URL is hardcoded to `http://127.0.0.1:8000`.

---

## 6. How to Run

### Prerequisites
```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn fastapi uvicorn requests
```

### Step 1 - Run the notebook
Run all cells in order. This requires the following CSVs to already exist in the working directory:
- `all_matches.csv`
- `world_cup_teams.csv`
- `world_cup_26_games.csv`
This will produce:
- `global_teams_profiles.csv`
- `world_cup_26_baselines.csv` (required by the backend)
> **Note:** Cell 2 attempts to fetch a live data file from GitHub. If there's no internet access, it falls back to existing historical data automatically.

### Step 2 - Start the backend
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
Make sure `world_cup_26_games.csv` and `world_cup_26_baselines.csv` are in the same directory as `app.py`.

### Step 3 - Open the frontend
Open `index.html` directly in a browser (or serve it via any static file server). It will connect to `http://127.0.0.1:8000` automatically. Select a match from the dropdown to begin.

---

## 7. Accuracy Measures During the World Cup

Several mechanisms are in place to monitor and validate model performance as the tournament progresses:
 
1. **Backtested 3-class accuracy, confusion matrix, and classification report** (Cell 7–8) on 2021–present matches establish a baseline performance benchmark before the tournament starts.
2. **Predicted vs. actual class distribution comparison** — checks whether the model systematically over/under-predicts home wins, draws, or away wins relative to real-world base rates.
3. **`eval_live_preds()` (Cell 10)** — designed to be re-run periodically during the tournament:
   - **Accuracy**: % of matches where the highest-probability outcome matched the actual result.
   - **Brier score**: mean squared error between predicted probability vector and the one-hot actual outcome (target < 0.55 — lower is better, 0 = perfect).
   - **Log loss**: penalizes confident-but-wrong predictions more heavily than Brier score.
   - To use this during the tournament, update the `results_test` DataFrame with real `match_id` → `actual_outcome` (`H`/`D`/`A`) pairs as matches conclude.
4. **Symmetric two-pass inference** for WC26 baselines reduces home-bias artifacts that could otherwise inflate or deflate probabilities for specific matchups.
5. **Draw sensitivity calibration (0.79 multiplier)** was tuned against the test set to correct for the Stage 2 model's tendency to over-predict draws — this should be periodically re-validated as live results come in, since the optimal multiplier may drift.
6. **Live probability sanity bounds** — `compute_live_probabilities` floors all probabilities at 0.01 and renormalizes, preventing degenerate 0% or negative values during extreme game states (e.g., two red cards).

## 8. Future Improvements

### Modeling
- **Replace the fixed `draw_sensitivity = 0.79` constant** with a learned or dynamically calibrated value (e.g., isotonic regression / Platt scaling on the Stage 2 output) rather than a single hand-tuned scalar.
- **Joint 3-class model or ordinal approach** as an alternative/ensemble to the two-stage decoupled design, then compare calibration (Brier/log loss) against the current pipeline.
- **Add unused engineered features** (`home_power_score`, `away_power_score`, `market_value_ratio`, `scoring_delta`, `defensive_delta`, `goal_diff_abs`) back into `feature_columns` and run feature-importance / ablation studies — several were computed but never used in training.
- **Recency-weighted form stats** — weight recent matches (e.g., last 6–12 months) more heavily than the full 2021+ window when computing `goals_scored_per_game`, `win_ratio`, etc.
- **Player-level features** — incorporate injuries/suspensions to key players, which can swing single-match probabilities significantly but aren't currently modeled at all.
- **Cross-validation instead of a single train/test split** to get more robust performance estimates, especially given the relatively small WC-specific sample size.
- **Tournament-stage-aware modeling** — knockout matches (extra time/penalties possible) have different draw dynamics than group stage; the model currently doesn't distinguish.
### Live Engine
- **Real possession/shot data fallback** — currently `home_possession`/`away_possession` default to 50/50 and shot stats default to 0 until ESPN's boxscore populates them, which can cause a visible "jump" in live probabilities once real data arrives.
- **xG (expected goals) integration** — if available from a data provider, would be a far stronger live signal than raw shot counts.
- **Tune momentum/time-decay coefficients** (`0.10` possession weight, `0.015` shot-threat weight, `1.2` exponent in `holding_power`) using historical live win-probability data rather than hand-set constants.
- **Injury time handling** — `current_minute` calculation doesn't explicitly account for stoppage time beyond parsing the displayed clock, which could affect the `time_rem_ratio` calculation near half/full time.
- **Caching/rate-limiting** for ESPN API calls if multiple users hit `/match/{game_id}` simultaneously, to avoid redundant external requests.

### Engineering
- **Move hardcoded API URL** (`http://127.0.0.1:8000` in `index.html`) to a config value or environment-based detection for deployment beyond localhost.
- **Persist live probability history server-side** (currently only stored client-side in browser memory) so the probability-over-time chart survives page refreshes and can be shared/analyzed across users.
- **Automated re-evaluation pipeline** — schedule `eval_live_preds()` to run automatically as match results come in (e.g., via a cron job pulling completed-match data from ESPN), rather than manual `results_test` updates.
- **Error handling/UI feedback** — `index.html` currently logs fetch errors to console only; a user-facing error/retry state would improve robustness during ESPN API outages.

# 👤 Author
Built by Yashish Eriki
Data Science & Analytics Enthusiast
LinkedIn: https://www.linkedin.com/in/yashisheriki/
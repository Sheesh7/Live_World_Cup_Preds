from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from engine import fetch_espn_data, compute_live_probabilities

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load CSVs once at startup
games_df = pd.read_csv("world_cup_26_games.csv")
baselines_df = pd.read_csv("world_cup_26_baselines.csv")
baselines_df["ESPN_ID"] = baselines_df["ESPN_ID"].astype(str).str.strip()

@app.get("/matches")
def get_matches():
    return games_df[["home_team", "away_team", "ESPN_ID"]].to_dict(orient="records")

@app.get("/match/{game_id}")
def get_match(game_id: str):
    row = baselines_df[baselines_df["ESPN_ID"] == game_id]
    if row.empty:
        priors = [50.0, 30.0, 20.0]
    else:
        priors = [float(row["Pre_Home_Win_Pct"].values[0]),
                  float(row["Pre_Draw_Pct"].values[0]),
                  float(row["Pre_Away_Win_Pct"].values[0])]

    stats = fetch_espn_data(game_id)
    if stats is None:
        return {"error": "Could not fetch match data"}

    p_home, p_draw, p_away = compute_live_probabilities(stats, priors)
    return {**stats, "p_home": round(p_home, 2), "p_draw": round(p_draw, 2), "p_away": round(p_away, 2), "priors": priors}

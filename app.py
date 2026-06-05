import streamlit as st
import pandas as pd
import time
import os
from engine import fetch_espn_data, compute_live_probabilities

st.set_page_config(
    page_title="Live Match Analytical Dashboard",
    page_icon="⚽",
    layout="wide"
)
st.title("⚽ Live Match Win-Probability Tracker")
st.caption("Real-time predictive analytics powered by live boxscore metrics.")

fixtures_csv = "world_cup_26_games.csv"
baselines_csv = "world_cup_26_baselines.csv"

@st.cache_data
def load_csv_data(filepath):
    """Safely loads data frames from local storage."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df.columns = [col.strip() for col in df.columns]
        return df
    return None

games_df = load_csv_data(fixtures_csv)
baselines_df = load_csv_data(baselines_csv)

if games_df is None or baselines_df is None:
    st.error(f"⚠️ Data file synchronization issue. Verify both `{fixtures_csv}` and `{baselines_csv}` exist in this project directory.")
    st.stop()

id_col_baselines = [col for col in baselines_df.columns if 'id' in col.lower()][0]
baselines_df[id_col_baselines] = baselines_df[id_col_baselines].astype(str).str.strip()

col_mapping = {col.lower(): col for col in games_df.columns}
home_col = col_mapping.get('home_team', games_df.columns[0])
away_col = col_mapping.get('away_team', games_df.columns[1])
id_col_fixtures = col_mapping.get('game_id', col_mapping.get('espn_id', games_df.columns[-1]))

dropdown_options = []
id_lookup_map = {}

for _, row in games_df.iterrows():
    m_home = row['home_team']
    m_away = row['away_team']
    m_id = str(row['ESPN_ID']).strip()
    
    display_label = f"🏆 {m_home} vs {m_away} (ID: {m_id})"
    dropdown_options.append(display_label)
    id_lookup_map[display_label] = m_id

selected_match_label = st.selectbox("🎯 Select Live Match to Analyze:", dropdown_options)
game_id = id_lookup_map[selected_match_label]

match_priors = baselines_df[baselines_df['ESPN_ID'].astype(str).str.strip() == game_id]

if not match_priors.empty:
    baseline_priors = [
        float(match_priors['Pre_Home_Win_Pct'].values[0]),
        float(match_priors['Pre_Draw_Pct'].values[0]),
        float(match_priors['Pre_Away_Win_Pct'].values[0])
    ]
else: 
    st.warning("⚠️ Match ID not found in baselines database. Defaulting to standard 50-30-20 tournament split.")
    baseline_priors = [50.0, 30.0, 20.0]

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Minute", "Home Win %", "Draw %", "Away Win %"])
if "current_game" not in st.session_state:
    st.session_state.current_game = game_id

if st.session_state.current_game != game_id:
    st.session_state.history = pd.DataFrame(columns=["Minute", "Home Win %", "Draw %", "Away Win %"])
    st.session_state.current_game = game_id

run_live_updates = st.checkbox("Enable Live Real-Time Polling (10s intervals)", value=True)
dashboard_container = st.empty()

while True:
    stats = fetch_espn_data(game_id)
    
    if stats is None:
        st.error("❌ Unable to connect or parse payload for the specified Game ID.")
        break
        
    p_home, p_draw, p_away = compute_live_probabilities(stats, baseline_priors)
    current_min = stats['minute']
    
    if not st.session_state.history['Minute'].isin([current_min]).any() and stats['is_live_or_dead']:
        new_row = pd.DataFrame([{
            "Minute": current_min,
            "Home Win %": round(p_home, 2),
            "Draw %": round(p_draw, 2),
            "Away Win %": round(p_away, 2)
        }])
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True).sort_values("Minute")

    with dashboard_container.container():
        st.info(f"📋 **Model Pre-Match Priors:** {stats['home_team']}: {baseline_priors[0]}% | Draw: {baseline_priors[1]}% | {stats['away_team']}: {baseline_priors[2]}%")

        st.markdown(f"### ⏱️ Minute: {current_min}' | Status: `{stats['status']}`")
        
        col_h, col_vs, col_a = st.columns([3, 1, 3])
        with col_h:
            st.metric(label=f"🏠 {stats['home_team']}", value=stats['home_score'], delta=f"Live Prob: {p_home:.1f}%")
        with col_vs:
            st.markdown("<h1 style='text-align: center; margin: 0;'>VS</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: gray; font-size: 0.8rem;'>Draw Prob: {p_draw:.1f}%</p>", unsafe_allow_html=True)
        with col_a:
            st.metric(label=f"🚀 {stats['away_team']}", value=stats['away_score'], delta=f"Live Prob: {p_away:.1f}%")
            
        st.markdown("---")

        st.subheader("📈 Live Predictive Percentage Shift")
        if not st.session_state.history.empty:
            chart_data = st.session_state.history.set_index("Minute")
            st.line_chart(chart_data, height=350)
        else:
            st.info("Waiting for live match data logging to initialize visualization sequence...")
            
        st.markdown("---")

        st.subheader("📊 Match Core Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.markdown(f"**Possession Percentage**")
            st.write(f"🏠 {stats['home_possession']}% vs {stats['away_possession']}% 🚀")
            st.progress(stats['home_possession'] / 100.0)
            
        with m_col2:
            st.markdown(f"**Attacking Volume**")
            st.write(f"Total Shot Attempts: `{stats['home_shots']}` vs `{stats['away_shots']}`")
            st.write(f"Shots On Target: `{stats['home_sot']}` vs `{stats['away_sot']}`")
            
        with m_col3:
            st.markdown(f"**Disciplinary Penalties**")
            st.write(f"🟨 Yellows: `{stats['home_yellows']}` | `{stats['away_yellows']}`")
            st.write(f"🟥 Reds: `{stats['home_reds']}` | `{stats['away_reds']}`")

    if not run_live_updates or stats['status'] in ['Full Time', 'FT', 'Final', 'Ended']:
        break
        
    time.sleep(10)
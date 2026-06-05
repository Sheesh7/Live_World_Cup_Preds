# test_engine_verbose.py
from engine import fetch_espn_data, compute_live_probabilities

# ─── ACTIVE TEST ID SELECTION ─────────────────────────────────────────
# Note: Use a valid ID from a live match today or a recently completed game 
# to see full stats. If using a future game, possession defaults to 50/50 and shots to 0.
test_id = "760415"  # Replace with an active game ID if you want to see live shifts!

print(f"📡 Requesting deep telemetry pull for Game ID: {test_id}...\n")
game_stats = fetch_espn_data(test_id)

if game_stats is None:
    print("❌ ERROR: The scraper could not retrieve data. The ID may not exist, or ESPN's server timed out.")
else:
    print("==================================================")
    print("       LIVE ENGINE DEEP DIAGNOSTIC CHECK          ")
    print("==================================================")
    
    print("\n[1] CORE MATCH STATE")
    print(f"    • Matchup:     {game_stats['home_team']} vs {game_stats['away_team']}")
    print(f"    • Status/State: {game_stats['status']}")
    print(f"    • Match Clock:  {game_stats['minute']}'")
    print(f"    • Data Flag:   Is Live/Finished? -> {game_stats['is_live_or_dead']}")
    
    print("\n[2] SCOREBOARD & METRICS DATA")
    print(f"    • Goals:        [{game_stats['home_team']}] {game_stats['home_score']} - {game_stats['away_score']} [{game_stats['away_team']}]")
    print(f"    • Possession:   [{game_stats['home_team']}] {game_stats['home_possession']}% vs {game_stats['away_possession']}% [{game_stats['away_team']}]")
    
    print("\n[3] MICRO-ATTACKING THREATS")
    print(f"    • Total Shots:  [{game_stats['home_team']}] {game_stats['home_shots']} vs {game_stats['away_shots']} [{game_stats['away_team']}]")
    print(f"    • Shots on Tgt: [{game_stats['home_team']}] {game_stats['home_sot']} vs {game_stats['away_sot']} [{game_stats['away_team']}]")
    
    print("\n[4] DISCIPLINARY MODIFIERS")
    print(f"    • Yellow Cards: [{game_stats['home_team']}] {game_stats['home_yellows']} | [{game_stats['away_team']}] {game_stats['away_yellows']}")
    print(f"    • Red Cards:    [{game_stats['home_team']}] {game_stats['home_reds']} | [{game_stats['away_team']}] {game_stats['away_reds']}")
    
    print("\n" + "─" * 50)
    
    # ─── MATH VERIFICATION ENGINE ─────────────────────────────────────
    # Let's pass mock model priors (e.g., 50% Home Win, 30% Draw, 20% Away Win)
    # to observe how your threat matrix handles the data weights.
    mock_priors = (50.0, 30.0, 20.0)
    h, d, a = compute_live_probabilities(game_stats, mock_priors)
    
    print("\n[5] LIVE MATHEMATICAL ENGINE REFLECTION")
    print(f"    • Input Priors: Home: 50.0% | Draw: 30.0% | Away: 20.0%")
    print(f"    • Engine Output: Home: {h:.2f}% | Draw: {d:.2f}% | Away: {a:.2f}%")
    print("==================================================")
import requests
import numpy as np

def fetch_espn_data(game_id):
    """
    Queries ESPN's API service.
    Handles scheduled, live, and completed matches.
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event={game_id}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return None
        data = response.json()
        if 'header' not in data or 'competitions' not in data.get('header', {}):
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/intl.friendly/summary?event={game_id}"
            response = requests.get(url, timeout=5)
            data = response.json()
        header = data.get('header', {})
        competitions = header.get('competitions', [{}])
        if not competitions:
            return None
        competition = competitions[0]
        status_info = competition.get('status',{})
        competitors = competition.get('competitors', [])

        if len(competitors) < 2:
            return None
        
        h_team_data = competitors[0]
        a_team_data = competitors[1]

        state_str = status_info.get('type', {}).get('state', 'pre')
        display_clock = status_info.get('displayClock', status_info.get('clock', 0))

        if status_info.get('type',{}).get('description') in ['Halftime', 'HT']:
            current_minute = 45
        elif status_info.get('type', {}).get('description') in ['Full Time', 'FT', 'Final']:
            current_minute = 90
        else:
            base_clock = str(display_clock).split('+')[0]
            clean_clock = ''.join(filter(str.isdigit, str(display_clock).split('.')[0]))
            current_minute = int(clean_clock) if clean_clock else 0
        match_info = {
            'minute': current_minute,
            'status': status_info.get('type', {}).get('description', 'SCHEDULED'),
            'home_team': competition.get('competitors', [{}])[0].get('team', {}).get('displayName', 'Home'),
            'away_team': competition.get('competitors', [{}])[1].get('team', {}).get('displayName', 'Away'),
            'is_live_or_dead': status_info.get('type', {}).get('state', 'pre') != 'pre',
            'home_score': int(h_team_data.get('score',0)) if state_str != 'pre' else 0, 'away_score': int(a_team_data.get('score',0)) if state_str != 'pre' else 0,
            'home_possession': 50.0, 'away_possession': 50.0,
            'home_shots': 0, 'away_shots': 0,
            'home_sot': 0, 'away_sot': 0,
            'home_yellows': 0, 'away_yellows': 0,
            'home_reds': 0, 'away_reds': 0
        }
        if not match_info['is_live_or_dead']:
            return match_info
        
        boxscore = data.get('boxscore', {})
        teams_data = boxscore.get('teams', [])
        if len(teams_data) >= 2:
            for team_profile in teams_data:
                name = team_profile.get('team', {}).get('displayName', '')
                prefix = 'home_' if name == match_info['home_team'] else 'away_'
                
                for stat in team_profile.get('statistics', []):
                    stat_name = stat.get('name')
                    val = stat.get('displayValue', '0')
                    print(f"{stat_name} + {val}")

                    if stat_name == 'possessionPct':
                        match_info[f'{prefix}possession'] = float(val.replace('%', ''))
                    elif stat_name == 'totalShots': 
                        match_info[f'{prefix}shots'] = int(val)
                    elif stat_name == 'shotsOnTarget':
                        match_info[f'{prefix}sot'] = int(val)
                    elif stat_name == 'yellowCards':
                        match_info[f'{prefix}yellows'] = int(val)
                    elif stat_name == 'redCards':
                        match_info[f'{prefix}reds'] = int(val)
        return match_info
    except Exception:
        return None

def compute_live_probabilities(stats, priors):
    if not stats['is_live_or_dead']:
        return priors[0], priors[1], priors[2]
    
    ph, pd, pa = priors[0] / 100.0, priors[1] / 100.0, priors[2] / 100.0

    ph = max(0.01, ph - (0.07 * stats['home_reds']) - (0.015 * stats['home_yellows']))
    pa = max(0.01, pa - (0.07 * stats['away_reds']) - (0.015 * stats['away_yellows']))

    poss_delta = (stats['home_possession'] - stats['away_possession']) / 100.0

    home_threat = stats['home_shots'] + (stats['home_sot'] * 2.0)
    away_threat = stats['away_shots'] + (stats['away_sot'] * 2.0)
    threat_delta = home_threat - away_threat

    momentum_modifier = (poss_delta * 0.10) + (threat_delta * 0.015)
    ph = max(0.01, ph + momentum_modifier)
    pa = max(0.01, pa - momentum_modifier)

    time_rem_ratio = max(0.0, (90 - stats['minute']) / 90.0)
    gd = stats['home_score'] - stats['away_score']

    if gd > 0:  # Home Team Leading
        holding_power = 1.0 - (time_rem_ratio ** (1.2 * gd))
        live_ph = ph + (1.0 - ph) * holding_power
        live_pa = pa * (time_rem_ratio ** (gd + 1))
        live_pd = pd * time_rem_ratio
    elif gd < 0:  # Away Team Leading
        holding_power = 1.0 - (time_rem_ratio ** (1.2 * abs(gd)))
        live_pa = pa + (1.0 - pa) * holding_power
        live_ph = ph * (time_rem_ratio ** (abs(gd) + 1))
        live_pd = pd * time_rem_ratio
    else:  # Match is Drawn
        draw_gravity = 1.0 - time_rem_ratio
        live_pd = pd + (1.0 - pd) * draw_gravity
        live_ph = ph * time_rem_ratio
        live_pa = pa * time_rem_ratio
        
    total_distribution = live_ph + live_pd + live_pa
    return (live_ph / total_distribution) * 100, (live_pd / total_distribution) * 100, (live_pa / total_distribution) * 100

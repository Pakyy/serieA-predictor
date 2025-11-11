print("\n\n*** FILE database.py VERSIONE FINALE CORRETTA ***\n\n")

from supabase import create_client, Client
import os
from dotenv import load_dotenv
import pandas as pd
import json
from datetime import datetime

# Carica le variabili d'ambiente
load_dotenv()

# Inizializza il client Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = None
try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Client Supabase inizializzato.")
    else:
        print("⚠️  Client Supabase non inizializzato per mancanza di credenziali.")
except Exception as e:
    print(f"❌ ERRORE CRITICO durante l'inizializzazione di Supabase: {e}")

# Definizioni per ELO e squadre promosse
ELO_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'elo_cache.json')
NEWLY_PROMOTED_2025 = {'Pisa', 'Sassuolo', 'Cremonese'}

# ============================================================
# FUNZIONI ELO (REINSERITE PER COMPATIBILITÀ)
# ============================================================

def update_elo_cache(season='2025-26', force=False):
    """
    Questa funzione ora serve solo per compatibilità.
    L'ELO viene letto direttamente dalle righe della tabella 'matches'.
    Tuttavia, main.py la chiama all'avvio, quindi deve esistere.
    """
    print("✅ Funzione 'update_elo_cache' chiamata (per compatibilità). L'ELO viene letto dalle partite.")
    # Si potrebbe implementare una logica di cache se necessario, ma per ora non serve.
    return {}, True

# ============================================================
# FUNZIONE PRINCIPALE PER RECUPERARE LE STATS
# ============================================================

def get_latest_team_stats(team_name: str, current_season='2025-26'):
    """
    Recupera le features pre-calcolate per una squadra dalla loro ultima partita giocata.
    Se non trova partite nella stagione corrente, cerca nella precedente.
    """
    if not supabase:
        raise ConnectionError("Supabase client non è disponibile.")

    def fetch_last_match_stats(team, season):
        try:
            home_response = supabase.table('matches').select('*').eq('home_team', team).eq('season', season).order('date', desc=True).limit(1).execute()
            away_response = supabase.table('matches').select('*').eq('away_team', team).eq('season', season).order('date', desc=True).limit(1).execute()
            
            last_home = home_response.data[0] if home_response.data else None
            last_away = away_response.data[0] if away_response.data else None
            
            if not last_home and not last_away: return None
            if last_home and last_away: return last_home if last_home['date'] > last_away['date'] else last_away
            return last_home or last_away
        except Exception as e:
            print(f"❌ ERRORE durante la query a Supabase per {team} in stagione {season}: {e}")
            raise

    last_match = fetch_last_match_stats(team_name, current_season)
    used_season = current_season

    if not last_match:
        previous_season = f"{int(current_season[:4])-1}-{int(current_season[-2:])-1}"
        print(f"⚠️ Dati non trovati per {team_name} in {current_season}. Fallback a {previous_season}.")
        last_match = fetch_last_match_stats(team_name, previous_season)
        used_season = previous_season

    if not last_match:
        print(f"❌ Nessun dato trovato per {team_name}. Ritorno valori di default.")
        raise ValueError(f"Nessun dato trovato per {team_name} nelle stagioni recenti.")

    is_home = last_match['home_team'] == team_name
    stats = {
        'elo': float(last_match['home_elo_before' if is_home else 'away_elo_before']),
        'form_5': float(last_match['home_form_5' if is_home else 'away_form_5']),
        'goals_avg_5': float(last_match['home_goals_avg_5' if is_home else 'away_goals_avg_5']),
        'conceded_avg_5': float(last_match['home_conceded_avg_5' if is_home else 'away_conceded_avg_5']),
        'shots_avg_5': float(last_match.get('home_shots_avg_5' if is_home else 'away_shots_avg_5', 10.0)),
        'xG_avg_rolling': float(last_match.get('home_xG_avg_rolling' if is_home else 'away_xG_avg_rolling', 1.2)),
        'xGA_avg_rolling': float(last_match.get('home_xGA_avg_rolling' if is_home else 'away_xGA_avg_rolling', 1.2)),
    }

    print(f"📊 Statistiche trovate per {team_name} (da stagione {used_season}, data {last_match['date']}):")
    print(f"   ELO: {stats['elo']:.0f}, Form: {stats['form_5']}")

    return stats
import os
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from pathlib import Path

# --- CARICAMENTO ROBUSTO DELLE CREDENZIALI ---
# Costruisce un percorso assoluto al file .env nella root del progetto
project_root = Path(__file__).parent.parent
dotenv_path = project_root / '.env'

# Controlla se il file .env esiste prima di caricarlo
if dotenv_path.exists():
    print(f"✅ Trovato file .env in: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"❌ ERRORE CRITICO: File .env non trovato nel percorso atteso: {dotenv_path}")
    print("   Assicurati che il file .env sia nella cartella principale del progetto.")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# --- FINE CARICAMENTO CREDENZIALI ---

# Schema Manuale (tutto minuscolo per corrispondere a PostgreSQL)
SCHEMA_MANUALE = [
    'date', 'season', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result',
    'home_elo_before', 'away_elo_before', 'elo_diff',
    'home_form_5', 'away_form_5', 'form_diff',
    'home_goals_avg_5', 'away_goals_avg_5',
    'home_conceded_avg_5', 'away_conceded_avg_5',
    'home_shots_avg_5', 'away_shots_avg_5',
    'home_xg_avg_rolling', 'away_xg_avg_rolling',
    'home_xga_avg_rolling', 'away_xga_avg_rolling'  # 'A' -> 'a'
]

def upload_dataframe_to_supabase(df: pd.DataFrame, table_name: str, batch_size=100):
    # Controllo critico all'inizio della funzione
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Upload annullato: SUPABASE_URL o SUPABASE_KEY non sono state caricate correttamente.")
        return

    try:
        print(f"Utilizzo dello schema manuale per la tabella '{table_name}'.")
        df_filtered = df[[col for col in SCHEMA_MANUALE if col in df.columns]]
        print("✅ DataFrame filtrato.")

        url = f"{SUPABASE_URL}/rest/v1/{table_name}"
        headers = {
            'apikey': SUPABASE_KEY,
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'Content-Type': 'application/json'
        }
        
        print(f"Cancellazione dati da '{table_name}'...")
        delete_response = requests.delete(f"{url}?season=neq.0000-00", headers={k: v for k, v in headers.items() if k != 'Content-Type'})
        delete_response.raise_for_status()
        print("✅ Dati esistenti cancellati.")

        df_sanitized = df_filtered.replace([np.inf, -np.inf], None).astype(object).where(pd.notnull(df_filtered), None)
        records = df_sanitized.to_dict(orient='records')
        
        total_records = len(records)
        num_batches = (total_records + batch_size - 1) // batch_size
        print(f"\nCaricamento di {total_records} record in {num_batches} batch...")

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch = records[start:end]
            
            response = requests.post(url, headers=headers, json=batch)
            if response.status_code >= 400:
                print(f"❌ ERRORE NEL BATCH {i+1}: Status Code {response.status_code}")
                print("Dettagli errore:", response.text)
            
            response.raise_for_status()
            print(f"  ✅ Batch {i+1}/{num_batches} uploaded ({len(batch)} records)")
        
        print(f"\n✅ UPLOAD COMPLETATO per la tabella '{table_name}'!")

    except Exception as e:
        print(f"\n❌ Errore durante il processo di upload: {e}")

if __name__ == '__main__':
    # Questo blocco viene eseguito solo se chiami "python data/upload_to_supabase.py" direttamente
    print("--- Esecuzione in modalità standalone ---")
    df_to_upload = pd.read_csv(project_root / 'data' / 'processed' / 'matches_with_all_features.csv')
    df_to_upload['date'] = pd.to_datetime(df_to_upload['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    upload_dataframe_to_supabase(df_to_upload, 'matches')
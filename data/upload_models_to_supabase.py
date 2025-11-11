#!/usr/bin/env python3
"""
Upload models to Supabase Storage (one-time operation)
"""

from supabase import create_client
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# --- MODIFICA QUI: Usa la SUPABASE_SERVICE_KEY ---
# La service_role key ha pieni privilegi e bypassa le policy RLS,
# ideale per operazioni di backend come l'upload di modelli.
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY") # Usiamo la service_role key
)

model_dir = Path('models')

# Create bucket if doesn't exist
try:
    # Aggiungi un'opzione per rendere il bucket pubblico se necessario
    supabase.storage.create_bucket('models', options={'public': True})
    print("✅ Bucket 'models' creato.")
except Exception as e:
    # L'errore "Bucket exists" è previsto se il bucket è già stato creato
    # Ogni altro errore indica un problema (es. permessi)
    if "The resource already exists" in str(e): # Messaggio specifico per bucket già esistente
        print("ℹ️  Bucket 'models' esiste già.")
    else:
        print(f"❌ Errore durante la creazione del bucket: {e}")


# Files to upload
files = [
    'model_home_goals_v1.pkl',
    'model_away_goals_v1.pkl',
    'model_hda.pkl',
    'label_encoder_hda.pkl',
    'features_v1.json',
    'model_metadata_v1.json'
]

print("\n📤 Uploading models to Supabase Storage...")

for filename in files:
    filepath = model_dir / filename
    
    if not filepath.exists():
        print(f"⚠️  {filename} non trovato, salto il caricamento.")
        continue
    
    with open(filepath, 'rb') as f:
        try:
            # Tenta di caricare o aggiornare il file
            supabase.storage.from_('models').upload(
                filename, 
                f.read(), # Leggi il contenuto del file
                file_options={"upsert": "true"}  # Sovrascrivi se esiste
            )
            print(f"✅ Caricato: {filename}")
        except Exception as e:
            print(f"❌ Fallito {filename}: {e}")

print("\n✅ Caricamento completato!")
print("\nURL dei modelli:")
base_url = f"{os.getenv('SUPABASE_URL')}/storage/v1/object/public/models"
for filename in files:
    print(f"  {base_url}/{filename}")
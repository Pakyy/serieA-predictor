# Complete Data Pipeline

Questo script orchestrato (`pipeline_complete.py`) esegue automaticamente tutte le fasi di processing dei dati:

## Workflow

1. **Scraping xG data** (da Understat)
2. **Cleaning dati base** (da CSV raw)
3. **Integrazione xG** con dati base
4. **Calcolo ELO ratings**
5. **Calcolo features** (form, rolling stats, rest days, league position)
6. **Calcolo xG rolling features**
7. **Training modelli ML** (regressione + classificazione HDA)
8. **Salvataggio** (dataset + modelli)

## Uso

### Esecuzione completa (con scraping)
```bash
cd "/Users/paky/VSCode Projects/SerieA_Prediction"
source venv/bin/activate
python data/pipeline_complete.py
```

### Esecuzione senza scraping (usa dati esistenti)
```bash
python data/pipeline_complete.py --skip-scrape
```

### Esecuzione senza upload Supabase
```bash
python data/pipeline_complete.py --skip-upload
```

### Entrambe le opzioni
```bash
python data/pipeline_complete.py --skip-scrape --skip-upload
```

## Output Files

Dopo l'esecuzione, troverai:

- `data/processed/matches_clean.csv` - Dati puliti base
- `data/processed/matches_with_all_features.csv` - Dataset completo con tutte le features
- `data/processed/understat_xg_data.csv` - Dati xG scraped
- `models/model_home_goals_v1.pkl` - Modello regressione home goals
- `models/model_away_goals_v1.pkl` - Modello regressione away goals
- `models/model_hda.pkl` - Classificatore H/D/A
- `models/label_encoder_hda.pkl` - Encoder per label H/D/A
- `models/features_v1.json` - Metadata features
- `models/model_metadata_v1.json` - Metadata modelli

## Note

- Il pipeline gestisce automaticamente errori e fallback (es: se scraping fallisce, usa dati esistenti)
- Tutti i path sono relativi al progetto, funziona da qualsiasi directory
- I modelli vengono addestrati su train set fino a 2024-06-30, test su dati successivi

## Troubleshooting

Se vedi errori:
1. Verifica che i file CSV raw esistano in `data/raw/`
2. Verifica che le dipendenze siano installate (`pip install -r requirements.txt`)
3. Per problemi di scraping, usa `--skip-scrape` e usa dati esistenti


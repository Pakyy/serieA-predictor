import pandas as pd
from datetime import datetime

def load_and_clean_season(filepath, season):
    """
    Carica e pulisce un CSV stagione - SOLO stats, NO odds
    """
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, encoding='latin1')
    
    # SOLO colonne statistiche - NO odds/betting
    cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 
                    'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'FTR']
    
    # Check se colonne esistono (alcuni CSV potrebbero essere diversi)
    available_cols = [col for col in cols_to_keep if col in df.columns]
    missing_cols = [col for col in cols_to_keep if col not in df.columns]
    
    if missing_cols:
        print(f"  ⚠️  Missing columns: {missing_cols}")
    
    df = df[available_cols].copy()
    
    # Rinomina colonne in modo chiaro
    rename_map = {
        'Date': 'date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_goals',
        'FTAG': 'away_goals',
        'HS': 'home_shots',
        'AS': 'away_shots',
        'HST': 'home_shots_on_target',
        'AST': 'away_shots_on_target',
        'HC': 'home_corners',
        'AC': 'away_corners',
        'HF': 'home_fouls',
        'AF': 'away_fouls',
        'FTR': 'result'
    }
    
    df = df.rename(columns=rename_map)
    
    # Converti date (formato potrebbe variare)
    try:
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    except:
        try:
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')
        except:
            print(f"  ⚠️  Date format issue, trying automatic parsing")
            df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    
    # Aggiungi season
    df['season'] = season
    
    # Drop righe con dati mancanti (partite non giocate)
    initial_rows = len(df)
    df = df.dropna(subset=['home_goals', 'away_goals'])
    dropped_rows = initial_rows - len(df)
    
    if dropped_rows > 0:
        print(f"  Dropped {dropped_rows} incomplete matches")
    
    # Convert goals to int
    df['home_goals'] = df['home_goals'].astype(int)
    df['away_goals'] = df['away_goals'].astype(int)
    
    print(f"  ✅ Loaded {len(df)} matches")
    
    return df

def main():
    print("="*60)
    print("CLEANING SERIE A DATA - STATS ONLY (NO ODDS)")
    print("="*60)
    
    # Carica tutte le stagioni
    df_21_22 = load_and_clean_season('raw/SerieA_2021-22.csv', '2021-22')
    df_22_23 = load_and_clean_season('raw/SerieA_2022-23.csv', '2022-23')
    df_23_24 = load_and_clean_season('raw/SerieA_2023-24.csv', '2023-24')
    df_24_25 = load_and_clean_season('raw/SerieA_2024-25.csv', '2024-25')
    df_25_26 = load_and_clean_season('raw/SerieA_2025-26.csv', '2025-26')
    
    # Concatena tutto
    df_all = pd.concat([df_21_22, df_22_23, df_23_24, df_24_25, df_25_26], ignore_index=True)
    
    # Sort per data
    df_all = df_all.sort_values('date').reset_index(drop=True)
    
    # Info finali
    print("\n" + "="*60)
    print("FINAL DATASET")
    print("="*60)
    print(f"Total matches: {len(df_all)}")
    print(f"Date range: {df_all['date'].min().date()} to {df_all['date'].max().date()}")
    print(f"Unique teams: {df_all['home_team'].nunique()}")
    print(f"\nTeams:")
    for i, team in enumerate(sorted(df_all['home_team'].unique()), 1):
        print(f"  {i:2d}. {team}")
    
    print(f"\nColumns: {list(df_all.columns)}")
    print(f"\nSample data:")
    print(df_all.head(3))
    
    # Salva
    output_path = 'processed/matches_clean.csv'
    df_all.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
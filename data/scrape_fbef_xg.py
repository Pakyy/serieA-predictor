import asyncio
import aiohttp
from understat import Understat
import pandas as pd
from datetime import datetime
import os

async def get_league_matches(season):
    """
    Fetch tutti i match Serie A per una stagione
    """
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        
        print(f"Fetching Serie A {season}...")
        
        try:
            # Get league matches
            matches = await understat.get_league_results(
                'Serie_A',
                season
            )
            
            print(f"  ✅ Found {len(matches)} matches")
            return matches
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None


async def scrape_multiple_seasons():
    """
    Scarica multiple seasons
    """
    # Understat seasons sono anno inizio (2023 = 2023/24)
    seasons = [2021, 2022, 2023, 2024, 2025]
    
    all_matches = []
    
    for season in seasons:
        matches = await get_league_matches(season)
        
        if matches:
            # Converti in DataFrame
            df = pd.DataFrame(matches)
            
            # Aggiungi season label
            df['season'] = f"{season}-{str(season+1)[-2:]}"
            
            all_matches.append(df)
            
            print(f"  Season {season}: {len(df)} matches\n")
            
            # Small delay
            await asyncio.sleep(1)
    
    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        return combined
    
    return None


def clean_understat_data(df):
    """
    Pulisci dati understat
    """
    print("\nCleaning data...")
    
    # Rinomina colonne
    rename_map = {
        'datetime': 'date',
        'h': 'home_team_dict',
        'a': 'away_team_dict',
        'goals': 'goals_dict',
        'xG': 'xG_dict'
    }
    
    df = df.rename(columns=rename_map)
    
    # Extract team names
    df['home_team'] = df['home_team_dict'].apply(lambda x: x['title'] if isinstance(x, dict) else None)
    df['away_team'] = df['away_team_dict'].apply(lambda x: x['title'] if isinstance(x, dict) else None)
    
    # Extract goals
    df['home_goals'] = df['goals_dict'].apply(lambda x: int(x['h']) if isinstance(x, dict) else None)
    df['away_goals'] = df['goals_dict'].apply(lambda x: int(x['a']) if isinstance(x, dict) else None)
    
    # Extract xG
    df['home_xG'] = df['xG_dict'].apply(lambda x: float(x['h']) if isinstance(x, dict) else None)
    df['away_xG'] = df['xG_dict'].apply(lambda x: float(x['a']) if isinstance(x, dict) else None)
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    
    # Select final columns
    cols_final = ['date', 'season', 'home_team', 'away_team', 
                  'home_goals', 'away_goals', 'home_xG', 'away_xG']
    
    df_clean = df[cols_final].copy()
    
    # Sort by date
    df_clean = df_clean.sort_values('date').reset_index(drop=True)
    
    print(f"✅ Cleaned {len(df_clean)} matches")
    
    return df_clean


async def main():
    print("="*60)
    print("UNDERSTAT xG SCRAPING - SERIE A")
    print("="*60)
    
    # Scrape
    df_raw = await scrape_multiple_seasons()
    
    if df_raw is None:
        print("\n❌ Scraping failed")
        return
    
    # Clean
    df_clean = clean_understat_data(df_raw)
    
    # Stats
    print("\n" + "="*60)
    print("FINAL DATASET")
    print("="*60)
    print(f"Total matches: {len(df_clean)}")
    print(f"Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
    print(f"Seasons: {df_clean['season'].unique()}")
    
    print("\nxG Statistics:")
    print(f"  Home xG avg: {df_clean['home_xG'].mean():.2f}")
    print(f"  Away xG avg: {df_clean['away_xG'].mean():.2f}")
    print(f"  Total xG avg: {(df_clean['home_xG'] + df_clean['away_xG']).mean():.2f}")
    
    print("\nSample data:")
    print(df_clean.head(10))
    
    # Save
    base_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(base_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, 'understat_xg_data.csv')
    df_clean.to_csv(output_path, index=False)
    print(f"\n✅ Saved xG dataset to: {output_path}")
    
    return df_clean


if __name__ == "__main__":
    # Run async function
    df = asyncio.run(main())
#!/usr/bin/env python3
"""
Complete Data Pipeline - Orchestrates all data processing steps:
1. Scrape xG data from Understat
2. Clean base match data
3. Integrate xG with base data
4. Calculate ELO ratings
5. Calculate all features (form, rolling stats, etc.)
6. Integrate xG rolling features
7. Train models (regression + HDA classification)
8. Save everything

Usage:
    python data/pipeline_complete.py [--skip-scrape] [--skip-upload]
"""

import os
import sys
import argparse
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import existing modules (will be imported when needed)


def calculate_elo_rating(df, k_factor=20, initial_rating=1500, regression_factor=0.75):
    """
    Calculate ELO ratings for all teams, with regression to the mean between seasons.
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    elo_ratings = {}
    home_elo_before = []
    away_elo_before = []
    current_season = None
    
    for _, row in df.iterrows():
        season = row['season']
        home_team = row['home_team']
        away_team = row['away_team']
        result = row['result']
        
        # Check for season change and apply regression
        if season != current_season:
            if current_season is not None:  # Apply regression if not the very first season
                print(f"New season ({season}), applying ELO regression...")
                for team in elo_ratings:
                    elo_ratings[team] = (elo_ratings[team] * regression_factor) + (initial_rating * (1 - regression_factor))
            current_season = season

        # Initialize teams if they appear for the first time
        if home_team not in elo_ratings:
            elo_ratings[home_team] = initial_rating
        if away_team not in elo_ratings:
            elo_ratings[away_team] = initial_rating
            
        # Save ELO BEFORE the match
        home_elo_before.append(elo_ratings[home_team])
        away_elo_before.append(elo_ratings[away_team])
        
        # Calculate expected scores
        expected_home = 1 / (1 + 10 ** ((elo_ratings[away_team] - elo_ratings[home_team]) / 400))
        expected_away = 1 - expected_home
        
        # Determine actual scores from result
        if result == 'H':
            actual_home, actual_away = 1.0, 0.0
        elif result == 'A':
            actual_home, actual_away = 0.0, 1.0
        else:  # Draw
            actual_home, actual_away = 0.5, 0.5
        
        # Update ratings
        elo_ratings[home_team] += k_factor * (actual_home - expected_home)
        elo_ratings[away_team] += k_factor * (actual_away - expected_away)
    
    # Add ELO columns to the dataframe
    df['home_elo_before'] = home_elo_before
    df['away_elo_before'] = away_elo_before
    df['elo_diff'] = df['home_elo_before'] - df['away_elo_before']
    
    return df, elo_ratings



def calculate_form(df, window=5):
    """
    Calculate form by looking back at the last N matches from the date of each game.
    This method is stateless and more robust.
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create a long-format dataframe of all matches played by each team with points earned
    home_matches = df[['date', 'season', 'home_team', 'result']].rename(columns={'home_team': 'team'})
    home_matches['points'] = home_matches['result'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
    
    away_matches = df[['date', 'season', 'away_team', 'result']].rename(columns={'away_team': 'team'})
    away_matches['points'] = away_matches['result'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))
    
    # Combine and sort chronologically
    all_matches = pd.concat([home_matches, away_matches]).sort_values('date').reset_index(drop=True)

    # Group by team for efficient lookup
    team_matches_lookup = all_matches.groupby('team')
    
    home_form_list = []
    away_form_list = []
    
    # Iterate through each match to calculate the form at that point in time
    for _, row in df.iterrows():
        match_date = row['date']
        season = row['season']
        home_team = row['home_team']
        away_team = row['away_team']
        
        # --- Calculate Home Team Form ---
        try:
            home_history = team_matches_lookup.get_group(home_team)
            # Filter for past matches within the same season
            home_past_season_games = home_history[
                (home_history['date'] < match_date) & 
                (home_history['season'] == season)
            ]
            # Get points from the last N games
            form_points = home_past_season_games.tail(window)['points'].sum()
            home_form_list.append(form_points if not home_past_season_games.empty else 0)
        except KeyError:
            home_form_list.append(0)  # Team has no history
            
        # --- Calculate Away Team Form ---
        try:
            away_history = team_matches_lookup.get_group(away_team)
            # Filter for past matches within the same season
            away_past_season_games = away_history[
                (away_history['date'] < match_date) & 
                (away_history['season'] == season)
            ]
            # Get points from the last N games
            form_points = away_past_season_games.tail(window)['points'].sum()
            away_form_list.append(form_points if not away_past_season_games.empty else 0)
        except KeyError:
            away_form_list.append(0)  # Team has no history

    df['home_form_5'] = home_form_list
    df['away_form_5'] = away_form_list
    df['form_diff'] = df['home_form_5'] - df['away_form_5']
    
    return df


def calculate_rest_days(df):
    """
    Calculate days between matches for each team
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    last_match_date = {}
    home_rest = []
    away_rest = []
    
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        match_date = row['date']
        
        if home_team in last_match_date:
            days = (match_date - last_match_date[home_team]).days
            home_rest.append(days)
        else:
            home_rest.append(None)
        
        if away_team in last_match_date:
            days = (match_date - last_match_date[away_team]).days
            away_rest.append(days)
        else:
            away_rest.append(None)
        
        last_match_date[home_team] = match_date
        last_match_date[away_team] = match_date
    
    df['home_rest_days'] = home_rest
    df['away_rest_days'] = away_rest
    df['rest_days_diff'] = df['home_rest_days'] - df['away_rest_days']
    
    return df


def calculate_league_position(df):
    """
    Calculate league position for each team before each match
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    season_points = {}
    home_position = []
    away_position = []
    home_points = []
    away_points = []
    
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        season = row['season']
        result = row['result']
        
        if season not in season_points:
            season_points[season] = {}
        
        if home_team not in season_points[season]:
            season_points[season][home_team] = 0
        if away_team not in season_points[season]:
            season_points[season][away_team] = 0
        
        # Get current points BEFORE match
        current_home_points = season_points[season][home_team]
        current_away_points = season_points[season][away_team]
        
        # Calculate positions
        season_table = sorted(season_points[season].items(), 
                             key=lambda x: x[1], reverse=True)
        
        home_pos = next((i+1 for i, (team, pts) in enumerate(season_table) 
                        if team == home_team), None)
        away_pos = next((i+1 for i, (team, pts) in enumerate(season_table) 
                        if team == away_team), None)
        
        home_position.append(home_pos)
        away_position.append(away_pos)
        home_points.append(current_home_points)
        away_points.append(current_away_points)
        
        # Update points AFTER match
        if result == 'H':
            season_points[season][home_team] += 3
        elif result == 'A':
            season_points[season][away_team] += 3
        else:  # Draw
            season_points[season][home_team] += 1
            season_points[season][away_team] += 1
    
    df['home_position'] = home_position
    df['away_position'] = away_position
    df['position_diff'] = df['away_position'] - df['home_position']
    df['home_points'] = home_points
    df['away_points'] = away_points
    df['points_diff'] = df['home_points'] - df['away_points']
    
    return df


def calculate_rolling_stats(df, windows=[3, 5]):
    """
    Calculate rolling averages for goals, shots, etc.
    """
    team_stats = {team: {
        'goals_scored': [],
        'goals_conceded': [],
        'shots': []
    } for team in df['home_team'].unique()}
    
    for window in windows:
        home_cols = {
            f'home_goals_avg_{window}': [],
            f'home_conceded_avg_{window}': [],
            f'home_shots_avg_{window}': []
        }
        away_cols = {
            f'away_goals_avg_{window}': [],
            f'away_conceded_avg_{window}': [],
            f'away_shots_avg_{window}': []
        }
        
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Calculate averages BEFORE match
            home_recent_goals = team_stats[home_team]['goals_scored'][-window:]
            home_recent_conceded = team_stats[home_team]['goals_conceded'][-window:]
            home_recent_shots = team_stats[home_team]['shots'][-window:]
            
            away_recent_goals = team_stats[away_team]['goals_scored'][-window:]
            away_recent_conceded = team_stats[away_team]['goals_conceded'][-window:]
            away_recent_shots = team_stats[away_team]['shots'][-window:]
            
            home_cols[f'home_goals_avg_{window}'].append(
                np.mean(home_recent_goals) if len(home_recent_goals) >= window else None
            )
            home_cols[f'home_conceded_avg_{window}'].append(
                np.mean(home_recent_conceded) if len(home_recent_conceded) >= window else None
            )
            home_cols[f'home_shots_avg_{window}'].append(
                np.mean(home_recent_shots) if len(home_recent_shots) >= window else None
            )
            
            away_cols[f'away_goals_avg_{window}'].append(
                np.mean(away_recent_goals) if len(away_recent_goals) >= window else None
            )
            away_cols[f'away_conceded_avg_{window}'].append(
                np.mean(away_recent_conceded) if len(away_recent_conceded) >= window else None
            )
            away_cols[f'away_shots_avg_{window}'].append(
                np.mean(away_recent_shots) if len(away_recent_shots) >= window else None
            )
            
            # Update stats AFTER match
            team_stats[home_team]['goals_scored'].append(row['home_goals'])
            team_stats[home_team]['goals_conceded'].append(row['away_goals'])
            team_stats[home_team]['shots'].append(row.get('home_shots', 0))
            
            team_stats[away_team]['goals_scored'].append(row['away_goals'])
            team_stats[away_team]['goals_conceded'].append(row['home_goals'])
            team_stats[away_team]['shots'].append(row.get('away_shots', 0))
        
        # Add columns
        for col_name, values in {**home_cols, **away_cols}.items():
            df[col_name] = values
    
    return df


def calculate_xg_rolling(df, window=15, max_age_months=12):
    """
    Calculate xG rolling averages (last N matches, max X months old)
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    team_xg_for = {}
    team_xg_against = {}
    
    home_xg_avg = []
    away_xg_avg = []
    home_xga_avg = []
    away_xga_avg = []
    
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        match_date = row['date']
        
        # Initialize teams
        if home_team not in team_xg_for:
            team_xg_for[home_team] = []
            team_xg_against[home_team] = []
        if away_team not in team_xg_for:
            team_xg_for[away_team] = []
            team_xg_against[away_team] = []
        
        # Get recent xG (last N matches, max X months)
        cutoff_date = match_date - timedelta(days=max_age_months*30)
        
        home_xg_recent = [
            (date, xg) for date, xg in team_xg_for[home_team]
            if date >= cutoff_date
        ][-window:]
        
        home_xga_recent = [
            (date, xga) for date, xga in team_xg_against[home_team]
            if date >= cutoff_date
        ][-window:]
        
        away_xg_recent = [
            (date, xg) for date, xg in team_xg_for[away_team]
            if date >= cutoff_date
        ][-window:]
        
        away_xga_recent = [
            (date, xga) for date, xga in team_xg_against[away_team]
            if date >= cutoff_date
        ][-window:]
        
        # Calculate averages (None if <3 matches)
        home_xg_avg.append(np.mean([xg for _, xg in home_xg_recent]) if len(home_xg_recent) >= 3 else None)
        home_xga_avg.append(np.mean([xga for _, xga in home_xga_recent]) if len(home_xga_recent) >= 3 else None)
        away_xg_avg.append(np.mean([xg for _, xg in away_xg_recent]) if len(away_xg_recent) >= 3 else None)
        away_xga_avg.append(np.mean([xga for _, xga in away_xga_recent]) if len(away_xga_recent) >= 3 else None)
        
        # Update history AFTER calculation
        if pd.notna(row.get('home_xG')):
            team_xg_for[home_team].append((match_date, row['home_xG']))
            team_xg_against[home_team].append((match_date, row['away_xG']))
            
            team_xg_for[away_team].append((match_date, row['away_xG']))
            team_xg_against[away_team].append((match_date, row['home_xG']))
    
    df['home_xG_avg_rolling'] = home_xg_avg
    df['away_xG_avg_rolling'] = away_xg_avg
    df['home_xGA_avg_rolling'] = home_xga_avg
    df['away_xGA_avg_rolling'] = away_xga_avg
    
    return df


def integrate_xg_data(df_main, df_xg):
    """
    Integrate xG data with main dataset
    """
    # Standardize team names
    team_mapping = {
        'AC Milan': 'Milan',
        'Parma Calcio 1913': 'Parma',
    }
    
    df_xg['home_team'] = df_xg['home_team'].replace(team_mapping)
    df_xg['away_team'] = df_xg['away_team'].replace(team_mapping)
    
    # Round dates to day
    df_main['date_only'] = df_main['date'].dt.date
    df_xg['date_only'] = df_xg['date'].dt.date
    
    # Merge
    df_merged = df_main.merge(
        df_xg[['date_only', 'home_team', 'away_team', 'home_xG', 'away_xG']],
        left_on=['date_only', 'home_team', 'away_team'],
        right_on=['date_only', 'home_team', 'away_team'],
        how='left',
        indicator=True
    )
    
    # Drop helper column
    df_merged = df_merged.drop(columns=['date_only', '_merge'])
    
    print(f"✅ xG integrated: {(df_merged['home_xG'].notna()).sum()} / {len(df_merged)} matches have xG")
    
    return df_merged


def train_models(df, model_dir):
    """
    Train regression models (home/away goals) and HDA classifier
    """
    from xgboost import XGBRegressor, XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
    
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    # Feature list
    base_features = ['home_elo_before', 'away_elo_before', 'elo_diff',
                     'home_form_5', 'away_form_5', 'form_diff',
                     'home_goals_avg_5', 'away_goals_avg_5',
                     'home_conceded_avg_5', 'away_conceded_avg_5',
                     'home_shots_avg_5', 'away_shots_avg_5']
    
    xg_features = ['home_xG_avg_rolling', 'away_xG_avg_rolling',
                   'home_xGA_avg_rolling', 'away_xGA_avg_rolling']
    
    features_model = base_features + xg_features
    
    # Drop rows with missing features
    df_clean = df.dropna(subset=features_model).copy()
    print(f"Dataset after dropping NaN: {len(df_clean)} matches")
    
    # Train/Test split (temporal)
    train_cutoff = '2024-06-30'
    train_df = df_clean[df_clean['date'] <= train_cutoff].copy()
    test_df = df_clean[df_clean['date'] > train_cutoff].copy()
    
    print(f"Train: {len(train_df)} matches")
    print(f"Test:  {len(test_df)} matches")
    
    X_train = train_df[features_model]
    y_train_home = train_df['home_goals']
    y_train_away = train_df['away_goals']
    
    X_test = test_df[features_model]
    y_test_home = test_df['home_goals']
    y_test_away = test_df['away_goals']
    y_test_result = test_df['result']
    
    # ============================================================
    # Train Regression Models
    # ============================================================
    print("\nTraining regression models (home/away goals)...")
    
    model_home = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model_away = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model_home.fit(X_train, y_train_home)
    model_away.fit(X_train, y_train_away)
    
    # Evaluate regression
    pred_home = np.maximum(model_home.predict(X_test), 0)
    pred_away = np.maximum(model_away.predict(X_test), 0)
    
    mae_home = mean_absolute_error(y_test_home, pred_home)
    mae_away = mean_absolute_error(y_test_away, pred_away)
    
    print(f"✅ Regression models trained")
    print(f"   Test MAE home: {mae_home:.3f}")
    print(f"   Test MAE away: {mae_away:.3f}")
    
    # ============================================================
    # Train HDA Classifier
    # ============================================================
    print("\nTraining HDA classifier...")
    
    le = LabelEncoder()
    y_train_hda_enc = le.fit_transform(train_df['result'])
    y_test_hda_enc = le.transform(test_df['result'])
    
    model_hda = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        subsample=0.9,
        colsample_bytree=0.8
    )
    
    model_hda.fit(X_train, y_train_hda_enc)
    
    # Evaluate classifier
    preds_label_enc = model_hda.predict(X_test)
    preds_label = le.inverse_transform(preds_label_enc)
    acc_hda = accuracy_score(y_test_result, preds_label)
    
    print(f"✅ HDA classifier trained")
    print(f"   Test accuracy: {acc_hda:.3f}")
    print(classification_report(y_test_result, preds_label))
    
    # ============================================================
    # Save Models
    # ============================================================
    os.makedirs(model_dir, exist_ok=True)
    
    # Save regression models
    joblib.dump(model_home, os.path.join(model_dir, 'model_home_goals_v1.pkl'))
    joblib.dump(model_away, os.path.join(model_dir, 'model_away_goals_v1.pkl'))
    
    # Save HDA classifier
    joblib.dump(model_hda, os.path.join(model_dir, 'model_hda.pkl'))
    joblib.dump(le, os.path.join(model_dir, 'label_encoder_hda.pkl'))
    
    # Save feature list
    with open(os.path.join(model_dir, 'features_v1.json'), 'w') as f:
        json.dump({
            'features': features_model,
            'n_features': len(features_model),
            'base_features': base_features,
            'xg_features': xg_features
        }, f, indent=2)
    
    # Save metadata
    metadata = {
        'model_version': 'v1.0',
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_samples': len(train_df),
        'test_samples': len(test_df),
        'mae_home': float(mae_home),
        'mae_away': float(mae_away),
        'hda_accuracy': float(acc_hda),
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 1.0
        }
    }
    
    with open(os.path.join(model_dir, 'model_metadata_v1.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ All models saved to {model_dir}")
    
    return model_home, model_away, model_hda


def main(skip_scrape=False, skip_upload=False):
    """
    Main pipeline orchestrator
    """
    print("="*60)
    print("COMPLETE DATA PIPELINE")
    print("="*60)
    
    data_dir = PROJECT_ROOT / 'data'
    processed_dir = data_dir / 'processed'
    model_dir = PROJECT_ROOT / 'models'
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # ============================================================
    # STEP 1: Scrape xG data
    # ============================================================
    if not skip_scrape:
        print("\n" + "="*60)
        print("STEP 1: Scraping xG data from Understat")
        print("="*60)
        
        try:
            # Import here to handle async properly
            from data.scrape_fbef_xg import main as scrape_xg_main
            df_xg = asyncio.run(scrape_xg_main())
            if df_xg is None:
                raise ValueError("Scraping returned None")
        except Exception as e:
            print(f"❌ Error in xG scraping: {e}")
            print("⚠️  Trying to load existing xG data...")
            xg_path = processed_dir / 'understat_xg_data.csv'
            if xg_path.exists():
                df_xg = pd.read_csv(xg_path)
                df_xg['date'] = pd.to_datetime(df_xg['date'])
                print(f"✅ Loaded existing xG data: {len(df_xg)} matches")
            else:
                raise FileNotFoundError("No xG data available. Run without --skip-scrape")
    else:
        print("\n⏭️  Skipping xG scraping (using existing data)")
        xg_path = processed_dir / 'understat_xg_data.csv'
        if xg_path.exists():
            df_xg = pd.read_csv(xg_path)
            df_xg['date'] = pd.to_datetime(df_xg['date'])
            print(f"✅ Loaded existing xG data: {len(df_xg)} matches")
        else:
            raise FileNotFoundError("No xG data available. Run without --skip-scrape")
    
    # ============================================================
    # STEP 2: Clean base match data
    # ============================================================
    print("\n" + "="*60)
    print("STEP 2: Cleaning base match data")
    print("="*60)
    
    # Import and run cleaning
    from data.clean_data import load_and_clean_season
    
    # Load all seasons
    df_21_22 = load_and_clean_season(str(data_dir / 'raw' / 'SerieA_2021-22.csv'), '2021-22')
    df_22_23 = load_and_clean_season(str(data_dir / 'raw' / 'SerieA_2022-23.csv'), '2022-23')
    df_23_24 = load_and_clean_season(str(data_dir / 'raw' / 'SerieA_2023-24.csv'), '2023-24')
    df_24_25 = load_and_clean_season(str(data_dir / 'raw' / 'SerieA_2024-25.csv'), '2024-25')
    df_25_26 = load_and_clean_season(str(data_dir / 'raw' / 'SerieA_2025-26.csv'), '2025-26')
    
    # Concatenate
    df_main = pd.concat([df_21_22, df_22_23, df_23_24, df_24_25, df_25_26], ignore_index=True)
    df_main = df_main.sort_values('date').reset_index(drop=True)
    
    # Ensure result column exists (map from goals if needed)
    if 'result' not in df_main.columns:
        df_main['result'] = df_main.apply(
            lambda row: 'H' if row['home_goals'] > row['away_goals'] 
            else ('A' if row['away_goals'] > row['home_goals'] else 'D'), 
            axis=1
        )
    
    # Save cleaned data
    output_clean = processed_dir / 'matches_clean.csv'
    df_main.to_csv(output_clean, index=False)
    
    df_main['date'] = pd.to_datetime(df_main['date'])
    print(f"✅ Cleaned and loaded data: {len(df_main)} matches")
    
    # ============================================================
    # STEP 3: Integrate xG data
    # ============================================================
    print("\n" + "="*60)
    print("STEP 3: Integrating xG data")
    print("="*60)
    
    df = integrate_xg_data(df_main, df_xg)
    
    # ============================================================
    # STEP 4: Calculate ELO ratings
    # ============================================================
    print("\n" + "="*60)
    print("STEP 4: Calculating ELO ratings")
    print("="*60)
    
    df, final_elo = calculate_elo_rating(df)
    print(f"✅ ELO calculated for {len(final_elo)} teams")
    
    # ============================================================
    # STEP 5: Calculate form and rolling stats
    # ============================================================
    print("\n" + "="*60)
    print("STEP 5: Calculating form and rolling statistics")
    print("="*60)
    
    df = calculate_form(df, window=5)
    df = calculate_rolling_stats(df, windows=[3, 5])
    df = calculate_rest_days(df)
    df = calculate_league_position(df)
    print(f"✅ Form, rolling stats, rest days, and league positions calculated")
    
    # ============================================================
    # STEP 6: Calculate xG rolling features
    # ============================================================
    print("\n" + "="*60)
    print("STEP 6: Calculating xG rolling features")
    print("="*60)
    
    df = calculate_xg_rolling(df, window=15, max_age_months=12)
    print(f"✅ xG rolling features calculated")
    print(f"   Non-null xG features: {df['home_xG_avg_rolling'].notna().sum()} / {len(df)}")
    
    # ============================================================
    # STEP 7: Save intermediate dataset
    # ============================================================
    output_path = processed_dir / 'matches_with_all_features.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved dataset with all features: {output_path}")
    print(f"   Total matches: {len(df)}")
    print(f"   Total features: {len(df.columns)}")
    
    # ============================================================
    # STEP 8: Train models
    # ============================================================
    print("\n" + "="*60)
    print("STEP 8: Training ML models")
    print("="*60)
    
    train_models(df, model_dir)
    
# ============================================================
# STEP 9: Upload to Supabase (optional)
# ============================================================
    if not skip_upload:
        print("\n" + "="*60)
        print("STEP 9: Uploading to Supabase")
        print("="*60)
        
        try:
            from data.upload_to_supabase import upload_dataframe_to_supabase

            # Creiamo una copia del DataFrame per l'upload
            df_for_upload = df.copy()
            df_for_upload['date'] = pd.to_datetime(df_for_upload['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # --- Correzione Case-Sensitivity ---
            # Rendi tutti i nomi delle colonne minuscoli per evitare errori con PostgreSQL
            df_for_upload.columns = [col.lower() for col in df_for_upload.columns]
            # --- Fine Correzione ---

            print(f"Uploading {len(df_for_upload)} rows to Supabase table 'matches'...")
            upload_dataframe_to_supabase(df_for_upload, 'matches')
            print("✅ Data successfully uploaded to Supabase.")

        except ImportError:
            print("⚠️ Could not import 'upload_dataframe_to_supabase'.")
            print("   Make sure 'data/upload_to_supabase.py' has this function and is in the correct path.")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            print("   Please ensure your .env file is correctly set up with SUPABASE_URL and SUPABASE_KEY.")
    else:
        print("\n⏭️  Skipping Supabase upload.")

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)

    # ============================================================
    # DEBUGGING STEP: Verify final form values from local CSV
    # ============================================================
    print("\n" + "="*60)
    print("DEBUG: Verifying latest form values in generated CSV...")
    print("="*60)
    try:
        final_df = pd.read_csv(output_path, parse_dates=['date'])
        teams_to_check = ['Lecce', 'Napoli']
        for team_name in teams_to_check:
            last_match = final_df[
                (final_df['home_team'] == team_name) | (final_df['away_team'] == team_name)
            ].sort_values('date', ascending=False)
            
            if not last_match.empty:
                last_match = last_match.iloc[0]
                form = last_match['home_form_5'] if last_match['home_team'] == team_name else last_match['away_form_5']
                print(f"-> {team_name}'s last match date in data: {last_match['date'].date()}")
                print(f"  -> Calculated form for this match: {form}")
            else:
                print(f"-> No matches found for {team_name} in the dataset.")
    except Exception as e:
        print(f"Could not perform debug check: {e}")

    print("\nOutput files:")
    print(f"  - {processed_dir / 'matches_with_all_features.csv'}")

    print(f"  - {model_dir / 'model_home_goals_v1.pkl'}")
    print(f"  - {model_dir / 'model_away_goals_v1.pkl'}")
    print(f"  - {model_dir / 'model_hda.pkl'}")
    print(f"  - {model_dir / 'label_encoder_hda.pkl'}")
    print(f"  - {model_dir / 'features_v1.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Complete data pipeline')
    parser.add_argument('--skip-scrape', action='store_true', help='Skip xG scraping (use existing data)')
    parser.add_argument('--skip-upload', action='store_true', help='Skip Supabase upload')
    
    args = parser.parse_args()
    
    main(skip_scrape=args.skip_scrape, skip_upload=args.skip_upload)
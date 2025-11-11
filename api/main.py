import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import requests # Nuovo import
import tempfile # Nuovo import
from pathlib import Path # Già presente, ma assicuriamo
from datetime import datetime
from scipy.stats import poisson

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv # Nuovo import per .env in produzione

print("=== MAIN.PY LOADED FRESH ===")

# Carica le variabili d'ambiente (necessario in produzione per SUPABASE_URL)
load_dotenv()

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import database functions
from api.database import get_latest_team_stats, update_elo_cache


# ============================================================
# MODEL LOADING (from Supabase Storage in production)
# ============================================================

def download_models_from_supabase():
    """
    Download models from Supabase Storage
    Used in production (Render)
    """
    print("🔄 Downloading models from Supabase Storage...")
    
    # Assicurati che SUPABASE_URL sia presente nel .env o nell'ambiente
    supabase_url = os.getenv('SUPABASE_URL')
    if not supabase_url:
        raise ValueError("SUPABASE_URL non è impostato nelle variabili d'ambiente.")
        
    base_url = f"{supabase_url}/storage/v1/object/public/models"
    
    model_files = [
        'model_home_goals_v1.pkl',
        'model_away_goals_v1.pkl',
        'model_hda.pkl',
        'label_encoder_hda.pkl',
        'features_v1.json',
        'model_metadata_v1.json'
    ]
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())
    
    for filename in model_files:
        url = f"{base_url}/{filename}"
        response = requests.get(url)
        
        if response.status_code == 200:
            filepath = temp_dir / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ Failed to download {filename}: {response.status_code}")
            raise Exception(f"Model download failed: {filename}")
    
    print("✅ All models downloaded")
    return temp_dir

def load_models():
    """
    Load models from local (dev) or Supabase (production)
    """
    # Check if running in production (Render sets PORT env var)
    is_production = os.getenv('PORT') is not None
    
    if is_production:
        print("🌐 Production mode: downloading models from Supabase")
        model_dir = download_models_from_supabase()
    else:
        print("💻 Development mode: loading models from local")
        model_dir = Path(BASE_DIR) / 'models' # Usa BASE_DIR per il percorso locale
    
    # Load models
    model_home = joblib.load(model_dir / 'model_home_goals_v1.pkl')
    model_away = joblib.load(model_dir / 'model_away_goals_v1.pkl')
    
    # Load HDA classifier (con gestione dell'assenza)
    model_hda = None
    label_encoder = None
    try:
        model_hda = joblib.load(model_dir / 'model_hda.pkl')
        label_encoder = joblib.load(model_dir / 'label_encoder_hda.pkl')
        print("✅ HDA classifier loaded")
    except FileNotFoundError:
        print("⚠️  model_hda.pkl not found, HDA predictions disabled")
    except Exception as e:
        print(f"❌ Error loading HDA classifier: {e}, HDA predictions disabled")

    with open(model_dir / 'features_v1.json') as f:
        features_metadata = json.load(f)
        feature_names = features_metadata['features']
    
    return model_home, model_away, model_hda, label_encoder, feature_names


# Initialize FastAPI
app = FastAPI(
    title="Serie A Match Predictor API",
    description="Predicts Serie A 2025-26 match outcomes using ML",
    version="1.0"
)

# Load models at startup
print("🚀 Loading models...")
# Assegniamo le variabili globali qui
model_home_global, model_away_global, model_hda_global, label_encoder_hda_global, REQUIRED_FEATURES_global = load_models()

# Assegniamo ai nomi originali usati nel codice per compatibilità
model_home = model_home_global
model_away = model_away_global
model_hda = model_hda_global
label_encoder_hda = label_encoder_hda_global
REQUIRED_FEATURES = REQUIRED_FEATURES_global

print("✅ Models loaded successfully")


# Initialize ELO cache on startup (if needed)
print("Initializing ELO cache...")
try:
    update_elo_cache(season='2025-26', force=False)
except Exception as e:
    print(f"⚠️  ELO cache initialization warning: {e}")


# Serie A 2025-26 teams
SERIE_A_2025_26 = {
    'Atalanta', 'Bologna', 'Cagliari', 'Cremonese', 'Como',
    'Fiorentina', 'Genoa', 'Inter', 'Juventus', 'Lazio',
    'Lecce', 'Milan', 'Napoli', 'Parma', 'Pisa', 
    'Roma', 'Sassuolo', 'Torino', 'Udinese', 'Verona'
}

# Derby matches (high unpredictability)
DERBIES = {
    ('Inter', 'Milan'), ('Milan', 'Inter'),
    ('Roma', 'Lazio'), ('Lazio', 'Roma'),
}

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    
class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    predicted_result: str
    probabilities: dict
    expected_goals: dict
    confidence: float
    is_derby: bool = False
    debug_info: dict

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_team_features(team_name: str, is_home: bool):
    """
    Fetch latest features for a team from database
    """
    try:
        stats = get_latest_team_stats(team_name, current_season='2025-26')
        return stats
    except Exception as e:
        print(f"❌ Error fetching stats for {team_name}: {e}")
        # Fallback to defaults
        return {
            'elo': 1500,
            'form_5': 6,
            'goals_avg_5': 1.3,
            'conceded_avg_5': 1.2,
            'shots_avg_5': 11.0,
            'xG_avg_rolling': 1.4,
            'xGA_avg_rolling': 1.3
        }


def calculate_feature_vector(home_team: str, away_team: str):
    """
    Build complete feature vector for prediction
    """
    home_features = get_team_features(home_team, is_home=True)
    away_features = get_team_features(away_team, is_home=False)
    
    features = {
        'home_elo_before': home_features['elo'],
        'away_elo_before': away_features['elo'],
        'elo_diff': home_features['elo'] - away_features['elo'],
        
        'home_form_5': home_features['form_5'],
        'away_form_5': away_features['form_5'],
        'form_diff': home_features['form_5'] - away_features['form_5'],
        
        'home_goals_avg_5': home_features['goals_avg_5'],
        'away_goals_avg_5': away_features['goals_avg_5'],
        
        'home_conceded_avg_5': home_features['conceded_avg_5'],
        'away_conceded_avg_5': away_features['conceded_avg_5'],
        
        'home_shots_avg_5': home_features['shots_avg_5'],
        'away_shots_avg_5': away_features['shots_avg_5'],
        
        'home_xG_avg_rolling': home_features['xG_avg_rolling'],
        'away_xG_avg_rolling': away_features['xG_avg_rolling'],
        
        'home_xGA_avg_rolling': home_features['xGA_avg_rolling'],
        'away_xGA_avg_rolling': away_features['xGA_avg_rolling']
    }
    
    df = pd.DataFrame([features])
    df = df[REQUIRED_FEATURES]
    
    return df


def adjust_goal_means(pred_home_goals: float, pred_away_goals: float, home_features: dict, away_features: dict):
    """
    Adjust the raw predicted goal means using signal-based biases so probabilities reflect clear mismatches.
    Signals used: elo_diff and xG avg difference (light weight), with safe caps.
    """
    print("*** DEBUG adjusted goals called", pred_home_goals, pred_away_goals, home_features.get('elo'), away_features.get('elo'))
    elo_diff = float(home_features['elo'] - away_features['elo'])
    xg_diff = float(home_features.get('xG_avg_rolling', 0.0) - away_features.get('xG_avg_rolling', 0.0))

    # Weights (conservative): elo has stronger impact than xG diff
    alpha_elo = 0.35   # per 400 Elo points, ~35% adjustment
    alpha_xg  = 0.08   # per 1 xG, ~8% adjustment

    elo_term = alpha_elo * (elo_diff / 400.0)
    xg_term  = alpha_xg * xg_diff

    # Total bias multiplier for home; away gets opposite
    bias = elo_term + xg_term

    # Cap bias to avoid extremes
    bias = max(min(bias, 0.25), -0.25)

    adj_home = pred_home_goals * max(0.5, (1.0 + bias))
    adj_away = pred_away_goals * max(0.5, (1.0 - bias))

    # Minimum floor to keep Poisson stable
    adj_home = max(adj_home, 0.05)
    adj_away = max(adj_away, 0.05)

    return adj_home, adj_away


def goals_to_result_probs(home_goals, away_goals, max_goals=8):
    """
    Convert predicted goals to 1X2 probabilities using Poisson distribution
    """
    home_probs = [poisson.pmf(i, home_goals) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_goals) for i in range(max_goals + 1)]
    
    prob_home_win = 0.0
    prob_draw = 0.0
    prob_away_win = 0.0
    
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = home_probs[h] * away_probs[a]
            
            if h > a:
                prob_home_win += prob
            elif h == a:
                prob_draw += prob
            else:
                prob_away_win += prob
    
    total = prob_home_win + prob_draw + prob_away_win
    
    return {
        'H': prob_home_win / total,
        'D': prob_draw / total,
        'A': prob_away_win / total
    }


def calibrate_predictions(probs, pred_home_goals, pred_away_goals):
    """
    Calibrate probabilities: allow sharper predictions when teams are clearly unequal,
    but smooth in case of virtual parity.
    """
    # Conservative blend only for balanced games
    goal_diff = abs(pred_home_goals - pred_away_goals)
    uniform = {'H': 0.333, 'D': 0.333, 'A': 0.333}

    if goal_diff <= 0.6:   # smooth only for tight games
        blend_factor = 0.35
    elif goal_diff <= 1.2: # still a bit conservative
        blend_factor = 0.15
    else:                  # don't smooth if clearly different
        blend_factor = 0.0

    calibrated = {}
    for key in probs:
        calibrated[key] = blend_factor * probs[key] + (1 - blend_factor) * uniform[key]

    # Renormalize
    total = sum(calibrated.values())
    return {k: round(v / total, 3) for k, v in calibrated.items()}


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Serie A 2025-26 Match Predictor API",
        "version": "1.0",
        "season": "2025-26",
        "endpoints": {
            "/predict": "POST - Predict match outcome",
            "/health": "GET - Health check",
            "/teams": "GET - List available teams"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "season": "2025-26",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/teams")
def list_teams():
    """Return list of Serie A 2025-26 teams"""
    return {
        "teams": sorted(list(SERIE_A_2025_26)),
        "season": "2025-26",
        "count": len(SERIE_A_2025_26)
    }


@app.post("/update-elo")
def force_update_elo(season: str = '2025-26'):
    """
    Force update ELO ratings for all teams (bypasses cache check)
    Useful to run at least once a week or after new matches are added
    """
    try:
        elo_ratings, was_updated = update_elo_cache(season, force=True)
        return {
            "status": "success",
            "message": f"ELO ratings updated for {len(elo_ratings)} teams",
            "season": season,
            "teams_count": len(elo_ratings),
            "updated": was_updated,
            "elo_ratings": elo_ratings
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update ELO: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
def predict_match(request: PredictionRequest):
    """
    Predict outcome of a Serie A 2025-26 match (direct classification version, con postprocessing)
    """
    # Validate teams
    if request.home_team not in SERIE_A_2025_26:
        raise HTTPException(
            status_code=400,
            detail=f"{request.home_team} is not in Serie A 2025-26"
        )
    if request.away_team not in SERIE_A_2025_26:
        raise HTTPException(
            status_code=400,
            detail=f"{request.away_team} is not in Serie A 2025-26"
        )
    if model_hda is None: # Usa model_hda globale
        raise HTTPException(status_code=500, detail="No classification model available (model_hda.pkl missing)")
    try:
        # 1. Build feature vector
        home_features = get_team_features(request.home_team, is_home=True)
        away_features = get_team_features(request.away_team, is_home=False)
        X = calculate_feature_vector(request.home_team, request.away_team)
        # 2. Predict probabilities (order: ['H', 'D', 'A'] encoded!)
        probas = model_hda.predict_proba(X)[0] # Usa model_hda globale
        # Ricava la label più probabile COME int
        top_int = int(np.argmax(probas))
        # Decodifica nel label stringa con label_encoder_hda
        if label_encoder_hda: # Usa label_encoder_hda globale
            predicted_result = label_encoder_hda.inverse_transform([top_int])[0]
            # Le classi stringa per mappare le probabilità finali:
            hda_labels = list(label_encoder_hda.classes_)
        else:
            # fallback: usa direttamente le classi dal modello (string o int)
            hda_labels = list(getattr(model_hda, 'classes_', ['H','D','A']))
            predicted_result = hda_labels[top_int] if top_int < len(hda_labels) else 'H'
        # Mappatura probs (mantiene ordine corretto)
        probs = {k: float(v) for k, v in zip(hda_labels, probas)}
        special_case = None
        # 3. Derby handling: blend toward uniform
        is_derby = (request.home_team, request.away_team) in DERBIES
        if is_derby:
            uniform = {k: 1.0/3.0 for k in probs}
            blend = 0.3
            probs = {k: (1-blend)*probs[k] + blend*uniform[k] for k in probs}
            special_case = "derby"
        # 4. Draw-friendly adjustment for balanced matches
        elo_diff = abs(home_features['elo'] - away_features['elo'])
        form_diff = abs(home_features['form_5'] - away_features['form_5'])
        if not is_derby and elo_diff < 20 and form_diff < 2:
            boost = 0.10
            pD0 = probs.get('D', 0)
            delta = boost
            total_oth = (probs.get('H',0) + probs.get('A',0))
            if total_oth > 0:
                scale = (1-delta)/total_oth
                probs['H'] = probs.get('H',0)*scale
                probs['A'] = probs.get('A',0)*scale
            probs['D'] = min(1.0, pD0+delta)
            special_case = "draw_bias"
        # Norm e rounding
        tot = sum(probs.values())
        probs = {k: round(v/tot, 3) for k,v in probs.items()}
        confidence = max(probs.values()) * 100
        # Assicura expected_goals = dict in output
        expected_goals = {}
        response = PredictionResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            predicted_result=predicted_result,
            probabilities=probs,
            expected_goals=expected_goals,
            confidence=round(confidence, 1),
            is_derby=is_derby,
            debug_info={
                'home_elo': round(home_features['elo'], 0),
                'away_elo': round(away_features['elo'], 0),
                'home_form': home_features['form_5'],
                'away_form': away_features['form_5'],
                'special_case': special_case
            }
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
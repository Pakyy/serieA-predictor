# ⚽ Serie A Match Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Machine Learning web application** that predicts Serie A match outcomes using historical data, Elo ratings, and Expected Goals (xG) statistics.

🔗 **[Live Demo](https://your-app.streamlit.app)** | 📊 **[API Docs](https://your-api.onrender.com/docs)** | 📝 **[Blog Post](#)**

---

## 🎯 Features

- 🤖 **ML-powered predictions** using XGBoost classifier
- 📈 **53% accuracy** (vs 33% random baseline)
- ⚽ **xG integration** from Understat API
- 🏆 **Elo rating system** with seasonal regression
- 🔄 **Auto-updates** weekly via GitHub Actions
- 🌐 **REST API** with FastAPI
- 🎨 **Interactive web UI** with Streamlit

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 53.0% |
| **Home Win Recall** | 62% |
| **Draw Recall** | 20% |
| **Away Win Recall** | 66% |
| **Test Set Size** | 423 matches |

**Better than**:
- ✅ Random guess (33%)
- ✅ Always predict home win (45%)
- ✅ Baseline models (48-50%)

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────┐
│  Data Sources                                       │
│  - Understat (xG data)                             │
│  - Football-Data.org (match history)               │
└─────────────┬───────────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────────┐
│  Data Pipeline (Python)                            │
│  - Scraping & cleaning                             │
│  - Feature engineering (Elo, form, rolling stats) │
│  - XGBoost training                                │
└─────────────┬───────────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────────┐
│  Database (Supabase PostgreSQL)                    │
│  - Match history                                   │
│  - Team statistics                                 │
│  - Model metadata                                  │
└─────────────┬───────────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────────┐
│  REST API (FastAPI)                                │
│  - /predict endpoint                               │
│  - /teams endpoint                                 │
│  - Deployed on Render                              │
└─────────────┬───────────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────────┐
│  Web UI (Streamlit)                                │
│  - Team selection                                  │
│  - Interactive predictions                         │
│  - Probability visualizations                      │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.11+
pip
Supabase account (free tier)
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/serieA-predictor.git
cd serieA-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your Supabase credentials
```

### Run Data Pipeline
```bash
# Full pipeline: scrape, process, train
python data/pipeline_complete.py

# Skip scraping (use existing data)
python data/pipeline_complete.py --skip-scrape

# Skip Supabase upload
python data/pipeline_complete.py --skip-upload
```

### Run API Locally
```bash
cd api
uvicorn main:app --reload
# Visit http://localhost:8000/docs
```

### Run Streamlit UI
```bash
streamlit run app/streamlit_app.py
# Visit http://localhost:8501
```

---

## 📦 Project Structure
```
serieA-predictor/
├── api/                    # FastAPI backend
│   ├── main.py            # API endpoints
│   └── database.py        # Supabase connection
├── app/                    # Streamlit frontend
│   └── streamlit_app.py   # Web UI
├── data/                   # Data pipeline
│   ├── pipeline_complete.py  # Main pipeline orchestrator
│   ├── scrape_fbref_xg.py    # xG scraping
│   ├── clean_data.py         # Data cleaning
│   └── upload_to_supabase.py # DB upload
├── models/                 # Trained ML models
│   ├── model_hda.pkl      # HDA classifier
│   └── features_v1.json   # Feature metadata
├── notebooks/              # Jupyter notebooks (EDA)
├── scripts/                # Utility scripts
│   └── weekly_update.py   # Auto-update script
├── .github/workflows/      # GitHub Actions
│   └── weekly_update.yml  # Weekly data refresh
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 🔬 Features & Methodology

### Feature Engineering

**16 features** used for prediction:

1. **Elo Ratings** (dynamic, regression between seasons)
   - `home_elo_before`, `away_elo_before`, `elo_diff`

2. **Team Form** (last 5 matches)
   - `home_form_5`, `away_form_5`, `form_diff`

3. **Rolling Statistics** (last 5 matches)
   - Goals scored/conceded avg
   - Shots avg

4. **Expected Goals** (xG, last 15 matches)
   - `home_xG_avg_rolling`, `away_xG_avg_rolling`
   - `home_xGA_avg_rolling`, `away_xGA_avg_rolling`

### Model Training

- **Algorithm**: XGBoost Classifier (multi-class)
- **Target**: H/D/A (Home win / Draw / Away win)
- **Class Weighting**: Balanced (addresses class imbalance)
- **Train/Test Split**: Temporal (2021-2024 train, 2024-25 test)
- **Validation**: Time-series cross-validation

### Hyperparameters
```python
XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8
)
```

---

## 🔄 Automated Updates

Data is **automatically updated** every Monday via GitHub Actions:

1. Scrapes latest matches from Understat
2. Recalculates features (Elo, form, xG)
3. Retrains models
4. Uploads to Supabase
5. Triggers API redeploy

**Manual trigger**: GitHub Actions tab → Run workflow

---

## 🌐 Deployment

### API (Render)
```bash
# Deployed automatically on git push
# URL: https://seriea-predictor-api.onrender.com
```

### UI (Streamlit Cloud)
```bash
# Deployed automatically on git push
# URL: https://seriea-predictor.streamlit.app
```

### Database (Supabase)

- PostgreSQL database
- Storage for ML models
- Free tier (500MB)

---

## 📈 Future Improvements

- [ ] Improve Draw prediction (target: 30% recall)
- [ ] Add player-level features (injuries, suspensions)
- [ ] Implement ensemble methods
- [ ] Add more leagues (Premier League, La Liga)
- [ ] Real-time odds comparison
- [ ] Historical performance tracking

---

## 🛠️ Tech Stack

**Machine Learning**:
- Python 3.11
- XGBoost
- Scikit-learn
- Pandas, NumPy

**Backend**:
- FastAPI
- Supabase (PostgreSQL)
- Pydantic

**Frontend**:
- Streamlit
- Plotly

**Deployment**:
- Render (API)
- Streamlit Cloud (UI)
- GitHub Actions (CI/CD)

**Data Sources**:
- Understat (xG)
- Football-Data.org (historical matches)

---

## 📝 API Documentation

### Endpoints

#### `GET /health`
Health check

#### `GET /teams`
List available Serie A teams

#### `POST /predict`
Predict match outcome

**Request**:
```json
{
  "home_team": "Inter",
  "away_team": "Milan"
}
```

**Response**:
```json
{
  "home_team": "Inter",
  "away_team": "Milan",
  "predicted_result": "H",
  "probabilities": {
    "H": 0.583,
    "D": 0.252,
    "A": 0.165
  },
  "expected_goals": {
    "home": 1.85,
    "away": 1.12
  },
  "confidence": 58.3
}
```

**Interactive docs**: [API Docs](https://your-api.onrender.com/docs)

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 👤 Author

**Your Name**

- LinkedIn: [your-profile](https://linkedin.com/in/yourname)
- GitHub: [@yourusername](https://github.com/yourusername)
- Portfolio: [yourportfolio.com](https://yourportfolio.com)

---

## 🙏 Acknowledgments

- [Understat](https://understat.com/) for xG data
- [Football-Data.org](https://www.football-data.co.uk/) for historical match data
- Serie A for being the best league ⚽

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/serieA-predictor?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/serieA-predictor?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/serieA-predictor?style=social)

**Built with ❤️ by a Data Science enthusiast**
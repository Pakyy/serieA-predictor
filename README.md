# ⚽ Serie A Match Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

**Machine Learning web application** that predicts Serie A match outcomes using historical data, Elo ratings, and Expected Goals (xG) statistics.

🔗 **[Live Demo](https://seriea-predictor-paky.streamlit.app/)** | 📊 **[API Docs](https://seriea-predictor.onrender.com)** 

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

## 👤 Author

**Your Name**

- LinkedIn: [myprofile](https://www.linkedin.com/in/pasquale-gravante-01075616b/)
- GitHub: [@pakyy](https://github.com/Pakyy)

---

## 🙏 Acknowledgments

- [Understat](https://understat.com/) for xG data
- [Football-Data.org](https://www.football-data.co.uk/) for historical match data
- Serie A for being the best league ⚽

---

**Built with ❤️ by a Data Science enthusiast**
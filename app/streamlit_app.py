import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Config
st.set_page_config(
    page_title="⚽ Serie A Predictor",
    page_icon="⚽",
    layout="wide"
)

API_URL = "https://seriea-predictor.onrender.com" # TODO: Sostituisci!

# ============================================================
# FUNCTIONS
# ============================================================

@st.cache_data(ttl=3600)
def get_teams():
    """Fetch team list from API"""
    response = requests.get(f"{API_URL}/teams")
    if response.status_code == 200:
        return response.json()['teams']
    return []


def predict_match(home_team, away_team):
    """Get prediction from API"""
    response = requests.post(
        f"{API_URL}/predict",
        json={"home_team": home_team, "away_team": away_team}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code}")
        return None


def plot_probabilities(probs):
    """Create probability bar chart"""
    labels = ['Home Win', 'Draw', 'Away Win']
    values = [probs['H'], probs['D'], probs['A']]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v*100:.1f}%" for v in values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Match Outcome Probabilities",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=400,
        showlegend=False
    )
    
    return fig


def plot_confidence_gauge(confidence):
    """Create confidence gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Model Confidence"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 60], 'color': "gray"},
                {'range': [60, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


# ============================================================
# HEADER
# ============================================================

st.title("⚽ Serie A Match Predictor")
st.markdown("Predict Serie A match outcomes using ML")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app predicts **Serie A 2025-26** match results using:
    - 🎯 **XGBoost** classifier
    - 📊 **Elo ratings**
    - 📈 **xG (Expected Goals)** from Understat
    - 🔥 **Team form** & rolling stats
    
    **Model Performance**:
    - Accuracy: ~53%
    - Better than random (33%)
    """)
    
    st.markdown("---")
    st.markdown("Made by Paky")
    st.markdown("[GitHub](https://github.com/Pakyy)")

# ============================================================
# MAIN APP
# ============================================================

# Load teams
teams = get_teams()

if not teams:
    st.error("⚠️ Could not load teams from API")
    st.stop()

# Team selection
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Home Team")
    home_team = st.selectbox(
        "Select home team",
        options=teams,
        index=teams.index('Inter') if 'Inter' in teams else 0,
        key='home'
    )

with col2:
    st.subheader("✈️ Away Team")
    away_team = st.selectbox(
        "Select away team",
        options=teams,
        index=teams.index('Milan') if 'Milan' in teams else 1,
        key='away'
    )

# Predict button
st.markdown("---")

if st.button("🔮 PREDICT MATCH", type="primary", use_container_width=True):
    
    if home_team == away_team:
        st.error("❌ Home and Away teams must be different!")
    else:
        with st.spinner("🤖 Computing prediction..."):
            result = predict_match(home_team, away_team)
        
        if result:
            st.success("✅ Prediction complete!")
            
            # Display match
            st.markdown("## 🏟️ Match Prediction")
            
            # Main result
            result_map = {
                'H': f"🏆 **{home_team} WIN**",
                'D': "🤝 **DRAW**",
                'A': f"🏆 **{away_team} WIN**"
            }
            
            st.markdown(f"### {result_map[result['predicted_result']]}")
            
            # Metrics row
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Home Win",
                    f"{result['probabilities']['H']*100:.1f}%"
                )
            
            with col2:
                st.metric(
                    "Draw",
                    f"{result['probabilities']['D']*100:.1f}%"
                )
            
            with col3:
                st.metric(
                    "Away Win",
                    f"{result['probabilities']['A']*100:.1f}%"
                )
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_probs = plot_probabilities(result['probabilities'])
                st.plotly_chart(fig_probs, use_container_width=True)
            
            with col2:
                fig_conf = plot_confidence_gauge(result['confidence'])
                st.plotly_chart(fig_conf, use_container_width=True)
            
            
            # Derby indicator
            if result.get('is_derby'):
                st.info("🔥 This is a **DERBY** match! Predictions are less reliable.")
            

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.caption("⚠️ Predictions are for entertainment purposes only")
"""
Streamlit UI for the Tennis-Prediction library.
Wraps data.data_loader + an XGBoost model into an interactive app.

Deploy notes:
  - Needs data files in ./data_atp/ : atp_players.csv + atp_matches_YYYY.csv
  - Main file path on Streamlit: app.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), "python"))

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

from data.data_loader import matches_data_loader, encode_data

st.set_page_config(page_title="Tennis Prediction", page_icon="🎾", layout="wide")
st.title("🎾 Tennis Prediction")
st.caption("XGBoost on pre-match player statistics (Vincent Auriau's Tennis-Prediction library).")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_atp")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

FEAT_1 = ["Ranking_1", "Ranking_Points_1", "Height_1", "Victories_Percentage_1",
          "Clay_Victories_Percentage_1", "Grass_Victories_Percentage_1",
          "Hard_Victories_Percentage_1", "Aces_Percentage_1",
          "First_Serve_Success_Percentage_1", "Overall_Win_on_Serve_Percentage_1",
          "BreakPoint_Saved_Percentage_1", "games_fatigue_1"]
FEAT_2 = [c.replace("_1", "_2") for c in FEAT_1]
META = ["tournament_level", "round", "best_of", "Winner"]

with st.sidebar:
    st.header("Settings")
    from_year = st.slider("Use data from year", 1990, 2024, 2015,
                          help="Later = faster to load. Each year adds ~1 min of processing.")
    st.caption(f"Reads atp_matches_*.csv from ./data_atp (year ≥ {from_year}).")

@st.cache_resource(show_spinner="Loading data & training (first run is slow)…")
def load_and_train(from_year):
    df = matches_data_loader(
        path_to_data=DATA_DIR, path_to_cache=CACHE_DIR,
        keep_values_from_year=from_year, flush_cache=True,
        get_match_statistics=True, get_reversed_match_data=True,
        match_type=["main_atp"],
    )
    df = df[META + FEAT_1 + FEAT_2].dropna(axis=0)
    enc = encode_data(df)
    y = enc["Winner"].values
    X = enc.drop(columns=["Winner"]).select_dtypes(include=[np.number])
    feat_names = list(X.columns)
    Xtr, Xte, ytr, yte = train_test_split(X.values, y, test_size=0.25, shuffle=False)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, eval_metric="logloss")
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    return {
        "acc": accuracy_score(yte, p > 0.5),
        "ll": log_loss(yte, np.clip(p, 1e-9, 1 - 1e-9)),
        "n": len(X), "n_test": len(Xte),
        "importance": sorted(zip(feat_names, model.feature_importances_),
                             key=lambda t: -t[1])[:10],
    }

if not os.path.isdir(DATA_DIR) or not os.path.exists(os.path.join(DATA_DIR, "atp_players.csv")):
    st.error("Missing data. Put atp_players.csv and atp_matches_YYYY.csv in ./data_atp/")
    st.stop()

res = load_and_train(from_year)

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{res['acc']:.1%}")
c2.metric("Log-loss", f"{res['ll']:.4f}")
c3.metric("Test matches", f"{res['n_test']:,}")

st.subheader("Top features")
st.dataframe(pd.DataFrame([(n, round(v, 3)) for n, v in res["importance"]],
                          columns=["Feature", "Importance"]),
             hide_index=True, use_container_width=True)

st.caption("Pre-match features only. More years = better model but slower load.")

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
    from_year = st.slider("Use data from year", 2020, 2026, 2025,
                          help="Later = faster to load. Each year adds processing time. "
                               "Must match the atp_matches_YYYY.csv files you uploaded.")
    st.caption(f"Reads atp_matches_*.csv from ./data_atp (year ≥ {from_year}).")

@st.cache_resource(show_spinner="Loading data & training (first run can take a few minutes)…")
def load_and_train(from_year):
    cache_marker = os.path.join(CACHE_DIR, f"matches_data_{from_year}.csv")
    df = matches_data_loader(
        path_to_data=DATA_DIR, path_to_cache=CACHE_DIR,
        keep_values_from_year=from_year,
        flush_cache=not os.path.exists(cache_marker),
        get_match_statistics=True, get_reversed_match_data=True,
        match_type=["main_atp"],
    )
    # --- build a "latest stats" profile per player from their most recent match ---
    # Each row has player 1's pre-match stats in *_1 cols with Name_1, likewise _2.
    profile = {}          # player name -> dict of their side-1 feature values
    order = ["tournament_date"]
    d_sorted = df.sort_values("tournament_date")
    p1_cols = ["Name_1"] + FEAT_1
    p2_cols = ["Name_2"] + FEAT_2
    for _, r in d_sorted[p1_cols].dropna().iterrows():
        profile[r["Name_1"]] = {f: r[f] for f in FEAT_1}
    for _, r in d_sorted[p2_cols].dropna().iterrows():
        # store under the same feature keys (strip _2 -> _1 naming) only if newer/unseen
        name = r["Name_2"]
        vals = {FEAT_1[i]: r[FEAT_2[i]] for i in range(len(FEAT_1))}
        profile[name] = vals  # later rows overwrite -> most recent wins

    dd = df[META + FEAT_1 + FEAT_2].dropna(axis=0)
    enc = encode_data(dd)
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
        "model": model, "feat_names": feat_names,
        "profile": profile, "players": sorted(profile.keys()),
    }

if not os.path.isdir(DATA_DIR) or not os.path.exists(os.path.join(DATA_DIR, "atp_players.csv")):
    st.error("Missing data. Put atp_players.csv and atp_matches_YYYY.csv in ./data_atp/")
    st.stop()

res = load_and_train(from_year)

tab_pred, tab_perf = st.tabs(["⚔️ Predict a matchup", "📊 Model performance"])

with tab_perf:
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{res['acc']:.1%}")
    c2.metric("Log-loss", f"{res['ll']:.4f}")
    c3.metric("Test matches", f"{res['n_test']:,}")
    st.subheader("Top features")
    st.dataframe(pd.DataFrame([(n, round(v, 3)) for n, v in res["importance"]],
                              columns=["Feature", "Importance"]),
                 hide_index=True, use_container_width=True)
    st.caption("Pre-match features only. More years = better model but slower load.")

with tab_pred:
    players = res["players"]
    profile = res["profile"]
    model = res["model"]
    feat_names = res["feat_names"]
    st.caption(f"{len(players)} players known (from the years loaded). "
               "Type to search. Only players active in the loaded years appear.")
    ca, cb = st.columns(2)
    pa = ca.selectbox("Player A", players, index=0)
    pb = cb.selectbox("Player B", players, index=min(1, len(players) - 1))
    cs, cbo = st.columns(2)
    surface = cs.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
    best_of = cbo.radio("Best of", [3, 5], horizontal=True)

    if pa == pb:
        st.warning("Pick two different players.")
    else:
        # assemble a feature row in the exact order the model was trained on
        row = {}
        for i, f in enumerate(FEAT_1):
            row[f] = profile[pa][f]
        for i, f in enumerate(FEAT_2):
            row[f] = profile[pb][FEAT_1[i]]
        row["best_of"] = best_of
        # surface / tournament_level are encoded columns; fill what the model knows
        feat_row = {}
        for name in feat_names:
            if name in row:
                feat_row[name] = row[name]
            elif name == f"tournament_surface_{surface}" or name == surface:
                feat_row[name] = 1
            else:
                feat_row[name] = 0
        X_one = pd.DataFrame([[feat_row.get(n, 0) for n in feat_names]], columns=feat_names)
        prob = float(model.predict_proba(X_one.values)[:, 1][0])
        st.metric(f"P({pa} beats {pb}) on {surface}", f"{prob:.1%}")
        st.progress(prob)
        st.caption("⚠️ Model leans heavily on ranking with few years of data. "
                   "Treat as a rough estimate, not betting advice.")

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqi_forecasting.predict import load_predictor


st.set_page_config(page_title="AQI Forecasting", layout="wide")
st.title("AQI Forecasting Dashboard")

model_dir = st.sidebar.text_input("Model directory", "models/aqi_tree")
uploaded = st.file_uploader("Upload recent readings CSV", type=["csv"])

st.caption("Upload recent readings with Datetime and pollutant/weather columns. For tree models, at least 25 recent rows are recommended so lag features can be built.")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Input Preview")
    st.dataframe(df.tail(30), use_container_width=True)

    if st.button("Predict AQI", type="primary"):
        try:
            predictor = load_predictor(model_dir)
            result = predictor.predict_with_category(df)
            col1, col2 = st.columns(2)
            col1.metric("Predicted AQI", f"{result['predicted_aqi']:.2f}")
            col2.metric("Category", result["category"])
        except Exception as exc:
            st.error(str(exc))
else:
    st.info("Train a model first, then upload a CSV of recent readings to predict AQI.")

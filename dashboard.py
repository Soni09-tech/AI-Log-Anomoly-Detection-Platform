"""
Streamlit dashboard for the Log Anomaly Detection Platform.

Run locally with:
    pip install streamlit pandas plotly
    streamlit run dashboard.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT
OUTPUTS_DIR = ROOT / "outputs" if (ROOT / "outputs").exists() else ROOT

st.set_page_config(page_title="Log Anomaly Detection Platform", layout="wide")
st.title("🔍 AI Log Anomaly Detection Platform")

RESULTS_PATH = OUTPUTS_DIR / "isolation_forest_results.csv"
TEMPLATES_PATH = DATA_DIR / "templates.csv"
PARSED_PATH = DATA_DIR / "parsed_logs.csv"


@st.cache_data
def load_data():
    results = pd.read_csv(RESULTS_PATH)
    templates = pd.read_csv(TEMPLATES_PATH)
    parsed = pd.read_csv(PARSED_PATH)
    return results, templates, parsed


try:
    results, templates, parsed = load_data()
except FileNotFoundError:
    st.error(
        "No results found yet. Run the pipeline first:\n\n"
        "```\npython generate_logs.py\n"
        "python log_parser.py\n"
        "python features.py\n"
        "python train_isolation_forest.py\n```"
    )
    st.stop()

# --- Summary metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sessions scored", len(results))
col2.metric("Flagged anomalies", int(results["predicted_anomaly"].sum()))
col3.metric("True anomalies", int(results["label"].sum()))
correct = ((results["predicted_anomaly"] == 1) & (results["label"] == 1)).sum()
col4.metric("Correctly caught", int(correct))

st.divider()

# --- Anomaly score distribution ---
st.subheader("Anomaly score distribution")
st.bar_chart(results.set_index("block_id")["anomaly_score"].sort_values(ascending=False).head(50))

st.divider()

# --- Flagged sessions table ---
st.subheader("Flagged sessions")
flagged = results[results["predicted_anomaly"] == 1].sort_values("anomaly_score", ascending=False)
st.dataframe(flagged, use_container_width=True)

# --- Drill-down into a session's raw events ---
st.subheader("Drill down into a session")
selected_block = st.selectbox("Choose a block/session ID", results["block_id"].tolist())
session_events = parsed[parsed["block_id"] == selected_block].merge(
    templates, left_on="event_id", right_on="event_id", how="left"
)
st.dataframe(session_events[["date", "time", "level", "event_id", "template"]], use_container_width=True)

row = results[results["block_id"] == selected_block].iloc[0]
if row["predicted_anomaly"] == 1:
    st.error(f"⚠️ Flagged as anomalous (score: {row['anomaly_score']:.3f})")
else:
    st.success(f"✅ Flagged as normal (score: {row['anomaly_score']:.3f})")

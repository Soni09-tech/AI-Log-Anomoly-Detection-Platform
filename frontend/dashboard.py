from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import build_inference_feature_matrix
from log_parser import decode_log_bytes, parse_log_file

MODEL_PATH = ROOT / "isolation_forest.joblib"
DEFAULT_API_URL = "http://127.0.0.1:8001"
try:
    configured_api_url = st.secrets.get(
        "API_URL",
        os.getenv("API_URL", DEFAULT_API_URL),
    )
except FileNotFoundError:
    configured_api_url = os.getenv("API_URL", DEFAULT_API_URL)
API_URL = (
    configured_api_url.strip().rstrip("/")
    if isinstance(configured_api_url, str) and configured_api_url.strip()
    else DEFAULT_API_URL
)

st.set_page_config(page_title="AI Log Anomaly Detection Platform", page_icon="🔎", layout="wide")
st.markdown(
    """
    <style>
    .stButton > button[kind="primary"] {
        background-color: #ff6b5f;
        color: #ffffff;
        border: 1px solid #ff6b5f;
        font-weight: 700;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #e85a50;
        border-color: #e85a50;
        color: #ffffff;
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(255, 107, 95, 0.35);
        border-radius: 0.75rem;
        padding: 1rem;
        background: rgba(255, 107, 95, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("AI-Powered Log Anomaly Detection Platform")
st.caption("Upload text-based logs or data files to identify anomalous sessions.")

try:
    health_response = requests.get(f"{API_URL}/health", timeout=3)
    backend_online = health_response.ok and health_response.json().get("status") == "healthy"
except requests.exceptions.RequestException:
    backend_online = False
except ValueError:
    backend_online = False

if backend_online:
    st.sidebar.success("🟢 Backend Online")
else:
    st.sidebar.error("🔴 Backend Offline")
st.sidebar.caption(f"API: {API_URL}")


@st.cache_resource
def load_local_model() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "model" not in artifact or "feature_cols" not in artifact:
        raise ValueError("The model artifact must include 'model' and 'feature_cols'.")
    return artifact


def local_predict(raw_bytes: bytes) -> dict[str, Any]:
    if not raw_bytes.strip():
        raise ValueError("The uploaded log file is empty.")
    raw_text = decode_log_bytes(raw_bytes)
    artifact = load_local_model()
    with tempfile.TemporaryDirectory(prefix="logscan_") as temporary_directory:
        temp_dir = Path(temporary_directory)
        raw_path = temp_dir / "uploaded.log"
        parsed_path = temp_dir / "parsed_logs.csv"
        templates_path = temp_dir / "templates.csv"
        labels_path = temp_dir / "labels.csv"
        raw_path.write_text(raw_text, encoding="utf-8")

        parse_log_file(raw_path, parsed_path, templates_path)
        parsed_df = pd.read_csv(parsed_path)
        if parsed_df.empty:
            raise ValueError("No parseable log entries were found in the upload.")
        block_ids = parsed_df["block_id"].astype(str).drop_duplicates()
        pd.DataFrame({"block_id": block_ids, "label": 0}).to_csv(labels_path, index=False)
        feature_df = build_inference_feature_matrix(
            parsed_path,
            labels_path,
            templates_path,
            ROOT / "templates.csv",
            artifact["feature_cols"],
        )

    feature_cols = artifact["feature_cols"]
    features = feature_df[feature_cols].fillna(0)
    model = artifact["model"]
    scores = model.decision_function(features)
    predictions = (model.predict(features) == -1).astype(int)
    results = [
        {
            "block_id": str(block_id),
            "anomaly_score": round(float(score), 6),
            "predicted_anomaly": int(prediction),
            "status": "Anomaly" if prediction else "Normal",
        }
        for block_id, score, prediction in zip(feature_df["block_id"], scores, predictions)
    ]
    return {
        "status": "success",
        "total_sessions": len(results),
        "anomalies_detected": int(sum(item["predicted_anomaly"] for item in results)),
        "results": results,
    }


def call_api(upload: Any) -> dict[str, Any]:
    response = requests.post(
        f"{API_URL}/predict",
        files={"file": (upload.name, upload.getvalue(), upload.type or "text/plain")},
        timeout=120,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"Backend returned HTTP {response.status_code}: {detail}")
        return {}
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("The API returned an invalid response.")
    return result


upload = st.file_uploader(
    "Upload a text-based file",
    help="Any file extension is accepted. Text lines without session IDs are analyzed as one file session; binary files are not supported.",
)
if not backend_online:
    st.info("The backend is unavailable. You can still run inference with the local model.")
mode = st.radio(
    "Inference mode",
    ["Run with Local Model", "Use FastAPI Backend"],
    index=0 if not backend_online else 1,
    horizontal=True,
)

if upload and st.button("Analyze logs", type="primary"):
    try:
        with st.spinner("Parsing log sessions and scoring anomalies..."):
            if mode == "Run with Local Model":
                st.session_state["scan_result"] = local_predict(upload.getvalue())
            else:
                st.session_state["scan_result"] = call_api(upload)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the prediction API: {exc}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Could not analyze this log file: {exc}")

result = st.session_state.get("scan_result")
if result and result.get("status") == "success":
    result_rows = result.get("results", [])
    col1, col2 = st.columns(2)
    col1.metric("Sessions analyzed", result.get("total_sessions", 0))
    col2.metric("Anomalies detected", result.get("anomalies_detected", 0))
    results_df = pd.DataFrame(result_rows)
    if not results_df.empty:
        st.subheader("Session predictions")
        display_df = results_df.rename(
            columns={
                "block_id": "Session ID",
                "anomaly_score": "Anomaly Score",
                "status": "Status",
            }
        )
        st.dataframe(
            display_df[["Session ID", "Anomaly Score", "Status"]],
            use_container_width=True,
            hide_index=True,
        )
        if "anomaly_score" in results_df:
            chart = px.bar(
                results_df,
                x="block_id",
                y="anomaly_score",
                color="status",
                title="Anomaly score by session",
                color_discrete_map={"Normal": "#4caf83", "Anomaly": "#ff6b5f"},
            )
            st.plotly_chart(chart, use_container_width=True)
elif result and result.get("status") != "success":
    st.error("The prediction service returned an unexpected response.")

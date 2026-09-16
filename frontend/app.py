from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "isolation_forest.joblib"
RESULTS_PATH = ROOT / "isolation_forest_results.csv"
PARSED_PATH = ROOT / "parsed_logs.csv"
TEMPLATES_PATH = ROOT / "templates.csv"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
PALETTE = {
    "bg": "#0f1715",
    "card": "#162320",
    "header": "#00ffcc",
    "body": "#a3c1ad",
    "gold": "#ffcc00",
    "muted": "#5a6f68",
    "danger": "#ff8066",
}

st.set_page_config(
    page_title="AI Log Anomaly Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    :root {{ color-scheme: dark; }}
    @keyframes scanline {{ from {{ transform: translateX(-100%); }} to {{ transform: translateX(100%); }} }}
    @keyframes pulseMint {{ 0%,100% {{ box-shadow: 0 0 8px rgba(0,255,204,.12); }} 50% {{ box-shadow: 0 0 22px rgba(0,255,204,.34); }} }}
    @keyframes menuPopup {{
        0% {{ opacity:0; transform:translateY(-10px) scale(.94); transform-origin:top center; }}
        70% {{ opacity:1; transform:translateY(2px) scale(1.015); }}
        100% {{ opacity:1; transform:translateY(0) scale(1); }}
    }}
    @keyframes optionPopup {{
        0% {{ opacity:0; transform:translateY(-6px) scale(.96); }}
        100% {{ opacity:1; transform:translateY(0) scale(1); }}
    }}
    .stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background:{PALETTE["bg"]}!important; color:{PALETTE["body"]}!important;
    }}
    .block-container {{ padding-top:1.8rem; max-width:1450px; }}
    h1,h2,h3,h4,h5,h6 {{ color:{PALETTE["header"]}!important; letter-spacing:.02em; }}
    p,li,label,small,span,[data-testid="stMarkdownContainer"] {{ color:{PALETTE["body"]}!important; }}
    [data-testid="stSidebar"],[data-testid="stSidebarContent"],[data-testid="stSidebarUserContent"] {{
        background:{PALETTE["card"]}!important;
    }}
    [data-testid="stSidebar"] h1 {{ text-shadow:0 0 12px rgba(0,255,204,.35); }}
    [data-baseweb="popover"], [role="listbox"] {{
        background:{PALETTE["card"]}!important; color:{PALETTE["body"]}!important;
        border:1px solid {PALETTE["header"]}!important; border-radius:12px;
        box-shadow:0 14px 34px rgba(0,0,0,.28), 0 0 18px rgba(0,255,204,.14);
        overflow:hidden; transform-origin:top center;
        animation:menuPopup .34s cubic-bezier(.22,1,.36,1) both;
    }}
    [role="listbox"] [role="option"] {{
        background:{PALETTE["card"]}!important; color:{PALETTE["body"]}!important;
        opacity:0; transform-origin:center;
        animation:optionPopup .26s cubic-bezier(.22,1,.36,1) both;
    }}
    [role="listbox"] [role="option"]:nth-child(1) {{ animation-delay:.04s; }}
    [role="listbox"] [role="option"]:nth-child(2) {{ animation-delay:.08s; }}
    [role="listbox"] [role="option"]:nth-child(3) {{ animation-delay:.12s; }}
    [role="listbox"] [role="option"]:nth-child(4) {{ animation-delay:.16s; }}
    [role="listbox"] [role="option"]:nth-child(5) {{ animation-delay:.20s; }}
    [role="listbox"] [role="option"]:hover,
    [role="listbox"] [role="option"][aria-selected="true"] {{
        background:{PALETTE["header"]}!important; color:{PALETTE["bg"]}!important;
        box-shadow:inset 3px 0 0 {PALETTE["gold"]};
    }}
    [data-baseweb="select"] > div {{ border-color:{PALETTE["header"]}!important; transition:all .25s cubic-bezier(.22,1,.36,1); }}
    [data-baseweb="select"] > div:hover {{ box-shadow:0 0 16px rgba(0,255,204,.25); transform:translateY(-1px); }}
    [data-testid="stMetric"] {{
        background:linear-gradient(135deg,{PALETTE["card"]},#1b302a)!important;
        border:1px solid {PALETTE["header"]}; border-radius:16px; padding:1rem;
        animation:pulseMint 3.5s ease-in-out infinite;
    }}
    [data-testid="stMetricValue"] {{ color:{PALETTE["gold"]}!important; font-weight:800; }}
    [data-testid="stMetricLabel"],[data-testid="stMetricDelta"] {{ color:{PALETTE["body"]}!important; }}
    .stButton > button {{
        background:linear-gradient(90deg,{PALETTE["header"]},#6affd9)!important;
        color:{PALETTE["bg"]}!important; border:0; border-radius:10px; font-weight:800;
        transition:all .25s cubic-bezier(.22,1,.36,1);
    }}
    .stButton > button:hover {{ background:{PALETTE["gold"]}!important; transform:translateY(-2px); box-shadow:0 0 18px rgba(255,204,0,.3); }}
    input,textarea,[data-baseweb="input"],[data-baseweb="select"] {{
        background:{PALETTE["card"]}!important; color:{PALETTE["body"]}!important;
        border-color:{PALETTE["header"]}!important;
    }}
    [data-testid="stFileUploader"],[data-testid="stFileUploaderDropzone"],[data-testid="stDataFrame"],
    [data-testid="stExpander"],[data-testid="stAlert"] {{
        background:{PALETTE["card"]}!important; border-radius:14px; border-color:{PALETTE["muted"]}!important;
    }}
    code,pre,[data-testid="stCode"] {{ background:{PALETTE["card"]}!important; color:{PALETTE["gold"]}!important; }}
    hr {{ border-color:{PALETTE["muted"]}!important; }}
    .hero {{ padding:1.4rem 1.6rem; border:1px solid {PALETTE["header"]}; border-radius:18px;
        background:linear-gradient(110deg,#162320,#20352d); position:relative; overflow:hidden; }}
    .hero:after {{ content:""; position:absolute; inset:0; width:35%; background:rgba(0,255,204,.08);
        transform:skewX(-20deg); animation:scanline 5s linear infinite; pointer-events:none; }}
    .badge {{ color:{PALETTE["gold"]}; font-weight:800; letter-spacing:.1em; font-size:.75rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeError):
            pass
    return pd.DataFrame(columns=columns)


@st.cache_resource
def load_model_artifact() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    try:
        artifact = joblib.load(MODEL_PATH)
        return artifact if isinstance(artifact, dict) else None
    except (OSError, ValueError):
        return None


def plot_theme(fig: Any, title: str) -> Any:
    fig.update_layout(
        title=title,
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card"],
        font={"color": PALETTE["body"], "family": "Segoe UI"},
        title_font={"color": PALETTE["header"]},
        xaxis={"tickfont": {"color": PALETTE["body"]}, "gridcolor": PALETTE["muted"]},
        yaxis={"tickfont": {"color": PALETTE["body"]}, "gridcolor": PALETTE["muted"]},
        legend={"font": {"color": PALETTE["body"]}},
        hoverlabel={"bgcolor": PALETTE["card"], "font": {"color": PALETTE["body"]}},
        margin={"l": 20, "r": 20, "t": 52, "b": 40},
    )
    return fig


def call_predict_api(uploaded_file: Any) -> dict[str, Any]:
    response = requests.post(
        f"{BACKEND_URL}/predict",
        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "text/plain")},
        timeout=120,
    )
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Backend returned HTTP {response.status_code}: {detail}")
    return response.json()


def make_pdf(result: dict[str, Any]) -> bytes:
    lines = [
        "AI LOG ANOMALY DETECTION - SCAN REPORT",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"File: {result.get('filename', 'Unknown')}",
        f"Risk: {result.get('risk_score', 'Unknown')} | Index: {result.get('risk_index', 0)}",
        f"Lines: {result.get('scanned_lines', 0)} | Sessions: {result.get('sessions_analyzed', 0)}",
        f"Anomalies: {result.get('anomalies_detected', 0)}",
        "",
        "THREAT BREAKDOWN",
    ]
    lines += [f"{name}: {count}" for name, count in result.get("threat_summary", {}).items()]
    lines.append("")
    lines.append("FLAGGED SESSIONS")
    lines += [f"{x['block_id']} | {x['threat_type']} | {x['anomaly_score']}" for x in result.get("flagged_sessions", [])[:25]]

    def escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content = "BT /F1 9 Tf 42 750 Td " + " ".join(f"({escape(line)}) Tj 0 -13 Td" for line in lines) + " ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
        f"<< /Length {len(content.encode('latin-1', errors='replace'))} >>\nstream\n{content}\nendstream".encode("latin-1", errors="replace"),
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    startxref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    pdf.extend("".join(f"{offset:010d} 00000 n \n" for offset in offsets[1:]).encode())
    pdf.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF".encode())
    return bytes(pdf)


if "scan_result" not in st.session_state:
    st.session_state.scan_result = None

parsed_df = load_csv(PARSED_PATH, ["block_id", "event_id", "level", "date", "time"])
results_df = load_csv(RESULTS_PATH, ["block_id", "label", "anomaly_score", "predicted_anomaly"])
templates_df = load_csv(TEMPLATES_PATH, ["event_id", "template"])
model_artifact = load_model_artifact()
result = st.session_state.scan_result

st.sidebar.markdown("# 🛡️ SentinelLog AI")
st.sidebar.markdown('<div class="badge">SOC INTELLIGENCE CONSOLE</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Dashboard", "Live Log Analyzer", "Parsed Logs Explorer", "Model Metrics"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("### SYSTEM STATUS")
st.sidebar.success("Detection engine ready")
st.sidebar.caption("Isolation Forest • Drain parser • API online")

if page in {"Home", "Dashboard"}:
    st.markdown('<div class="hero"><div class="badge">REAL-TIME THREAT MONITORING</div><h1>AI Log Anomaly Detection Platform</h1><p>Detect unusual sessions, classify threats, and prioritize response from one focused SOC workspace.</p></div>', unsafe_allow_html=True)
    if page == "Home":
        st.subheader("Platform Overview")
        st.markdown("Upload raw logs in **Live Log Analyzer** or review the existing pipeline output below.")
    scan = result
    if scan:
        parsed = pd.DataFrame(scan.get("parsed_logs", []))
        total = int(scan.get("scanned_lines", 0))
        anomalies = int(scan.get("anomalies_detected", 0))
        flagged_sessions = pd.DataFrame(scan.get("flagged_sessions", []))
        values = [f"{total:,}", f"{anomalies:,}", scan.get("risk_score", "Unknown"), f"{scan.get('sessions_analyzed', 0):,}"]
    else:
        parsed = parsed_df
        total = len(parsed)
        anomalies = int(results_df["predicted_anomaly"].sum()) if "predicted_anomaly" in results_df else 0
        values = [f"{total:,}", f"{anomalies:,}", "Low" if total else "Unknown", f"{results_df['block_id'].nunique():,}" if "block_id" in results_df else "0"]
        flagged_sessions = results_df[results_df.get("predicted_anomaly", pd.Series(dtype=int)) == 1] if not results_df.empty else pd.DataFrame()
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Logs Scanned", "Anomalies Flagged", "Threat Risk", "Sessions Analyzed"], values):
        col.metric(label, value)
    st.markdown("---")
    if not flagged_sessions.empty and "anomaly_score" in flagged_sessions:
        chart = flagged_sessions.head(50).copy()
        chart["block_id"] = chart["block_id"].astype(str)
        fig = px.bar(chart, x="block_id", y="anomaly_score", color="threat_type" if "threat_type" in chart else None, color_discrete_sequence=[PALETTE["gold"], PALETTE["header"], PALETTE["danger"]])
        st.plotly_chart(plot_theme(fig, "Anomaly Trend • Highest Risk Sessions"), use_container_width=True)
    elif not results_df.empty and "anomaly_score" in results_df:
        chart = results_df.sort_values("block_id").head(80)
        st.plotly_chart(plot_theme(px.line(chart, x="block_id", y="anomaly_score", color_discrete_sequence=[PALETTE["header"]]), "Anomaly Trend • Model Scores"), use_container_width=True)
    else:
        st.info("Run the pipeline or upload logs to generate anomaly trends.")
    if page == "Dashboard":
        left, right = st.columns([1.15, 1])
        with left:
            st.subheader("Threat Distribution")
            if scan and scan.get("threat_summary"):
                threat_df = pd.DataFrame([{"Threat": k, "Count": v} for k, v in scan["threat_summary"].items()])
                st.plotly_chart(plot_theme(px.pie(threat_df, names="Threat", values="Count", color_discrete_sequence=[PALETTE["gold"], PALETTE["header"], PALETTE["danger"], PALETTE["body"]]), "Detected Threat Categories"), use_container_width=True)
            else:
                st.info("Threat categories appear after a live scan.")
        with right:
            st.subheader("Operational Status")
            st.success("Parser engine operational")
            st.success("Model artifact loaded" if model_artifact else "Model artifact unavailable")
            st.warning("Alerts triggered" if scan and scan.get("alert", {}).get("triggered") else "No high-risk alert triggered")

elif page == "Live Log Analyzer":
    st.title("🔍 Live Log Analyzer")
    st.markdown("Upload a raw log file to parse sessions, run Isolation Forest inference, and classify detected threats.")
    uploaded_file = st.file_uploader("Upload .log / .txt file", type=["log", "txt"])
    if uploaded_file and st.button("Run Anomaly Detection", type="primary"):
        try:
            with st.spinner("Parsing logs and executing inference..."):
                st.session_state.scan_result = call_predict_api(uploaded_file)
            st.success("Analysis complete.")
            st.rerun()
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            st.error(f"Inference failed: {exc}")
    if result:
        cols = st.columns(5)
        cols[0].metric("Lines Scanned", result.get("scanned_lines", 0))
        cols[1].metric("Anomalies", result.get("anomalies_detected", 0))
        cols[2].metric("Risk Level", result.get("risk_score", "Unknown"))
        cols[3].metric("Risk Index", result.get("risk_index", 0))
        cols[4].metric("Alert", "TRIGGERED" if result.get("alert", {}).get("triggered") else "Clear")
        st.subheader("Threat Breakdown")
        st.dataframe(pd.DataFrame([{"Threat Type": k, "Count": v} for k, v in result.get("threat_summary", {}).items()]), hide_index=True, use_container_width=True)
        st.subheader("Flagged Sessions")
        st.dataframe(pd.DataFrame(result.get("flagged_sessions", [])), hide_index=True, use_container_width=True)
        st.download_button("Export Scan Report as PDF", make_pdf(result), "scan_report.pdf", "application/pdf")

elif page == "Parsed Logs Explorer":
    st.title("📂 Parsed Logs Explorer")
    source = pd.DataFrame(result.get("parsed_logs", [])) if result else parsed_df
    query = st.text_input("Search parsed logs by keyword", placeholder="block id, level, event...")
    levels = st.multiselect("Filter by log level", sorted(source["level"].dropna().unique().tolist()) if "level" in source else [])
    if source.empty:
        st.info("No parsed records available. Run a live scan or generate parsed_logs.csv.")
    else:
        filtered = source
        if query:
            mask = filtered.astype(str).apply(lambda row: row.str.contains(query, case=False, na=False).any(), axis=1)
            filtered = filtered[mask]
        if levels:
            filtered = filtered[filtered["level"].isin(levels)]
        st.caption(f"{len(filtered):,} records shown")
        st.dataframe(filtered, hide_index=True, use_container_width=True)
        if not templates_df.empty:
            with st.expander("Event template dictionary"):
                st.dataframe(templates_df, hide_index=True, use_container_width=True)

else:
    st.title("⚙️ Model Metrics & Configuration")
    if not model_artifact:
        st.error("Model artifact not found or invalid: isolation_forest.joblib")
    else:
        model = model_artifact.get("model")
        feature_cols = model_artifact.get("feature_cols", [])
        cols = st.columns(5)
        cols[0].metric("Algorithm", type(model).__name__)
        cols[1].metric("Estimators", getattr(model, "n_estimators", "N/A"))
        cols[2].metric("Contamination", getattr(model, "contamination", "N/A"))
        cols[3].metric("Features", len(feature_cols))
        cols[4].metric("Random State", getattr(model, "random_state", "N/A"))
        st.subheader("Isolation Forest Parameters")
        st.code("\n".join(f"{key}: {value}" for key, value in {
            "max_samples": getattr(model, "max_samples", "N/A"),
            "max_features": getattr(model, "max_features", "N/A"),
            "bootstrap": getattr(model, "bootstrap", "N/A"),
            "feature_columns": ", ".join(feature_cols),
        }.items()))

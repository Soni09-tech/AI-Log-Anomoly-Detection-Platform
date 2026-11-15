from __future__ import annotations

import html
import logging
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from streaming_inference import analyze_log_stream

LOGGER = logging.getLogger(__name__)
MODEL_PATH = PROJECT_ROOT / "isolation_forest.joblib"
PARSED_PATH = PROJECT_ROOT / "parsed_logs.csv"
RESULTS_PATH = PROJECT_ROOT / "isolation_forest_results.csv"
TEMPLATES_PATH = PROJECT_ROOT / "templates.csv"
MAX_UPLOAD_MB = 200
PAGES = (
    "Dashboard",
    "Live Log Analyzer",
    "Parsed Logs Explorer",
    "Model Metrics",
)

try:
    configured_api_url = st.secrets.get(
        "API_URL",
        os.getenv("API_URL", "http://127.0.0.1:8001"),
    )
except FileNotFoundError:
    configured_api_url = os.getenv("API_URL", "http://127.0.0.1:8001")
API_URL = (
    configured_api_url.strip().rstrip("/")
    if isinstance(configured_api_url, str) and configured_api_url.strip()
    else "http://127.0.0.1:8001"
)

PALETTE = {
    "background": "#0b1411",
    "surface": "#13241f",
    "surface_alt": "#162e27",
    "border": "#1f3d34",
    "green": "#00e676",
    "amber": "#ffcc00",
    "white": "#ffffff",
    "muted": "#90a4ae",
    "red": "#ff6b6b",
}

st.set_page_config(
    page_title="SentinelLog AI | SOC Intelligence Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    f"""
    <style>
    :root {{ color-scheme: dark; }}
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background: {PALETTE["background"]};
        color: {PALETTE["white"]};
    }}
    .block-container {{ max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem; }}
    h1, h2, h3, h4 {{ color: {PALETTE["white"]} !important; letter-spacing: -.02em; }}
    p, label, small, [data-testid="stMarkdownContainer"] {{
        color: {PALETTE["muted"]};
    }}
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
        background: #0e1a17;
        border-right: 1px solid {PALETTE["border"]};
    }}
    [data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
    [data-testid="stMetric"] {{
        background: {PALETTE["surface"]};
        border: 1px solid {PALETTE["border"]};
        border-radius: 8px;
        padding: 1.1rem 1.2rem;
    }}
    [data-testid="stMetricLabel"] {{ color: {PALETTE["muted"]} !important; }}
    [data-testid="stMetricValue"] {{ color: {PALETTE["white"]} !important; }}
    [data-testid="stDataFrame"], [data-testid="stFileUploader"],
    [data-testid="stExpander"] {{
        background: {PALETTE["surface"]};
        border: 1px solid {PALETTE["border"]};
        border-radius: 8px;
    }}
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    textarea, input {{
        background: {PALETTE["surface"]} !important;
        border-color: {PALETTE["border"]} !important;
        color: {PALETTE["white"]} !important;
    }}
    .stButton > button, .stDownloadButton > button {{
        border: 1px solid {PALETTE["green"]};
        border-radius: 7px;
        background: {PALETTE["green"]};
        color: #07120d;
        font-weight: 750;
        min-height: 2.7rem;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: {PALETTE["amber"]};
        background: {PALETTE["amber"]};
        color: #07120d;
    }}
    .soc-brand {{
        color: {PALETTE["white"]};
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -.03em;
    }}
    .soc-subtitle, .eyebrow {{
        color: {PALETTE["green"]};
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .15em;
    }}
    .status-card, .soc-card, .hero {{
        border: 1px solid {PALETTE["border"]};
        border-radius: 8px;
        background: {PALETTE["surface"]};
        padding: 1rem 1.1rem;
    }}
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        border: 1px solid rgba(0, 230, 118, .32);
        border-radius: 999px;
        padding: .38rem .68rem;
        background: rgba(0, 230, 118, .09);
        color: {PALETTE["green"]};
        font-size: .78rem;
        font-weight: 750;
    }}
    .status-pill.offline {{
        border-color: rgba(255, 107, 107, .35);
        background: rgba(255, 107, 107, .08);
        color: {PALETTE["red"]};
    }}
    .hero {{
        padding: 1.5rem 1.7rem;
        background: linear-gradient(115deg, {PALETTE["surface"]}, #173128);
        margin-bottom: 1.2rem;
    }}
    .hero h1 {{ margin: .45rem 0 .4rem; font-size: clamp(1.7rem, 3vw, 2.35rem); }}
    .hero p {{ margin: 0; max-width: 850px; }}
    .metric-card {{
        min-height: 112px;
        border: 1px solid {PALETTE["border"]};
        border-radius: 8px;
        background: {PALETTE["surface"]};
        padding: 1rem 1.1rem;
    }}
    .metric-label {{ color: {PALETTE["muted"]}; font-size: .82rem; }}
    .metric-value {{
        color: {PALETTE["white"]};
        font-size: 1.8rem;
        line-height: 1.35;
        font-weight: 800;
    }}
    .metric-value.amber {{ color: {PALETTE["amber"]}; }}
    .section-heading {{ color: {PALETTE["white"]}; font-size: 1.08rem; font-weight: 750; }}
    .status-line {{ padding: .43rem 0; color: {PALETTE["muted"]}; }}
    .status-dot {{ color: {PALETTE["green"]}; font-weight: 900; padding-right: .4rem; }}
    .risk-chip {{
        display: inline-block;
        border-radius: 999px;
        background: rgba(255, 204, 0, .12);
        color: {PALETTE["amber"]};
        padding: .2rem .65rem;
        font-size: 1rem;
        font-weight: 800;
    }}
    hr {{ border-color: {PALETTE["border"]}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=list(columns))
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        LOGGER.exception("Could not read dashboard data file %s", path)
        st.warning(f"Could not read `{path.name}`: {exc}")
        return pd.DataFrame(columns=list(columns))


@st.cache_resource(show_spinner=False)
def load_model_artifact() -> dict[str, Any] | None:
    if not MODEL_PATH.is_file():
        return None
    try:
        artifact = joblib.load(MODEL_PATH)
    except Exception as exc:
        LOGGER.exception("Could not load model artifact %s", MODEL_PATH)
        st.warning(f"The local model could not be loaded: {exc}")
        return None
    if (
        not isinstance(artifact, dict)
        or "model" not in artifact
        or not isinstance(artifact.get("feature_cols"), list)
        or not artifact["feature_cols"]
    ):
        st.warning("The model artifact is missing its model or feature columns.")
        return None
    return artifact


def backend_is_online() -> bool:
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        payload = response.json()
        return (
            response.ok
            and isinstance(payload, dict)
            and payload.get("status") == "online"
            and payload.get("engine") == "SentinelLog AI"
        )
    except (requests.RequestException, ValueError):
        return False


class BackendUnavailableError(RuntimeError):
    pass


class MultipartFileBody:
    def __init__(self, stream: Any, filename: str, size: int, boundary: str) -> None:
        safe_name = Path(filename).name.replace('"', "_").replace("\r", "_").replace("\n", "_")
        self.stream = stream
        self.size = size
        self.prefix = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
            "Content-Type: text/plain\r\n\r\n"
        ).encode("utf-8")
        self.suffix = f"\r\n--{boundary}--\r\n".encode("ascii")

    def __len__(self) -> int:
        return len(self.prefix) + self.size + len(self.suffix)

    def __iter__(self):
        self.stream.seek(0)
        yield self.prefix
        while chunk := self.stream.read(1024 * 1024):
            yield chunk
        yield self.suffix


def call_predict_api(
    filename: str,
    stream: Any,
    size: int,
    progress_callback: Any = None,
) -> dict[str, Any]:
    boundary = f"sentinellog-{os.urandom(16).hex()}"
    body = MultipartFileBody(stream, filename, size, boundary)
    if progress_callback is None:
        request_body = body
    else:
        class ProgressBody:
            def __len__(self) -> int:
                return len(body)

            def __iter__(self):
                sent = 0
                for chunk in body:
                    sent += len(chunk)
                    progress_callback(sent, len(body), 0, "upload", 0, 0)
                    yield chunk

        request_body = ProgressBody()
    try:
        response = requests.post(
            f"{API_URL}/predict",
            data=request_body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=(3, 120),
        )
    except requests.RequestException as exc:
        raise BackendUnavailableError(f"FastAPI request failed: {exc}") from exc
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if response.status_code >= 500:
            raise BackendUnavailableError(
                f"FastAPI returned HTTP {response.status_code}: {detail}"
            )
        raise RuntimeError(f"FastAPI returned HTTP {response.status_code}: {detail}")
    result = response.json()
    if not isinstance(result, dict) or result.get("status") != "success":
        raise ValueError("FastAPI returned an invalid prediction response.")
    return result


def local_predict(
    filename: str,
    stream: Any,
    artifact: dict[str, Any],
    size: int,
    progress_callback: Any = None,
) -> dict[str, Any]:
    stream.seek(0)
    result = analyze_log_stream(
        stream,
        artifact,
        TEMPLATES_PATH,
        total_bytes=size,
        progress_callback=progress_callback,
    )
    result["filename"] = filename
    return result


def add_template_content(data: pd.DataFrame, templates: pd.DataFrame) -> pd.DataFrame:
    source = data.copy()
    if "content" not in source:
        if {"event_id", "template"}.issubset(templates.columns):
            template_lookup = templates.drop_duplicates("event_id").set_index("event_id")["template"]
            source["content"] = source["event_id"].map(template_lookup).fillna("")
        else:
            source["content"] = ""
    for column in ("block_id", "event_id", "level", "date", "time", "content"):
        if column not in source:
            source[column] = ""
    return source


def metric_card(label: str, value: Any, accent: bool = False) -> None:
    value_class = "metric-value amber" if accent else "metric-value"
    if label == "Threat Risk":
        safe_value = html.escape(str(value))
        value_html = f'<span class="risk-chip">{safe_value}</span>'
    else:
        value_html = f'<div class="{value_class}">{html.escape(str(value))}</div>'
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f"{value_html}</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def initial_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parsed = load_csv(PARSED_PATH, ("block_id", "event_id", "level", "date", "time"))
    results = load_csv(
        RESULTS_PATH,
        ("block_id", "label", "anomaly_score", "predicted_anomaly"),
    )
    templates = load_csv(TEMPLATES_PATH, ("event_id", "template"))
    return add_template_content(parsed, templates), results, templates


def render_dashboard(
    scan: dict[str, Any] | None,
    saved_parsed: pd.DataFrame,
    saved_results: pd.DataFrame,
    model_ready: bool,
    api_online: bool,
) -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">REAL-TIME THREAT MONITORING</div>
          <h1>AI Log Anomaly Detection Platform</h1>
          <p>Detect unusual sessions, classify threats, and prioritize response from one focused SOC workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if scan:
        parsed = add_template_content(
            pd.DataFrame(scan.get("parsed_logs", [])),
            templates_df,
        )
        results = pd.DataFrame(scan.get("results", []))
        scanned = int(scan.get("scanned_lines", len(parsed)))
        anomalies = int(scan.get("anomalies_detected", 0))
        sessions = int(scan.get("sessions_analyzed", scan.get("total_sessions", 0)))
        risk = scan.get("risk_score", "Unknown")
        threat_summary = scan.get("threat_summary", {})
    else:
        parsed = saved_parsed
        results = saved_results.copy()
        scanned = len(parsed)
        anomalies = (
            int(pd.to_numeric(results["predicted_anomaly"], errors="coerce").fillna(0).sum())
            if "predicted_anomaly" in results
            else 0
        )
        sessions = (
            int(results["block_id"].nunique())
            if "block_id" in results
            else int(parsed["block_id"].nunique()) if "block_id" in parsed else 0
        )
        if "predicted_anomaly" in results and len(results):
            prediction_values = pd.to_numeric(
                results["predicted_anomaly"], errors="coerce"
            ).fillna(0)
            anomaly_rate = float(prediction_values.mean())
            risk = "High" if anomaly_rate >= 0.20 else "Medium" if anomaly_rate >= 0.08 else "Low"
        else:
            risk = "Unknown"
        threat_summary = {}

    metric_columns = st.columns(4)
    for column, label, value, accent in zip(
        metric_columns,
        ("Logs Scanned", "Anomalies Flagged", "Threat Risk", "Sessions Analyzed"),
        (f"{scanned:,}", f"{anomalies:,}", risk, f"{sessions:,}"),
        (False, True, True, False),
    ):
        with column:
            metric_card(label, value, accent)

    st.markdown("")
    st.markdown('<div class="section-heading">Anomaly Trend • Highest Risk Sessions</div>', unsafe_allow_html=True)
    if not results.empty and {"block_id", "anomaly_score"}.issubset(results.columns):
        chart_data = results.copy()
        chart_data["anomaly_score"] = pd.to_numeric(chart_data["anomaly_score"], errors="coerce")
        chart_data = chart_data.dropna(subset=["anomaly_score"]).sort_values(
            "anomaly_score", ascending=False
        ).head(40)
        figure = go.Figure(
            go.Bar(
                x=chart_data["block_id"].astype(str),
                y=chart_data["anomaly_score"],
                marker_color=PALETTE["amber"],
                hovertemplate="Session %{x}<br>Anomaly score %{y:.4f}<extra></extra>",
            )
        )
        figure.update_layout(
            height=330,
            margin={"l": 18, "r": 18, "t": 18, "b": 70},
            paper_bgcolor=PALETTE["background"],
            plot_bgcolor=PALETTE["surface"],
            font={"color": PALETTE["muted"]},
            xaxis_title="Block ID",
            yaxis_title="Anomaly score",
            xaxis={"gridcolor": PALETTE["border"], "tickangle": -35},
            yaxis={"gridcolor": PALETTE["border"], "zerolinecolor": PALETTE["border"]},
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    else:
        st.info("Upload a log file to generate anomaly scores.")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="soc-card"><div class="section-heading">Threat Distribution</div>', unsafe_allow_html=True)
        if threat_summary:
            threat_data = pd.DataFrame(
                [{"Threat": name, "Sessions": count} for name, count in threat_summary.items()]
            )
            pie = px.pie(
                threat_data,
                names="Threat",
                values="Sessions",
                color_discrete_sequence=[PALETTE["amber"], PALETTE["green"], "#5bc0eb", PALETTE["red"]],
            )
            pie.update_layout(
                height=250,
                margin={"l": 8, "r": 8, "t": 12, "b": 8},
                paper_bgcolor=PALETTE["surface"],
                font={"color": PALETTE["muted"]},
                legend={"font": {"color": PALETTE["muted"]}},
            )
            st.plotly_chart(pie, width="stretch", config={"displayModeBar": False})
        else:
            st.markdown(
                '<div class="status-line">Threat categories appear after a live scan.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="soc-card"><div class="section-heading">Operational Status</div>', unsafe_allow_html=True)
        checks = (
            ("Parser engine operational", True),
            ("Model artifact loaded", model_ready),
            ("FastAPI backend online", api_online),
            (
                "No high-risk alert triggered",
                not bool(scan and scan.get("alert", {}).get("triggered")),
            ),
        )
        for label, healthy in checks:
            dot_color = PALETTE["green"] if healthy else PALETTE["amber"]
            glyph = "✓" if healthy else "!"
            st.markdown(
                f'<div class="status-line"><span class="status-dot" style="color:{dot_color}">{glyph}</span>'
                f"{html.escape(label)}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def render_live_analyzer(
    api_online: bool,
    model_artifact: dict[str, Any] | None,
) -> None:
    st.title("📂 Live Log Analyzer")
    st.caption(
        "Upload a raw log file to parse sessions, run Isolation Forest inference, and classify detected threats."
    ) 
    st.info("Text-based .log and .txt files up to 200 MB are supported. Binary files are rejected clearly.")
    engine_options = ["FastAPI Backend", "Local Model"]
    default_engine = 0 if api_online else 1
    selected_engine = st.radio(
        "Inference engine",
        engine_options,
        index=default_engine,
        horizontal=True,
        disabled=not model_artifact and not api_online,
    )
    uploaded_file = st.file_uploader(
        "Upload .log / .txt file",
        type=["log", "txt"],
        help="Text-based log files are processed in 25,000-line batches. Maximum size: 200 MB.",
        max_upload_size=MAX_UPLOAD_MB,
    )
    run_clicked = st.button(
        "Run Threat Analysis",
        type="primary",
        disabled=uploaded_file is None or (
            selected_engine == "FastAPI Backend" and not api_online
        ) or (selected_engine == "Local Model" and not model_artifact),
    )
    if run_clicked and uploaded_file is not None:
        progress = st.progress(0, text="Preparing the log stream...")
        file_size = int(getattr(uploaded_file, "size", 0))

        def update_progress(
            processed_bytes: int,
            total_bytes: int | None,
            parsed_lines: int,
            phase: str,
            scored_sessions: int,
            total_sessions: int,
        ) -> None:
            if phase == "upload":
                fraction = processed_bytes / total_bytes if total_bytes else 0.0
                percent = int(min(0.7, max(0.0, fraction * 0.7)) * 100)
                message = f"Streaming upload: {fraction:.0%}"
            elif phase == "parsing":
                fraction = processed_bytes / total_bytes if total_bytes else 0.0
                percent = int(min(0.7, max(0.0, fraction * 0.7)) * 100)
                message = f"Parsed {parsed_lines:,} log lines"
            elif phase == "inference":
                fraction = scored_sessions / total_sessions if total_sessions else 1.0
                percent = 70 + int(min(1.0, max(0.0, fraction)) * 29)
                message = f"Scoring sessions: {scored_sessions:,} / {total_sessions:,}"
            else:
                percent = 100
                message = f"Completed {parsed_lines:,} log lines"
            progress.progress(percent, text=message)

        try:
            if file_size == 0:
                raise ValueError("The uploaded file is empty.")
            if file_size > MAX_UPLOAD_MB * 1024 * 1024:
                raise ValueError("The uploaded file exceeds the 200 MB limit.")
            if selected_engine == "FastAPI Backend":
                with st.spinner("Sending the file to FastAPI and analyzing sessions..."):
                    try:
                        scan = call_predict_api(
                            uploaded_file.name,
                            uploaded_file,
                            file_size,
                            update_progress,
                        )
                    except BackendUnavailableError as exc:
                        if model_artifact is None:
                            raise RuntimeError(f"Could not reach FastAPI: {exc}") from exc
                        st.warning("FastAPI became unreachable. Continuing with the local model.")
                        scan = local_predict(
                            uploaded_file.name,
                            uploaded_file,
                            model_artifact,
                            file_size,
                            update_progress,
                        )
            else:
                with st.spinner("Parsing log sessions and scoring anomalies locally..."):
                    if model_artifact is None:
                        raise RuntimeError("The local model is unavailable.")
                    scan = local_predict(
                        uploaded_file.name,
                        uploaded_file,
                        model_artifact,
                        file_size,
                        update_progress,
                    )
            st.session_state["scan_result"] = scan
            st.success(
                f"Analysis complete: {scan['sessions_analyzed']:,} sessions, "
                f"{scan['anomalies_detected']:,} anomalies."
            )
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            st.error(f"Analysis failed: {exc}")
        except Exception as exc:
            LOGGER.exception("Unexpected threat-analysis failure")
            st.error(f"Unexpected analysis error: {exc}")
        finally:
            progress.empty()

    scan = st.session_state.get("scan_result")
    if scan:
        st.divider()
        columns = st.columns(4)
        for column, label, value in zip(
            columns,
            ("Lines Scanned", "Sessions Analyzed", "Anomalies Flagged", "Threat Risk"),
            (
                f"{scan.get('scanned_lines', 0):,}",
                f"{scan.get('sessions_analyzed', 0):,}",
                f"{scan.get('anomalies_detected', 0):,}",
                scan.get("risk_score", "Unknown"),
            ),
        ):
            with column:
                st.metric(label, value)
        flagged = pd.DataFrame(scan.get("flagged_sessions", []))
        st.subheader("High-risk sessions")
        if flagged.empty:
            st.success("No anomalous sessions were flagged.")
        else:
            st.dataframe(
                flagged,
                hide_index=True,
                width="stretch",
                height=min(420, 42 + len(flagged) * 35),
            )
        incident_columns = [
            "block_id",
            "severity",
            "anomaly_score",
            "threat_type",
            "timestamp",
            "duration_seconds",
            "error_count",
        ]
        incident_csv = flagged.reindex(columns=incident_columns).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Incident Report",
            incident_csv,
            file_name="sentinellog-incident-report.csv",
            mime="text/csv",
        )
        if scan.get("incident_report_truncated"):
            st.warning(
                "The downloadable report contains the 10,000 highest-scoring flagged sessions; "
                "the anomaly metric includes every flagged session."
            )


def render_parsed_logs(
    scan: dict[str, Any] | None,
    saved_parsed: pd.DataFrame,
    templates: pd.DataFrame,
) -> None:
    st.title("📁 Parsed Logs Explorer")
    st.caption("Search parsed events, filter by severity, and inspect session content.")
    source = (
        add_template_content(pd.DataFrame(scan.get("parsed_logs", [])), templates)
        if scan
        else saved_parsed.copy()
    )
    if source.empty:
        st.info("No parsed records are available. Upload a log file or provide parsed_logs.csv.")
        return
    query = st.text_input(
        "Search parsed logs by keyword",
        placeholder="Search parsed logs by keyword...",
    )
    level = st.selectbox("Filter by log level", ("ALL", "INFO", "WARN", "ERROR"))
    filtered = source
    if query:
        matches = filtered.astype(str).apply(
            lambda column: column.str.contains(query, case=False, na=False, regex=False)
        )
        filtered = filtered[matches.any(axis=1)]
    if level != "ALL" and "level" in filtered:
        filtered = filtered[filtered["level"].astype(str).str.upper() == level]
    columns = ["block_id", "event_id", "level", "date", "time", "content"]
    st.caption(f"Showing {len(filtered):,} records")
    st.dataframe(
        filtered.reindex(columns=columns),
        hide_index=True,
        width="stretch",
        height=520,
    )


def render_model_metrics(model_artifact: dict[str, Any] | None) -> None:
    st.title("⚙️ Model Metrics & Configuration")
    if model_artifact is None:
        st.error(f"Model artifact unavailable: `{MODEL_PATH.name}`")
        return
    model = model_artifact["model"]
    feature_columns = model_artifact["feature_cols"]
    values = (
        ("Algorithm", type(model).__name__),
        ("Estimators", getattr(model, "n_estimators", "N/A")),
        ("Contamination", getattr(model, "contamination", "N/A")),
        ("Features", len(feature_columns)),
        ("Random State", getattr(model, "random_state", "N/A")),
    )
    metric_columns = st.columns(len(values))
    for column, (label, value) in zip(metric_columns, values):
        with column:
            metric_card(label, value)
    st.subheader("Isolation Forest Parameters")
    parameters = {
        "max_samples": getattr(model, "max_samples", "N/A"),
        "max_features": getattr(model, "max_features", "N/A"),
        "bootstrap": getattr(model, "bootstrap", "N/A"),
        "feature_columns": feature_columns,
    }
    st.json(parameters, expanded=True)


parsed_df, results_df, templates_df = initial_data()
model_artifact = load_model_artifact()
api_online = backend_is_online()

st.sidebar.markdown('<div class="soc-brand">🛡️ SentinelLog AI</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="soc-subtitle">SOC INTELLIGENCE CONSOLE</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
page = st.sidebar.selectbox("Navigation", PAGES, index=0)

engine_ready = api_online or model_artifact is not None
status_class = "status-pill" if engine_ready else "status-pill offline"
status_label = "Detection engine ready" if engine_ready else "Detection engine offline"
api_status = "Online" if api_online else "Offline"
model_status = "ready" if model_artifact is not None else "unavailable"
st.sidebar.markdown(
    f'<div class="status-card"><span class="{status_class}">● {status_label}</span>'
    f'<div class="status-line">Isolation Forest • Drain parser • API: {api_status}</div>'
    f'<div class="status-line">Local model: {model_status}</div></div>',
    unsafe_allow_html=True,
)

scan_result = st.session_state.get("scan_result")
if page == "Dashboard":
    render_dashboard(scan_result, parsed_df, results_df, model_artifact is not None, api_online)
elif page == "Live Log Analyzer":
    render_live_analyzer(api_online, model_artifact)
elif page == "Parsed Logs Explorer":
    render_parsed_logs(scan_result, parsed_df, templates_df)
else:
    render_model_metrics(model_artifact)

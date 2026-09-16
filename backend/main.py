from __future__ import annotations

import logging
import re
import tempfile
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from features import build_feature_matrix
from log_parser import parse_log_file

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "isolation_forest.joblib"
TEMP_DIR = ROOT / "tmp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_SUFFIXES = {".log", ".txt"}
logger = logging.getLogger("ai_log_anomaly")

app = FastAPI(title="AI Log Anomaly Detection API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

THREAT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SQL Injection", ("union select", "or 1=1", "drop table", "information_schema", "select * from", "sqlmap")),
    ("Brute Force Login", ("failed password", "authentication failure", "invalid user", "login failed", "too many attempts")),
    ("Server Error", ("status=500", "http 500", "internal server error", "traceback", "exception", "error")),
    ("Suspicious Command", ("cmd.exe", "powershell", "/bin/sh", "wget ", "curl ", "base64")),
)


def load_model_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "model" not in artifact or "feature_cols" not in artifact:
        raise ValueError("Model artifact must contain 'model' and 'feature_cols'.")
    if not isinstance(artifact["feature_cols"], list) or not artifact["feature_cols"]:
        raise ValueError("Model artifact contains no feature columns.")
    return artifact


def classify_threat(lines: list[str]) -> str:
    text = "\n".join(lines).lower()
    for threat, indicators in THREAT_RULES:
        if any(indicator in text for indicator in indicators):
            return threat
    return "Unclassified Anomaly"


def extract_block_from_line(line: str) -> str | None:
    match = re.search(r"\bblk_-?\d+\b", line)
    return match.group(0) if match else None


def build_threat_lookup(raw_text: str) -> dict[str, str]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for line in raw_text.splitlines():
        block_id = extract_block_from_line(line)
        if block_id:
            grouped[block_id].append(line)
    return {block_id: classify_threat(lines) for block_id, lines in grouped.items()}


def compute_risk(avg_score: float, anomaly_rate: float) -> tuple[str, float]:
    risk_index = max(0.0, avg_score + (anomaly_rate * 3.0))
    if risk_index >= 2.0:
        return "Critical", risk_index
    if risk_index >= 1.3:
        return "High", risk_index
    if risk_index >= 0.7:
        return "Medium", risk_index
    return "Low", risk_index


def trigger_security_alert(risk_level: str, filename: str, anomalies: int) -> dict[str, Any]:
    """Replace this integration point with SMTP, Telegram, or an incident API."""
    triggered = risk_level in {"High", "Critical"} and anomalies > 0
    if triggered:
        logger.warning("Security alert triggered: file=%s risk=%s anomalies=%d", filename, risk_level, anomalies)
    return {"triggered": triggered, "channels": ["email", "telegram"] if triggered else []}


@app.get("/")
def read_root() -> dict[str, Any]:
    try:
        load_model_artifact()
    except (FileNotFoundError, ValueError, OSError) as exc:
        return {"status": "degraded", "model_loaded": False, "error": str(exc)}
    return {"status": "online", "model_loaded": True, "model_name": MODEL_PATH.name}


@app.get("/health")
def health_check() -> dict[str, Any]:
    try:
        load_model_artifact()
    except (FileNotFoundError, ValueError, OSError) as exc:
        return {"status": "degraded", "model_loaded": False, "error": str(exc)}
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
async def predict_logs(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only .log and .txt files are supported.")

    try:
        artifact = load_model_artifact()
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Model is unavailable: {exc}") from exc

    raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the 10 MB limit.")
    if not raw_bytes.strip():
        raise HTTPException(status_code=400, detail="Uploaded log file is empty.")
    raw_text = raw_bytes.decode("utf-8", errors="replace")

    with tempfile.TemporaryDirectory(prefix="logscan_", dir=str(TEMP_DIR)) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        raw_path = tmp_dir / f"{uuid.uuid4().hex}_{filename}"
        parsed_path = tmp_dir / "parsed_logs.csv"
        template_path = tmp_dir / "templates.csv"
        labels_path = tmp_dir / "labels.csv"
        raw_path.write_text(raw_text, encoding="utf-8")

        try:
            parse_log_file(str(raw_path), str(parsed_path), str(template_path))
            parsed_df = pd.read_csv(parsed_path)
        except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse uploaded log file: {exc}") from exc
        if parsed_df.empty:
            raise HTTPException(status_code=400, detail="No parseable log entries were found.")

        block_ids = sorted(parsed_df["block_id"].astype(str).unique().tolist())
        pd.DataFrame({"block_id": block_ids, "label": 0}).to_csv(labels_path, index=False)
        try:
            feature_df = build_feature_matrix(str(parsed_path), str(labels_path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=500, detail=f"Feature extraction failed: {exc}") from exc

        feature_cols = artifact["feature_cols"]
        missing_cols = [column for column in feature_cols if column not in feature_df.columns]
        if missing_cols:
            raise HTTPException(status_code=500, detail=f"Model features mismatch: missing columns {missing_cols}.")

        model = artifact["model"]
        X = feature_df[feature_cols].fillna(0)
        try:
            decision_scores = -model.decision_function(X)
            predictions = (model.predict(X) == -1).astype(int)
        except (ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

        threat_lookup = build_threat_lookup(raw_text)
        results_df = pd.DataFrame(
            {
                "block_id": feature_df["block_id"].astype(str),
                "anomaly_score": decision_scores,
                "predicted_anomaly": predictions,
            }
        )
        results_df["threat_type"] = results_df["block_id"].map(threat_lookup).fillna("Unclassified Anomaly")
        flagged = results_df[results_df["predicted_anomaly"] == 1].sort_values("anomaly_score", ascending=False)
        anomaly_count = int(len(flagged))
        anomaly_rate = float(predictions.mean()) if len(predictions) else 0.0
        average_score = float(decision_scores.mean()) if len(decision_scores) else 0.0
        max_score = float(decision_scores.max()) if len(decision_scores) else 0.0
        risk_level, risk_index = compute_risk(average_score, anomaly_rate)
        alert = trigger_security_alert(risk_level, filename, anomaly_count)

        flagged_sessions = [
            {
                "block_id": str(row.block_id),
                "anomaly_score": round(float(row.anomaly_score), 6),
                "predicted_anomaly": int(row.predicted_anomaly),
                "threat_type": str(row.threat_type),
            }
            for row in flagged.head(50).itertuples(index=False)
        ]
        return {
            "status": "success",
            "filename": filename,
            "model_used": MODEL_PATH.name,
            "scanned_lines": int(len(parsed_df)),
            "sessions_analyzed": int(feature_df["block_id"].nunique()),
            "anomalies_detected": anomaly_count,
            "anomaly_rate": round(anomaly_rate, 4),
            "average_anomaly_score": round(average_score, 6),
            "max_anomaly_score": round(max_score, 6),
            "risk_score": risk_level,
            "risk_index": round(risk_index, 4),
            "alert": alert,
            "threat_summary": dict(Counter(item["threat_type"] for item in flagged_sessions)),
            "flagged_sessions": flagged_sessions,
            "parsed_logs": parsed_df.astype(str).to_dict(orient="records"),
            "feature_columns": feature_cols,
        }

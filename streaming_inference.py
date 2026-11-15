from __future__ import annotations

import heapq
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable

import pandas as pd

from log_parser import iter_log_chunks, parse_log_line

LOGGER = logging.getLogger(__name__)
LINES_PER_CHUNK = 25_000
SCORING_BATCH_SIZE = 25_000
PARSED_PREVIEW_LIMIT = 10_000
PARSED_PREVIEW_BYTES_LIMIT = 2 * 1024 * 1024
RESULT_PREVIEW_LIMIT = 500
INCIDENT_REPORT_LIMIT = 10_000

THREAT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SQL Injection", ("union select", "or 1=1", "drop table", "information_schema", "sqlmap")),
    ("Brute Force Login", ("failed password", "authentication failure", "invalid user", "login failed", "too many attempts")),
    ("Server Error", ("status=500", "http 500", "internal server error", "traceback", "exception", "error")),
    ("Suspicious Command", ("cmd.exe", "powershell", "/bin/sh", "wget ", "curl ", "base64")),
)


def _read_templates(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Known event templates are missing: {path}")
    templates = pd.read_csv(path)
    if not {"event_id", "template"}.issubset(templates.columns):
        raise ValueError("The known templates file must contain event_id and template columns.")
    return {
        str(row.template): int(row.event_id)
        for row in templates.itertuples(index=False)
    }


def _timestamp(date: str, time: str) -> str:
    if not date:
        return ""
    return f"{date} {time}".strip()


def _duration_seconds(start: str, end: str) -> float:
    if not start or not end:
        return 0.0
    if " " in start and " " in end:
        start_date, start_time = start.split(" ", 1)
        end_date, end_time = end.split(" ", 1)
        if (
            len(start_date) == len(end_date) == 6
            and start_date.isdigit()
            and end_date.isdigit()
            and start_time.isdigit()
            and end_time.isdigit()
        ):
            try:
                first = datetime.strptime(start_date + start_time, "%y%m%d%H%M%S")
                last = datetime.strptime(end_date + end_time, "%y%m%d%H%M%S")
                return max(0.0, (last - first).total_seconds())
            except ValueError:
                return 0.0
    for value in (start, end):
        if value.isdigit() and len(value) == 12:
            continue
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
    try:
        if start.isdigit() and end.isdigit() and len(start) == len(end) == 12:
            first = datetime.strptime(start, "%y%m%d%H%M%S")
            last = datetime.strptime(end, "%y%m%d%H%M%S")
        else:
            first = datetime.fromisoformat(start.replace("Z", "+00:00"))
            last = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return max(0.0, (last - first).total_seconds())
    except (ValueError, TypeError):
        return 0.0


def _threats_for_line(line: str) -> set[str]:
    normalized = line.lower()
    return {
        name
        for name, indicators in THREAT_RULES
        if any(indicator in normalized for indicator in indicators)
    }


def _session_threat(threats: set[str]) -> str:
    for name, _ in THREAT_RULES:
        if name in threats:
            return name
    return "Unclassified Anomaly"


def _make_model_row(
    block_id: str,
    session: dict[str, Any],
    feature_columns: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "block_id": block_id,
        "n_events": session["n_events"],
        "n_distinct_events": len(session["distinct_events"]),
        "warn_err_ratio": session["warn_err_count"] / session["n_events"],
    }
    for column in feature_columns:
        if column.startswith("evt_"):
            try:
                event_id = int(column[4:])
            except ValueError:
                continue
            row[column] = session["event_counts"].get(event_id, 0)
    return row


def analyze_log_stream(
    stream: BinaryIO,
    artifact: dict[str, Any],
    templates_path: Path,
    *,
    total_bytes: int | None = None,
    progress_callback: Callable[[int, int | None, int, str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Parse text in bounded line batches and score sessions in model batches."""
    model = artifact.get("model")
    feature_columns = artifact.get("feature_cols")
    if model is None or not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("The model artifact must include a model and feature columns.")
    template_to_id = _read_templates(templates_path)
    sessions: dict[str, dict[str, Any]] = {}
    preview: list[dict[str, Any]] = []
    preview_bytes = 0
    line_count = 0
    bytes_processed = 0
    last_progress_bytes = 0

    def report_progress(read_bytes: int) -> None:
        nonlocal bytes_processed, last_progress_bytes
        bytes_processed += read_bytes
        if (
            progress_callback is not None
            and bytes_processed - last_progress_bytes >= 2 * 1024 * 1024
        ):
            last_progress_bytes = bytes_processed
            progress_callback(bytes_processed, total_bytes, line_count, "parsing", 0, 0)

    for lines in iter_log_chunks(stream, LINES_PER_CHUNK, on_bytes_read=report_progress):
        for line in lines:
            parsed = parse_log_line(line)
            if parsed is None:
                continue
            line_count += 1
            template = str(parsed["template"])
            event_id = template_to_id.get(template)
            if len(preview) < PARSED_PREVIEW_LIMIT:
                preview_row = {
                    key: str(value)
                    for key, value in parsed.items()
                    if key != "template"
                }
                preview_row["event_id"] = str(event_id if event_id is not None else -1)
                row_size = sum(len(value.encode("utf-8")) for value in preview_row.values())
                if preview_bytes + row_size <= PARSED_PREVIEW_BYTES_LIMIT:
                    preview.append(preview_row)
                    preview_bytes += row_size
            block_id = str(parsed["block_id"])
            if event_id is None:
                distinct_token: Any = ("unknown", template)
            else:
                distinct_token = event_id
            session = sessions.get(block_id)
            if session is None:
                session = {
                    "n_events": 0,
                    "warn_err_count": 0,
                    "event_counts": Counter(),
                    "distinct_events": set(),
                    "first_timestamp": "",
                    "last_timestamp": "",
                    "errors": 0,
                    "warnings": 0,
                    "threats": set(),
                }
                sessions[block_id] = session
            session["n_events"] += 1
            session["distinct_events"].add(distinct_token)
            if event_id is not None:
                session["event_counts"][event_id] += 1
            level = str(parsed["level"]).upper()
            if level in {"WARN", "ERROR"}:
                session["warn_err_count"] += 1
            if level == "ERROR":
                session["errors"] += 1
            elif level == "WARN":
                session["warnings"] += 1
            timestamp = _timestamp(str(parsed["date"]), str(parsed["time"]))
            if timestamp:
                if not session["first_timestamp"] or timestamp < session["first_timestamp"]:
                    session["first_timestamp"] = timestamp
                if not session["last_timestamp"] or timestamp > session["last_timestamp"]:
                    session["last_timestamp"] = timestamp
            session["threats"].update(_threats_for_line(line))

    if line_count == 0:
        raise ValueError("No non-empty log lines were found in this file.")

    session_count = len(sessions)
    anomaly_count = 0
    score_sum = 0.0
    maximum_score = float("-inf")
    threat_summary: Counter[str] = Counter()
    results_heap: list[tuple[float, int, dict[str, Any]]] = []
    incidents_heap: list[tuple[float, int, dict[str, Any]]] = []
    batch_ids: list[str] = []
    batch_rows: list[dict[str, Any]] = []
    sequence = 0

    def score_batch() -> None:
        nonlocal anomaly_count, score_sum, maximum_score, sequence
        if not batch_ids:
            return
        model_frame = pd.DataFrame(batch_rows, columns=["block_id", *feature_columns])
        model_features = model_frame[feature_columns].fillna(0)
        anomaly_scores = -model.decision_function(model_features)
        predictions = model.predict(model_features)
        for index, block_id in enumerate(batch_ids):
            session = sessions[block_id]
            anomaly_score = float(anomaly_scores[index])
            is_anomaly = int(predictions[index] == -1)
            score_sum += anomaly_score
            maximum_score = max(maximum_score, anomaly_score)
            timestamp = session["last_timestamp"]
            duration = _duration_seconds(session["first_timestamp"], timestamp)
            severity = (
                "Critical" if session["errors"] > 0
                else "High" if session["warnings"] > 0
                else "Medium"
            )
            item = {
                "block_id": block_id,
                "anomaly_score": round(anomaly_score, 6),
                "predicted_anomaly": is_anomaly,
                "status": "Anomaly" if is_anomaly else "Normal",
                "severity": severity if is_anomaly else "Normal",
                "threat_type": _session_threat(session["threats"]) if is_anomaly else "",
                "timestamp": timestamp,
                "duration_seconds": round(duration, 3),
                "error_count": int(session["errors"]),
                "warning_count": int(session["warnings"]),
            }
            heapq.heappush(results_heap, (anomaly_score, sequence, item))
            if len(results_heap) > RESULT_PREVIEW_LIMIT:
                heapq.heappop(results_heap)
            if is_anomaly:
                anomaly_count += 1
                threat_summary[item["threat_type"]] += 1
                heapq.heappush(incidents_heap, (anomaly_score, sequence, item))
                if len(incidents_heap) > INCIDENT_REPORT_LIMIT:
                    heapq.heappop(incidents_heap)
            sequence += 1
        if progress_callback is not None:
            progress_callback(
                bytes_processed,
                total_bytes,
                line_count,
                "inference",
                sequence,
                session_count,
            )
        batch_ids.clear()
        batch_rows.clear()

    for block_id, session in sessions.items():
        batch_ids.append(block_id)
        batch_rows.append(_make_model_row(block_id, session, feature_columns))
        if len(batch_ids) >= SCORING_BATCH_SIZE:
            score_batch()
    score_batch()

    anomaly_rate = anomaly_count / session_count if session_count else 0.0
    if anomaly_rate >= 0.20:
        risk = "High"
    elif anomaly_rate >= 0.08:
        risk = "Medium"
    else:
        risk = "Low"
    top_results = [
        item for _, _, item in sorted(results_heap, key=lambda value: value[0], reverse=True)
    ]
    flagged_sessions = [
        item for _, _, item in sorted(incidents_heap, key=lambda value: value[0], reverse=True)
    ]
    if progress_callback is not None:
        progress_callback(
            total_bytes or bytes_processed,
            total_bytes,
            line_count,
            "complete",
            session_count,
            session_count,
        )
    return {
        "status": "success",
        "filename": "",
        "scanned_lines": line_count,
        "sessions_analyzed": session_count,
        "total_sessions": session_count,
        "anomalies_detected": anomaly_count,
        "anomaly_rate": round(anomaly_rate, 6),
        "average_anomaly_score": round(score_sum / session_count, 6),
        "max_anomaly_score": round(maximum_score, 6),
        "risk_score": risk,
        "risk_index": round(anomaly_rate, 6),
        "alert": {"triggered": risk == "High" and anomaly_count > 0, "channels": []},
        "threat_summary": dict(threat_summary),
        "results": top_results,
        "flagged_sessions": flagged_sessions,
        "parsed_logs": preview,
        "parsed_logs_truncated": line_count > len(preview),
        "results_truncated": session_count > len(top_results),
        "incident_report_truncated": anomaly_count > len(flagged_sessions),
        "feature_columns": feature_columns,
    }


def analyze_log_path(
    path: Path,
    artifact: dict[str, Any],
    templates_path: Path,
) -> dict[str, Any]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        result = analyze_log_stream(
            stream,
            artifact,
            templates_path,
            total_bytes=size,
        )
    result["filename"] = path.name
    return result

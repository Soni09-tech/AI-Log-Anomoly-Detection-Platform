from __future__ import annotations

import logging
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException
from starlette.requests import ClientDisconnect

from streaming_inference import analyze_log_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "isolation_forest.joblib"
TEMPLATES_PATH = PROJECT_ROOT / "templates.csv"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
logger = logging.getLogger("sentinellog")


class RequestBodyTooLarge(Exception):
    pass


class UploadSizeLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/predict"
        ):
            await self.app(scope, receive, send)
            return

        bytes_received = 0
        response_started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal bytes_received
            message = await receive()
            if message["type"] == "http.request":
                bytes_received += len(message.get("body", b""))
                if bytes_received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if not response_started:
                body = b'{"detail":"Request exceeds the 200 MB upload limit."}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode("ascii")),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})


app = FastAPI(title="SentinelLog AI Engine", version="1.0.0")
app.add_middleware(
    UploadSizeLimitMiddleware,
    max_bytes=MAX_UPLOAD_BYTES + 64 * 1024,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_model_artifact() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    try:
        artifact = joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.exception("Could not load model artifact at %s", MODEL_PATH)
        raise ValueError(f"Model artifact could not be loaded: {exc}") from exc
    if (
        not isinstance(artifact, dict)
        or "model" not in artifact
        or not isinstance(artifact.get("feature_cols"), list)
        or not artifact["feature_cols"]
    ):
        raise ValueError("Model artifact must include a model and non-empty feature_cols.")
    if not callable(getattr(artifact["model"], "predict", None)) or not callable(
        getattr(artifact["model"], "decision_function", None)
    ):
        raise ValueError("Model must provide predict() and decision_function().")
    return artifact


def _safe_filename(filename: str | None) -> str:
    return (filename or "uploaded.log").replace("\\", "/").rsplit("/", 1)[-1] or "uploaded.log"


async def _write_upload(upload: UploadFile, destination: Path) -> int:
    total = 0
    try:
        with destination.open("wb") as target:
            while chunk := await upload.read(COPY_CHUNK_BYTES):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Log input exceeds the 200 MB upload limit.",
                    )
                target.write(chunk)
    except OSError as exc:
        logger.exception("Could not spool uploaded log to temporary storage")
        raise HTTPException(status_code=507, detail="Temporary storage is unavailable.") from exc
    return total


async def _write_raw_request(request: Request, destination: Path) -> int:
    total = 0
    try:
        with destination.open("wb") as target:
            async for chunk in request.stream():
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Log input exceeds the 200 MB upload limit.",
                    )
                target.write(chunk)
    except OSError as exc:
        logger.exception("Could not spool raw request to temporary storage")
        raise HTTPException(status_code=507, detail="Temporary storage is unavailable.") from exc
    except ClientDisconnect as exc:
        raise HTTPException(status_code=400, detail="The upload ended before the request completed.") from exc
    return total


@app.get("/")
def read_root() -> dict[str, str]:
    return {"status": "online", "engine": "SentinelLog AI", "version": "1.0.0"}


@app.get("/health")
def health_check() -> dict[str, Any]:
    try:
        load_model_artifact()
    except Exception as exc:
        logger.exception("Health check could not load model")
        return {"status": "degraded", "model_loaded": False, "error": str(exc)}
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
async def predict_logs(request: Request) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            request_size = int(content_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from exc
        if request_size > MAX_UPLOAD_BYTES + 64 * 1024:
            raise HTTPException(status_code=413, detail="Log input exceeds the 200 MB upload limit.")

    try:
        artifact = load_model_artifact()
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.exception("Prediction model is unavailable")
        raise HTTPException(status_code=503, detail=f"Prediction model unavailable: {exc}") from exc

    content_type = request.headers.get("content-type", "").lower()
    with tempfile.TemporaryDirectory(prefix="sentinellog_") as temporary_directory:
        raw_path = Path(temporary_directory) / "uploaded.log"
        filename = "uploaded.log"
        if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            try:
                form = await request.form(
                    max_files=1,
                    max_fields=1,
                    max_part_size=MAX_UPLOAD_BYTES,
                )
            except (MultiPartException, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"Invalid upload form: {exc}") from exc
            uploaded = form.get("file")
            if isinstance(uploaded, UploadFile):
                filename = _safe_filename(uploaded.filename)
                if Path(filename).suffix.lower() not in {".log", ".txt"}:
                    await uploaded.close()
                    raise HTTPException(
                        status_code=400,
                        detail="Upload a .log or .txt text file.",
                    )
                try:
                    uploaded_bytes = await _write_upload(uploaded, raw_path)
                finally:
                    await uploaded.close()
            else:
                text = form.get("text")
                if not isinstance(text, str):
                    raise HTTPException(
                        status_code=400,
                        detail="Provide a .log/.txt file or a text form field.",
                    )
                encoded = text.encode("utf-8")
                if len(encoded) > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Log input exceeds the 200 MB upload limit.",
                    )
                raw_path.write_bytes(encoded)
                uploaded_bytes = len(encoded)
        else:
            uploaded_bytes = await _write_raw_request(request, raw_path)

        if uploaded_bytes == 0:
            raise HTTPException(status_code=400, detail="Log input is empty.")
        if not raw_path.is_file():
            raise HTTPException(status_code=400, detail="No log content was uploaded.")

        try:
            result = await run_in_threadpool(
                analyze_log_path,
                raw_path,
                artifact,
                TEMPLATES_PATH,
            )
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            logger.info("Log analysis rejected for %s: %s", filename, exc)
            raise HTTPException(status_code=400, detail=f"Could not analyze uploaded log: {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected inference failure for %s", filename)
            raise HTTPException(status_code=500, detail="Log inference failed unexpectedly.") from exc

    result["filename"] = filename
    result["model_used"] = MODEL_PATH.name
    return result

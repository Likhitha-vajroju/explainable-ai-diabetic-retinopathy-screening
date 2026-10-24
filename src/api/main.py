"""FastAPI application exposing image analysis through existing project modules."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import tempfile
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

from fastapi import FastAPI, File, HTTPException, Path as ApiPath, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.api.screening_history import ScreeningHistoryStore
from src.preprocessing.image_pipeline import preprocess_image
from src.reporting.screening_report import (
    create_screening_report,
    write_screening_report,
)


LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG", "TIFF", "BMP"}
SUPPORTED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/bmp",
}
CONTENT_TYPES_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "TIFF": "image/tiff",
    "BMP": "image/bmp",
}

ReviewerDecision = Literal[
    "ai_prediction_accepted",
    "ai_prediction_modified",
    "image_requires_manual_review",
    "image_ungradable",
]


class ScreeningRecordCreate(BaseModel):
    anonymous_patient_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    dr_level: int = Field(ge=0, le=4)
    referable: bool
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    image_quality_score: float = Field(ge=0, le=100, allow_inf_nan=False)


class ReviewerDecisionUpdate(BaseModel):
    reviewer_decision: ReviewerDecision


def _configured_temperature() -> float | None:
    value = os.environ.get("DR_CALIBRATION_TEMPERATURE")
    if value is None or not value.strip():
        return None
    try:
        temperature = float(value)
    except ValueError as error:
        raise RuntimeError("DR_CALIBRATION_TEMPERATURE must be a positive number.") from error
    if not 0 < temperature < float("inf"):
        raise RuntimeError("DR_CALIBRATION_TEMPERATURE must be finite and positive.")
    return temperature


def load_model_checkpoint(checkpoint_path: str | Path) -> tuple[Any, Any, Any]:
    from src.dr_classification.classifier import load_model_checkpoint as load_checkpoint

    return load_checkpoint(checkpoint_path)


def _encode_file(path: Path, media_type: str) -> dict[str, str]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "media_type": media_type,
        "base64": encoded,
        "data_url": f"data:{media_type};base64,{encoded}",
    }


def _validate_image(data: bytes) -> tuple[str, tuple[int, int]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                image_format = image.format
                dimensions = image.size
                if image_format not in SUPPORTED_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail=(
                            "Unsupported image format. Upload a JPEG, PNG, TIFF, "
                            "or BMP fundus image."
                        ),
                    )
                if (
                    dimensions[0] < 32
                    or dimensions[1] < 32
                    or dimensions[0] * dimensions[1] > MAX_IMAGE_PIXELS
                ):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Image dimensions must be at least 32x32 and no more "
                            "than 25 megapixels."
                        ),
                    )
                image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file is not a valid, readable retinal image.",
        ) from error
    except Image.DecompressionBombWarning as error:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the supported pixel limit.",
        ) from error
    assert image_format is not None
    return image_format, dimensions


def create_app(
    checkpoint_path: str | Path | None = None,
    calibration_temperature: float | None = None,
    history_db_path: str | Path | None = None,
) -> FastAPI:
    configured_checkpoint = Path(
        checkpoint_path
        if checkpoint_path is not None
        else os.environ.get(
            "DR_CHECKPOINT",
            "models/checkpoints/dr_resnet18/best_model.pt",
        )
    )
    configured_temperature = (
        calibration_temperature
        if calibration_temperature is not None
        else _configured_temperature()
    )
    configured_history_path = Path(
        history_db_path
        if history_db_path is not None
        else os.environ.get("DR_HISTORY_DB", "data/screening_history.sqlite3")
    )
    history_store = ScreeningHistoryStore(configured_history_path)
    analysis_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        history_store.initialize()
        application.state.model = None
        application.state.config = None
        application.state.device = None
        application.state.model_error = None
        application.state.model_status = "model_unavailable"
        application.state.model_loading_task = None
        if configured_checkpoint.is_file():
            application.state.model_status = "loading"

            async def load_configured_model() -> None:
                try:
                    model, config, device = await run_in_threadpool(
                        load_model_checkpoint,
                        configured_checkpoint,
                    )
                except Exception as error:
                    application.state.model_error = str(error)
                    application.state.model_status = "model_unavailable"
                    LOGGER.exception("Could not load DR classifier checkpoint.")
                else:
                    application.state.model = model
                    application.state.config = config
                    application.state.device = device
                    application.state.model_status = "ready"

            application.state.model_loading_task = asyncio.create_task(
                load_configured_model()
            )
        else:
            application.state.model_error = (
                f"Classifier checkpoint not found: {configured_checkpoint}"
            )
            LOGGER.error(application.state.model_error)
        try:
            yield
        finally:
            loading_task = application.state.model_loading_task
            if loading_task is not None and not loading_task.done():
                loading_task.cancel()
                try:
                    await loading_task
                except asyncio.CancelledError:
                    pass

    application = FastAPI(
        title="AI-assisted DR screening API",
        description=(
            "Research prototype API. Outputs are not a definitive diagnosis "
            "and must not determine treatment."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip()
            for origin in os.environ.get(
                "DR_CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Content-Type"],
    )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        LOGGER.error(
            "Unhandled API error for %s %s.",
            request.method,
            request.url.path,
            exc_info=(type(error), error, error.__traceback__),
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check the service logs for details."},
        )

    @application.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {
                        "loc": item["loc"],
                        "msg": item["msg"],
                        "type": item["type"],
                    }
                    for item in error.errors()
                ]
            },
        )

    @application.get("/api/health")
    async def health() -> dict[str, Any]:
        ready = application.state.model is not None
        return {
            "status": "ready" if ready else application.state.model_status,
            "model_ready": ready,
            "model_loading": application.state.model_status == "loading",
            "model_error": application.state.model_error,
            "calibration_available": configured_temperature is not None,
        }

    @application.post("/api/screening-history", status_code=201)
    async def create_screening_history_record(
        record: ScreeningRecordCreate,
    ) -> dict[str, Any]:
        return await run_in_threadpool(
            history_store.create_record,
            **record.model_dump(),
        )

    @application.get("/api/screening-history/{anonymous_patient_id}")
    async def get_screening_history(
        anonymous_patient_id: str = ApiPath(
            min_length=1,
            max_length=64,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
        ),
    ) -> dict[str, Any]:
        records = await run_in_threadpool(
            history_store.get_patient_records,
            anonymous_patient_id,
        )
        return {
            "anonymous_patient_id": anonymous_patient_id,
            "records": records,
        }

    @application.put("/api/screening-history/{record_id}/review")
    async def update_reviewer_decision(
        decision: ReviewerDecisionUpdate,
        record_id: int = ApiPath(gt=0),
    ) -> dict[str, Any]:
        record = await run_in_threadpool(
            history_store.record_reviewer_decision,
            record_id,
            decision.reviewer_decision,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="Screening record not found.")
        return record

    @application.post("/api/analyze")
    async def analyze(image: UploadFile = File(...)) -> dict[str, Any]:
        if application.state.model is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The DR classifier is unavailable. Configure a valid trained "
                    "checkpoint with DR_CHECKPOINT and restart the API."
                ),
            )

        content_type = (image.content_type or "").lower()
        if content_type not in SUPPORTED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Unsupported image type. Upload JPEG, PNG, TIFF, or BMP.",
            )
        chunks: list[bytes] = []
        total_bytes = 0
        while chunk := await image.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Image upload exceeds the 15 MB limit.",
                )
            chunks.append(chunk)
        image_bytes = b"".join(chunks)
        if not image_bytes:
            raise HTTPException(status_code=422, detail="The uploaded image is empty.")
        image_format, original_size = _validate_image(image_bytes)
        if CONTENT_TYPES_BY_FORMAT[image_format] != content_type:
            raise HTTPException(
                status_code=415,
                detail="The image content does not match its declared image type.",
            )

        def process_request() -> dict[str, Any]:
            try:
                with tempfile.TemporaryDirectory(prefix="dr-screening-") as temporary:
                    workspace = Path(temporary)
                    image_path = workspace / f"uploaded.{image_format.lower()}"
                    image_path.write_bytes(image_bytes)
                    output_dir = workspace / "artifacts"
                    result = create_screening_report(
                        image_path,
                        application.state.model,
                        application.state.config,
                        application.state.device,
                        output_dir,
                        calibration_temperature=configured_temperature,
                    )
                    processed = preprocess_image(
                        image_path,
                        application.state.config.preprocessing,
                    )
                    processed_path = workspace / "preprocessed.png"
                    processed.enhanced_image.save(processed_path, format="PNG")

                    artifact_paths: dict[str, tuple[Path, str]] = {
                        "preprocessed_image": (processed_path, "image/png"),
                    }
                    for name, relative_name in (
                        ("gradcam", "gradcam.png"),
                        ("retinal_evidence", "retinal_evidence.png"),
                    ):
                        artifact_path = output_dir / relative_name
                        if artifact_path.is_file():
                            artifact_paths[name] = (artifact_path, "image/png")

                    result["image"]["format"] = image_format
                    if result["gradcam"]["status"] == "available":
                        result["gradcam"]["visualization_path"] = "gradcam.png"
                    if result["retinal_lesion_evidence"]["status"] == "available":
                        result["retinal_lesion_evidence"]["visualization_path"] = (
                            "retinal_evidence.png"
                        )
                    result["artifacts"] = {
                        name: name
                        for name in (
                            *artifact_paths,
                            "screening_report_pdf",
                            "screening_report_markdown",
                            "screening_report_json",
                        )
                    }
                    report_paths = write_screening_report(result, output_dir)
                    artifact_paths.update(
                        {
                            "screening_report_pdf": (
                                report_paths["pdf"],
                                "application/pdf",
                            ),
                            "screening_report_markdown": (
                                report_paths["markdown"],
                                "text/markdown; charset=utf-8",
                            ),
                            "screening_report_json": (
                                report_paths["json"],
                                "application/json",
                            ),
                        }
                    )
                    artifacts = {
                        name: _encode_file(path, media_type)
                        for name, (path, media_type) in artifact_paths.items()
                    }
                    return {
                        "result": result,
                        "artifacts": artifacts,
                        "upload": {
                            "format": image_format,
                            "width": original_size[0],
                            "height": original_size[1],
                        },
                    }
            except HTTPException:
                raise
            except Exception as error:
                LOGGER.exception("Image analysis failed.")
                raise HTTPException(
                    status_code=500,
                    detail="Image analysis failed. Check the service logs for details.",
                ) from error

        try:
            async with analysis_lock:
                return await run_in_threadpool(process_request)
        finally:
            await image.close()

    return application


app = create_app()

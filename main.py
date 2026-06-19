from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from gemini_service import (
    GeminiConfigurationError,
    GeminiServiceError,
    classify_image_with_gemini,
)
from model_service import InvalidImageError, load_model_bundle, predict_from_bytes

MODEL_FILENAME = "final_fake_real_swin_model.pth"
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str


class PredictionResponse(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    decision_source: Literal["final_classifier"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup and keep them in app state."""
    model_path = Path(__file__).resolve().parent / MODEL_FILENAME
    app.state.model_bundle = load_model_bundle(model_path)
    yield


app = FastAPI(
    title="Fake/Real Image Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict_image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    # Validate using real file bytes (handled in model_service via PIL),
    # because some mobile clients send unreliable content types.
    payload = await file.read()
    await file.close()

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        _, probability = predict_from_bytes(payload, app.state.model_bundle)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        # Hide internal details while returning a stable API error shape.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal error.",
        ) from exc

    # Always run Gemini for every valid image. Swin probability is retained as metadata.
    try:
        gemini_label = await classify_image_with_gemini(payload)
    except GeminiConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except GeminiServiceError as exc:
        logger.exception("Gemini final-classifier failure inside /predict_image.")
        # Fail safely with a clear backend error if final classifier cannot provide
        # a reliable binary label.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Final classification service is temporarily unavailable. Please retry.",
        ) from exc

    return PredictionResponse(
        label=gemini_label,
        probability=round(probability, 4),
        decision_source="final_classifier",
    )

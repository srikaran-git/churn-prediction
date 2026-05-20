# src/api/app.py

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.models.predictor import ChurnPredictor
from src.models.schemas import CustomerFeatures, ErrorResponse, PredictionResult
from src.utils.config_loader import load_config
from src.utils.exceptions import PredictionError
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

predictor = ChurnPredictor(model_path=config["paths"]["model_output"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting up — loading pipeline...")
    predictor.load()
    logger.info("Pipeline loaded. API ready.")
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn from Telco data.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Override FastAPI's default 422 format to match ErrorResponse."""
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="validation_error",
            detail=str(exc.errors()),
        ).model_dump(),
    )


@app.exception_handler(PredictionError)
async def prediction_error_handler(
    request: Request, exc: PredictionError
) -> JSONResponse:
    """Handle known prediction failures."""
    logger.error("PredictionError on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="prediction_error",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all — prevents raw tracebacks reaching the client."""
    logger.error(
        "Unhandled exception on %s: %s", request.url.path, exc, exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            detail="An unexpected error occurred.",
        ).model_dump(),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root() -> dict:
    logger.info("Root endpoint called")
    return {"message": "Churn Prediction API is running"}


@app.get(
    "/health",
    tags=["Health"],
    responses={503: {"model": ErrorResponse, "description": "Model not ready"}},
)
def health_check() -> dict:
    """Returns 200 only when the model is loaded and ready to serve."""
    logger.info("Health check called")
    if predictor.pipeline is None:
        logger.warning("Health check failed — pipeline not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )
    return {"status": "ok", "model_loaded": True}


@app.post(
    "/predict",
    response_model=PredictionResult,
    tags=["Predictions"],
    responses={
        422: {"model": ErrorResponse, "description": "Invalid input fields"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
)
def predict(features: CustomerFeatures) -> PredictionResult:
    logger.info("POST /predict called")
    return predictor.predict(features)
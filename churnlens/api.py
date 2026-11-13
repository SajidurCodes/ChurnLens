"""Local FastAPI application for the frozen ChurnLens model."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from churnlens.inference import PROJECT_ROOT, PredictionService
from churnlens.schemas import PredictionRequest


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model_version: str
    churn_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    threshold: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    flagged_for_review: bool


def create_app(root: Path = PROJECT_ROOT) -> FastAPI:
    root = Path(root)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.predictor = PredictionService(root)
        try:
            yield
        finally:
            del application.state.predictor

    application = FastAPI(
        title="ChurnLens",
        version="0.1.0",
        description=(
            "Offline churn benchmark served locally. Inputs must follow "
            "the source dataset's observation-window definitions. "
            "Probabilities are estimates; flags indicate review priority."
        ),
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        details = [
            {
                "loc": list(error["loc"]),
                "msg": error["msg"],
                "type": error["type"],
            }
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    @application.get("/", response_class=FileResponse, include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(
            root / "web" / "index.html",
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/health")
    def health(request: Request) -> dict:
        predictor = request.app.state.predictor
        return {
            "status": "ready",
            "model_version": predictor.model_version,
        }

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: PredictionRequest,
        request: Request,
    ) -> dict:
        return request.app.state.predictor.predict(payload.model_dump())

    return application


app = create_app()
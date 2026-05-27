"""FastAPI application for alloy property forward and backward prediction."""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api import backward_service, forward_service
from api.schemas import (
    BackwardCandidate,
    BackwardSearchRequest,
    BackwardSearchResponse,
    CompositionRequest,
    ForwardPredictResponse,
    HealthResponse,
    PairSearchRequest,
    PairSearchResponse,
)

_startup_errors: list[str] = []


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_cors_origins() -> list[str]:
    defaults = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    extra = os.getenv("CORS_ALLOW_ORIGINS", "")
    parsed = [o.strip() for o in extra.split(",") if o.strip()]
    # Keep order stable while removing duplicates.
    return list(dict.fromkeys(defaults + parsed))


def _preload_forward_on_startup() -> bool:
    # Render free instances can be sensitive to heavy cold-start work.
    # Default to lazy forward loading there unless explicitly overridden.
    on_render = _env_flag("RENDER", False)
    return _env_flag("PRELOAD_FORWARD_MODELS", default=not on_render)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_errors
    _startup_errors = []
    if _preload_forward_on_startup():
        try:
            forward_service.load_models()
        except Exception as e:
            _startup_errors.append(f"Forward models: {e}")
    try:
        backward_service.load_pool()
    except Exception as e:
        _startup_errors.append(f"Backward pool: {e}")
    yield


app = FastAPI(
    title="Alloy Property Prediction API",
    description="Forward prediction (composition -> properties) and backward search (targets -> candidate alloys).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_build_cors_origins(),
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    forward_loaded = forward_service.is_loaded()
    backward_loaded = backward_service.is_loaded()
    status = "ok" if forward_loaded else "degraded"
    if not forward_loaded and not backward_loaded:
        status = "error"
    forward_targets = None
    pool_rows = None
    if forward_loaded:
        forward_targets = len(forward_service.get_trainer().best_models)
    if backward_loaded:
        pool_rows = len(backward_service.get_pool())
    return HealthResponse(
        status=status,
        forward_models_loaded=forward_loaded,
        backward_pool_loaded=backward_loaded,
        forward_targets=forward_targets,
        pool_rows=pool_rows,
    )


@app.post("/api/forward/predict", response_model=ForwardPredictResponse)
def forward_predict(body: CompositionRequest):
    if not forward_service.is_loaded():
        try:
            forward_service.load_models()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Forward models not loaded. Ensure wrought_alloys_final.csv and "
                    "hyperparams_config.json are available on the server. "
                    f"Load error: {e}"
                ),
            ) from e
    try:
        trainer = forward_service.get_trainer()
        composition = body.to_dict()
        predictions = forward_service.predict_profile(composition, trainer)
        return ForwardPredictResponse(composition=composition, predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/backward/search", response_model=BackwardSearchResponse)
def backward_search(body: BackwardSearchRequest):
    if not backward_service.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Synthetic pool not loaded. Run 06_generate_synthetic_wrought.ipynb to create synthetic_wrought.csv.",
        )
    try:
        raw = backward_service.search_targets(body.targets, top_k=body.top_k)
        candidates = [
            BackwardCandidate(
                composition=c["composition"],
                properties=c["properties"],
                recipe=c["recipe"],
                total_error=c["total_error"],
            )
            for c in raw
        ]
        return BackwardSearchResponse(targets=body.targets, candidates=candidates)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/backward/pair-search", response_model=PairSearchResponse)
def backward_pair_search(body: PairSearchRequest):
    if not backward_service.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Synthetic pool not loaded. Run 06_generate_synthetic_wrought.ipynb to create synthetic_wrought.csv.",
        )
    try:
        result = backward_service.pair_search(
            body.property_a,
            body.value_a,
            body.property_b,
            body.value_b,
            tolerance=body.tolerance,
        )
        if result is None:
            return PairSearchResponse(
                found=False,
                message=f"No candidate found within +/-{body.tolerance * 100:.1f}% tolerance.",
            )
        return PairSearchResponse(found=True, candidate=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

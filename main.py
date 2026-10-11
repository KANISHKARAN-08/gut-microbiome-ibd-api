
import os
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

app = FastAPI(
    title="Gut Microbiome IBD Predictor API",
    version="1.0.0",
    description="Research prototype; not for clinical diagnosis."
)

# For initial integration testing. Restrict this to your
# actual frontend domain before public production use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL_REPO = "Kanish08/gut-microbiome-ibd-model"
MODEL_FILE = "ibd_random_forest_v1.joblib"

model_bundle = None
model_load_error = None


def load_model():
    global model_bundle, model_load_error

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is missing.")

    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        token=token
    )
    bundle = joblib.load(path)

    required = {"model", "features", "threshold"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise RuntimeError("Unexpected model bundle format.")

    model_bundle = bundle
    model_load_error = None


@app.on_event("startup")
def startup():
    global model_load_error
    try:
        load_model()
        print(
            f"Model loaded: {len(model_bundle['features'])} features; "
            f"threshold={model_bundle['threshold']}"
        )
    except Exception as exc:
        model_load_error = str(exc)
        print(f"Model loading failed: {exc}")


class PredictionInput(BaseModel):
    # Each key must match a selected genus name exactly.
    abundances: dict[str, float]


@app.get("/")
def root():
    return {
        "service": "Gut Microbiome IBD Predictor",
        "status": "online",
        "model_loaded": model_bundle is not None,
        "docs": "/docs",
        "disclaimer": "Research use only; not a clinical diagnosis."
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_bundle is not None
    }


@app.get("/features")
def features():
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable. Check server logs and HF_TOKEN."
        )
    return {
        "features": model_bundle["features"],
        "count": len(model_bundle["features"]),
        "threshold": float(model_bundle["threshold"])
    }


@app.post("/predict")
def predict(payload: PredictionInput):
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable. Check server logs and HF_TOKEN."
        )

    expected = model_bundle["features"]
    received = payload.abundances

    missing = [name for name in expected if name not in received]
    extra = [name for name in received if name not in expected]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": extra,
                "expected_count": len(expected)
            }
        )

    try:
        values = [float(received[name]) for name in expected]
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="All abundances must be numeric."
        )

    X = np.asarray(values, dtype=float).reshape(1, -1)

    if not np.isfinite(X).all() or (X < 0).any():
        raise HTTPException(
            status_code=422,
            detail="Abundances must be finite and non-negative."
        )

    model = model_bundle["model"]
    classes = list(model.classes_)
    if 1 not in classes:
        raise HTTPException(
            status_code=500,
            detail="Model does not contain the IBD-positive class."
        )

    ibd_index = classes.index(1)
    probability = float(model.predict_proba(X)[0][ibd_index])
    threshold = float(model_bundle["threshold"])

    return {
        "ibd_probability": probability,
        "ibd_probability_percent": round(probability * 100, 2),
        "threshold": threshold,
        "predicted_label": (
            "IBD-like model output"
            if probability >= threshold
            else "Healthy-like model output"
        ),
        "features_used": len(expected),
        "disclaimer": (
            "Research prediction only; not a clinical diagnosis."
        )
    }

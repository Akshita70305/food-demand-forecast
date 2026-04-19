from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import mlflow.sklearn
import numpy as np
import pandas as pd
from app.schemas import PredictRequest, PredictResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global model store ──
model_store = {}

FEATURES = [
    "year", "month", "day_of_year", "day_of_week",
    "is_harvest_season", "is_lean_season",
    "arrivals", "num_markets",
    "price_lag_7", "price_lag_14", "price_lag_30",
    "arrivals_lag_7",
    "price_roll_7", "price_roll_30", "arrivals_roll_7"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: load model once ──
    logger.info("Loading model from MLflow registry...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    try:
        model = mlflow.sklearn.load_model("models:/cumin-forecaster/1")
        model_store["model"]   = model
        model_store["name"]    = "cumin-forecaster"
        model_store["version"] = "1"
        logger.info("Model loaded successfully ✅")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    # ── Shutdown ──
    model_store.clear()
    logger.info("Model unloaded.")

app = FastAPI(
    title="Cumin Price Forecasting API",
    description="Predicts cumin (Jeera) modal price in Rs/Quintal using MLflow registered model.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in model_store
    }

@app.get("/model-info")
def model_info():
    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name":    model_store["name"],
        "model_version": model_store["version"],
        "features":      FEATURES,
        "target":        "modal_price (Rs/Quintal)"
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        # Build feature dataframe in correct order
        data = pd.DataFrame([{f: getattr(request, f) for f in FEATURES}])
        prediction = model_store["model"].predict(data)[0]
        # Clip to realistic range
        prediction = float(np.clip(prediction, 1000, 100000))
        return PredictResponse(
            predicted_price=round(prediction, 2),
            unit="Rs/Quintal",
            model_name=model_store["name"],
            model_version=model_store["version"]
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
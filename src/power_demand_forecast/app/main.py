from __future__ import annotations

from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/model.joblib")

app = FastAPI(title="Power Demand Forecast TR", version="0.1.0")


class PredictRequest(BaseModel):
    timestamp: datetime
    temperature_c: float = Field(..., description="Outside temperature in Celsius")


class PredictResponse(BaseModel):
    predicted_demand_mw: float


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("Model yok. Önce eğit: python -m src.power_demand_forecast.models.train")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    ts = req.timestamp
    row = {
        "hour": ts.hour,
        "dayofweek": ts.weekday(),
        "is_weekend": 1 if ts.weekday() >= 5 else 0,
        "temperature_c": req.temperature_c,
    }
    X = pd.DataFrame([row])
    model = load_model()
    y = float(model.predict(X)[0])
    return PredictResponse(predicted_demand_mw=y)

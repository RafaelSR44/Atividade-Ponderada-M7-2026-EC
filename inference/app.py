"""Backend de inferência: carrega o artefato treinado e serve predições via HTTP."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from features import WINDOW, predict_path

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/btc_model.joblib"))
DATA_PATH = Path(os.getenv("DATA_PATH", "data/btc_usd_daily.csv"))
MAX_HORIZON = 7

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["artifact"] = joblib.load(MODEL_PATH)
    print(f"[api] modelo carregado de {MODEL_PATH} "
          f"(treinado em {state['artifact']['trained_at']})", flush=True)
    yield
    state.clear()


app = FastAPI(title="BTC Forecast API", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    closes: list[float] = Field(..., min_length=WINDOW,
                                description=f"últimos fechamentos em USD (mín. {WINDOW}), do mais antigo ao mais recente")
    horizon: int = Field(1, ge=1, le=MAX_HORIZON, description="quantos dias à frente prever")


def _forecast(closes: list[float], horizon: int, last_date: str | None = None) -> dict:
    try:
        path = predict_path(state["artifact"]["model"], closes, horizon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if last_date:
        dates = pd.date_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=horizon)
        for p, d in zip(path, dates):
            p["date"] = str(d.date())
    return {
        "last_close": closes[-1],
        "last_date": last_date,
        "horizon": horizon,
        "predictions": [{k: round(v, 6) if k == "predicted_log_return" else (round(v, 2) if isinstance(v, float) else v)
                         for k, v in p.items()} for p in path],
    }


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "artifact" in state}


@app.get("/model/info")
def model_info():
    a = state["artifact"]
    return {k: a[k] for k in ("window", "feature_names", "target", "trained_at",
                              "data_last_date", "sklearn_version", "metrics")}


@app.post("/predict")
def predict(req: PredictRequest):
    return _forecast(req.closes, req.horizon)


@app.get("/predict/latest")
def predict_latest(horizon: int = Query(1, ge=1, le=MAX_HORIZON)):
    """Usa os últimos fechamentos do CSV montado no container."""
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail=f"CSV não encontrado em {DATA_PATH}")
    df = pd.read_csv(DATA_PATH).tail(WINDOW)
    return _forecast(df["Close"].tolist(), horizon, last_date=df["Date"].iloc[-1])

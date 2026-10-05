"""Treina o modelo de previsão do fechamento do BTC no dia seguinte.

Entrada : data/btc_usd_daily.csv
Saída   : model/btc_model.joblib  (artefato carregado pela API)
          model/metrics.json      (métricas de avaliação)
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

from features import FEATURE_NAMES, WINDOW, features_from_window

DATA_PATH = Path(os.getenv("DATA_PATH", "data/btc_usd_daily.csv"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
TEST_FRACTION = 0.2


def build_dataset(closes: np.ndarray):
    X, y, last_close, next_close = [], [], [], []
    for t in range(WINDOW - 1, len(closes) - 1):
        X.append(features_from_window(closes[t - WINDOW + 1 : t + 1]))
        y.append(np.log(closes[t + 1] / closes[t]))
        last_close.append(closes[t])
        next_close.append(closes[t + 1])
    return np.array(X), np.array(y), np.array(last_close), np.array(next_close)


def new_model():
    return GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=42,
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date")
    closes = df["Close"].to_numpy(dtype=float)
    print(f"[dados] {len(df)} dias | {df.Date.iloc[0].date()} -> {df.Date.iloc[-1].date()}")

    X, y, last_close, next_close = build_dataset(closes)
    split = int(len(X) * (1 - TEST_FRACTION))
    print(f"[dados] {len(X)} amostras | treino={split} teste={len(X) - split} (split cronológico)")

    # 1) Avaliação: treina só no passado, testa no período mais recente
    model = new_model().fit(X[:split], y[:split])
    pred_close = last_close[split:] * np.exp(model.predict(X[split:]))
    real = next_close[split:]
    naive = last_close[split:]  # baseline: amanhã = hoje

    def mae(p):
        return float(np.mean(np.abs(p - real)))

    def mape(p):
        return float(np.mean(np.abs(p - real) / real) * 100)

    real_dir = np.sign(real - naive)
    pred_dir = np.sign(pred_close - naive)
    metrics = {
        "test_period": [str(df.Date.iloc[WINDOW + split].date()), str(df.Date.iloc[-1].date())],
        "n_train": int(split),
        "n_test": int(len(X) - split),
        "model_mae_usd": round(mae(pred_close), 2),
        "model_mape_pct": round(mape(pred_close), 3),
        "naive_mae_usd": round(mae(naive), 2),
        "naive_mape_pct": round(mape(naive), 3),
        "direction_accuracy_pct": round(float(np.mean(real_dir == pred_dir) * 100), 2),
    }
    print("[avaliação]", json.dumps(metrics, indent=2, ensure_ascii=False))

    # 2) Modelo final: re-treina com todo o histórico para servir na API
    final_model = new_model().fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "window": WINDOW,
        "feature_names": FEATURE_NAMES,
        "target": "log(close[t+1] / close[t])",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_last_date": str(df.Date.iloc[-1].date()),
        "sklearn_version": sklearn.__version__,
        "metrics": metrics,
    }
    joblib.dump(artifact, MODEL_DIR / "btc_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    size_kb = (MODEL_DIR / "btc_model.joblib").stat().st_size / 1024
    print(f"[artefato] salvo em {MODEL_DIR / 'btc_model.joblib'} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()

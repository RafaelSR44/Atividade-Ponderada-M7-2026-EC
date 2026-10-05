"""Aplicação cliente: lê o histórico, pede uma previsão ao backend e mostra o resultado."""
import csv
import os
import sys

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_PATH = os.getenv("DATA_PATH", "data/btc_usd_daily.csv")
HORIZON = int(os.getenv("HORIZON", "3"))
WINDOW = 14


def main() -> None:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    print(f"[client] GET /health -> {health}")
    if not health.get("model_loaded"):
        sys.exit("[client] backend sem modelo carregado")

    with open(DATA_PATH, newline="") as f:
        rows = list(csv.DictReader(f))[-WINDOW:]
    closes = [float(r["Close"]) for r in rows]
    print(f"[client] enviando {len(closes)} fechamentos ({rows[0]['Date']} -> {rows[-1]['Date']}), horizon={HORIZON}")

    resp = requests.post(f"{API_URL}/predict", json={"closes": closes, "horizon": HORIZON}, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    print(f"[client] POST /predict -> HTTP {resp.status_code}")
    print(f"[client] último fechamento real: US$ {body['last_close']:,.2f} ({rows[-1]['Date']})")
    for i, p in enumerate(body["predictions"], start=1):
        var = (p["predicted_close"] / body["last_close"] - 1) * 100
        print(f"[client]   D+{i}: US$ {p['predicted_close']:,.2f}  ({var:+.2f}% vs último)")


if __name__ == "__main__":
    main()

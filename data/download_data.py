"""Baixa o histórico diário do BTC-USD e salva em data/btc_usd_daily.csv.

Fonte principal: Yahoo Finance (via yfinance).
Uso:  python data/download_data.py [--start 2015-01-01]
"""
import argparse
from datetime import date
from pathlib import Path

import yfinance as yf

OUT = Path(__file__).resolve().parent / "btc_usd_daily.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=date.today().isoformat(), help="exclusivo (não inclui o dia)")
    args = parser.parse_args()

    df = yf.download("BTC-USD", start=args.start, end=args.end, interval="1d",
                     auto_adjust=False, progress=False)
    if df.empty:
        raise SystemExit("Nenhum dado retornado pelo Yahoo Finance.")

    # yfinance devolve colunas MultiIndex (campo, ticker) -> achata para só o campo
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"
    df.index = df.index.strftime("%Y-%m-%d")
    df.round(2).to_csv(OUT)

    print(f"Salvo {len(df)} linhas em {OUT}")
    print(f"Período: {df.index[0]} -> {df.index[-1]}")
    print(df.tail())


if __name__ == "__main__":
    main()

"""Engenharia de features compartilhada entre treino e inferência.

O mesmo arquivo é copiado para os dois containers, garantindo que o modelo
receba exatamente as mesmas features nos dois lados.
"""
import numpy as np

WINDOW = 14  # quantos fechamentos são necessários para montar 1 vetor de features
N_LAGS = 7

FEATURE_NAMES = (
    [f"ret_lag_{i}" for i in range(1, N_LAGS + 1)]
    + ["close_over_ma7", "close_over_ma14", "vol_7d"]
)


def features_from_window(closes) -> np.ndarray:
    """Recebe os últimos WINDOW fechamentos (mais antigo -> mais recente)
    e devolve o vetor de features para prever o retorno do dia seguinte."""
    c = np.asarray(closes, dtype=float)
    if c.shape != (WINDOW,):
        raise ValueError(f"esperado {WINDOW} fechamentos, recebido {c.shape}")
    if np.any(c <= 0):
        raise ValueError("fechamentos precisam ser positivos")

    log_ret = np.diff(np.log(c))          # 13 retornos diários
    lags = log_ret[::-1][:N_LAGS]         # ret_lag_1 = retorno de ontem p/ hoje
    last = c[-1]
    return np.concatenate([
        lags,
        [last / c[-7:].mean() - 1, last / c.mean() - 1, log_ret[-7:].std()],
    ])


def predict_path(model, closes, horizon: int = 1):
    """Previsão recursiva: prevê o dia seguinte, junta na janela e repete."""
    window = list(map(float, closes[-WINDOW:]))
    out = []
    for _ in range(horizon):
        x = features_from_window(window).reshape(1, -1)
        r = float(model.predict(x)[0])
        nxt = window[-1] * float(np.exp(r))
        out.append({"predicted_log_return": r, "predicted_close": nxt})
        window = window[1:] + [nxt]
    return out

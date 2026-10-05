# Container de TREINO: lê o CSV, treina e grava o artefato em /app/model
# Build a partir da raiz do repo (contexto ".") para conseguir copiar common/
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    DATA_PATH=/app/data/btc_usd_daily.csv \
    MODEL_DIR=/app/model

COPY training/requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

COPY common/features.py training/train.py ./

CMD ["python", "train.py"]

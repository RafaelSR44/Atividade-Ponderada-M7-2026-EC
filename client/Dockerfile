# Aplicação CLIENTE: script que consome a API de inferência
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir --root-user-action=ignore requests==2.32.3
COPY client/client.py .

CMD ["python", "client.py"]

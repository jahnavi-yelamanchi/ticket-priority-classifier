FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRIAGE_ARTIFACTS_PATH=/models

WORKDIR /app

COPY requirements.runtime.txt .
RUN pip install --no-cache-dir -r requirements.runtime.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

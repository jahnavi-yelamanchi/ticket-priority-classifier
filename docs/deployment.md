# Deploying Triage

## Modal (production)

Authenticate once with `modal setup`, then run the pipeline in order:

```bash
modal run modal_app/train.py
modal run modal_app/export.py --run-id <run-id-printed-by-training>
modal deploy modal_app/service.py
```

Training and export share the `triage-model-artifacts` Modal Volume. Export writes `production.json`, so the deployed service always loads the promoted INT8 ONNX artifact rather than an arbitrary checkpoint.

The Modal deployment exposes:

```text
GET  /              one-page interactive demo
GET  /health        model readiness
POST /predict       priority prediction
GET  /metrics       recorded evaluation and optimization metrics
GET  /docs          OpenAPI documentation
```

## Local API

The service expects a promoted artifact at `$TRIAGE_ARTIFACTS_PATH` (default `/models`).

```bash
export TRIAGE_ARTIFACTS_PATH="$PWD/artifacts"
uvicorn app.main:app --reload
```

## Docker

Build the API image and mount the directory containing `production.json` and `runs/`:

```bash
docker build -t triage-api .
docker run --rm -p 8000:8000 -v "$PWD/artifacts:/models:ro" triage-api
```

Open `http://localhost:8000` to use the demo and `http://localhost:8000/docs` for the API contract.

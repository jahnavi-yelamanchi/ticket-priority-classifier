# Triage

Triage is a deliberately small support-ticket priority classifier. Paste one ticket; receive one of four operational priorities: `low`, `medium`, `high`, or `urgent`.

The project is designed to demonstrate the full lifecycle of a compact NLP model without turning into a dashboard: supervised fine-tuning, ONNX export, INT8 CPU inference, a typed API, and a one-page public demo.

## Status

The classifier is trained, exported to ONNX, dynamically quantized to INT8, and deployed on Modal. The public demo and API use the promoted INT8 artifact from the Modal Volume.

Live demo: [Triage on Modal](https://jahnavi-yelamanchi--triage-ticket-priority-classifier-fa-e7151d.modal.run)

## Target architecture

```text
public ticket text
       |
       v
Modal FastAPI app  --->  ONNX Runtime INT8 classifier
       |                         ^
       v                         |
one-page web demo          Modal Volume artifacts
                                 ^
                                 |
                    Modal GPU fine-tuning job
```

## Repository layout

```text
app/        FastAPI service and static demo assets
modal_app/  Modal training, export, and deployment entry points
data/       Dataset metadata only; raw and processed data stay untracked
docs/       Design and implementation notes
tests/      Unit and API tests
```

## Intended endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /predict` | Classify a support ticket and return priority, confidence, and probabilities. |
| `GET /health` | Report service and model readiness. |
| `GET /metrics` | Return recorded evaluation and inference metrics. |

## Design direction

The demo is an original dark editorial interface informed by the supplied Framer reference: a near-black canvas, assertive white display type, charcoal product panels, white pill actions, a blue keyboard-focus state, and a single violet spotlight card. It calls the deployed `/predict` endpoint directly and fills its result/metric panels only with live API data. It uses no Framer branding, invented benchmark claims, or LLM-generated rationales. See [the design brief](docs/design-brief.md).

## Run the pipeline

```bash
# Run the Modal GPU fine-tuning job
modal run modal_app/train.py

# Export its FP32 checkpoint, quantize to INT8, and select it for serving
modal run modal_app/export.py --run-id <training-run-id>

# Deploy the API after a production model has been selected
modal deploy modal_app/service.py
```

The training command creates a timestamped FP32 checkpoint in the `triage-model-artifacts` Modal Volume. Export records both FP32 and INT8 artifact size plus P50/P95 CPU latency, then stores the selected production run in the same Volume. The Modal deployment serves `/health`, `/predict`, `/metrics`, and interactive API docs at `/docs`.

## Run and deploy

```bash
# Local API (requires a promoted artifact at TRIAGE_ARTIFACTS_PATH)
export TRIAGE_ARTIFACTS_PATH="$PWD/artifacts"
uvicorn app.main:app --reload

# Docker API (mount the promoted model artifact directory)
docker build -t triage-api .
docker run --rm -p 8000:8000 -v "$PWD/artifacts:/models:ro" triage-api
```

For the complete Modal training → export → deploy sequence, see [deployment instructions](docs/deployment.md).

## Dataset and measured results

Training uses the public Hugging Face dataset [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets). Source priority variants are normalized into the stable API vocabulary: `very_low` and `low` become `low`; `normal` and `medium` become `medium`; `high` remains `high`; `very_high`, `critical`, and `urgent` become `urgent`.

The reproducible split is stratified 80/10/10 with seed `42`. The completed run (`20260714T010514Z`) trained on 49,411 tickets, validated on 6,176, and evaluated on 6,178 held-out tickets.

| Priority | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Low | 11,638 | 1,455 | 1,455 |
| Medium | 18,702 | 2,338 | 2,338 |
| High | 17,540 | 2,192 | 2,193 |
| Urgent | 1,531 | 191 | 192 |

The held-out test Macro F1 is **0.533** (accuracy **0.550**). This result is reported as measured, including the dataset's substantial urgent-class imbalance.

| Artifact | Model size | P50 latency | P95 latency |
| --- | ---: | ---: | ---: |
| FP32 ONNX | 255.5 MB | 32.750 ms | 92.882 ms |
| INT8 ONNX | 64.3 MB | 17.154 ms | 51.626 ms |

The benchmarks were recorded with ONNX Runtime CPU inference on the Modal export worker. INT8 reduced the stored model from 267,961,451 to 67,387,526 bytes while also lowering measured latency.

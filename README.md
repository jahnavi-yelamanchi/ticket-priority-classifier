# Triage

Triage is a deliberately small support-ticket priority classifier. Paste one ticket; receive one of four operational priorities: `low`, `medium`, `high`, or `urgent`.

The project is designed to demonstrate the full lifecycle of a compact NLP model without turning into a dashboard: supervised fine-tuning, ONNX export, INT8 CPU inference, a typed API, and a one-page public demo.

## Status

Repository scaffolding is in place. The next implementation milestones add the dataset pipeline, Modal GPU training, evaluation, optimized artifact, FastAPI service, and demo UI.

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

## Planned commands

```bash
# Run the Modal GPU fine-tuning job
modal run modal_app/train.py

# Export its FP32 checkpoint, quantize to INT8, and select it for serving
modal run modal_app/export.py --run-id 20260713T000000Z

# Deploy the API after a production model has been selected
modal deploy modal_app/service.py
```

The training command creates a timestamped FP32 checkpoint in the `triage-model-artifacts` Modal Volume. Export records both FP32 and INT8 artifact size plus P50/P95 CPU latency, then stores the selected production run in the same Volume. The Modal deployment serves `/health`, `/predict`, `/metrics`, and interactive API docs at `/docs`.

## Dataset and metrics

Training uses the public Hugging Face dataset [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets). Its source priority values are normalized into the stable API vocabulary: `low`, `medium`, `high`, and `urgent` (`critical` maps to `urgent`).

The reproducible split is stratified 80/10/10 with seed `42`. The training run will record source metadata, class counts, Macro F1, per-class confusion matrix, ONNX model size, and P50/P95 CPU latency. Results will be reported only after they are measured.

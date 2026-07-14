"""Export a trained Triage run to ONNX, quantize it, and record CPU benchmarks.

Run ``modal run modal_app/export.py --run-id <training-run-id>`` after the
training job completes. The optimized run becomes the Volume's production model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median

import modal

from app.benchmarking import percentile
from modal_app.train import APP_NAME, ARTIFACTS_PATH, VOLUME_NAME

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
export_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26,<3.0",
        "onnx>=1.16,<2.0",
        "onnxscript>=0.1,<1.0",
        "onnxruntime>=1.19,<2.0",
        "torch>=2.4,<3.0",
        "transformers>=4.45,<5.0",
    )
    .add_local_python_source("app", "modal_app")
)


def _benchmark(session, inputs: dict[str, object]) -> dict[str, float]:
    for _ in range(25):
        session.run(None, inputs)
    timings: list[float] = []
    for _ in range(150):
        started = time.perf_counter()
        session.run(None, inputs)
        timings.append((time.perf_counter() - started) * 1_000)
    return {"p50_ms": round(float(median(timings)), 3), "p95_ms": round(percentile(timings, 0.95), 3)}


@app.function(image=export_image, timeout=30 * 60, volumes={ARTIFACTS_PATH: model_volume})
def export_and_quantize(run_id: str) -> dict[str, object]:
    """Create a dynamically quantized ONNX model and production pointer."""

    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    run_path = Path(ARTIFACTS_PATH) / "runs" / run_id
    source_path = run_path / "fp32"
    if not source_path.exists():
        raise FileNotFoundError(f"No FP32 checkpoint found for training run {run_id!r}.")

    tokenizer = AutoTokenizer.from_pretrained(source_path)
    model = AutoModelForSequenceClassification.from_pretrained(source_path).eval()
    sample = tokenizer(
        "The production checkout is failing for every customer.",
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    class LogitsOnly(torch.nn.Module):
        def __init__(self, wrapped_model):
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, input_ids, attention_mask):
            return self.wrapped_model(input_ids=input_ids, attention_mask=attention_mask).logits

    fp32_onnx = run_path / "model-fp32.onnx"
    torch.onnx.export(
        LogitsOnly(model),
        (sample["input_ids"], sample["attention_mask"]),
        fp32_onnx,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    int8_onnx = run_path / "model-int8.onnx"
    quantize_dynamic(fp32_onnx, int8_onnx, weight_type=QuantType.QInt8)

    ort_inputs = {name: sample[name].cpu().numpy() for name in ("input_ids", "attention_mask")}
    fp32_session = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(int8_onnx), providers=["CPUExecutionProvider"])
    optimization = {
        "fp32": {"size_bytes": fp32_onnx.stat().st_size, **_benchmark(fp32_session, ort_inputs)},
        "int8": {"size_bytes": int8_onnx.stat().st_size, **_benchmark(int8_session, ort_inputs)},
    }

    metrics_path = run_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["optimization"] = optimization
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (Path(ARTIFACTS_PATH) / "production.json").write_text(
        json.dumps({"run_id": run_id, "model": "model-int8.onnx"}, indent=2), encoding="utf-8"
    )
    model_volume.commit()
    return {"run_id": run_id, "optimization": optimization}


@app.local_entrypoint()
def main(run_id: str) -> None:
    """Submit export and quantization for the named training run."""

    print(json.dumps(export_and_quantize.remote(run_id), indent=2))

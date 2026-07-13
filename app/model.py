"""ONNX Runtime model loader for the promoted Modal Volume artifact."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.prediction import prediction_payload

DEFAULT_ARTIFACTS_PATH = "/models"


class ModelNotReadyError(RuntimeError):
    """Raised until a training run has been exported and promoted."""


class TriageModel:
    """Lazy singleton around one promoted tokenizer and INT8 ONNX session."""

    def __init__(self, artifacts_path: str | Path | None = None) -> None:
        self.artifacts_path = Path(artifacts_path or os.getenv("TRIAGE_ARTIFACTS_PATH", DEFAULT_ARTIFACTS_PATH))
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._labels: list[str] | None = None
        self._metrics: dict[str, object] | None = None
        self._run_id: str | None = None

    @property
    def ready(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        """Load the production pointer, tokenizer, metrics, and optimized model once."""

        if self.ready:
            return
        pointer_path = self.artifacts_path / "production.json"
        if not pointer_path.exists():
            raise ModelNotReadyError("No promoted model exists yet. Run training and export first.")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        run_id = pointer.get("run_id")
        model_name = pointer.get("model")
        if not isinstance(run_id, str) or not isinstance(model_name, str):
            raise ModelNotReadyError("The production model pointer is malformed.")

        run_path = self.artifacts_path / "runs" / run_id
        model_path = run_path / model_name
        tokenizer_path = run_path / "fp32"
        labels_path = run_path / "labels.json"
        metrics_path = run_path / "metrics.json"
        required_paths = (model_path, tokenizer_path, labels_path, metrics_path)
        if not all(path.exists() for path in required_paths):
            raise ModelNotReadyError(f"Production artifact {run_id!r} is incomplete.")

        import onnxruntime as ort
        from transformers import AutoTokenizer

        labels_by_id = json.loads(labels_path.read_text(encoding="utf-8"))
        try:
            labels = [labels_by_id[str(index)] for index in range(len(labels_by_id))]
        except KeyError as error:
            raise ModelNotReadyError("The production label mapping is malformed.") from error
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._labels = labels
        self._metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self._run_id = run_id

    def predict(self, ticket: str) -> dict[str, object]:
        """Run one text through the promoted ONNX model."""

        self.load()
        assert self._session is not None
        assert self._tokenizer is not None
        assert self._labels is not None
        encoded = self._tokenizer(ticket, truncation=True, max_length=256, return_tensors="np")
        accepted_inputs = {input_metadata.name for input_metadata in self._session.get_inputs()}
        session_inputs = {name: value for name, value in encoded.items() if name in accepted_inputs}
        logits = self._session.run(None, session_inputs)[0][0].tolist()
        return prediction_payload(self._labels, logits)

    def health(self) -> dict[str, object]:
        """Return a safe readiness payload without loading a model on status requests."""

        if self.ready:
            return {"status": "ready", "run_id": self._run_id}
        return {"status": "not_ready"}

    def metrics(self) -> dict[str, object]:
        """Return measured training and optimization metadata for the promoted run."""

        self.load()
        assert self._metrics is not None
        return self._metrics

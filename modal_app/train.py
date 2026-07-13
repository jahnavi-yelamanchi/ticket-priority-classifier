"""Modal GPU fine-tuning entry point for the Triage classifier.

Run with ``modal run modal_app/train.py``. The job writes a versioned FP32
checkpoint and evaluation report to the shared Modal Volume. ONNX export and
quantization are deliberately a separate follow-up command.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import modal

from modal_app.data import LABELS, class_counts, load_source_records, stratified_split

APP_NAME = "triage-ticket-priority-classifier"
VOLUME_NAME = "triage-model-artifacts"
ARTIFACTS_PATH = "/models"
BASE_MODEL = "distilbert-base-uncased"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate>=1.0,<2.0",
        "datasets>=3.0,<4.0",
        "scikit-learn>=1.5,<2.0",
        "torch>=2.4,<3.0",
        "transformers>=4.45,<5.0",
    )
)


def _dataset_from_records(records, tokenizer, label_to_id):
    """Create a tokenized Hugging Face dataset without exposing source columns."""

    from datasets import Dataset

    dataset = Dataset.from_dict(
        {
            "text": [record.text for record in records],
            "label": [label_to_id[record.label] for record in records],
        }
    )
    return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=256), batched=True)


@app.function(
    image=training_image,
    gpu="T4",
    timeout=60 * 60,
    volumes={ARTIFACTS_PATH: model_volume},
)
def train() -> dict[str, object]:
    """Fine-tune DistilBERT and persist one reproducible FP32 training run."""

    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    records = load_source_records()
    splits = stratified_split(records)
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_dataset = _dataset_from_records(splits["train"], tokenizer, label_to_id)
    validation_dataset = _dataset_from_records(splits["validation"], tokenizer, label_to_id)
    test_dataset = _dataset_from_records(splits["test"], tokenizer, label_to_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    def compute_metrics(prediction):
        predictions = np.argmax(prediction.predictions, axis=1)
        references = prediction.label_ids
        return {
            "macro_f1": float(f1_score(references, predictions, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(references, predictions)),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/triage-training",
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=25,
            report_to="none",
            seed=42,
        ),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    prediction = trainer.predict(test_dataset)
    predicted_ids = np.argmax(prediction.predictions, axis=1)
    reference_ids = prediction.label_ids

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_path = Path(ARTIFACTS_PATH) / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=False)
    trainer.save_model(str(run_path / "fp32"))
    tokenizer.save_pretrained(str(run_path / "fp32"))

    metrics = {
        "run_id": run_id,
        "trained_at": datetime.now(UTC).isoformat(),
        "base_model": BASE_MODEL,
        "labels": list(LABELS),
        "split_seed": 42,
        "split_counts": {name: class_counts(partition) for name, partition in splits.items()},
        "test": {
            "macro_f1": float(f1_score(reference_ids, predicted_ids, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(reference_ids, predicted_ids)),
            "confusion_matrix": confusion_matrix(reference_ids, predicted_ids, labels=range(len(LABELS))).tolist(),
            "classification_report": classification_report(
                reference_ids,
                predicted_ids,
                labels=range(len(LABELS)),
                target_names=list(LABELS),
                output_dict=True,
                zero_division=0,
            ),
        },
    }
    (run_path / "labels.json").write_text(json.dumps(id_to_label, indent=2), encoding="utf-8")
    (run_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model_volume.commit()
    return metrics


@app.local_entrypoint()
def main() -> None:
    """Submit the GPU training job from the local Modal CLI."""

    metrics = train.remote()
    print(json.dumps(metrics, indent=2))

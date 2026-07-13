"""Prediction math that stays independent from serving and model runtimes."""

from __future__ import annotations

from math import exp
from typing import Sequence


def softmax(logits: Sequence[float]) -> list[float]:
    """Convert model logits to stable probabilities."""

    if not logits:
        raise ValueError("Model output cannot be empty.")
    maximum = max(logits)
    values = [exp(logit - maximum) for logit in logits]
    total = sum(values)
    return [value / total for value in values]


def prediction_payload(labels: Sequence[str], logits: Sequence[float]) -> dict[str, object]:
    """Create the public response shape from ordered labels and raw logits."""

    if len(labels) != len(logits):
        raise ValueError("Model output count does not match the label vocabulary.")
    probabilities = softmax(logits)
    winning_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "priority": labels[winning_index],
        "confidence": probabilities[winning_index],
        "probabilities": dict(zip(labels, probabilities, strict=True)),
    }

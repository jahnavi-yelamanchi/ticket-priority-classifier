"""Dataset loading and normalization for Triage model training.

The source dataset uses customer-support records with a priority field. This module
keeps source-specific column names at the boundary and exposes the four stable API
labels used by the model and service.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import Any, Iterable, Mapping, Sequence

LABELS = ("low", "medium", "high", "urgent")
DATASET_ID = "Tobi-Bueck/customer-support-tickets"
DATASET_SPLIT = "train"
SPLIT_SEED = 42

_TEXT_COLUMNS = ("ticket_description", "ticket", "description", "body", "text")
_PRIORITY_COLUMNS = ("ticket_priority", "priority", "severity")
_PRIORITY_ALIASES = {
    "very_low": "low",
    "low": "low",
    "medium": "medium",
    "normal": "medium",
    "high": "high",
    "very_high": "urgent",
    "urgent": "urgent",
    "critical": "urgent",
}


@dataclass(frozen=True)
class TicketRecord:
    """One normalized training example."""

    text: str
    label: str


def _normalized_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_priority(value: Any) -> str:
    """Map source labels such as ``Critical`` to the public label vocabulary."""

    normalized = _normalized_name(str(value))
    try:
        return _PRIORITY_ALIASES[normalized]
    except KeyError as error:
        expected = ", ".join(sorted(_PRIORITY_ALIASES))
        raise ValueError(f"Unsupported ticket priority {value!r}; expected one of {expected}.") from error


def find_source_columns(column_names: Iterable[str]) -> tuple[str, str]:
    """Return source text and priority columns, accepting common naming variants."""

    names = {_normalized_name(name): name for name in column_names}
    text_column = next((names[name] for name in _TEXT_COLUMNS if name in names), None)
    priority_column = next((names[name] for name in _PRIORITY_COLUMNS if name in names), None)
    if not text_column or not priority_column:
        raise ValueError(
            "Dataset must include a ticket text column "
            f"({_TEXT_COLUMNS}) and priority column ({_PRIORITY_COLUMNS}); received {sorted(column_names)}."
        )
    return text_column, priority_column


def normalize_records(
    records: Iterable[Mapping[str, Any]], text_column: str, priority_column: str
) -> list[TicketRecord]:
    """Remove blank tickets and map all supported priorities to stable labels."""

    normalized: list[TicketRecord] = []
    for record in records:
        text = str(record.get(text_column, "")).strip()
        if not text:
            continue
        normalized.append(TicketRecord(text=text, label=normalize_priority(record[priority_column])))
    if not normalized:
        raise ValueError("No non-empty support tickets were found in the dataset.")
    return normalized


def stratified_split(
    records: Sequence[TicketRecord], seed: int = SPLIT_SEED
) -> dict[str, list[TicketRecord]]:
    """Create deterministic 80/10/10 train/validation/test partitions."""

    counts = Counter(record.label for record in records)
    too_small = [label for label, count in counts.items() if count < 3]
    if too_small:
        raise ValueError(f"Each class needs at least 3 records for stratified splitting; too small: {too_small}.")

    grouped: dict[str, list[TicketRecord]] = {label: [] for label in LABELS}
    for record in records:
        grouped[record.label].append(record)

    random = Random(seed)
    train: list[TicketRecord] = []
    validation: list[TicketRecord] = []
    test: list[TicketRecord] = []
    for label in LABELS:
        group = grouped[label]
        random.shuffle(group)
        holdout_size = max(2, round(len(group) * 0.2))
        validation_size = holdout_size // 2
        test_size = holdout_size - validation_size
        train.extend(group[holdout_size:])
        validation.extend(group[:validation_size])
        test.extend(group[validation_size : validation_size + test_size])

    random.shuffle(train)
    random.shuffle(validation)
    random.shuffle(test)
    return {"train": train, "validation": validation, "test": test}


def class_counts(records: Sequence[TicketRecord]) -> dict[str, int]:
    """Return an ordered, zero-filled label distribution for documentation and metrics."""

    counts = Counter(record.label for record in records)
    return {label: counts[label] for label in LABELS}


def load_source_records(dataset_id: str = DATASET_ID, split: str = DATASET_SPLIT) -> list[TicketRecord]:
    """Load the public source dataset through Hugging Face Datasets.

    Keeping this import inside the function lets unit tests exercise normalization
    without installing the full ML stack.
    """

    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split)
    text_column, priority_column = find_source_columns(dataset.column_names)
    return normalize_records(dataset, text_column, priority_column)

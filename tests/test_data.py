import unittest

from modal_app.data import (
    LABELS,
    TicketRecord,
    class_counts,
    find_source_columns,
    normalize_priority,
    normalize_records,
    stratified_split,
)


class DatasetNormalizationTests(unittest.TestCase):
    def test_normalize_priority_maps_source_variants(self) -> None:
        self.assertEqual(normalize_priority("Critical"), "urgent")
        self.assertEqual(normalize_priority("normal"), "medium")
        self.assertEqual(normalize_priority("LOW"), "low")

    def test_find_source_columns_accepts_human_readable_names(self) -> None:
        self.assertEqual(
            find_source_columns(["Ticket Description", "Ticket Priority"]),
            ("Ticket Description", "Ticket Priority"),
        )

    def test_normalize_records_discards_blank_ticket_text(self) -> None:
        records = normalize_records(
            [
                {"body": "  Login page returns 500  ", "priority": "High"},
                {"body": " ", "priority": "Low"},
            ],
            "body",
            "priority",
        )
        self.assertEqual(records, [TicketRecord(text="Login page returns 500", label="high")])

    def test_stratified_split_is_reproducible_and_preserves_labels(self) -> None:
        records = [
            TicketRecord(text=f"{label}-{index}", label=label)
            for label in LABELS
            for index in range(10)
        ]
        first = stratified_split(records, seed=7)
        second = stratified_split(records, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(sum(len(partition) for partition in first.values()), len(records))
        self.assertTrue(
            all(class_counts(partition)[label] > 0 for partition in first.values() for label in LABELS)
        )

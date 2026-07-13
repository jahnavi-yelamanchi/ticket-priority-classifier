import unittest

from app.prediction import prediction_payload, softmax


class PredictionMathTests(unittest.TestCase):
    def test_softmax_returns_normalized_values(self) -> None:
        probabilities = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[2], probabilities[1])

    def test_prediction_payload_returns_winner_and_all_probabilities(self) -> None:
        payload = prediction_payload(["low", "medium", "high", "urgent"], [0.0, 1.0, 2.0, 5.0])
        self.assertEqual(payload["priority"], "urgent")
        self.assertEqual(set(payload["probabilities"]), {"low", "medium", "high", "urgent"})
        self.assertAlmostEqual(sum(payload["probabilities"].values()), 1.0)

    def test_prediction_payload_rejects_misaligned_model_outputs(self) -> None:
        with self.assertRaises(ValueError):
            prediction_payload(["low", "medium"], [0.0])

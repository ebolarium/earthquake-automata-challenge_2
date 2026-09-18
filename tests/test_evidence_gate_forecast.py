import unittest
from pathlib import Path

import numpy as np

from etas_challenge.evidence_gate_forecast import (
    assemble_forecast,
    evidence_hurdle_weight,
    NumpyFastSlowEnsemble,
    reallocate_rates,
    support_restricted_mix,
)


class EvidenceGateForecastTest(unittest.TestCase):
    def test_neural_reallocation_preserves_count(self):
        rates = np.array([1.0, 2.0, 3.0])
        result = reallocate_rates(rates, np.array([-1.0, 0.0, 1.0]))
        self.assertAlmostEqual(np.sum(result), np.sum(rates))
        self.assertTrue(np.all(result > 0))

    def test_support_restriction_is_exact_outside_active_cells(self):
        etas = np.arange(1.0, 101.0)
        safe = np.full(100, 2.0)
        expert = np.linspace(0.5, 2.5, 100)
        result, active = support_restricted_mix(safe, expert, etas)
        self.assertTrue(np.array_equal(result[~active], safe[~active]))
        self.assertAlmostEqual(np.sum(result), np.sum(safe))

    def test_bf20_gate_starts_exactly_safe(self):
        etas = np.arange(1.0, 101.0)
        safe = np.full(100, 2.0)
        neural = np.linspace(0.5, 2.5, 100)
        result = assemble_forecast(etas, safe, neural, 0.0)
        self.assertEqual(result.gate_weight, 0.0)
        self.assertTrue(np.array_equal(result.gated_rates, safe))
        self.assertEqual(evidence_hurdle_weight(np.log(20.0)), 0.0)

    def test_gate_activates_only_beyond_hurdle(self):
        self.assertGreater(evidence_hurdle_weight(np.log(20.0) + 0.1), 0.0)

    def test_frozen_numpy_ensemble_loads_and_is_deterministic(self):
        root = Path(__file__).resolve().parents[1]
        model = NumpyFastSlowEnsemble.load(
            root / "models/evidence-gate/california-ch014-ensemble-v1.npz"
        )
        context = np.zeros((model.context_events, 5), dtype=np.float32)
        candidates = np.array([[0.0, 0.0], [0.25, -0.5]], dtype=np.float64)
        first = model.logits(context, candidates, batch_size=1)
        second = model.logits(context, candidates, batch_size=2)
        self.assertEqual(first.shape, (2,))
        self.assertTrue(np.all(np.isfinite(first)))
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()

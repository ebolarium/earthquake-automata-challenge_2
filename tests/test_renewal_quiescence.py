import unittest

import numpy as np

from etas_challenge.renewal_quiescence import bpt_overdue_score
from etas_challenge.renewal_quiescence import expected_reset_weight_gr
from etas_challenge.renewal_quiescence import magnitude_reset_weight
from etas_challenge.renewal_quiescence import update_expected_hazard_age


class RenewalQuiescenceTest(unittest.TestCase):
    def test_magnitude_mark_is_smooth_and_capped(self):
        weights = magnitude_reset_weight(np.array([2.5, 4.5, 6.5, 7.0]), 6.5, 0.5)
        np.testing.assert_allclose(weights, [0.01, 0.1, 1.0, 1.0])

    def test_gr_expectation_matches_numerical_quadrature(self):
        beta = 2.1471442086213064
        expected = expected_reset_weight_gr(beta, 2.5, 5.5, 0.5)
        x = np.linspace(0.0, 20.0, 1_000_001)
        mark = np.minimum(1.0, np.exp(0.5 * np.log(10.0) * (x - 3.0)))
        integrate = getattr(np, "trapezoid", None) or np.trapz
        numerical = integrate(beta * np.exp(-beta * x) * mark, x)
        self.assertAlmostEqual(expected, numerical, places=9)

    def test_hazard_age_uses_soft_posterior_reset(self):
        age = update_expected_hazard_age(
            np.array([1.0, 1.0]),
            np.array([0.2, 0.2]),
            np.array([0.0, np.log(2.0)]),
        )
        np.testing.assert_allclose(age, [1.2, 0.6])

    def test_bpt_score_is_zero_soon_after_reset_and_positive_when_overdue(self):
        score = bpt_overdue_score(np.array([0.0, 0.05, 1.0, 2.0]), 0.5)
        self.assertEqual(score[0], 0.0)
        self.assertEqual(score[1], 0.0)
        self.assertGreater(score[2], 0.0)
        self.assertGreater(score[3], 0.0)


if __name__ == "__main__":
    unittest.main()

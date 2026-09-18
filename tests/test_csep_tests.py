import unittest

import numpy as np

from etas_challenge.csep_tests import csep_daily_report
from etas_challenge.csep_tests import paired_r_test
from etas_challenge.csep_tests import poisson_log_likelihood


class CSEPTestsTest(unittest.TestCase):
    def test_equal_forecasts_have_zero_r_statistic(self):
        rates = np.array([0.2, 0.3, 0.5])
        report = paired_r_test(np.array([0, 1, 0]), rates, rates)
        self.assertAlmostEqual(report["observed_log_likelihood_ratio"], 0.0)
        self.assertIsNone(report["one_sided_p"])

    def test_report_is_deterministic_and_contains_standard_tests(self):
        cells = np.array([0, 1, 1], dtype=np.int32)
        baseline = np.array([0.5, 0.7, 0.2])
        challenger = np.array([0.4, 0.8, 0.2])
        first = csep_daily_report(
            cells, baseline, challenger, identity="run:catalog", simulations=100
        )
        second = csep_daily_report(
            cells, baseline, challenger, identity="run:catalog", simulations=100
        )
        self.assertEqual(first, second)
        self.assertIn("n_test_two_sided_p", first["challenger"])
        self.assertIn("l_test_lower_tail_p", first["challenger"])
        self.assertIn("one_sided_p", first["r_test"])

    def test_poisson_log_likelihood_rejects_nonpositive_rates(self):
        with self.assertRaises(ValueError):
            poisson_log_likelihood(np.array([0]), np.array([0.0]))


if __name__ == "__main__":
    unittest.main()

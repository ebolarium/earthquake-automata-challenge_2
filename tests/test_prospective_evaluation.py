import unittest
from pathlib import Path

from etas_challenge.prospective_evaluation import build_evaluation, evaluation_markdown


class ProspectiveEvaluationTest(unittest.TestCase):
    def dashboard(self):
        summary = {
            "days": 2,
            "events": 4,
            "total_gain": 0.04,
            "mean_igpe": 0.01,
            "relative_factor": 1.010050167,
        }
        empty = {
            "days": 0,
            "events": 0,
            "total_gain": 0.0,
            "mean_igpe": None,
            "relative_factor": None,
        }
        return {
            "generated_at": "2026-09-06T00:05:00+00:00",
            "pipeline_status": "ok",
            "protocol": {
                "protocol_id": "ch008-three-region-dry-run-v1",
                "mode": "dry_run",
                "counts_toward_prospective_claim": False,
                "minimum_events": 1,
                "settled_score_delay_days": 7,
            },
            "dry_run": {"phase": "running", "planned_days": 14},
            "provisional": summary,
            "final": empty,
            "latest_target_start": "2026-09-07T00:00:00+00:00",
            "forecast_current": True,
            "published_regions": 1,
            "open_incidents": 0,
            "last_incident_at": None,
            "last_open_incident_at": None,
            "pooled_primary_claim_status": "eligible",
            "regions": [{
                "region_id": "test-region",
                "name": "Test Region",
                "catalog_source": "Test catalog",
                "minimum_magnitude": 4.0,
                "minimum_depth_km": 0.0,
                "maximum_depth_km": 40.0,
                "latest_catalog": None,
                "latest_forecast": None,
                "operations": {
                    "primary_eligible": True,
                    "missed_region_days": 0,
                    "current_consecutive_missed_days": 0,
                    "longest_consecutive_missed_days": 0,
                },
                "provisional": summary,
                "final": empty,
            }],
            "daily_scores": [],
        }

    def test_dry_run_is_never_presented_as_prospective_win(self):
        root = Path(__file__).resolve().parents[1]
        result = build_evaluation(self.dashboard(), root=root)
        self.assertEqual(
            result["evidence_status"]["claim_assessment"],
            "no_prospective_claim_permitted_in_current_dry_run",
        )
        self.assertEqual(
            result["live_results"]["descriptive_result"]["descriptive_direction"],
            "challenger_higher_observed_event_density",
        )
        self.assertEqual(
            result["primary_metric"]["event_formula"],
            "IG_i = ln(lambda_CH008(i) / lambda_ETAS(i))",
        )
        self.assertIn("no_prospective_claim_permitted", evaluation_markdown(result))

    def test_final_scores_are_preferred_after_settlement(self):
        dashboard = self.dashboard()
        dashboard["final"] = {
            "days": 1,
            "events": 2,
            "total_gain": -0.02,
            "mean_igpe": -0.01,
            "relative_factor": 0.990049833,
        }
        result = build_evaluation(
            dashboard, root=Path(__file__).resolve().parents[1]
        )
        self.assertEqual(
            result["evidence_status"]["preferred_score_revision_for_description"],
            "final",
        )
        self.assertEqual(
            result["live_results"]["descriptive_result"]["descriptive_direction"],
            "etas_higher_observed_event_density",
        )


if __name__ == "__main__":
    unittest.main()

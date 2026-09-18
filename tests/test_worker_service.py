import json
from http import HTTPStatus
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from etas_challenge.worker_service import create_server


class Newsletter:
    def active_count(self):
        return 12

    def subscribe(self, email, locale):
        if "@" not in email:
            raise ValueError(email)
        return type("Result", (), {"status": "pending", "confirmation_sent": True})()

    def confirm(self, token):
        return "confirmed" if token == "valid" else "invalid"

    def unsubscribe(self, token):
        return "unsubscribed" if token == "valid" else "invalid"


class WorkerServiceTest(unittest.TestCase):
    def start_server(self, result, dashboard=None, forecast_map=None, newsletter=None):
        static = Path(__file__).resolve().parents[1] / "prospective_web" / "static"
        try:
            server = create_server(
                "127.0.0.1", 0, lambda: result,
                None if dashboard is None else lambda: dashboard,
                static,
                None if forecast_map is None else lambda region: forecast_map(region),
                newsletter,
            )
        except PermissionError:
            self.skipTest("local sockets are unavailable in this sandbox")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: thread.join(timeout=2))
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def test_healthy_database(self):
        base = self.start_server((True, None))
        with urllib.request.urlopen(f"{base}/health") as response:
            payload = json.load(response)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], "prospective-worker")

    def test_unavailable_database(self):
        base = self.start_server((False, "database_unavailable"))
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(f"{base}/health")
        self.assertEqual(raised.exception.code, HTTPStatus.SERVICE_UNAVAILABLE)

    def test_serves_dashboard_and_static_application(self):
        dashboard = {
            "schema_version": 2,
            "generated_at": "2026-09-04T08:00:00+00:00",
            "pipeline_status": "ok",
            "protocol": {
                "protocol_id": "ch008-three-region-dry-run-v1",
                "mode": "dry_run",
                "counts_toward_prospective_claim": False,
                "minimum_events": 1,
                "settled_score_delay_days": 7,
            },
            "dry_run": {"phase": "running", "planned_days": 14},
            "provisional": {
                "days": 1, "events": 1, "total_gain": 0.01,
                "mean_igpe": 0.01, "relative_factor": 1.01005,
            },
            "final": {
                "days": 0, "events": 0, "total_gain": 0.0,
                "mean_igpe": None, "relative_factor": None,
            },
            "regions": [],
            "daily_scores": [],
        }
        base = self.start_server(
            (True, None), dashboard,
            lambda region: {"region_id": region, "layers": {}},
        )
        with urllib.request.urlopen(f"{base}/") as response:
            self.assertIn(b"Multi-Region ETAS Test", response.read())
            self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        with urllib.request.urlopen(f"{base}/api/dashboard") as response:
            self.assertEqual(json.load(response), dashboard)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        with urllib.request.urlopen(f"{base}/api/evaluation.json") as response:
            evaluation = json.load(response)
            self.assertEqual(evaluation["schema"], "multi-region-etas-evaluation-v1")
            self.assertFalse(
                evaluation["evidence_status"]["counts_toward_prospective_claim"]
            )
        with urllib.request.urlopen(f"{base}/ai-evaluation") as response:
            self.assertEqual(response.headers.get_content_type(), "text/markdown")
            self.assertIn(b"no_prospective_claim_permitted", response.read())
        with urllib.request.urlopen(f"{base}/llms.txt") as response:
            self.assertIn(b"Evaluation JSON", response.read())
        with urllib.request.urlopen(f"{base}/robots.txt") as response:
            self.assertEqual(response.read(), b"User-agent: *\nAllow: /\n")
        with urllib.request.urlopen(f"{base}/api/forecast-map?region=california-relm") as response:
            self.assertEqual(response.status, HTTPStatus.OK)
        with urllib.request.urlopen(f"{base}/about.html") as response:
            content = response.read()
            self.assertIn("Multi-Region ETAS Test".encode(), content)
            self.assertIn(b"Machine-readable evidence", content)
        with urllib.request.urlopen(f"{base}/en/") as response:
            self.assertIn(b"Multi-Region ETAS Test", response.read())
        with urllib.request.urlopen(f"{base}/en/about.html") as response:
            content = response.read()
            self.assertIn(b"Multi-Region ETAS Test", content)
            self.assertIn(b"Machine-readable evidence", content)

    def test_forecast_map_requires_region_and_handles_unknown_region(self):
        def reader(region):
            if region != "california-relm":
                raise LookupError(region)
            return {"region_id": region, "layers": {}}

        base = self.start_server((True, None), forecast_map=reader)
        with self.assertRaises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"{base}/api/forecast-map")
        self.assertEqual(missing.exception.code, HTTPStatus.BAD_REQUEST)
        with self.assertRaises(urllib.error.HTTPError) as unknown:
            urllib.request.urlopen(f"{base}/api/forecast-map?region=unknown")
        self.assertEqual(unknown.exception.code, HTTPStatus.NOT_FOUND)

    def test_newsletter_count_subscription_confirmation_and_unsubscribe(self):
        base = self.start_server((True, None), newsletter=Newsletter())
        with urllib.request.urlopen(f"{base}/api/newsletter") as response:
            self.assertEqual(json.load(response)["subscribers"], 12)
        request = urllib.request.Request(
            f"{base}/api/newsletter",
            data=json.dumps({"email": "a@example.com", "locale": "tr"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, HTTPStatus.ACCEPTED)
            self.assertEqual(json.load(response), {"status": "pending"})
        with urllib.request.urlopen(f"{base}/newsletter/confirm?token=valid") as response:
            self.assertIn("Abonelik doğrulandı".encode(), response.read())
        unsubscribe = urllib.request.Request(
            f"{base}/newsletter/unsubscribe?token=valid", data=b"", method="POST"
        )
        with urllib.request.urlopen(unsubscribe) as response:
            self.assertIn("Abonelik sonlandırıldı".encode(), response.read())
        one_click = urllib.request.Request(
            f"{base}/api/newsletter/unsubscribe?token=valid", data=b"", method="POST"
        )
        with urllib.request.urlopen(one_click) as response:
            self.assertEqual(response.status, HTTPStatus.NO_CONTENT)


if __name__ == "__main__":
    unittest.main()

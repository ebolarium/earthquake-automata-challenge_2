from datetime import date
import json
import unittest

from etas_challenge.newsletter import ResendClient, confirmation_message, normalize_email
from etas_challenge.newsletter_report import render_daily_report


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return b'{"id":"message-1"}'


class NewsletterTest(unittest.TestCase):
    def test_normalizes_valid_email_and_rejects_invalid_email(self):
        self.assertEqual(normalize_email(" Person@Example.COM "), "person@example.com")
        for value in ("missing-at", "x@localhost", "x y@example.com", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_email(value)

    def test_resend_client_sets_auth_user_agent_and_idempotency(self):
        captured = {}

        def opener(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return Response()

        client = ResendClient("re_test", opener=opener)
        message_id = client.send(
            {"from": "hello@bboga.com", "to": ["a@example.com"], "subject": "x"},
            "daily-1",
        )
        request = captured["request"]
        self.assertEqual(message_id, "message-1")
        self.assertEqual(request.get_header("Authorization"), "Bearer re_test")
        self.assertEqual(request.get_header("Idempotency-key"), "daily-1")
        self.assertIn("earthquake-automata", request.get_header("User-agent"))
        self.assertEqual(json.loads(request.data)["to"], ["a@example.com"])

    def test_confirmation_and_report_are_localized(self):
        message = confirmation_message(
            "hello@bboga.com", "a@example.com", "tr", "https://example.com/confirm"
        )
        self.assertIn("doğrula", message["subject"])
        self.assertIn("https://example.com/confirm", message["html"])
        dashboard = {
            "pipeline_status": "ok", "latest_target_start": "2026-09-04T00:00:00+00:00",
            "published_regions": 3, "open_incidents": 0,
            "provisional": {"mean_igpe": 0.01},
            "regions": [{"region_id": "chile", "name": "Chile"}],
            "daily_scores": [{
                "region_id": "chile", "target_date": "2026-09-02",
                "revision": "provisional", "event_count": 1, "mean_igpe": 0.01,
                "csep": {
                    "challenger": {
                        "n_test_two_sided_p": 0.42,
                        "l_test_lower_tail_p": 0.31,
                    },
                    "r_test": {"one_sided_p": 0.08},
                },
            }],
        }
        subject, body = render_daily_report(
            dashboard, date(2026, 9, 3), "en", "https://example.com", "https://example.com/u"
        )
        self.assertIn("Daily Status", subject)
        self.assertIn("Pipeline operational", body)
        self.assertIn("N-test p", body)
        self.assertIn("0.420", body)
        self.assertIn("https://example.com/u", body)
        self.assertIn('href="https://example.com/"', body)
        _, turkish_body = render_daily_report(
            dashboard, date(2026, 9, 3), "tr", "https://example.com", "https://example.com/u"
        )
        self.assertIn('href="https://example.com/tr/"', turkish_body)


if __name__ == "__main__":
    unittest.main()

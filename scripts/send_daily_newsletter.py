#!/usr/bin/env python3
"""Send the morning prospective status report to confirmed subscribers."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etas_challenge.newsletter import ResendClient, ResendError, public_base_url  # noqa: E402
from etas_challenge.newsletter_report import render_daily_report  # noqa: E402
from etas_challenge.prospective_dashboard import read_dashboard  # noqa: E402
from etas_challenge.prospective_runtime import read_active_protocol_id  # noqa: E402


TURKEY_TIME = timezone(timedelta(hours=3))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-date", type=date.fromisoformat)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    database_url = os.environ.get("DATABASE_URL")
    api_key = os.environ.get("RESEND_API_KEY")
    base_url = public_base_url(os.environ.get("NEWSLETTER_PUBLIC_BASE_URL", ""))
    sender = os.environ.get(
        "NEWSLETTER_FROM", "Multi-Region ETAS Test <hello@bboga.com>"
    )
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    if not api_key and not args.dry_run:
        raise SystemExit("RESEND_API_KEY is required")

    import psycopg

    report_date = args.report_date or datetime.now(TURKEY_TIME).date()
    dashboard = read_dashboard(database_url, read_active_protocol_id(database_url))
    client = None if args.dry_run else ResendClient(api_key)
    sent = 0
    failures = 0
    with psycopg.connect(database_url, connect_timeout=5) as connection:
        connection.execute(
            """
            INSERT INTO prospective.newsletter_deliveries (subscriber_id, report_date)
            SELECT subscriber_id, %s FROM prospective.newsletter_subscribers
            WHERE status = 'active'
            ON CONFLICT (subscriber_id, report_date) DO NOTHING
            """,
            (report_date,),
        )
        connection.commit()
        recipients = connection.execute(
            """
            SELECT d.delivery_id, s.subscriber_id, s.email, s.locale,
                   s.confirmation_token_sha256, d.attempt_count
            FROM prospective.newsletter_deliveries d
            JOIN prospective.newsletter_subscribers s USING (subscriber_id)
            WHERE d.report_date = %s AND d.status <> 'sent'
              AND d.attempt_count < 3 AND s.status = 'active'
            ORDER BY s.subscriber_id
            """,
            (report_date,),
        ).fetchall()
        for delivery_id, subscriber_id, email, locale, token, attempts in recipients:
            unsubscribe_url = f"{base_url}/newsletter/unsubscribe?token={token}"
            one_click_url = f"{base_url}/api/newsletter/unsubscribe?token={token}"
            subject, html_body = render_daily_report(
                dashboard, report_date, locale, base_url, unsubscribe_url
            )
            if args.dry_run:
                print(json.dumps({
                    "status": "dry_run", "subscriber_id": int(subscriber_id),
                    "locale": locale, "subject": subject,
                }, sort_keys=True))
                continue
            delivered = False
            for attempt in range(int(attempts), 3):
                connection.execute(
                    """
                    UPDATE prospective.newsletter_deliveries
                    SET attempt_count = attempt_count + 1, updated_at = now()
                    WHERE delivery_id = %s
                    """,
                    (delivery_id,),
                )
                connection.commit()
                try:
                    message_id = client.send(
                        {
                            "from": sender,
                            "to": [email],
                            "subject": subject,
                            "html": html_body,
                            "headers": {
                                "List-Unsubscribe": f"<{one_click_url}>",
                                "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
                            },
                        },
                        f"multi-region-etas-daily-{report_date.isoformat()}-{subscriber_id}",
                    )
                except Exception as exc:
                    connection.execute(
                        """
                        UPDATE prospective.newsletter_deliveries
                        SET status = 'failed', last_error = %s, updated_at = now()
                        WHERE delivery_id = %s
                        """,
                        (str(exc)[-1000:], delivery_id),
                    )
                    connection.commit()
                    if isinstance(exc, ResendError) and not exc.retryable:
                        break
                    if attempt < 2:
                        time.sleep((2, 10)[attempt])
                    continue
                connection.execute(
                    """
                    UPDATE prospective.newsletter_deliveries
                    SET status = 'sent', provider_message_id = %s, sent_at = now(),
                        last_error = NULL, updated_at = now()
                    WHERE delivery_id = %s
                    """,
                    (message_id, delivery_id),
                )
                connection.commit()
                sent += 1
                delivered = True
                break
            if not delivered:
                failures += 1
    print(json.dumps({
        "status": "ok" if failures == 0 else "failed",
        "report_date": report_date.isoformat(), "recipients": len(recipients),
        "sent": sent, "failures": failures,
    }, sort_keys=True))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

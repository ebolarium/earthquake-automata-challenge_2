"""Newsletter subscription, consent, and Resend delivery helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import formataddr
import hashlib
import json
import os
import re
import secrets
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EMAIL_PATTERN = re.compile(
    r"^[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
)
RESEND_ENDPOINT = "https://api.resend.com/emails"


def normalize_email(value: str) -> str:
    email = value.strip().lower()
    if len(email) > 254 or not EMAIL_PATTERN.fullmatch(email):
        raise ValueError("invalid email address")
    return email


def normalize_locale(value: str) -> str:
    return "en" if value == "en" else "tr"


def token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def public_base_url(value: str) -> str:
    result = value.strip().rstrip("/")
    if not result or not (result.startswith("https://") or result.startswith("http://localhost")):
        raise ValueError("NEWSLETTER_PUBLIC_BASE_URL must use https")
    return result


@dataclass(frozen=True, slots=True)
class SubscriptionResult:
    status: str
    confirmation_sent: bool


class ResendError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


class ResendClient:
    def __init__(self, api_key: str, opener=urlopen):
        if not api_key:
            raise ValueError("RESEND_API_KEY is required")
        self.api_key = api_key
        self.opener = opener

    def send(self, message: dict, idempotency_key: str) -> str:
        request = Request(
            RESEND_ENDPOINT,
            data=json.dumps(message, separators=(",", ":")).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
                "User-Agent": "earthquake-automata-challenge/0.1",
            },
        )
        try:
            with self.opener(request, timeout=15) as response:
                result = json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise ResendError(
                f"Resend HTTP {exc.code}: {detail}",
                retryable=exc.code == 429 or exc.code >= 500,
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise ResendError("Resend request failed", retryable=True) from exc
        message_id = result.get("id")
        if not isinstance(message_id, str) or not message_id:
            raise RuntimeError("Resend response did not contain a message id")
        return message_id


def confirmation_message(sender: str, email: str, locale: str, confirm_url: str) -> dict:
    if locale == "en":
        subject = "Confirm your Evidence-Gated Forecast daily report subscription"
        title = "Confirm your subscription"
        copy = "Use the button below to receive the Evidence-Gated Forecast Test status report each morning."
        action = "Confirm email address"
        ignore = "If you did not request this, you can ignore this email."
    else:
        subject = "Kanıt Kapılı Tahmin günlük durum raporu aboneliğini doğrula"
        title = "Aboneliğini doğrula"
        copy = "Her sabah Kanıt Kapılı Tahmin Testi durum raporunu almak için aşağıdaki düğmeyi kullan."
        action = "E-posta adresini doğrula"
        ignore = "Bu isteği sen yapmadıysan bu e-postayı yok sayabilirsin."
    html = f"""<!doctype html><html><body style="margin:0;background:#f4f6f5;color:#17201e;font-family:Arial,sans-serif">
<div style="max-width:560px;margin:0 auto;padding:36px 20px"><p style="font-size:11px;color:#087f7a;font-weight:700">EVIDENCE-GATED FORECAST TEST</p>
<h1 style="font-size:24px">{title}</h1><p style="font-size:14px;line-height:1.6">{copy}</p>
<p style="margin:28px 0"><a href="{confirm_url}" style="padding:11px 16px;background:#087f7a;color:#fff;text-decoration:none;border-radius:4px;font-weight:700">{action}</a></p>
<p style="font-size:11px;color:#68736f">{ignore}</p></div></body></html>"""
    return {"from": sender, "to": [email], "subject": subject, "html": html}


class NewsletterService:
    def __init__(
        self,
        database_url: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        sender: str | None = None,
        client: ResendClient | None = None,
    ):
        self.database_url = database_url
        configured_base = (
            base_url if base_url is not None else os.environ.get("NEWSLETTER_PUBLIC_BASE_URL", "")
        )
        self.base_url = public_base_url(configured_base) if configured_base else None
        self.sender = sender or os.environ.get(
            "NEWSLETTER_FROM", formataddr(("Evidence-Gated Forecast Test", "hello@bboga.com"))
        )
        configured_key = api_key if api_key is not None else os.environ.get("RESEND_API_KEY", "")
        self.client = client or (ResendClient(configured_key) if configured_key else None)

    def active_count(self) -> int:
        import psycopg

        with psycopg.connect(self.database_url, connect_timeout=5) as connection:
            return int(connection.execute(
                "SELECT count(*) FROM prospective.newsletter_subscribers WHERE status = 'active'"
            ).fetchone()[0])

    def subscribe(self, email_value: str, locale_value: str) -> SubscriptionResult:
        import psycopg

        if self.base_url is None or self.client is None:
            raise RuntimeError("newsletter delivery is not configured")
        email = normalize_email(email_value)
        locale = normalize_locale(locale_value)
        token = secrets.token_urlsafe(32)
        digest = token_sha256(token)
        now = datetime.now(timezone.utc)
        should_send = True
        with psycopg.connect(self.database_url, connect_timeout=5) as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (email,))
            existing = connection.execute(
                """
                SELECT status, updated_at FROM prospective.newsletter_subscribers
                WHERE email = %s
                """,
                (email,),
            ).fetchone()
            if existing is not None and existing[0] == "active":
                return SubscriptionResult("active", False)
            if existing is not None and existing[0] == "pending":
                should_send = existing[1] < now - timedelta(minutes=10)
                if not should_send:
                    return SubscriptionResult("pending", False)
            connection.execute(
                """
                INSERT INTO prospective.newsletter_subscribers
                    (email, locale, status, confirmation_token_sha256, updated_at)
                VALUES (%s, %s, 'pending', %s, %s)
                ON CONFLICT (email) DO UPDATE
                SET locale = EXCLUDED.locale, status = 'pending',
                    confirmation_token_sha256 = EXCLUDED.confirmation_token_sha256,
                    unsubscribed_at = NULL, updated_at = EXCLUDED.updated_at
                """,
                (email, locale, digest, now),
            )
            confirm_url = f"{self.base_url}/newsletter/confirm?token={token}"
            self.client.send(
                confirmation_message(self.sender, email, locale, confirm_url),
                f"ch008-confirm-{digest}",
            )
        return SubscriptionResult("pending", should_send)

    def confirm(self, token: str) -> str:
        import psycopg

        digest = token_sha256(token)
        with psycopg.connect(self.database_url, connect_timeout=5) as connection:
            row = connection.execute(
                """
                UPDATE prospective.newsletter_subscribers
                SET status = 'active', subscribed_at = COALESCE(subscribed_at, now()),
                    unsubscribed_at = NULL, updated_at = now()
                WHERE confirmation_token_sha256 = %s AND status IN ('pending', 'active')
                RETURNING status
                """,
                (digest,),
            ).fetchone()
        return "confirmed" if row is not None else "invalid"

    def unsubscribe(self, token: str) -> str:
        import psycopg

        digest = token if re.fullmatch(r"[0-9a-f]{64}", token) else token_sha256(token)
        with psycopg.connect(self.database_url, connect_timeout=5) as connection:
            row = connection.execute(
                """
                UPDATE prospective.newsletter_subscribers
                SET status = 'unsubscribed', unsubscribed_at = now(), updated_at = now()
                WHERE confirmation_token_sha256 = %s
                RETURNING status
                """,
                (digest,),
            ).fetchone()
        return "unsubscribed" if row is not None else "invalid"

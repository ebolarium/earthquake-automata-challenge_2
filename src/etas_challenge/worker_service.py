"""Health service for the prospective scheduler container."""

from __future__ import annotations

from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from etas_challenge.object_storage import ObjectStorageConfig
from etas_challenge.object_storage import storage_health
from etas_challenge.newsletter import NewsletterService
from etas_challenge.prospective_dashboard import read_dashboard
from etas_challenge.prospective_evaluation import build_evaluation
from etas_challenge.prospective_evaluation import evaluation_markdown
from etas_challenge.prospective_evaluation import llms_text
from etas_challenge.prospective_map import ForecastMapReader
from etas_challenge.prospective_runtime import read_active_protocol_id


DEFAULT_STATIC = Path(
    os.environ.get(
        "PROSPECTIVE_STATIC_DIR",
        Path.cwd() / "prospective_web" / "static",
    )
)


class ActiveForecastMapReader:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.readers = {}

    def __call__(self, region_id: str):
        protocol_id = read_active_protocol_id(self.database_url)
        reader = self.readers.get(protocol_id)
        if reader is None:
            reader = ForecastMapReader(self.database_url, protocol_id)
            self.readers[protocol_id] = reader
        return reader(region_id)


def database_health(database_url: str | None) -> tuple[bool, str | None]:
    if not database_url:
        return False, "database_url_missing"
    try:
        import psycopg

        with psycopg.connect(database_url, connect_timeout=3) as connection:
            connection.execute("SELECT 1")
    except Exception:
        return False, "database_unavailable"
    return True, None


def combined_health(
    database_url: str | None,
    require_object_storage: bool,
) -> tuple[bool, str | None]:
    healthy, reason = database_health(database_url)
    if not healthy or not require_object_storage:
        return healthy, reason
    try:
        config = ObjectStorageConfig.from_environment()
    except ValueError:
        return False, "object_storage_config_invalid"
    return storage_health(config)


def handler_factory(
    checker, dashboard_reader=None, static_dir=None, map_reader=None,
    newsletter_service=None,
):
    static_root = Path(static_dir or DEFAULT_STATIC).resolve()

    class WorkerRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_root), **kwargs)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/health":
                self._serve_health()
                return
            if path == "/api/dashboard":
                self._serve_dashboard()
                return
            if path == "/api/evaluation.json":
                self._serve_evaluation_json()
                return
            if path == "/ai-evaluation":
                self._serve_ai_evaluation()
                return
            if path == "/llms.txt":
                self._text_response(
                    HTTPStatus.OK, llms_text(self._public_base_url()), "text/plain"
                )
                return
            if path == "/robots.txt":
                self._text_response(
                    HTTPStatus.OK, "User-agent: *\nAllow: /\n", "text/plain"
                )
                return
            if path == "/api/forecast-map":
                self._serve_forecast_map()
                return
            if path == "/api/newsletter":
                self._serve_newsletter_status()
                return
            if path == "/newsletter/confirm":
                self._serve_newsletter_confirmation()
                return
            if path == "/newsletter/unsubscribe":
                self._serve_unsubscribe_page()
                return
            if path in ("/", "/en", "/en/"):
                self.path = "/en/index.html"
            elif path in ("/about.html", "/en/about.html"):
                self.path = "/en/about.html"
            elif path in ("/tr", "/tr/"):
                self.path = "/index.html"
            elif path == "/tr/about.html":
                self.path = "/about.html"
            elif path not in (
                "/index.html", "/styles.css", "/about.css", "/app.js",
                "/app-en.js", "/en/index.html"
            ):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            super().do_GET()

        def do_POST(self):
            path = urlsplit(self.path).path
            if path == "/api/newsletter":
                self._serve_newsletter_subscription()
                return
            if path == "/api/newsletter/unsubscribe":
                self._serve_newsletter_unsubscribe(one_click=True)
                return
            if path == "/newsletter/unsubscribe":
                self._serve_newsletter_unsubscribe(one_click=False)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def end_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
            )
            super().end_headers()

        def _serve_health(self):
            healthy, reason = checker()
            body = {
                "status": "ok" if healthy else "degraded",
                "service": "prospective-worker",
            }
            if reason:
                body["reason"] = reason
            commit = os.environ.get("SOURCE_COMMIT")
            if commit:
                body["source_commit"] = commit
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _serve_dashboard(self):
            if dashboard_reader is None:
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "dashboard_unavailable")
                return
            try:
                body = dashboard_reader()
            except Exception:
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "dashboard_unavailable")
                return
            encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _public_base_url(self):
            return os.environ.get(
                "PUBLIC_BASE_URL",
                os.environ.get("NEWSLETTER_PUBLIC_BASE_URL", "https://etas.bboga.com"),
            ).rstrip("/")

        def _evaluation(self):
            if dashboard_reader is None:
                raise RuntimeError("dashboard unavailable")
            return build_evaluation(
                dashboard_reader(),
                root=Path.cwd(),
                public_base_url=self._public_base_url(),
            )

        def _serve_evaluation_json(self):
            try:
                body = self._evaluation()
            except Exception:
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "evaluation_unavailable")
                return
            self._json_response(HTTPStatus.OK, body)

        def _serve_ai_evaluation(self):
            try:
                body = evaluation_markdown(self._evaluation())
            except Exception:
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "evaluation_unavailable")
                return
            self._text_response(HTTPStatus.OK, body, "text/markdown")

        def _serve_forecast_map(self):
            if map_reader is None:
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "forecast_map_unavailable")
                return
            query = parse_qs(urlsplit(self.path).query)
            region_id = query.get("region", [""])[0]
            if not region_id:
                self._json_error(HTTPStatus.BAD_REQUEST, "region_required")
                return
            try:
                body = map_reader(region_id)
            except LookupError:
                self._json_error(HTTPStatus.NOT_FOUND, "forecast_map_not_found")
                return
            except Exception:
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "forecast_map_unavailable")
                return
            encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _serve_newsletter_status(self):
            if newsletter_service is None:
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "newsletter_unavailable")
                return
            try:
                count = newsletter_service.active_count()
            except Exception:
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "newsletter_unavailable")
                return
            self._json_response(HTTPStatus.OK, {"status": "ok", "subscribers": count})

        def _serve_newsletter_subscription(self):
            if newsletter_service is None:
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "newsletter_unavailable")
                return
            if self.headers.get_content_type() != "application/json":
                self._json_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "json_required")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4096:
                    raise ValueError("invalid body length")
                body = json.loads(self.rfile.read(length))
                if body.get("company"):
                    self._json_response(HTTPStatus.ACCEPTED, {"status": "pending"})
                    return
                result = newsletter_service.subscribe(
                    str(body.get("email", "")), str(body.get("locale", "en"))
                )
            except (ValueError, json.JSONDecodeError):
                self._json_error(HTTPStatus.BAD_REQUEST, "invalid_subscription")
                return
            except Exception:
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "newsletter_unavailable")
                return
            self._json_response(
                HTTPStatus.ACCEPTED,
                {"status": "pending"},
            )

        def _serve_newsletter_confirmation(self):
            token = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
            if newsletter_service is None or not token or len(token) > 128:
                self._html_message("Bağlantı geçersiz", "The link is invalid.")
                return
            try:
                result = newsletter_service.confirm(token)
            except Exception:
                self._html_message("İşlem tamamlanamadı", "Please try again later.", error=True)
                return
            if result == "confirmed":
                self._html_message("Abonelik doğrulandı", "Your subscription is confirmed.")
            else:
                self._html_message("Bağlantı geçersiz", "The link is invalid.", error=True)

        def _serve_unsubscribe_page(self):
            token = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
            if not token or len(token) > 128:
                self._html_message("Bağlantı geçersiz", "The link is invalid.", error=True)
                return
            action = f"/newsletter/unsubscribe?token={html.escape(token, quote=True)}"
            body = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Multi-Region ETAS Newsletter</title><link rel="stylesheet" href="/styles.css"></head>
<body><main class="subscription-page"><h1>Abonelikten çık</h1><p>Çok bölgeli ETAS testinin günlük raporlarını artık almak istemiyor musun?</p><form method="post" action="{action}"><button>Aboneliği sonlandır</button></form><small>Unsubscribe from the Multi-Region ETAS Test daily status report.</small></main></body></html>"""
            self._html_response(HTTPStatus.OK, body)

        def _serve_newsletter_unsubscribe(self, *, one_click):
            token = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
            if newsletter_service is None or not token or len(token) > 128:
                self._html_message("Bağlantı geçersiz", "The link is invalid.", error=True)
                return
            try:
                result = newsletter_service.unsubscribe(token)
            except Exception:
                self._html_message("İşlem tamamlanamadı", "Please try again later.", error=True)
                return
            if one_click and result == "unsubscribed":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
            elif result == "unsubscribed":
                self._html_message("Abonelik sonlandırıldı", "You have been unsubscribed.")
            else:
                self._html_message("Bağlantı geçersiz", "The link is invalid.", error=True)

        def _html_message(self, title, copy, error=False):
            body = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{html.escape(title)}</title><link rel="stylesheet" href="/styles.css"></head>
<body><main class="subscription-page {'error' if error else ''}"><span>MULTI-REGION ETAS PROSPECTIVE TEST</span><h1>{html.escape(title)}</h1><p>{html.escape(copy)}</p><a href="/">Dashboard</a></main></body></html>"""
            self._html_response(HTTPStatus.OK, body)

        def _html_response(self, status, body):
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json_response(self, status, body):
            encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _text_response(self, status, body, content_type):
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json_error(self, status, reason):
            self._json_response(status, {"status": "error", "reason": reason})

        def log_message(self, format, *args):
            print(f"{self.address_string()} - {format % args}", flush=True)

    return WorkerRequestHandler


def create_server(
    host: str, port: int, checker, dashboard_reader=None, static_dir=None, map_reader=None,
    newsletter_service=None,
):
    return ThreadingHTTPServer(
        (host, port), handler_factory(
            checker, dashboard_reader, static_dir, map_reader, newsletter_service
        )
    )


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8080"))
    database_url = os.environ.get("DATABASE_URL")
    require_storage = os.environ.get("REQUIRE_OBJECT_STORAGE", "0") == "1"
    map_reader = None
    newsletter_service = NewsletterService(database_url) if database_url else None
    if database_url:
        try:
            map_reader = ActiveForecastMapReader(database_url)
        except ValueError:
            pass
    server = create_server(
        host,
        port,
        lambda: combined_health(database_url, require_storage),
        (
            None
            if not database_url
            else lambda: read_dashboard(
                database_url, read_active_protocol_id(database_url)
            )
        ),
        map_reader=map_reader,
        newsletter_service=newsletter_service,
    )
    print(f"Prospective worker listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

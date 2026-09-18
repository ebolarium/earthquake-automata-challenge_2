"""Bilingual HTML status report for the evidence-gated forecast test."""

from __future__ import annotations

from datetime import date
import html
import math


def _number(value, digits=4):
    if value is None:
        return "—"
    prefix = "+" if value > 0 else ""
    return f"{prefix}{value:.{digits}f}"


def _probability(value):
    if value is None:
        return "—"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _latest_scores(dashboard: dict) -> tuple[str | None, list[dict]]:
    provisional = [
        row for row in dashboard.get("daily_scores", [])
        if row["revision"] == "provisional"
    ]
    if not provisional:
        return None, []
    target = max(row["target_date"] for row in provisional)
    return target, [row for row in provisional if row["target_date"] == target]


def render_daily_report(
    dashboard: dict,
    report_date: date,
    locale: str,
    dashboard_url: str,
    unsubscribe_url: str,
) -> tuple[str, str]:
    english = locale == "en"
    target_date, scores = _latest_scores(dashboard)
    region_names = {
        region["region_id"]: region["name"] for region in dashboard["regions"]
    }
    status_ok = dashboard["pipeline_status"] == "ok"
    mean = dashboard["provisional"]["mean_igpe"]
    factor = None if mean is None else math.exp(mean)
    subject = (
        f"Multi-Region ETAS Test Daily Status | {report_date.isoformat()}"
        if english else f"Çok Bölgeli ETAS Test Durumu | {report_date.isoformat()}"
    )
    labels = {
        "title": "Prospective test daily report" if english else "Prospektif test günlük raporu",
        "status": "Pipeline operational" if status_ok and english else "Review required" if english else "Pipeline çalışıyor" if status_ok else "Kontrol gerekli",
        "target": "Latest forecast target" if english else "Son tahmin hedefi",
        "regions": "Published regions" if english else "Yayınlanan bölgeler",
        "incidents": "Open incidents" if english else "Açık incident",
        "score": "Cumulative provisional IGPE" if english else "Kümülatif provisional IGPE",
        "factor": "Relative factor" if english else "Göreli oran",
        "latest": "Latest completed target day" if english else "Son tamamlanan hedef günü",
        "no_scores": "No completed target-day score yet." if english else "Henüz tamamlanmış hedef günü skoru yok.",
        "events": "events" if english else "olay",
        "n_test": "N-test p",
        "l_test": "L-test p",
        "r_test": "R-test p",
        "dashboard": "Open dashboard" if english else "Dashboard’u aç",
        "research": "Research forecast; not an earthquake warning." if english else "Araştırma tahminidir; deprem uyarısı değildir.",
        "unsubscribe": "Unsubscribe" if english else "Abonelikten çık",
    }
    score_cards = ""
    for score in scores:
        winner = (
            ("Challenger" if english else "Yeni model")
            if score["mean_igpe"] is not None and score["mean_igpe"] > 0
            else "ETAS" if score["mean_igpe"] is not None and score["mean_igpe"] < 0
            else "—"
        )
        csep = score.get("csep") or {}
        challenger = csep.get("challenger") or {}
        r_test = csep.get("r_test") or {}
        score_cards += (
            f"<table role=\"presentation\" style=\"width:100%;margin:0 0 10px;border-collapse:collapse;background:#fff;border:1px solid #d6dedb\">"
            f"<tr><td colspan=\"2\" style=\"padding:10px 11px;font-size:12px;font-weight:700\">{html.escape(region_names.get(score['region_id'], score['region_id']))}</td>"
            f"<td colspan=\"2\" style=\"padding:10px 11px;text-align:right;color:#68736f;font-size:11px\">{score['event_count']} {labels['events']} · {winner}</td></tr>"
            f"<tr style=\"color:#68736f;font-size:9px;text-align:center\"><td style=\"padding:7px 3px;border-top:1px solid #d6dedb\">IGPE</td><td style=\"padding:7px 3px;border-top:1px solid #d6dedb\">{labels['n_test']}</td><td style=\"padding:7px 3px;border-top:1px solid #d6dedb\">{labels['l_test']}</td><td style=\"padding:7px 3px;border-top:1px solid #d6dedb\">{labels['r_test']}</td></tr>"
            f"<tr style=\"text-align:center;font-size:12px\"><td style=\"padding:0 3px 10px;font-weight:700\">{_number(score['mean_igpe'])}</td><td style=\"padding:0 3px 10px\">{_probability(challenger.get('n_test_two_sided_p'))}</td><td style=\"padding:0 3px 10px\">{_probability(challenger.get('l_test_lower_tail_p'))}</td><td style=\"padding:0 3px 10px\">{_probability(r_test.get('one_sided_p'))}</td></tr></table>"
        )
    latest_block = (
        f"<h2 style=\"font-size:15px;margin:26px 0 8px\">{labels['latest']} · {target_date}</h2>"
        f"{score_cards}"
        if scores else f"<p style=\"margin-top:24px;color:#68736f\">{labels['no_scores']}</p>"
    )
    localized_dashboard_url = dashboard_url.rstrip("/") + ("/" if english else "/tr/")
    html_body = f"""<!doctype html><html><body style="margin:0;background:#f4f6f5;color:#17201e;font-family:Arial,sans-serif">
<div style="max-width:620px;margin:0 auto;padding:30px 18px"><p style="font-size:10px;color:#087f7a;font-weight:700">MULTI-REGION ETAS PROSPECTIVE TEST · {report_date.isoformat()}</p>
<h1 style="font-size:24px;margin:8px 0 4px">{labels['title']}</h1><p style="margin:0 0 22px;color:{'#27835b' if status_ok else '#cf5b4c'};font-weight:700">{labels['status']}</p>
<table style="width:100%;border-collapse:collapse;background:#fff;font-size:12px"><tbody>
<tr><td style="padding:10px;border-bottom:1px solid #d6dedb;color:#68736f">{labels['target']}</td><td style="padding:10px;border-bottom:1px solid #d6dedb;text-align:right;font-weight:700">{html.escape(str(dashboard.get('latest_target_start') or '—')[:10])}</td></tr>
<tr><td style="padding:10px;border-bottom:1px solid #d6dedb;color:#68736f">{labels['regions']}</td><td style="padding:10px;border-bottom:1px solid #d6dedb;text-align:right;font-weight:700">{dashboard['published_regions']}/{len(dashboard['regions'])}</td></tr>
<tr><td style="padding:10px;border-bottom:1px solid #d6dedb;color:#68736f">{labels['incidents']}</td><td style="padding:10px;border-bottom:1px solid #d6dedb;text-align:right;font-weight:700">{dashboard['open_incidents']}</td></tr>
<tr><td style="padding:10px;border-bottom:1px solid #d6dedb;color:#68736f">{labels['score']}</td><td style="padding:10px;border-bottom:1px solid #d6dedb;text-align:right;font-weight:700">{_number(mean)}</td></tr>
<tr><td style="padding:10px;color:#68736f">{labels['factor']}</td><td style="padding:10px;text-align:right;font-weight:700">{'—' if factor is None else f'{factor:.4f}×'}</td></tr>
</tbody></table>{latest_block}
<p style="margin:28px 0"><a href="{html.escape(localized_dashboard_url, quote=True)}" style="padding:10px 14px;background:#087f7a;color:#fff;text-decoration:none;border-radius:4px;font-weight:700;font-size:12px">{labels['dashboard']}</a></p>
<p style="color:#68736f;font-size:10px;line-height:1.5">{labels['research']} · <a href="{html.escape(unsubscribe_url, quote=True)}" style="color:#68736f">{labels['unsubscribe']}</a></p>
</div></body></html>"""
    return subject, html_body

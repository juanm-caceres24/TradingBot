"""Outbound HTTP JSON webhook dispatcher for external alert consumers (stdlib only)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import tradingbot
from tradingbot.config import WebhookConfig
from tradingbot.models import AlertEvent


@dataclass(frozen=True)
class WebhookDeliveryResult:
    """Outcome of one outbound webhook POST attempt."""

    webhook_name: str
    url: str
    success: bool
    status_code: int | None = None
    error: str | None = None


def _build_alert_envelope(alert: AlertEvent) -> dict[str, Any]:
    data = alert.to_dict()
    try:
        data["payload"] = json.loads(alert.payload_json) if alert.payload_json else {}
    except json.JSONDecodeError:
        data["payload"] = {}
    return {"event": "tradingbot.alert", "alert": data}


class WebhookDispatcher:
    """Dispatches AlertEvents as JSON HTTP POST payloads to configured external webhooks."""

    def __init__(self, webhooks: Sequence[WebhookConfig]) -> None:
        self._webhooks = tuple(w for w in webhooks if w.enabled)

    @property
    def enabled_count(self) -> int:
        return len(self._webhooks)

    def dispatch(self, alert: AlertEvent) -> list[WebhookDeliveryResult]:
        if not self._webhooks:
            return []

        body = json.dumps(_build_alert_envelope(alert), separators=(",", ":")).encode("utf-8")
        results: list[WebhookDeliveryResult] = []

        for wh in self._webhooks:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": f"TradingBot/{tradingbot.__version__}",
            }
            headers.update(wh.headers)

            if wh.auth_token_env:
                token = os.environ.get(wh.auth_token_env, "").strip()
                if not token:
                    results.append(
                        WebhookDeliveryResult(
                            webhook_name=wh.name,
                            url=wh.url,
                            success=False,
                            error=f"Environment variable '{wh.auth_token_env}' is not set",
                        )
                    )
                    continue
                headers[wh.auth_header] = token

            req = urllib.request.Request(wh.url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=wh.timeout_sec) as resp:
                    status = int(getattr(resp, "status", 200))
                    results.append(
                        WebhookDeliveryResult(
                            webhook_name=wh.name,
                            url=wh.url,
                            success=200 <= status < 300,
                            status_code=status,
                        )
                    )
            except urllib.error.HTTPError as exc:
                results.append(
                    WebhookDeliveryResult(
                        webhook_name=wh.name,
                        url=wh.url,
                        success=False,
                        status_code=int(exc.code),
                        error=f"HTTP {exc.code}",
                    )
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                results.append(
                    WebhookDeliveryResult(
                        webhook_name=wh.name,
                        url=wh.url,
                        success=False,
                        error=str(exc),
                    )
                )

        return results

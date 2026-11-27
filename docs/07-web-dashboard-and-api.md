# Web Dashboard, External Alerts API & WebSockets (`v0.6.0`)

This document describes the embedded 24/7 Web Dashboard (`src/tradingbot/web/`), the zero-dependency HTTP/1.1 + RFC 6455 WebSocket server, LAN/Tailscale deployment, and how external notification services (Telegram, WhatsApp, email) consume alerts.

---

## 1. Zero-Dependency Architecture

Mirroring `NeuralNetwork`'s POSIX socket inference server, the entire TradingBot stack (`v0.1.0` through `v0.9.0`) runs using **Python's standard library only** — zero `pip install`, zero Node.js/npm build steps, and zero external CDNs:

```text
Browser (LAN / Tailscale)                     External Bot (Telegram / WhatsApp / Email)
   │  GET / + /static/*                          │  GET /api/alerts?since_id=N
   │  WS  /ws/live                               │  POST /api/alerts/{id}/ack
   ▼                                             ▼  WS  /ws/alerts
┌────────────────────────────────────────────────────────────────────────────┐
│ DashboardServer (ThreadingHTTPServer + native RFC 6455 WebSocketHub)       │
│        │                                       │                           │
│        ├── reads RuntimeState (RAM)            ├── reads/acks SQLite Store │
│        └── subscribes to BotOrchestrator.EventBus                          │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Web Dashboard Features (`src/tradingbot/web/static/`)

- **Multi-Asset Watchlist Sidebar:** Displays live price, timeframe, latest action (`LONG`/`SHORT`/`HOLD`), confidence, and unread alert badges for every active asset; click any card to switch charts instantaneously.
- **HiDPI Interactive Financial Chart (`chart.js`):**
  - Sliding window of real candles (default `70`, wheel-zoomable `15..250`, drag-to-pan, one-click `Snap Live`).
  - Dynamic indicator overlays (`SMA`, `EMA`, shaded `BollingerBands`) honoring warm-up `null` gaps cleanly (raw early candles remain visible while indicator lines start once valid).
  - **Translucent Predicted Candles:** Rendered immediately past the vertical `FORECAST` divider with opacity decaying across horizon steps + dashed `TP` / `SL` reference lines.
  - Synchronized **Volume** histogram and dynamic subcharts (**RSI** with 30/70 guides; **LazyBear Squeeze Momentum** with 4-color momentum histogram and zero-line squeeze dots).
  - Live toggle chips to show/hide indicators and forecast bars on the fly.
- **Prediction, Analysis & Alerts Panel:** Shows recommended action, confidence, expected move %, estimated TP/SL, Risk-Reward ratio, model inference latency (µs), Softmax probabilities (`LONG`/`HOLD`/`SHORT`), per-model online/offline status, consecutive-bar alert streak, and live Alerts feed with one-click `Ack`.
- **Extensible Toolbar:** Includes `Poll Now` (`POST /api/actions/poll`) and a dedicated `#custom-actions` DOM container for future custom controls.

---

## 3. REST & WebSocket API Reference

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the responsive Web Dashboard UI |
| `GET` | `/health` | Server liveness (`status`, `version`, `assets_count`, `ws_clients`) |
| `GET` | `/api/config` | Safe public UI config (`title`, `visible_candles`, `api_enabled`, `version`) |
| `GET` | `/api/assets` | Watchlist summary for all running assets |
| `GET` | `/api/assets/{asset_id}/latest?visible=70` | Candles (`visible >= 5`), aligned indicators + `indicator_types`, latest prediction & analysis |
| `GET` | `/api/alerts?since_id=0&asset_id=&unacknowledged=true&limit=100` | Cursor-based Alerts API (oldest-first when `since_id` is given; `403` when `notifications.api_enabled=false`) |
| `POST` | `/api/alerts/{alert_id}/ack` | Acknowledge an alert (`200 OK`, `403 Forbidden`, or `404 Not Found`) |
| `POST` | `/api/actions/poll` | Force a thread-safe immediate poll cycle (`{"asset_id": "btc_1h"}` or `{}`) |
| `WS` | `/ws/live` | Real-time push (`connected` greeting followed by `candle_update`, `prediction`, `alert`, `status`) |
| `WS` | `/ws/alerts` | Real-time push (`connected` greeting followed by `alert` events only; `403` when alerts API is disabled) |

---

## 4. Running on a 24/7 Server over LAN / Tailscale

```bash
# Bind to all interfaces (LAN + Tailscale) on port 8000 (from configs/bot.json)
./scripts/live.sh configs/bot.json

# Or bind strictly to your Tailscale IP (recommended for private remote access)
./scripts/live.sh configs/bot.json --web-host 100.x.y.z --web-port 8000
```

Then open `http://100.x.y.z:8000` from any phone, laptop, or desktop on your Tailscale network.

---

## 5. External Alert Consumer Example (Python — Telegram / WhatsApp / Email Bridge)

```python
import json, time, urllib.request

BASE = "http://127.0.0.1:8000"
cursor_id = 0

while True:
    url = f"{BASE}/api/alerts?since_id={cursor_id}&unacknowledged=true"
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = json.loads(resp.read().decode())
    for alert in data["alerts"]:
        print(f"SEND NOTIFICATION: {alert['asset_id']} {alert['action']} ({alert['confidence']:.0%})")
        # Dispatch Telegram / WhatsApp / SMTP here...
        ack_req = urllib.request.Request(f"{BASE}/api/alerts/{alert['id']}/ack", method="POST")
        urllib.request.urlopen(ack_req, timeout=5).close()
        cursor_id = max(cursor_id, alert["id"])
    time.sleep(5)
```

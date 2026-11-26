/**
 * TradingBot Web Dashboard controller.
 * Connects to /api/* and /ws/live with automatic reconnect over LAN or Tailscale.
 */
(function () {
  "use strict";

  const state = {
    config: { title: "TradingBot Dashboard", visible_candles: 70 },
    assets: [],
    activeAssetId: null,
    activeDetail: null,
    alerts: [],
    alertsUnreadOnly: true,
    ws: null,
    reconnectDelayMs: 1000,
    refreshTimer: null,
  };

  const els = {
    title: document.getElementById("dashboard-title"),
    tickerMeta: document.getElementById("active-ticker-meta"),
    livePrice: document.getElementById("active-price-display"),
    actionPill: document.getElementById("active-action-pill"),
    visibleSelect: document.getElementById("visible-select"),
    btnTogglePred: document.getElementById("btn-toggle-pred"),
    btnSnapLive: document.getElementById("btn-snap-live"),
    btnPollNow: document.getElementById("btn-poll-now"),
    wsBadge: document.getElementById("ws-badge"),
    wsLabel: document.getElementById("ws-label"),
    watchlistCount: document.getElementById("watchlist-count"),
    watchlist: document.getElementById("watchlist-container"),
    indicatorToggles: document.getElementById("indicator-toggles"),
    crosshairReadout: document.getElementById("crosshair-readout"),
    latencyBadge: document.getElementById("latency-badge"),
    analysisAction: document.getElementById("analysis-action"),
    analysisConfidence: document.getElementById("analysis-confidence"),
    metricMove: document.getElementById("metric-move"),
    metricRr: document.getElementById("metric-rr"),
    metricTp: document.getElementById("metric-tp"),
    metricSl: document.getElementById("metric-sl"),
    analysisReason: document.getElementById("analysis-reason"),
    signalLatency: document.getElementById("signal-latency"),
    barLong: document.getElementById("bar-long"),
    barHold: document.getElementById("bar-hold"),
    barShort: document.getElementById("bar-short"),
    valLong: document.getElementById("val-long"),
    valHold: document.getElementById("val-hold"),
    valShort: document.getElementById("val-short"),
    alertStreakBadge: document.getElementById("alert-streak-badge"),
    nnHealthList: document.getElementById("nn-health-list"),
    errorBanner: document.getElementById("asset-error-banner"),
    alertsList: document.getElementById("alerts-list"),
    btnAlertsUnack: document.getElementById("btn-alerts-unack"),
    btnAlertsAll: document.getElementById("btn-alerts-all"),
  };

  function fmtNum(v, decimals) {
    if (v === null || v === undefined || !Number.isFinite(v)) return "--";
    if (decimals !== undefined) return v.toFixed(decimals);
    if (Math.abs(v) >= 1000) return v.toFixed(2);
    if (Math.abs(v) >= 1) return v.toFixed(4);
    return v.toFixed(6);
  }

  function fmtPct(v) {
    if (v === null || v === undefined || !Number.isFinite(v)) return "--";
    const pct = v * 100;
    return `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
  }

  function actionClass(action) {
    if (action === "LONG") return "long";
    if (action === "SHORT") return "short";
    return "neutral";
  }

  function fmtIsoShort(ms) {
    if (!ms) return "--";
    return new Date(ms).toISOString().replace("T", " ").slice(0, 16) + "Z";
  }

  const chart = new window.TradingChart(
    document.getElementById("main-chart"),
    function onHover(info) {
      if (!info) {
        els.crosshairReadout.textContent = "Hover chart to inspect OHLCV & indicators";
        return;
      }
      const c = info.candle;
      if (info.kind === "predicted") {
        els.crosshairReadout.textContent =
          `FORECAST +${c.step} | ${fmtIsoShort(c.open_time_ms)} | ` +
          `O:${fmtNum(c.open)} H:${fmtNum(c.high)} L:${fmtNum(c.low)} C:${fmtNum(c.close)} | ` +
          `conf ${(c.confidence * 100).toFixed(0)}%`;
        return;
      }
      let txt =
        `${fmtIsoShort(c.open_time_ms)} | O:${fmtNum(c.open)} H:${fmtNum(c.high)} ` +
        `L:${fmtNum(c.low)} C:${fmtNum(c.close)} V:${fmtNum(c.volume, 2)}`;
      if (info.indicators) {
        for (const [id, cols] of Object.entries(info.indicators)) {
          const val = cols.value !== undefined ? cols.value : cols.momentum;
          if (val !== null && val !== undefined && Number.isFinite(val)) {
            txt += ` | ${id}:${fmtNum(val, 2)}`;
          }
        }
      }
      els.crosshairReadout.textContent = txt;
    }
  );

  async function apiGet(path) {
    const resp = await fetch(path, { headers: { Accept: "application/json" } });
    if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${path}`);
    return resp.json();
  }

  async function apiPost(path, body) {
    const resp = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${path}`);
    return resp.json();
  }

  function renderWatchlist() {
    els.watchlistCount.textContent = String(state.assets.length);
    if (!state.assets.length) {
      els.watchlist.innerHTML = '<div class="empty-state">No active assets</div>';
      return;
    }
    els.watchlist.innerHTML = "";
    for (const a of state.assets) {
      const card = document.createElement("div");
      card.className = "watch-item" + (a.asset_id === state.activeAssetId ? " selected" : "");
      const act = a.action || "HOLD";
      const unreadHtml = a.unread_alerts > 0
        ? `<span class="unread-pill">${a.unread_alerts}</span>`
        : "";
      const confStr = a.confidence !== null && a.confidence !== undefined
        ? `${(a.confidence * 100).toFixed(0)}%`
        : "--";

      card.innerHTML = `
        <div class="watch-row">
          <span class="watch-symbol">${a.symbol}</span>
          <span class="watch-tf">${a.timeframe} ${unreadHtml}</span>
        </div>
        <div class="watch-row">
          <span class="watch-price">${fmtNum(a.last_price)}</span>
          <span class="action-pill ${actionClass(act)}">${act}</span>
        </div>
        <div class="watch-row watch-meta">
          <span>${a.asset_id}</span>
          <span>conf ${confStr}</span>
        </div>
      `;
      card.addEventListener("click", () => selectAsset(a.asset_id));
      els.watchlist.appendChild(card);
    }
  }

  function renderIndicatorChips(detail) {
    els.indicatorToggles.innerHTML = "";
    const plots = detail.indicator_plots || {};
    for (const [id, mode] of Object.entries(plots)) {
      if (mode === "none") continue;
      const active = !chart.hiddenIndicators.has(id);
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "ind-chip" + (active ? " active" : "");
      const swatch = chart.getIndicatorColor(id);
      chip.innerHTML = `<span class="ind-swatch" style="background:${swatch}"></span><span>${id}</span>`;
      chip.addEventListener("click", () => {
        chart.toggleIndicator(id);
        renderIndicatorChips(detail);
      });
      els.indicatorToggles.appendChild(chip);
    }
  }

  function renderDetail(detail) {
    state.activeDetail = detail;
    const candles = detail.candles || [];
    const lastBar = candles.length ? candles[candles.length - 1] : null;
    els.tickerMeta.textContent = `${detail.asset_id} · ${detail.provider} · ${detail.symbol} (${detail.timeframe}) · ${detail.mode}`;
    els.livePrice.textContent = lastBar ? fmtNum(lastBar.close) : "--";

    const pred = detail.prediction;
    const analysis = pred && pred.analysis ? pred.analysis : null;
    const action = analysis ? analysis.action : "HOLD";
    const cls = actionClass(action);

    els.actionPill.className = `action-pill ${cls}`;
    els.actionPill.textContent = action;

    els.analysisAction.className = `analysis-action ${cls}`;
    els.analysisAction.textContent = action;
    els.analysisConfidence.textContent = analysis ? `${(analysis.confidence * 100).toFixed(1)}%` : "--";
    els.metricMove.textContent = analysis ? fmtPct(analysis.expected_move_pct) : "--";
    els.metricMove.className = "metric-value " + (analysis && analysis.expected_move_pct > 0 ? "bullish" : (analysis && analysis.expected_move_pct < 0 ? "bearish" : ""));
    els.metricRr.textContent = analysis && analysis.risk_reward !== null ? `${analysis.risk_reward.toFixed(2)}x` : "--";
    els.metricTp.textContent = analysis && analysis.estimated_tp !== null ? fmtNum(analysis.estimated_tp) : "--";
    els.metricSl.textContent = analysis && analysis.estimated_sl !== null ? fmtNum(analysis.estimated_sl) : "--";
    els.analysisReason.textContent = analysis ? analysis.reason : "No prediction yet (awaiting warm-up or NN server).";
    els.latencyBadge.textContent = pred && pred.candle_latency_us !== null ? `${pred.candle_latency_us} \u00b5s` : "-- \u00b5s";

    // Signal probabilities
    const sig = pred && pred.signal ? pred.signal : null;
    if (sig) {
      const lp = sig.long_prob * 100;
      const hp = sig.hold_prob * 100;
      const sp = sig.short_prob * 100;
      els.barLong.style.width = `${lp.toFixed(1)}%`;
      els.barHold.style.width = `${hp.toFixed(1)}%`;
      els.barShort.style.width = `${sp.toFixed(1)}%`;
      els.valLong.textContent = `${lp.toFixed(1)}%`;
      els.valHold.textContent = `${hp.toFixed(1)}%`;
      els.valShort.textContent = `${sp.toFixed(1)}%`;
      els.signalLatency.textContent = sig.latency_us !== null ? `${sig.latency_us} \u00b5s` : "Softmax";
    } else {
      els.barLong.style.width = "0%";
      els.barHold.style.width = "0%";
      els.barShort.style.width = "0%";
      els.valLong.textContent = "--";
      els.valHold.textContent = "--";
      els.valShort.textContent = "--";
      els.signalLatency.textContent = "Disabled";
    }

    // NN Health & Alert Streak
    const st = detail.alert_status || {};
    els.alertStreakBadge.textContent = st.enabled
      ? `streak ${st.streak_count || 0}/${st.required_consecutive || 1}`
      : "alerts off";

    const health = detail.nn_health || {};
    const healthEntries = Object.entries(health);
    if (!healthEntries.length) {
      els.nnHealthList.innerHTML = '<div class="empty-state-sm">Charting-only (no NN enabled)</div>';
    } else {
      els.nnHealthList.innerHTML = healthEntries
        .map(([name, ok]) => `
          <div class="health-row">
            <span><span class="status-dot ${ok ? "ok" : "err"}"></span>${name}</span>
            <span class="${ok ? "bullish" : "bearish"}">${ok ? "ONLINE" : "OFFLINE"}</span>
          </div>
        `)
        .join("");
    }

    if (detail.last_error) {
      els.errorBanner.textContent = detail.last_error;
      els.errorBanner.classList.remove("hidden");
    } else {
      els.errorBanner.classList.add("hidden");
    }

    renderIndicatorChips(detail);
    chart.setData(detail);
  }

  function renderAlerts() {
    if (!state.alerts.length) {
      els.alertsList.innerHTML = '<div class="empty-state-sm">No alerts to display</div>';
      return;
    }
    els.alertsList.innerHTML = "";
    for (const a of state.alerts) {
      const item = document.createElement("div");
      const cls = actionClass(a.action);
      item.className = `alert-item ${cls}` + (a.acknowledged ? " acked" : "");
      const rr = a.payload && a.payload.risk_reward ? ` · R:R ${a.payload.risk_reward.toFixed(2)}` : "";
      const ackBtn = a.acknowledged
        ? '<span class="badge-muted">Acked</span>'
        : `<button class="mini-btn" data-ack="${a.id}" type="button">Ack</button>`;

      item.innerHTML = `
        <div class="alert-top">
          <span><strong>${a.asset_id}</strong> <span class="action-pill ${cls}">${a.action}</span></span>
          ${ackBtn}
        </div>
        <div class="alert-meta">
          #${a.id} · conf ${(a.confidence * 100).toFixed(0)}% · streak ${a.consecutive_hits}${rr}
        </div>
        <div class="alert-meta">${fmtIsoShort(a.anchor_open_time_ms)}</div>
      `;
      const btn = item.querySelector("[data-ack]");
      if (btn) {
        btn.addEventListener("click", async () => {
          try {
            await apiPost(`/api/alerts/${a.id}/ack`, {});
            await Promise.all([refreshAlerts(), refreshAssets()]);
          } catch (err) {
            console.error(err);
          }
        });
      }
      els.alertsList.appendChild(item);
    }
  }

  async function refreshAssets() {
    const data = await apiGet("/api/assets");
    state.assets = data.assets || [];
    if (!state.activeAssetId && state.assets.length > 0) {
      state.activeAssetId = state.assets[0].asset_id;
    }
    renderWatchlist();
  }

  async function refreshActiveDetail() {
    if (!state.activeAssetId) return;
    const reqBars = Math.max(chart.visibleCount + 40, 120);
    const detail = await apiGet(
      `/api/assets/${encodeURIComponent(state.activeAssetId)}/latest?visible=${reqBars}`
    );
    renderDetail(detail);
  }

  async function refreshAlerts() {
    const qs = state.alertsUnreadOnly ? "?unacknowledged=true&limit=50" : "?limit=50";
    try {
      const data = await apiGet(`/api/alerts${qs}`);
      state.alerts = data.alerts || [];
      renderAlerts();
    } catch (_err) {
      // Alerts API may be disabled in config
    }
  }

  async function selectAsset(assetId) {
    state.activeAssetId = assetId;
    chart.snapLive();
    renderWatchlist();
    await refreshActiveDetail();
  }

  function setWsBadge(mode, text) {
    els.wsBadge.className = `ws-badge ${mode}`;
    els.wsLabel.textContent = text;
  }

  function connectWebSocket() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/ws/live`;
    setWsBadge("connecting", "CONNECTING");
    const ws = new WebSocket(url);
    state.ws = ws;

    ws.onopen = () => {
      state.reconnectDelayMs = 1000;
      setWsBadge("connected", "LIVE");
    };

    ws.onmessage = (evt) => {
      let msg;
      try { msg = JSON.parse(evt.data); } catch (_e) { return; }
      if (msg.type === "connected") return;

      // Refresh affected views cleanly on live push events
      if (msg.type === "candle_update" || msg.type === "prediction" || msg.type === "status") {
        refreshAssets().catch(() => {});
        if (msg.asset_id === state.activeAssetId) {
          refreshActiveDetail().catch(() => {});
        }
      } else if (msg.type === "alert") {
        refreshAlerts().catch(() => {});
        refreshAssets().catch(() => {});
      }
    };

    ws.onclose = () => {
      setWsBadge("disconnected", "RECONNECTING");
      const delay = state.reconnectDelayMs;
      state.reconnectDelayMs = Math.min(10000, delay * 1.5);
      setTimeout(connectWebSocket, delay);
    };

    ws.onerror = () => {
      try { ws.close(); } catch (_e) {}
    };
  }

  function bindControls() {
    els.visibleSelect.addEventListener("change", async () => {
      const val = parseInt(els.visibleSelect.value, 10) || 70;
      chart.setVisibleCount(val);
      await refreshActiveDetail();
    });

    els.btnTogglePred.addEventListener("click", () => {
      const next = !chart.showPredicted;
      chart.setShowPredicted(next);
      els.btnTogglePred.textContent = next ? "Forecast: ON" : "Forecast: OFF";
      els.btnTogglePred.className = next ? "btn btn-active" : "btn";
    });

    els.btnSnapLive.addEventListener("click", () => chart.snapLive());

    els.btnPollNow.addEventListener("click", async () => {
      if (!state.activeAssetId) return;
      els.btnPollNow.disabled = true;
      try {
        await apiPost("/api/actions/poll", { asset_id: state.activeAssetId });
        await Promise.all([refreshAssets(), refreshActiveDetail(), refreshAlerts()]);
      } catch (err) {
        console.error(err);
      } finally {
        els.btnPollNow.disabled = false;
      }
    });

    els.btnAlertsUnack.addEventListener("click", async () => {
      state.alertsUnreadOnly = true;
      els.btnAlertsUnack.classList.add("active");
      els.btnAlertsAll.classList.remove("active");
      await refreshAlerts();
    });

    els.btnAlertsAll.addEventListener("click", async () => {
      state.alertsUnreadOnly = false;
      els.btnAlertsAll.classList.add("active");
      els.btnAlertsUnack.classList.remove("active");
      await refreshAlerts();
    });
  }

  async function init() {
    bindControls();
    try {
      const cfg = await apiGet("/api/config");
      state.config = cfg;
      if (cfg.title) {
        els.title.textContent = cfg.title;
        document.title = cfg.title;
      }
      if (cfg.visible_candles) {
        chart.setVisibleCount(cfg.visible_candles);
        const opt = Array.from(els.visibleSelect.options).find(
          (o) => parseInt(o.value, 10) === cfg.visible_candles
        );
        if (opt) els.visibleSelect.value = String(cfg.visible_candles);
      }
    } catch (err) {
      console.error("Failed to load config:", err);
    }

    await refreshAssets();
    await Promise.all([refreshActiveDetail(), refreshAlerts()]);
    connectWebSocket();

    // Periodic fallback sync in case a proxy buffers WebSocket traffic
    state.refreshTimer = setInterval(() => {
      refreshAssets().catch(() => {});
      refreshActiveDetail().catch(() => {});
    }, 30000);
  }

  init().catch(console.error);
})();

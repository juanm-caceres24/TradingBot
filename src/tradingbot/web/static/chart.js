/**
 * Zero-dependency HiDPI HTML5 Canvas financial chart engine for TradingBot.
 *
 * Renders synchronized panes:
 *  1. Main Price Pane: real OHLC candles, indicator overlays (SMA/EMA/Bollinger),
 *     vertical forecast divider, translucent predicted candles with decaying alpha,
 *     and estimated TP/SL lines.
 *  2. Volume Pane: aligned volume histogram.
 *  3. Dynamic Subchart Panes: RSI (30/70 guides) and LazyBear Squeeze Momentum
 *     (4-color momentum histogram + zero-line squeeze state dots).
 */
(function () {
  "use strict";

  const OVERLAY_PALETTE = [
    "#38bdf8",
    "#f59e0b",
    "#a855f7",
    "#ec4899",
    "#14b8a6",
    "#eab308",
  ];

  function fmtPrice(v) {
    if (v === null || v === undefined || !Number.isFinite(v)) return "--";
    if (Math.abs(v) >= 1000) return v.toFixed(2);
    if (Math.abs(v) >= 1) return v.toFixed(4);
    return v.toFixed(6);
  }

  function fmtTime(ms) {
    const d = new Date(ms);
    const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
    const dd = String(d.getUTCDate()).padStart(2, "0");
    const hh = String(d.getUTCHours()).padStart(2, "0");
    const mi = String(d.getUTCMinutes()).padStart(2, "0");
    return `${mm}-${dd} ${hh}:${mi}`;
  }

  class TradingChart {
    constructor(canvas, onHoverCallback, onZoomCallback) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.onHover = onHoverCallback || function () {};
      this.onZoom = onZoomCallback || function () {};

      this.candles = [];
      this.indicators = {};
      this.indicatorPlots = {};
      this.indicatorTypes = {};
      this.hiddenIndicators = new Set();
      this.predictedCandles = [];
      this.analysis = null;
      this.showPredicted = true;

      this.visibleCount = 70;
      this.panOffset = 0; // 0 = pinned live to rightmost bar
      this.hoverIndex = null;
      this.hoverY = null;

      this._dragging = false;
      this._dragStartX = 0;
      this._dragStartPan = 0;
      this._width = 800;
      this._height = 500;
      this._colorMap = {};

      this._bindEvents();
      this.resize();
    }

    getIndicatorColor(id) {
      if (!this._colorMap[id]) {
        const idx = Object.keys(this._colorMap).length % OVERLAY_PALETTE.length;
        this._colorMap[id] = OVERLAY_PALETTE[idx];
      }
      return this._colorMap[id];
    }

    setData(detail) {
      this.candles = detail.candles || [];
      this.indicators = detail.indicators || {};
      this.indicatorPlots = detail.indicator_plots || {};
      this.indicatorTypes = detail.indicator_types || {};
      const pred = detail.prediction || null;
      this.predictedCandles = pred && pred.predicted_candles ? pred.predicted_candles : [];
      this.analysis = pred && pred.analysis ? pred.analysis : null;
      this._clampPan();
      this.render();
    }

    setVisibleCount(count, notifyZoom) {
      const next = Math.max(15, Math.min(250, count));
      const changed = next !== this.visibleCount;
      this.visibleCount = next;
      this._clampPan();
      this.render();
      if (changed && notifyZoom) {
        this.onZoom(this.visibleCount);
      }
    }

    setShowPredicted(flag) {
      this.showPredicted = Boolean(flag);
      this.render();
    }

    toggleIndicator(id) {
      if (this.hiddenIndicators.has(id)) {
        this.hiddenIndicators.delete(id);
      } else {
        this.hiddenIndicators.add(id);
      }
      this.render();
    }

    snapLive() {
      this.panOffset = 0;
      this.render();
    }

    resize() {
      const parent = this.canvas.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this._width = Math.max(300, Math.floor(rect.width));
      this._height = Math.max(320, Math.floor(rect.height));
      this.canvas.width = Math.floor(this._width * dpr);
      this.canvas.height = Math.floor(this._height * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.render();
    }

    _clampPan() {
      const maxPan = Math.max(0, this.candles.length - 10);
      if (this.panOffset < 0) this.panOffset = 0;
      if (this.panOffset > maxPan) this.panOffset = maxPan;
    }

    _bindEvents() {
      if (typeof ResizeObserver !== "undefined" && this.canvas.parentElement) {
        const ro = new ResizeObserver(() => this.resize());
        ro.observe(this.canvas.parentElement);
      } else {
        window.addEventListener("resize", () => this.resize());
      }

      this.canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const step = e.deltaY > 0 ? 8 : -8;
        this.setVisibleCount(this.visibleCount + step, true);
      }, { passive: false });

      this.canvas.addEventListener("mousedown", (e) => {
        this._dragging = true;
        this._dragStartX = e.clientX;
        this._dragStartPan = this.panOffset;
      });

      window.addEventListener("mouseup", () => {
        this._dragging = false;
      });

      this.canvas.addEventListener("mousemove", (e) => {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (this._dragging) {
          const dx = e.clientX - this._dragStartX;
          const plotW = Math.max(100, this._width - 80);
          const totalSlots = this.visibleCount + (this.showPredicted ? this.predictedCandles.length : 0);
          const slotW = plotW / Math.max(1, totalSlots);
          const barDelta = Math.round(dx / Math.max(2, slotW));
          this.panOffset = this._dragStartPan + barDelta;
          this._clampPan();
          this.render();
          return;
        }

        this.hoverY = y;
        this._updateHoverFromX(x);
        this.render();
      });

      this.canvas.addEventListener("mouseleave", () => {
        this.hoverIndex = null;
        this.hoverY = null;
        this.onHover(null);
        this.render();
      });
    }

    _computeWindow() {
      const n = this.candles.length;
      const predBars = this.showPredicted ? this.predictedCandles : [];
      const endIdx = Math.max(0, n - this.panOffset);
      const startIdx = Math.max(0, endIdx - this.visibleCount);
      const realSlice = this.candles.slice(startIdx, endIdx);
      const showPredNow = this.showPredicted && this.panOffset === 0 && predBars.length > 0;
      const totalSlots = Math.max(1, realSlice.length + (showPredNow ? predBars.length : 0));
      return { startIdx, endIdx, realSlice, predBars: showPredNow ? predBars : [], totalSlots };
    }

    _updateHoverFromX(x) {
      const padLeft = 10;
      const padRight = 72;
      const plotW = this._width - padLeft - padRight;
      if (x < padLeft || x > padLeft + plotW) {
        this.hoverIndex = null;
        this.onHover(null);
        return;
      }
      const win = this._computeWindow();
      const slotW = plotW / win.totalSlots;
      const slotIdx = Math.max(0, Math.min(win.totalSlots - 1, Math.floor((x - padLeft) / slotW)));
      this.hoverIndex = slotIdx;

      if (slotIdx < win.realSlice.length) {
        const absIdx = win.startIdx + slotIdx;
        const c = win.realSlice[slotIdx];
        const indSnap = {};
        for (const [id, cols] of Object.entries(this.indicators)) {
          if (this.hiddenIndicators.has(id)) continue;
          indSnap[id] = {};
          for (const [col, arr] of Object.entries(cols)) {
            indSnap[id][col] = arr[absIdx];
          }
        }
        this.onHover({ kind: "real", candle: c, indicators: indSnap });
      } else {
        const pIdx = slotIdx - win.realSlice.length;
        const pc = win.predBars[pIdx];
        if (pc) {
          this.onHover({ kind: "predicted", candle: pc });
        }
      }
    }

    _activeSubcharts() {
      const subs = [];
      for (const [id, mode] of Object.entries(this.indicatorPlots)) {
        if (mode === "subchart" && !this.hiddenIndicators.has(id) && this.indicators[id]) {
          subs.push(id);
        }
      }
      return subs;
    }

    render() {
      const ctx = this.ctx;
      const W = this._width;
      const H = this._height;

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#090d16";
      ctx.fillRect(0, 0, W, H);

      if (!this.candles.length) {
        ctx.fillStyle = "#5d708f";
        ctx.font = "13px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Waiting for market candles...", W / 2, H / 2);
        return;
      }

      const padLeft = 10;
      const padRight = 72;
      const padTop = 10;
      const axisBottomH = 22;
      const plotW = Math.max(80, W - padLeft - padRight);
      const usableH = Math.max(200, H - padTop - axisBottomH);

      const subcharts = this._activeSubcharts();
      const volWeight = 0.14;
      const subWeight = subcharts.length > 0 ? Math.min(0.16, 0.34 / subcharts.length) : 0;
      const priceWeight = 1.0 - volWeight - subWeight * subcharts.length;

      const panes = [];
      let curY = padTop;
      const priceH = Math.floor(usableH * priceWeight);
      panes.push({ kind: "price", y: curY, h: priceH });
      curY += priceH;

      const volH = Math.floor(usableH * volWeight);
      panes.push({ kind: "volume", y: curY, h: volH });
      curY += volH;

      for (const subId of subcharts) {
        const sh = Math.floor(usableH * subWeight);
        panes.push({ kind: "subchart", id: subId, y: curY, h: sh });
        curY += sh;
      }

      const win = this._computeWindow();
      const slotW = plotW / win.totalSlots;
      const barW = Math.max(2, Math.min(18, Math.floor(slotW * 0.68)));
      const xCenter = (slotIdx) => padLeft + (slotIdx + 0.5) * slotW;

      // 1) Draw Main Price Pane
      const pricePane = panes[0];
      this._drawPricePane(ctx, pricePane, padLeft, plotW, barW, xCenter, win);

      // 2) Draw Volume Pane
      const volPane = panes[1];
      this._drawVolumePane(ctx, volPane, padLeft, plotW, barW, xCenter, win);

      // 3) Draw Indicator Subchart Panes
      for (let i = 2; i < panes.length; i++) {
        this._drawSubchartPane(ctx, panes[i], padLeft, plotW, barW, xCenter, win);
      }

      // 4) Bottom Time Axis
      this._drawTimeAxis(ctx, padLeft, plotW, H - axisBottomH, axisBottomH, xCenter, win);

      // 5) Crosshair
      if (this.hoverIndex !== null && this.hoverIndex < win.totalSlots) {
        const cx = xCenter(this.hoverIndex);
        ctx.save();
        ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx, padTop);
        ctx.lineTo(cx, H - axisBottomH);
        ctx.stroke();
        if (this.hoverY !== null && this.hoverY >= padTop && this.hoverY <= H - axisBottomH) {
          ctx.beginPath();
          ctx.moveTo(padLeft, this.hoverY);
          ctx.lineTo(padLeft + plotW, this.hoverY);
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    _drawPricePane(ctx, pane, padLeft, plotW, barW, xCenter, win) {
      let minP = Infinity;
      let maxP = -Infinity;

      for (const c of win.realSlice) {
        if (c.low < minP) minP = c.low;
        if (c.high > maxP) maxP = c.high;
      }
      for (const pc of win.predBars) {
        if (pc.low < minP) minP = pc.low;
        if (pc.high > maxP) maxP = pc.high;
      }
      if (this.showPredicted && this.analysis && this.panOffset === 0) {
        const tp = this.analysis.estimated_tp;
        const sl = this.analysis.estimated_sl;
        if (tp !== null && tp !== undefined && Number.isFinite(tp)) {
          if (tp < minP) minP = tp;
          if (tp > maxP) maxP = tp;
        }
        if (sl !== null && sl !== undefined && Number.isFinite(sl)) {
          if (sl < minP) minP = sl;
          if (sl > maxP) maxP = sl;
        }
      }
      // Include visible overlays in Y-axis range
      for (const [id, mode] of Object.entries(this.indicatorPlots)) {
        if (mode !== "overlay" || this.hiddenIndicators.has(id) || !this.indicators[id]) continue;
        for (const arr of Object.values(this.indicators[id])) {
          for (let i = win.startIdx; i < win.endIdx; i++) {
            const v = arr[i];
            if (v !== null && v !== undefined && Number.isFinite(v)) {
              if (v < minP) minP = v;
              if (v > maxP) maxP = v;
            }
          }
        }
      }
      if (!Number.isFinite(minP) || !Number.isFinite(maxP)) return;
      const span = Math.max(maxP - minP, maxP * 0.002, 1e-6);
      minP -= span * 0.07;
      maxP += span * 0.07;
      const yOf = (p) => pane.y + pane.h - ((p - minP) / (maxP - minP)) * pane.h;

      // Horizontal Grid & Right Price Axis
      const ticks = 5;
      ctx.font = "10px ui-monospace, monospace";
      ctx.textAlign = "left";
      for (let t = 0; t <= ticks; t++) {
        const frac = t / ticks;
        const val = maxP - frac * (maxP - minP);
        const yy = Math.round(pane.y + frac * pane.h) + 0.5;
        ctx.strokeStyle = "rgba(36, 50, 77, 0.45)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padLeft, yy);
        ctx.lineTo(padLeft + plotW, yy);
        ctx.stroke();

        ctx.fillStyle = "#8ea0bc";
        ctx.fillText(fmtPrice(val), padLeft + plotW + 6, yy + 3);
      }

      // Forecast Background Shading & Divider Line
      if (win.predBars.length > 0 && win.realSlice.length > 0) {
        const divX = (xCenter(win.realSlice.length - 1) + xCenter(win.realSlice.length)) / 2;
        ctx.fillStyle = "rgba(56, 189, 248, 0.045)";
        ctx.fillRect(divX, pane.y, padLeft + plotW - divX, pane.h);

        ctx.save();
        ctx.strokeStyle = "rgba(56, 189, 248, 0.5)";
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(divX, pane.y);
        ctx.lineTo(divX, pane.y + pane.h);
        ctx.stroke();
        ctx.restore();

        ctx.fillStyle = "rgba(56, 189, 248, 0.85)";
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText("FORECAST", divX + 6, pane.y + 13);
      }

      // Estimated TP / SL Horizontal Reference Lines (works in dual, candle-only, and signal-only modes)
      if (this.showPredicted && this.analysis && this.panOffset === 0 && win.realSlice.length > 0) {
        const startLineX = xCenter(
          win.predBars.length > 0
            ? Math.max(0, win.realSlice.length - 1)
            : Math.max(0, win.realSlice.length - Math.min(8, win.realSlice.length))
        );
        const drawLevel = (price, color, label) => {
          if (price === null || price === undefined || !Number.isFinite(price)) return;
          if (price < minP || price > maxP) return;
          const ly = yOf(price);
          ctx.save();
          ctx.strokeStyle = color;
          ctx.setLineDash([5, 4]);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(startLineX, ly);
          ctx.lineTo(padLeft + plotW, ly);
          ctx.stroke();
          ctx.fillStyle = color;
          ctx.font = "10px ui-monospace, monospace";
          ctx.fillText(`${label} ${fmtPrice(price)}`, startLineX + 4, ly - 3);
          ctx.restore();
        };
        drawLevel(this.analysis.estimated_tp, "rgba(34, 197, 94, 0.8)", "TP");
        drawLevel(this.analysis.estimated_sl, "rgba(239, 68, 68, 0.8)", "SL");
      }

      // Overlay Indicators (SMA, EMA, Bollinger Bands)
      for (const [id, mode] of Object.entries(this.indicatorPlots)) {
        if (mode !== "overlay" || this.hiddenIndicators.has(id) || !this.indicators[id]) continue;
        const cols = this.indicators[id];
        const color = this.getIndicatorColor(id);

        // Shaded Bollinger channel if upper + lower exist
        if (cols.upper && cols.lower) {
          ctx.save();
          ctx.fillStyle = "rgba(56, 189, 248, 0.07)";
          ctx.beginPath();
          let started = false;
          const lowerPts = [];
          for (let s = 0; s < win.realSlice.length; s++) {
            const u = cols.upper[win.startIdx + s];
            const l = cols.lower[win.startIdx + s];
            if (u !== null && l !== null && Number.isFinite(u) && Number.isFinite(l)) {
              const cx = xCenter(s);
              if (!started) { ctx.moveTo(cx, yOf(u)); started = true; }
              else { ctx.lineTo(cx, yOf(u)); }
              lowerPts.push([cx, yOf(l)]);
            }
          }
          for (let j = lowerPts.length - 1; j >= 0; j--) {
            ctx.lineTo(lowerPts[j][0], lowerPts[j][1]);
          }
          if (started) {
            ctx.closePath();
            ctx.fill();
          }
          ctx.restore();
        }

        for (const [colName, arr] of Object.entries(cols)) {
          ctx.save();
          ctx.strokeStyle = color;
          ctx.lineWidth = colName === "middle" || colName === "value" ? 1.5 : 1.0;
          if (colName === "upper" || colName === "lower") ctx.setLineDash([3, 3]);
          ctx.beginPath();
          let penDown = false;
          for (let s = 0; s < win.realSlice.length; s++) {
            const v = arr[win.startIdx + s];
            if (v === null || v === undefined || !Number.isFinite(v)) {
              penDown = false;
              continue;
            }
            const cx = xCenter(s);
            const cy = yOf(v);
            if (!penDown) { ctx.moveTo(cx, cy); penDown = true; }
            else { ctx.lineTo(cx, cy); }
          }
          ctx.stroke();
          ctx.restore();
        }
      }

      // Real Candles
      for (let s = 0; s < win.realSlice.length; s++) {
        const c = win.realSlice[s];
        const bull = c.close >= c.open;
        const color = bull ? "#22c55e" : "#ef4444";
        const cx = xCenter(s);
        const yO = yOf(c.open);
        const yC = yOf(c.close);
        const yH = yOf(c.high);
        const yL = yOf(c.low);

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(cx, yH);
        ctx.lineTo(cx, yL);
        ctx.stroke();

        const top = Math.min(yO, yC);
        const bodyH = Math.max(1.5, Math.abs(yC - yO));
        ctx.fillStyle = c.is_closed ? color : (bull ? "rgba(34,197,94,0.65)" : "rgba(239,68,68,0.65)");
        ctx.fillRect(cx - barW / 2, top, barW, bodyH);
      }

      // Translucent Predicted Candles (alpha proportional to step confidence)
      for (let p = 0; p < win.predBars.length; p++) {
        const pc = win.predBars[p];
        const slotIdx = win.realSlice.length + p;
        const bull = pc.close >= pc.open;
        const conf = Number.isFinite(pc.confidence) ? pc.confidence : 0.6;
        const fillAlpha = Math.max(0.18, Math.min(0.55, conf * 0.55));
        const strokeAlpha = Math.max(0.45, Math.min(0.95, conf * 0.95));

        const rgb = bull ? "34, 197, 94" : "239, 68, 68";
        const cx = xCenter(slotIdx);
        const yO = yOf(pc.open);
        const yC = yOf(pc.close);
        const yH = yOf(pc.high);
        const yL = yOf(pc.low);

        ctx.save();
        ctx.strokeStyle = `rgba(${rgb}, ${strokeAlpha.toFixed(2)})`;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(cx, yH);
        ctx.lineTo(cx, yL);
        ctx.stroke();

        const top = Math.min(yO, yC);
        const bodyH = Math.max(2, Math.abs(yC - yO));
        ctx.fillStyle = `rgba(${rgb}, ${fillAlpha.toFixed(2)})`;
        ctx.fillRect(cx - barW / 2, top, barW, bodyH);
        ctx.setLineDash([2, 2]);
        ctx.strokeRect(cx - barW / 2 + 0.5, top + 0.5, Math.max(1, barW - 1), Math.max(1, bodyH - 1));
        ctx.restore();
      }
    }

    _drawVolumePane(ctx, pane, padLeft, plotW, barW, xCenter, win) {
      ctx.strokeStyle = "#24324d";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padLeft, pane.y + 0.5);
      ctx.lineTo(padLeft + plotW, pane.y + 0.5);
      ctx.stroke();

      let maxV = 0;
      for (const c of win.realSlice) {
        if (c.volume > maxV) maxV = c.volume;
      }
      if (maxV <= 0) maxV = 1;

      const usableH = Math.max(8, pane.h - 14);
      for (let s = 0; s < win.realSlice.length; s++) {
        const c = win.realSlice[s];
        const bull = c.close >= c.open;
        const vh = (c.volume / maxV) * usableH;
        const cx = xCenter(s);
        ctx.fillStyle = bull ? "rgba(34, 197, 94, 0.35)" : "rgba(239, 68, 68, 0.35)";
        ctx.fillRect(cx - barW / 2, pane.y + pane.h - vh, barW, vh);
      }

      ctx.fillStyle = "#5d708f";
      ctx.font = "10px ui-monospace, monospace";
      ctx.textAlign = "left";
      ctx.fillText("VOL", padLeft + 4, pane.y + 12);
      ctx.fillText(maxV.toFixed(1), padLeft + plotW + 6, pane.y + 14);
    }

    _drawSubchartPane(ctx, pane, padLeft, plotW, barW, xCenter, win) {
      const id = pane.id;
      const cols = this.indicators[id];
      if (!cols) return;

      ctx.strokeStyle = "#24324d";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padLeft, pane.y + 0.5);
      ctx.lineTo(padLeft + plotW, pane.y + 0.5);
      ctx.stroke();

      ctx.fillStyle = "#8ea0bc";
      ctx.font = "10px ui-monospace, monospace";
      ctx.textAlign = "left";
      ctx.fillText(id.toUpperCase(), padLeft + 4, pane.y + 12);

      const indType = (this.indicatorTypes[id] || "").toLowerCase();

      // Authentic LazyBear Squeeze Momentum renderer
      if (indType === "squeeze_momentum" || (cols.momentum && cols.squeeze_state)) {
        let maxAbs = 0;
        for (let i = win.startIdx; i < win.endIdx; i++) {
          const v = cols.momentum[i];
          if (v !== null && v !== undefined && Number.isFinite(v)) {
            maxAbs = Math.max(maxAbs, Math.abs(v));
          }
        }
        if (maxAbs <= 0) maxAbs = 1;
        const midY = pane.y + pane.h / 2 + 3;
        const halfH = Math.max(8, pane.h / 2 - 12);

        // Zero line
        ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
        ctx.beginPath();
        ctx.moveTo(padLeft, midY);
        ctx.lineTo(padLeft + plotW, midY);
        ctx.stroke();

        for (let s = 0; s < win.realSlice.length; s++) {
          const absIdx = win.startIdx + s;
          const v = cols.momentum[absIdx];
          const prev = absIdx > 0 ? cols.momentum[absIdx - 1] : null;
          const sqz = cols.squeeze_state[absIdx];
          const cx = xCenter(s);

          if (v !== null && v !== undefined && Number.isFinite(v)) {
            const rising = prev === null || prev === undefined ? v >= 0 : v > prev;
            let barColor;
            if (v >= 0) barColor = rising ? "#22c55e" : "#15803d";
            else barColor = !rising ? "#ef4444" : "#991b1b";

            const bh = (Math.abs(v) / maxAbs) * halfH;
            ctx.fillStyle = barColor;
            if (v >= 0) ctx.fillRect(cx - barW / 2, midY - bh, barW, bh);
            else ctx.fillRect(cx - barW / 2, midY, barW, bh);
          }

          if (sqz !== null && sqz !== undefined && Number.isFinite(sqz)) {
            ctx.fillStyle = sqz > 0.5 ? "#ef4444" : (sqz < -0.5 ? "#22c55e" : "#64748b");
            ctx.beginPath();
            ctx.arc(cx, midY, 2.2, 0, Math.PI * 2);
            ctx.fill();
          }
        }
        return;
      }

      // Generic line subchart (RSI with 70/30 guides or any custom scalar subchart with min/max ticks)
      const isRsi = indType === "rsi" || (!indType && id.toLowerCase().includes("rsi"));
      let minV = isRsi ? 0 : Infinity;
      let maxV = isRsi ? 100 : -Infinity;
      if (!isRsi) {
        for (const arr of Object.values(cols)) {
          for (let i = win.startIdx; i < win.endIdx; i++) {
            const v = arr[i];
            if (v !== null && v !== undefined && Number.isFinite(v)) {
              if (v < minV) minV = v;
              if (v > maxV) maxV = v;
            }
          }
        }
        if (!Number.isFinite(minV)) { minV = 0; maxV = 1; }
        const sp = Math.max(maxV - minV, 1e-6);
        minV -= sp * 0.1;
        maxV += sp * 0.1;
      }

      const topPad = 14;
      const botPad = 6;
      const innerH = Math.max(10, pane.h - topPad - botPad);
      const yOf = (v) => pane.y + topPad + innerH - ((v - minV) / (maxV - minV)) * innerH;

      if (isRsi) {
        for (const lvl of [70, 30]) {
          const ly = yOf(lvl);
          ctx.save();
          ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(padLeft, ly);
          ctx.lineTo(padLeft + plotW, ly);
          ctx.stroke();
          ctx.fillStyle = "#5d708f";
          ctx.fillText(String(lvl), padLeft + plotW + 6, ly + 3);
          ctx.restore();
        }
      } else {
        ctx.fillStyle = "#5d708f";
        ctx.fillText(fmtPrice(maxV), padLeft + plotW + 6, pane.y + topPad + 4);
        ctx.fillText(fmtPrice(minV), padLeft + plotW + 6, pane.y + pane.h - 4);
      }

      const color = this.getIndicatorColor(id);
      for (const arr of Object.values(cols)) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        let penDown = false;
        for (let s = 0; s < win.realSlice.length; s++) {
          const v = arr[win.startIdx + s];
          if (v === null || v === undefined || !Number.isFinite(v)) {
            penDown = false;
            continue;
          }
          const cx = xCenter(s);
          const cy = yOf(v);
          if (!penDown) { ctx.moveTo(cx, cy); penDown = true; }
          else { ctx.lineTo(cx, cy); }
        }
        ctx.stroke();
        ctx.restore();
      }
    }

    _drawTimeAxis(ctx, padLeft, plotW, yTop, axisH, xCenter, win) {
      ctx.strokeStyle = "#24324d";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padLeft, yTop + 0.5);
      ctx.lineTo(padLeft + plotW, yTop + 0.5);
      ctx.stroke();

      const maxLabels = Math.max(2, Math.floor(plotW / 110));
      const step = Math.max(1, Math.ceil(win.totalSlots / maxLabels));
      ctx.fillStyle = "#8ea0bc";
      ctx.font = "10px ui-monospace, monospace";
      ctx.textAlign = "center";

      for (let s = 0; s < win.totalSlots; s += step) {
        let ts = null;
        if (s < win.realSlice.length) ts = win.realSlice[s].open_time_ms;
        else {
          const pc = win.predBars[s - win.realSlice.length];
          if (pc) ts = pc.open_time_ms;
        }
        if (ts === null) continue;
        ctx.fillText(fmtTime(ts), xCenter(s), yTop + 14);
      }
    }
  }

  window.TradingChart = TradingChart;
})();

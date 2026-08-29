/* Canvas chart primitives.
 *
 * No dependencies on purpose: this page has to load on a GPU box with no network,
 * behind a CSP that forbids remote scripts. Colours are read from the stylesheet's
 * CSS variables, so the charts follow the light/dark palette without a second
 * definition of it here.
 *
 * Every chart keeps its full series in memory and redraws on demand. A long
 * training run pushes thousands of points, so the line renderer decimates to at
 * most one segment per device pixel column -- keeping the min and the max of each
 * column, which preserves spikes that plain sampling would drop.
 */
(function (global) {
  "use strict";

  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (v || "").trim() || fallback;
  }

  const PALETTE = () => ({
    text: cssVar("--text", "#e6e9ef"),
    dim: cssVar("--text-dim", "#9aa4b2"),
    faint: cssVar("--text-faint", "#6b7482"),
    grid: cssVar("--border-soft", "#1c2027"),
    border: cssVar("--border", "#262b33"),
    attack: cssVar("--attack", "#ff6b6b"),
    defense: cssVar("--defense", "#4dabf7"),
    good: cssVar("--good", "#51cf66"),
    warn: cssVar("--warn", "#ffd43b"),
    accent: cssVar("--accent", "#9775fa"),
    panel: cssVar("--panel", "#14171c"),
  });

  function fmt(v, digits) {
    if (v === null || v === undefined || Number.isNaN(v)) return "--";
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(1);
    return v.toFixed(digits === undefined ? (a < 1 ? 3 : 2) : digits);
  }

  /* ------------------------------------------------------------------ */
  /* LineChart: N named series over a shared integer x (the round number) */
  /* ------------------------------------------------------------------ */
  class LineChart {
    /** @param {HTMLCanvasElement} canvas
     *  @param {object} opts  {series:[{key,label,color,dash,width,fill,axis}],
     *                         yFormat, xLabel, bands:[{from,to,color,label}],
     *                         zeroLine:bool, yMin, yMax, stacked:bool} */
    constructor(canvas, opts) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.opts = Object.assign({ yFormat: (v) => fmt(v), zeroLine: false }, opts);
      this.series = this.opts.series.map((s) => Object.assign({ points: [] }, s));
      this.byKey = {};
      this.series.forEach((s) => { this.byKey[s.key] = s; });
      this.hover = null;
      this._tip = null;
      this._bindHover();
      this._ro = new ResizeObserver(() => this.draw());
      this._ro.observe(canvas);
    }

    /** push(x, {key: value, ...}) -- missing keys simply get no point at this x. */
    push(x, values) {
      for (const key in values) {
        const s = this.byKey[key];
        const v = values[key];
        if (!s || v === null || v === undefined || Number.isNaN(v)) continue;
        s.points.push([x, +v]);
      }
    }

    reset() { this.series.forEach((s) => { s.points.length = 0; }); this.hover = null; }

    setVisible(key, on) {
      const s = this.byKey[key];
      if (s) { s.hidden = !on; this.draw(); }
    }

    _bindHover() {
      const wrap = this.canvas.parentElement;
      this.canvas.addEventListener("mousemove", (e) => {
        const r = this.canvas.getBoundingClientRect();
        this.hover = { x: e.clientX - r.left, y: e.clientY - r.top };
        this.draw();
      });
      this.canvas.addEventListener("mouseleave", () => {
        this.hover = null;
        if (this._tip) this._tip.style.opacity = 0;
        this.draw();
      });
      if (wrap && !wrap.querySelector(".chart-tip")) {
        this._tip = document.createElement("div");
        this._tip.className = "chart-tip";
        wrap.appendChild(this._tip);
      } else if (wrap) {
        this._tip = wrap.querySelector(".chart-tip");
      }
    }

    draw() {
      const c = this.canvas, ctx = this.ctx, P = PALETTE();
      const dpr = global.devicePixelRatio || 1;
      const w = c.clientWidth, h = c.clientHeight;
      if (!w || !h) return;
      if (c.width !== w * dpr || c.height !== h * dpr) {
        c.width = w * dpr; c.height = h * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const live = this.series.filter((s) => !s.hidden && s.points.length);
      const padL = 46, padR = 10, padT = 8, padB = 20;
      const plotW = Math.max(1, w - padL - padR), plotH = Math.max(1, h - padT - padB);

      if (!live.length) {
        ctx.fillStyle = P.faint;
        ctx.font = "11px " + cssVar("--mono", "monospace");
        ctx.textAlign = "center";
        ctx.fillText("waiting for data", w / 2, h / 2);
        return;
      }

      let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
      live.forEach((s) => s.points.forEach(([x, y]) => {
        if (x < xMin) xMin = x; if (x > xMax) xMax = x;
        if (y < yMin) yMin = y; if (y > yMax) yMax = y;
      }));
      (this.opts.bands || []).forEach((b) => {
        if (b.from !== undefined) { yMin = Math.min(yMin, b.from); yMax = Math.max(yMax, b.to); }
      });
      if (this.opts.yMin !== undefined) yMin = Math.min(yMin, this.opts.yMin);
      if (this.opts.yMax !== undefined) yMax = Math.max(yMax, this.opts.yMax);
      if (this.opts.zeroLine) { yMin = Math.min(yMin, 0); yMax = Math.max(yMax, 0); }
      if (xMax === xMin) xMax = xMin + 1;
      if (yMax === yMin) { yMax += Math.abs(yMax || 1) * 0.05 + 1e-6; yMin -= 1e-6; }
      const pad = (yMax - yMin) * 0.08;
      yMin -= pad; yMax += pad;

      const X = (x) => padL + ((x - xMin) / (xMax - xMin)) * plotW;
      const Y = (y) => padT + plotH - ((y - yMin) / (yMax - yMin)) * plotH;

      /* bands (e.g. the goal's target region) */
      (this.opts.bands || []).forEach((b) => {
        ctx.fillStyle = b.color;
        const y0 = Y(Math.max(b.from, b.to)), y1 = Y(Math.min(b.from, b.to));
        ctx.fillRect(padL, y0, plotW, Math.max(1, y1 - y0));
      });

      /* y grid */
      ctx.strokeStyle = P.grid; ctx.lineWidth = 1;
      ctx.fillStyle = P.faint;
      ctx.font = "10px " + cssVar("--mono", "monospace");
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      const ticks = 4;
      for (let i = 0; i <= ticks; i++) {
        const v = yMin + ((yMax - yMin) * i) / ticks;
        const y = Math.round(Y(v)) + 0.5;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
        ctx.fillText(this.opts.yFormat(v), padL - 6, y);
      }
      if (this.opts.zeroLine && yMin < 0 && yMax > 0) {
        ctx.strokeStyle = P.border; ctx.lineWidth = 1.5;
        const y = Math.round(Y(0)) + 0.5;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      }

      /* x labels: first and last round only -- the axis is dense and monotone */
      ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillStyle = P.faint;
      ctx.fillText(String(Math.round(xMin)), padL, padT + plotH + 5);
      ctx.textAlign = "right";
      ctx.fillText(String(Math.round(xMax)), w - padR, padT + plotH + 5);

      /* series */
      live.forEach((s) => {
        const pts = decimate(s.points, plotW);
        if (s.fill) {
          ctx.beginPath();
          pts.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
          ctx.lineTo(X(pts[pts.length - 1][0]), Y(Math.max(yMin, 0)));
          ctx.lineTo(X(pts[0][0]), Y(Math.max(yMin, 0)));
          ctx.closePath();
          ctx.fillStyle = s.fill;
          ctx.fill();
        }
        ctx.beginPath();
        ctx.lineWidth = s.width || 1.6;
        ctx.strokeStyle = s.color;
        ctx.setLineDash(s.dash || []);
        ctx.lineJoin = "round";
        pts.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
        ctx.stroke();
        ctx.setLineDash([]);
        if (pts.length === 1) {
          ctx.fillStyle = s.color;
          ctx.beginPath(); ctx.arc(X(pts[0][0]), Y(pts[0][1]), 2.5, 0, 7); ctx.fill();
        }
      });

      /* crosshair + tooltip */
      if (this.hover && this.hover.x > padL && this.hover.x < w - padR) {
        const xv = xMin + ((this.hover.x - padL) / plotW) * (xMax - xMin);
        const rows = [];
        let snapX = null;
        live.forEach((s) => {
          const p = nearest(s.points, xv);
          if (!p) return;
          if (snapX === null || Math.abs(p[0] - xv) < Math.abs(snapX - xv)) snapX = p[0];
          rows.push([s, p]);
        });
        if (snapX !== null) {
          const px = X(snapX);
          ctx.strokeStyle = P.border; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(px, padT); ctx.lineTo(px, padT + plotH); ctx.stroke();
          rows.forEach(([s, p]) => {
            if (Math.abs(p[0] - snapX) > (xMax - xMin) / plotW * 4) return;
            ctx.fillStyle = s.color;
            ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3, 0, 7); ctx.fill();
          });
          if (this._tip) {
            const lines = [(this.opts.xLabel || "round") + " " + Math.round(snapX)];
            rows.forEach(([s, p]) => {
              if (Math.abs(p[0] - snapX) > (xMax - xMin) / plotW * 4) return;
              lines.push(s.label + "  " + this.opts.yFormat(p[1]));
            });
            this._tip.textContent = lines.join("\n");
            this._tip.style.opacity = 1;
            const tw = this._tip.offsetWidth || 120;
            this._tip.style.left = Math.min(w - tw - 4, Math.max(0, px + 10)) + "px";
            this._tip.style.top = Math.max(0, this.hover.y - 10) + "px";
          }
        }
      }
    }
  }

  /** Keep min+max per pixel column: spikes survive, point count stays bounded. */
  function decimate(points, pixels) {
    if (points.length <= pixels * 2) return points;
    const xMin = points[0][0], xMax = points[points.length - 1][0];
    const span = xMax - xMin || 1;
    const buckets = new Map();
    for (const p of points) {
      const col = Math.floor(((p[0] - xMin) / span) * pixels);
      const b = buckets.get(col);
      if (!b) buckets.set(col, [p, p]);
      else {
        if (p[1] < b[0][1]) b[0] = p;
        if (p[1] > b[1][1]) b[1] = p;
      }
    }
    const out = [];
    Array.from(buckets.keys()).sort((a, b) => a - b).forEach((col) => {
      const [lo, hi] = buckets.get(col);
      if (lo[0] <= hi[0]) { out.push(lo); if (hi !== lo) out.push(hi); }
      else { out.push(hi); out.push(lo); }
    });
    return out;
  }

  function nearest(points, x) {
    if (!points.length) return null;
    let lo = 0, hi = points.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (points[mid][0] < x) lo = mid + 1; else hi = mid;
    }
    const a = points[Math.max(0, lo - 1)], b = points[lo];
    return Math.abs(a[0] - x) <= Math.abs(b[0] - x) ? a : b;
  }

  /* ------------------------------------------------------------------ */
  /* Sparkline: a tiny inline trend, no axes                             */
  /* ------------------------------------------------------------------ */
  function sparkline(canvas, values, color, opts) {
    const ctx = canvas.getContext("2d");
    const dpr = global.devicePixelRatio || 1;
    const w = canvas.clientWidth || 80, h = canvas.clientHeight || 22;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const vals = values.filter((v) => typeof v === "number" && !Number.isNaN(v));
    if (vals.length < 2) return;
    let lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    if (opts && opts.min !== undefined) lo = Math.min(lo, opts.min);
    if (opts && opts.max !== undefined) hi = Math.max(hi, opts.max);
    if (hi === lo) { hi += 1e-9; lo -= 1e-9; }
    ctx.beginPath();
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = color;
    vals.forEach((v, i) => {
      const x = (i / (vals.length - 1)) * (w - 2) + 1;
      const y = h - 2 - ((v - lo) / (hi - lo)) * (h - 4);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }

  /* ------------------------------------------------------------------ */
  /* Diverging colour ramp for the matrix heat cells                     */
  /* ------------------------------------------------------------------ */
  function heatColor(value, max, invert) {
    /* value in [0, max]; 0 -> neutral, max -> full attack red (or good green
       when inverted, e.g. for a detection rate where high is the defense
       winning). Alpha carries the magnitude so text stays readable. */
    if (value === null || value === undefined || Number.isNaN(value)) return "transparent";
    const t = Math.max(0, Math.min(1, max ? value / max : 0));
    const base = invert ? cssVar("--good", "#51cf66") : cssVar("--attack", "#ff6b6b");
    return `color-mix(in srgb, ${base} ${Math.round(t * 62)}%, transparent)`;
  }

  global.Charts = { LineChart, sparkline, heatColor, cssVar, fmt, PALETTE };
})(window);

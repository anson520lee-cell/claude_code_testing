/* Swing Research Terminal - single-page dashboard (vanilla JS, no build step). Research only. */
"use strict";

const state = { snap: null, session: null, sort: {}, filters: { tier: "", setup: "", hideEarnings: false, q: "" },
  charts: [], focusCharts: [], scannerSide: "long", scanMode: null };
const $ = (sel, el = document) => el.querySelector(sel);
let I18N = {};
const tr = (x) => (x === null || x === undefined ? "" : (I18N[x] ?? String(x)));
const app = $("#app");

// ------------------------------------------------------------------ formatting
const isNum = (v) => typeof v === "number" && isFinite(v);
const fixed = (v, d) => { const t = v.toFixed(d); return Number(t) === 0 ? (0).toFixed(d) : t; };  // no "-0"
const pct = (v, d = 1, sign = true) => (isNum(v) ? `${sign && Number(fixed(v * 100, d)) > 0 ? "+" : ""}${fixed(v * 100, d)}%` : "n/a");
const num = (v, d = 2) => (isNum(v) ? v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }) : "n/a");
const money = (v) => {
  if (!isNum(v)) return "n/a";
  const a = Math.abs(v);
  for (const [div, s] of [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]]) if (a >= div) return `$${(v / div).toFixed(1)}${s}`;
  return `$${v.toFixed(0)}`;
};
const cls = (v) => (isNum(v) ? (v > 0 ? "up" : v < 0 ? "down" : "") : "muted");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const regimeClass = (l) => "r-" + String(l || "").toLowerCase().replace(/\s+/g, "-");
const tier = (t) => `<span class="tier tier-${esc(t)}">${esc(t)}</span>`;
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

async function api(path, opts) {
  if (window.__PREVIEW__) return window.__PREVIEW__(path, opts);   // static preview build (scripts/export_preview.py)
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* not json */ }
    throw new Error(msg);
  }
  return r.json();
}

// ------------------------------------------------------------------ generic sortable table
function table(id, columns, rows, opts = {}) {
  const s = state.sort[id] || { key: opts.sortKey, dir: opts.sortDir || "desc" };
  const sorted = [...rows];
  if (s.key) {
    const col = columns.find((c) => c.key === s.key);
    const val = col && col.sort ? col.sort : (r) => r[s.key];
    sorted.sort((a, b) => {
      const x = val(a), y = val(b);
      const nx = x === null || x === undefined || (typeof x === "number" && !isFinite(x));
      const ny = y === null || y === undefined || (typeof y === "number" && !isFinite(y));
      if (nx && ny) return 0; if (nx) return 1; if (ny) return -1;
      const c = typeof x === "string" ? x.localeCompare(y) : x - y;
      return s.dir === "asc" ? c : -c;
    });
  }
  const head = columns.map((c) => `<th data-t="${id}" data-k="${c.key}" class="${c.n ? "n" : ""} ${s.key === c.key ? "sorted-" + s.dir : ""}">${esc(c.label)}</th>`).join("");
  const body = sorted.map((r) => `<tr class="${opts.onClick ? "click" : ""}" ${opts.rowAttr ? opts.rowAttr(r) : ""}>${columns.map((c) => `<td class="${c.n ? "n" : ""} ${c.wrap ? "wrap" : ""}">${c.render ? c.render(r) : esc(r[c.key])}</td>`).join("")}</tr>`).join("");
  return `<div class="tw"><table id="${id}"><thead><tr>${head}</tr></thead><tbody>${body || `<tr><td colspan="${columns.length}" class="empty">Nothing to show.</td></tr>`}</tbody></table></div>`;
}
document.addEventListener("click", (e) => {
  const th = e.target.closest("th[data-t]");
  if (th) {
    const id = th.dataset.t, key = th.dataset.k;
    const cur = state.sort[id] || {};
    state.sort[id] = { key, dir: cur.key === key && cur.dir === "desc" ? "asc" : "desc" };
    route();
    return;
  }
  const tr = e.target.closest("tr[data-sym]");
  if (tr && !e.target.closest("a")) location.hash = `#/ticker/${tr.dataset.sym}`;
});

// ------------------------------------------------------------------ boot
async function boot() {
  applyTheme(localStorageGet("theme") || "dark");
  state.scanMode = localStorageGet("scanMode") || "cards";
  try { I18N = await api("/api/i18n"); } catch (_) { I18N = {}; }
  api("/api/backtest").then((b) => { state.btFull = b && b.available ? b : null; }).catch(() => { state.btFull = null; });
  try {
    const { sessions } = await api("/api/sessions");
    const run = await api("/api/run/status");
    if (run.running) watchRun(true);
    const sel = $("#session");
    sel.innerHTML = sessions.map((s) => `<option>${s}</option>`).join("");
    if (!sessions.length) {
      app.innerHTML = run.running
        ? `<div class="card empty"><h2>Downloading market data…</h2><p>The first run downloads about two years of history for the universe and analyses it (2 to 4 minutes); this page opens the report when it is done.</p></div>`
        : `<div class="card empty"><h2>No data yet</h2><p>Press Update above, or run <code>python daily_run.py</code>. The first run downloads about two years of history (a few minutes).</p></div>`;
      return;
    }
    state.session = sessions[0];
    sel.onchange = () => loadSession(sel.value);
    await loadSession(state.session);
  } catch (err) {
    app.innerHTML = `<div class="card empty">Could not load: ${esc(err.message)}</div>`;
  }
}
async function loadSession(session) {
  app.innerHTML = `<div class="loading">Loading ${esc(session)}…</div>`;
  state.snap = await api(`/api/snapshot?session=${encodeURIComponent(session)}`);
  state.session = session;
  const reg = state.snap.regime;
  const b = $("#regime-badge");
  b.className = `badge ${regimeClass(reg.label)}`;
  b.textContent = `${tr(reg.label)} ${isNum(reg.composite) ? (reg.composite > 0 ? "+" : "") + reg.composite.toFixed(0) : ""}`;
  $("#foot-session").textContent = `Session ${session} · generated ${state.snap.generated_at}`;
  const w = state.snap.warnings || [];
  const banner = $("#banner");
  banner.classList.toggle("hidden", !w.length);
  banner.innerHTML = w.map(esc).join("<br>");
  route();
}
function localStorageGet(k) { try { return localStorage.getItem(k); } catch (_) { return null; } }
function localStorageSet(k, v) { try { localStorage.setItem(k, v); } catch (_) { /* private mode */ } }
function applyTheme(t) { document.documentElement.dataset.theme = t; localStorageSet("theme", t); }
$("#theme-btn").onclick = () => { applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"); route(); };
$("#jump").onsubmit = (e) => { e.preventDefault(); const v = $("#jump-input").value.trim().toUpperCase(); if (v) location.hash = `#/ticker/${v}`; };
$("#run-btn").onclick = async () => {
  if (window.__PREVIEW__) { const b = $("#banner"); b.classList.remove("hidden"); b.textContent = `This is a read-only preview of the ${state.session || "latest"} analysis; data cannot be refreshed here. Use the Swing Terminal app in the Dock on your Mac to update.`; return; }
  if (!confirm("Run the daily pipeline now? It downloads market data and takes a few minutes.")) return;
  try { await api("/api/run", { method: "POST" }); watchRun(false); } catch (err) { alert(err.message); }
};
// Poll a running pipeline. ``quiet``: started elsewhere (Dock launcher / schedule) - no alert, just refresh.
async function watchRun(quiet) {
  const btn = $("#run-btn"), banner = $("#banner");
  let st;
  try { st = await api("/api/run/status"); } catch (_) { setTimeout(() => watchRun(quiet), 5000); return; }
  btn.disabled = st.running;
  btn.textContent = st.running ? "Updating…" : "Run";
  if (st.running) {
    banner.classList.remove("hidden");
    banner.textContent = `Updating data: ${st.log.length ? st.log[st.log.length - 1] : "starting…"}`;
    setTimeout(() => watchRun(quiet), 3000);
    return;
  }
  if (!quiet && st.status) alert(`Run finished: ${st.status}${st.message ? " - " + st.message : ""}`);
  const { sessions } = await api("/api/sessions");
  const sel = $("#session");
  sel.innerHTML = sessions.map((x) => `<option>${x}</option>`).join("");
  sel.onchange = () => loadSession(sel.value);
  if (sessions.length) loadSession(sessions[0]);
  else { banner.classList.remove("hidden"); banner.textContent = `Run ${st.status || "ended"}: ${st.message || ""}`; }
}

// ------------------------------------------------------------------ router
window.addEventListener("hashchange", route);
function route() {
  if (!state.snap) return;
  closeFocus(true);
  disposeCharts(state.charts);
  const [, view = "overview", arg] = (location.hash || "#/overview").split("/");
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === view));
  const views = { overview, scanner, universe, catalysts, news, portfolio, research, quality, ticker };
  (views[view] || overview)(arg);
  window.scrollTo(0, 0);
}

// ------------------------------------------------------------------ chart helpers
// Futu-style palette: green up / red down, EMA9 blue, EMA21 red, EMA50 white, EMA200 green, EMA250 gold.
const PAL = () => ({
  up: cssVar("--up"), down: cssVar("--down"), neutral: cssVar("--neutral-line"),
  ema9: "#2f7df6", ema21: "#e8364f", ema50: cssVar("--neutral-line"), ema200: "#1db886", ema250: "#f4c21f",
  boll: "rgba(160,160,170,.55)", dif: "#2f7df6", dea: cssVar("--neutral-line"), rsi6: "#2f7df6", rsi14: cssVar("--neutral-line"),
});
const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
function makeChart(el, height, opts = {}) {
  const chart = LightweightCharts.createChart(el, {
    height, layout: { background: { type: "solid", color: "rgba(0,0,0,0)" }, textColor: cssVar("--muted"), fontSize: 11,
      fontFamily: getComputedStyle(document.body).fontFamily },
    grid: { vertLines: { visible: false }, horzLines: { color: cssVar("--line"), style: 1 } },
    rightPriceScale: { borderVisible: false }, timeScale: { borderVisible: false, visible: opts.timeVisible !== false },
    crosshair: { mode: 0, vertLine: { color: cssVar("--line3"), labelBackgroundColor: "#2a2a2e" }, horzLine: { color: cssVar("--line3"), labelBackgroundColor: "#2a2a2e" } },
    handleScroll: true, handleScale: true, ...opts.extra,
  });
  const observer = new ResizeObserver(() => { if (!chart._gone) chart.applyOptions({ width: el.clientWidth }); });
  observer.observe(el);
  chart._observer = observer;
  (opts.bucket || state.charts).push(chart);
  return chart;
}
function disposeCharts(list) {
  list.forEach((c) => { c._gone = true; c._observer.disconnect(); try { c.remove(); } catch (_) { /* gone */ } });
  list.length = 0;
}
function syncCharts(charts) {
  charts.forEach((c) => c.timeScale().subscribeVisibleLogicalRangeChange((range) => {
    if (!range) return;
    charts.forEach((o) => { if (o !== c) o.timeScale().setVisibleLogicalRange(range); });
  }));
}
function chartBlock(p, big = 420) {
  return `<div class="lg" id="${p}-lg-price"></div><div id="${p}-price" class="chart" style="height:${big}px"></div>
    <div class="lg" id="${p}-lg-vol"></div><div id="${p}-vol" class="chart small"></div>
    <div class="lg" id="${p}-lg-macd"></div><div id="${p}-macd" class="chart small"></div>
    <div class="lg" id="${p}-lg-rsi"></div><div id="${p}-rsi" class="chart small"></div>`;
}
const fmtVol = (v) => (!isNum(v) ? "n/a" : v >= 1e9 ? (v / 1e9).toFixed(2) + "B" : v >= 1e6 ? (v / 1e6).toFixed(2) + "M" : (v / 1e3).toFixed(1) + "K");
const timeKey = (t) => (typeof t === "object" && t ? `${t.year}-${String(t.month).padStart(2, "0")}-${String(t.day).padStart(2, "0")}` : t);

function renderTickerCharts(d, c, p = "c", bucket = undefined, big = 420, sub = 104) {
  const P = PAL(), dp = (d.last && d.last.close < 10) ? 3 : 2;
  const opt = { bucket };
  const price = makeChart($(`#${p}-price`), big, { ...opt, timeVisible: false });
  const candles = price.addCandlestickSeries({ upColor: P.up, downColor: P.down, borderVisible: false, wickUpColor: P.up, wickDownColor: P.down });
  candles.setData(d.candles);
  const line = (chart, data, color, width = 1.5, style = 0) => { const sr = chart.addLineSeries({ color, lineWidth: width, lineStyle: style, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false }); sr.setData(data); return sr; };
  line(price, d.bb_upper, P.boll, 1, 2); line(price, d.bb_lower, P.boll, 1, 2);
  line(price, d.ema250, P.ema250, 1.4); line(price, d.ema200, P.ema200, 1.4); line(price, d.ema50, P.ema50, 1.4); line(price, d.ema21, P.ema21, 1.5); line(price, d.ema9, P.ema9, 1.5);
  const recent = new Set(d.markers.slice(-3).map((m) => m.time + m.setup));
  candles.setMarkers(d.markers.map((m) => ({ time: m.time, position: m.direction === "long" ? "belowBar" : "aboveBar", color: m.direction === "long" ? P.up : P.down, shape: m.direction === "long" ? "arrowUp" : "arrowDown", text: recent.has(m.time + m.setup) ? m.setup.replace("_", " ") : "" })));
  if (d.volume_profile) { price.timeScale().applyOptions({ rightOffset: 9 }); vpOverlay(price, candles, $(`#${p}-price`), d.volume_profile); }
  if (c) {
    const pl = (v, color, title, style = 0) => candles.createPriceLine({ price: v, color, lineWidth: 1, lineStyle: style, axisLabelVisible: true, title });
    const pp = c.plan;
    pl(pp.entry, P.neutral, pp.conditional_entry ? "trigger" : "ref", 2); pl(pp.stop, P.down, "stop"); pl(pp.target1, P.up, "T1"); pl(pp.target2, P.up, "T2", 2);
  }
  const upAt = new Map(d.candles.map((k) => [k.time, k.close >= k.open]));
  const vol = makeChart($(`#${p}-vol`), sub, { ...opt, timeVisible: false });
  vol.addHistogramSeries({ priceFormat: { type: "volume" }, priceLineVisible: false, lastValueVisible: false })
    .setData(d.volume.map((x) => ({ time: x.time, value: x.value, color: upAt.get(x.time) ? P.up : P.down })));
  line(vol, d.mavol20, P.neutral, 1.2);
  const macd = makeChart($(`#${p}-macd`), sub, { ...opt, timeVisible: false });
  macd.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false }).setData(d.macd_hist.map((x) => ({ ...x, color: x.value >= 0 ? P.up : P.down })));
  line(macd, d.macd, P.dif, 1.3); line(macd, d.macd_signal, P.dea, 1.3);
  const rsi = makeChart($(`#${p}-rsi`), sub + 20, opt);
  const r6 = line(rsi, d.rsi6, P.rsi6, 1.3); line(rsi, d.rsi14, P.rsi14, 1.3);
  [80, 50, 20].forEach((v) => r6.createPriceLine({ price: v, color: cssVar("--line2"), lineWidth: 1, lineStyle: 2, axisLabelVisible: false }));
  const all = [price, vol, macd, rsi];
  all.forEach((x) => x.timeScale().fitContent());
  syncCharts(all);

  // Futu-style legends: values follow the crosshair, default = last session.
  const mp = (arr) => new Map(arr.map((x) => [x.time, x.value]));
  const M = { ema9: mp(d.ema9), ema21: mp(d.ema21), ema50: mp(d.ema50), ema200: mp(d.ema200), ema250: mp(d.ema250), vol: mp(d.volume),
    mavol: mp(d.mavol20), dif: mp(d.macd), dea: mp(d.macd_signal), hist: mp(d.macd_hist), rsi6: mp(d.rsi6), rsi14: mp(d.rsi14) };
  const K = new Map(d.candles.map((k) => [k.time, k]));
  const span = (label, color, v, digits = dp) => `<span style="color:${color}">${label}: ${num(v, digits)}</span>`;
  const draw = (t) => {
    const k = K.get(t) || d.candles[d.candles.length - 1];
    t = k.time;
    const lg = (id) => $(`#${p}-lg-${id}`);
    if (!lg("price")) return;
    lg("price").innerHTML = `<span class="t">EMA</span>${span("EMA9", P.ema9, M.ema9.get(t))}${span("EMA21", P.ema21, M.ema21.get(t))}${span("EMA50", P.ema50, M.ema50.get(t))}${span("EMA200", P.ema200, M.ema200.get(t))}${span("EMA250", P.ema250, M.ema250.get(t))}
      <span class="muted">${t} · O ${num(k.open, dp)} H ${num(k.high, dp)} L ${num(k.low, dp)} C <b class="${k.close >= k.open ? "up" : "down"}">${num(k.close, dp)}</b></span>`;
    lg("vol").innerHTML = `<span class="t">MAVOL</span><span style="color:${upAt.get(t) ? P.up : P.down}">VOL: ${fmtVol(M.vol.get(t))}</span><span style="color:${P.neutral}">MAVOL20: ${fmtVol(M.mavol.get(t))}</span>`;
    const h = M.hist.get(t);
    lg("macd").innerHTML = `<span class="t">MACD</span>${span("DIF", P.dif, M.dif.get(t), 3)}${span("DEA", P.dea, M.dea.get(t), 3)}<span style="color:${h >= 0 ? P.up : P.down}">MACD: ${num(h, 3)}</span>`;
    lg("rsi").innerHTML = `<span class="t">RSI</span>${span("RSI6", P.rsi6, M.rsi6.get(t), 2)}${span("RSI14", P.rsi14, M.rsi14.get(t), 2)}`;
  };
  draw(null);
  all.forEach((ch) => ch.subscribeCrosshairMove((param) => draw(param && param.time ? timeKey(param.time) : null)));
}

// ------------------------------------------------------------------ views: overview
function overview() {
  const s = state.snap, r = s.regime;
  const needle = isNum(r.composite) ? ((r.composite + 100) / 200) * 100 : 50;
  const comps = r.components.map((c) => {
    const w = isNum(c.score) ? Math.abs(c.score) / 2 * 50 : 0;
    const left = isNum(c.score) && c.score < 0 ? 50 - w : 50;
    const color = isNum(c.score) ? (c.score >= 0 ? "var(--up)" : "var(--down)") : "var(--dim)";
    return `<div class="comp"><div>${esc(tr(c.name))} <span class="muted">· ${c.weight}</span></div>
      <div class="zbar"><div class="mid"></div><div class="fill" style="left:${left}%;width:${w}%;background:${color}"></div></div>
      <div class="n mono ${cls(c.score)}">${isNum(c.score) ? (c.score > 0 ? "+" : "") + c.score.toFixed(2) : "n/a"}</div>
      <div class="ev">${c.evidence.map(esc).join(" ")}</div></div>`;
  }).join("");
  const b = r.breadth || {}, v = r.volatility || {}, rt = r.rates || {};
  const tops = (side) => s.candidates[side].slice(0, 6).map((c) => `<tr class="click" data-sym="${c.symbol}"><td><b>${c.symbol}</b></td><td><span class="spark" data-mini-sm="${esc(c.symbol)}"></span></td><td>${esc(c.setup_name)}</td><td class="n">${c.score.toFixed(0)}</td><td>${tier(c.tier)}</td><td class="n">${num(c.plan.reward_risk, 1)}R</td></tr>`).join("") || `<tr><td class="empty">None</td></tr>`;
  app.innerHTML = `
  <div class="grid g-side">
    <div class="grid">
      <div class="card"><h2>Market regime <small>${esc(s.session.date)} · ${esc(s.session.weekday)}${s.session.early_close ? " · early close" : ""}</small></h2>
        <div class="grid g4">
          <div class="kpi"><span class="l">Regime</span><span class="v"><span class="badge ${regimeClass(r.label)}" style="font-size:18px">${esc(r.label)}</span></span></div>
          <div class="kpi"><span class="l">Composite</span><span class="v num ${cls(r.composite)}">${isNum(r.composite) ? r.composite.toFixed(0) : "n/a"}</span><span class="note">${isNum(r.composite_5d_change) ? (Number(fixed(r.composite_5d_change, 0)) > 0 ? "+" : "") + fixed(r.composite_5d_change, 0) + " vs 5 sessions ago" : ""}</span></div>
          <div class="kpi"><span class="l">Risk budget (long / short)</span><span class="v num">×${r.long_risk_budget} / ×${r.short_risk_budget}</span></div>
          <div class="kpi"><span class="l">VIX · VIX/VIX3M</span><span class="v num">${num(v.vix, 1)} <span class="muted" style="font-size:14px">${num(v.vix_term, 2)}</span></span></div>
        </div>
        <div style="margin:14px 0 4px"><div class="gauge"><div class="needle" style="left:calc(${needle}% - 1px)"></div></div>
          <div class="gauge-labels"><span>Strong Bearish</span><span>Bearish</span><span>Neutral</span><span>Bullish</span><span>Strong Bullish</span></div></div>
        <p>${esc(r.guidance)}</p>
        ${comps}
        <p class="note">Composite = Σ weight × score / 2 (−100…+100). Methodology: <a href="#/research">Research</a>.</p>
      </div>
      <div class="card"><h2>Regime history <small>composite, last ${r.history.length} sessions</small></h2><div id="regime-chart" class="chart mid"></div></div>
      <div class="card"><h2>Indices · volatility · credit</h2>${table("idx", [
        { key: "symbol", label: "Symbol", render: (x) => `<b>${esc(x.symbol)}</b>` },
        { key: "close", label: "Close", n: 1, render: (x) => num(x.close) },
        { key: "ret_1d", label: "1D", n: 1, render: (x) => `<span class="${cls(x.ret_1d)}">${pct(x.ret_1d)}</span>` },
        { key: "ret_5d", label: "5D", n: 1, render: (x) => `<span class="${cls(x.ret_5d)}">${pct(x.ret_5d)}</span>` },
        { key: "ret_20d", label: "20D", n: 1, render: (x) => `<span class="${cls(x.ret_20d)}">${pct(x.ret_20d)}</span>` },
        { key: "ret_ytd", label: "YTD", n: 1, render: (x) => `<span class="${cls(x.ret_ytd)}">${pct(x.ret_ytd)}</span>` },
        { key: "vs_ema50", label: "vs EMA50", n: 1, render: (x) => pct(x.vs_ema50) },
        { key: "vs_ema200", label: "vs EMA200", n: 1, render: (x) => pct(x.vs_ema200) },
        { key: "from_52w_high", label: "From 52w high", n: 1, render: (x) => pct(x.from_52w_high) },
        { key: "ema_stack", label: "EMA stack", render: (x) => `<span class="${x.ema_stack === "bullish" ? "up" : x.ema_stack === "bearish" ? "down" : "muted"}">${x.ema_stack}</span>` },
      ], s.indices)}</div>
    </div>
    <div class="grid" style="align-content:start">
      <div class="card"><h2>Breadth <small>screened universe (${num(b.breadth_n, 0)} stocks)</small></h2>
        <div class="kv"><div><span>Above EMA50</span><span class="num">${num(b.pct_above_ema50, 0)}%</span></div>
        <div><span>Above EMA200</span><span class="num">${num(b.pct_above_ema200, 0)}%</span></div>
        <div><span>At 20D highs</span><span class="num up">${num(b.pct_new_high20, 0)}%</span></div>
        <div><span>At 20D lows</span><span class="num down">${num(b.pct_new_low20, 0)}%</span></div>
        <div><span>US 10Y</span><span class="num">${num(rt.us10y, 2)}%</span></div>
        <div><span>10Y Δ20 sessions</span><span class="num ${cls(-(rt.us10y_chg20_bp || 0))}">${num(rt.us10y_chg20_bp, 0)}bp</span></div>
        <div><span>2s10s</span><span class="num">${isNum(rt.curve_2s10s) ? (rt.curve_2s10s * 100).toFixed(0) + "bp" : "n/a"}</span></div>
        <div><span>VIX 5D change</span><span class="num ${cls(-(v.vix_chg5 || 0))}">${pct(v.vix_chg5, 0)}</span></div></div></div>
      <div class="card"><h2>Top long research candidates</h2><div class="tw"><table><tbody>${tops("long")}</tbody></table></div></div>
      <div class="card"><h2>Next events</h2><ul class="list">${(s.macro_events || []).slice(0, 6).map((e) => `<li><span class="mono">${e.event_date}</span> ${esc(e.event_type)} <span class="chip">${e.status}</span></li>`).join("") || "<li>None in the next 30 days</li>"}
        ${(s.upcoming_earnings || []).filter((e) => e.sessions_until <= 10).slice(0, 8).map((e) => `<li><span class="mono">${e.date}</span> <a href="#/ticker/${e.symbol}">${e.symbol}</a> earnings ${esc(e.timing || "(time n/a)")} <span class="chip">${e.status}</span></li>`).join("")}</ul></div>
    </div>
  </div>
  <div class="grid g2" style="margin-top:12px">
    <div class="card"><h2>Sector rotation <small>RS vs SPY: trend (x) vs momentum (y), 10-session tails</small></h2><svg id="rrg" class="rrg"></svg></div>
    <div class="card"><h2>Sector & industry ETFs</h2>${table("sec", [
      { key: "rank_20d", label: "#", n: 1 },
      { key: "symbol", label: "ETF", render: (x) => `<b>${x.symbol}</b> <span class="muted">${esc(tr(x.name))}</span>` },
      { key: "ret_5d", label: "5D", n: 1, render: (x) => `<span class="${cls(x.ret_5d)}">${pct(x.ret_5d)}</span>` },
      { key: "ret_20d", label: "20D", n: 1, render: (x) => `<span class="${cls(x.ret_20d)}">${pct(x.ret_20d)}</span>` },
      { key: "rel_20d", label: "Rel 20D", n: 1, render: (x) => `<span class="${cls(x.rel_20d)}">${pct(x.rel_20d)}</span>` },
      { key: "rel_60d", label: "Rel 60D", n: 1, render: (x) => `<span class="${cls(x.rel_60d)}">${pct(x.rel_60d)}</span>` },
      { key: "quadrant", label: "Quadrant", render: (x) => `<span class="${{ Leading: "up", Lagging: "down", Improving: "", Weakening: "warn" }[x.quadrant] || "muted"}">${x.quadrant}</span>` },
    ], s.sectors, { sortKey: "rank_20d", sortDir: "asc" })}</div>
  </div>
  ${marketExtras(s)}`;
  const el = $("#regime-chart");
  const ch = makeChart(el, 220);
  const line = ch.addBaselineSeries({ baseValue: { type: "price", price: 0 }, topLineColor: cssVar("--up"), bottomLineColor: cssVar("--down"),
    topFillColor1: "rgba(38,166,154,0.25)", topFillColor2: "rgba(38,166,154,0.02)", bottomFillColor1: "rgba(239,83,80,0.02)", bottomFillColor2: "rgba(239,83,80,0.25)", lineWidth: 2 });
  line.setData(r.history.filter((h) => isNum(h.composite)).map((h) => ({ time: h.date, value: h.composite })));
  for (const [p, t] of [[45, "Strong Bullish"], [15, "Bullish"], [-15, "Bearish"], [-45, "Strong Bearish"]])
    line.createPriceLine({ price: p, color: cssVar("--dim"), lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: t });
  ch.timeScale().fitContent();
  drawRRG($("#rrg"), s.sectors);
  drawAD(s);
  loadMinicharts([...s.candidates.long.slice(0, 6), ...s.candidates.short.slice(0, 6)]);
}

function drawRRG(svg, sectors) {
  const W = svg.clientWidth || 600, H = svg.clientHeight || 340, pad = 30;
  const pts = sectors.filter((x) => isNum(x.rs_trend) && isNum(x.rs_momentum));
  const xs = pts.flatMap((x) => x.rs_tail.filter(isNum)), ys = pts.flatMap((x) => x.mom_tail.filter(isNum));
  const mx = Math.max(0.02, ...xs.map(Math.abs)) * 1.15, my = Math.max(0.01, ...ys.map(Math.abs)) * 1.15;
  const X = (v) => pad + ((v + mx) / (2 * mx)) * (W - 2 * pad), Y = (v) => H - pad - ((v + my) / (2 * my)) * (H - 2 * pad);
  const quad = [["Leading", W - pad - 4, pad + 12, "end", "--up"], ["Weakening", W - pad - 4, H - pad - 4, "end", "--warn"],
    ["Lagging", pad + 4, H - pad - 4, "start", "--down"], ["Improving", pad + 4, pad + 12, "start", "--accent"]];
  let out = `<line x1="${X(0)}" y1="${pad}" x2="${X(0)}" y2="${H - pad}" stroke="${cssVar("--line")}"/><line x1="${pad}" y1="${Y(0)}" x2="${W - pad}" y2="${Y(0)}" stroke="${cssVar("--line")}"/>`;
  out += quad.map(([t, x, y, a, c]) => `<text x="${x}" y="${y}" text-anchor="${a}" fill="${cssVar(c)}" font-size="11" opacity="0.8">${tr(t)}</text>`).join("");
  for (const p of pts) {
    const color = cssVar({ Leading: "--up", Weakening: "--warn", Lagging: "--down", Improving: "--accent" }[p.quadrant] || "--muted");
    const tail = p.rs_tail.map((v, i) => [v, p.mom_tail[i]]).filter(([a, b]) => isNum(a) && isNum(b));
    if (tail.length > 1) out += `<polyline fill="none" stroke="${color}" stroke-opacity="0.45" stroke-width="1.5" points="${tail.map(([a, b]) => `${X(a)},${Y(b)}`).join(" ")}"/>`;
    out += `<circle cx="${X(p.rs_trend)}" cy="${Y(p.rs_momentum)}" r="4.5" fill="${color}"><title>${p.symbol} ${p.name}: RS trend ${pct(p.rs_trend)}, momentum ${pct(p.rs_momentum)}</title></circle>`;
    out += `<text x="${X(p.rs_trend) + 6}" y="${Y(p.rs_momentum) - 6}" fill="${cssVar("--fg")}" font-size="11">${p.symbol}</text>`;
  }
  out += `<text x="${W / 2}" y="${H - 6}" text-anchor="middle" fill="${cssVar("--dim")}" font-size="10">RS trend (RS / SMA50(RS) − 1) →</text>`;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = out;
}

// ------------------------------------------------------------------ views: scanner
function earningsOf(c) { return ((state.snap.deep_dive[c.symbol] || {}).catalysts || {}).next_earnings; }
function scanner() {
  const s = state.snap, f = state.filters, side = (s.candidates.short || []).length ? state.scannerSide : "long", mode = state.scanMode || "cards";
  const all = s.candidates[side];
  const setups = [...new Set(all.map((c) => c.setup))];
  const rows = all.filter((c) => (!f.tier || c.tier === f.tier) && (!f.setup || c.setup === f.setup)
    && (!f.hideEarnings || !c.penalties.earnings_in_window) && (!f.q || c.symbol.includes(f.q.toUpperCase())));
  state.wallList = rows;
  app.innerHTML = `
  <div class="wall-head"><div><div class="eyebrow">Session ${esc(s.session.date)} / <b>Regime ${esc(tr(s.regime.label))}</b></div>
    <h1 class="display">Candidates <em>swing</em></h1>
    <p>Ranked by research priority - not trade signals. Every card and row has a recent price preview; click to open the full chart and research.</p></div>
    <div class="filters" style="margin:0">
      ${(s.candidates.short || []).length ? `<div class="seg" id="seg-side"><button data-v="long" class="${side === "long" ? "active" : ""}">Long ${s.candidates.long.length}</button><button data-v="short" class="${side === "short" ? "active" : ""}">Short ${s.candidates.short.length}</button></div>` : `<span class="chip">Long only · ${s.candidates.long.length}</span>`}
      <div class="seg" id="seg-mode"><button data-v="cards" class="${mode === "cards" ? "active" : ""}">Cards</button><button data-v="table" class="${mode === "table" ? "active" : ""}">Table</button></div>
    </div></div>
  <div class="filters">
    <label>Tier <select id="f-tier"><option value="">All</option>${["A", "B", "C", "Watch"].map((t) => `<option value="${t}" ${f.tier === t ? "selected" : ""}>${t === "Watch" ? "Watch" : t}</option>`).join("")}</select></label>
    <label>Setup <select id="f-setup"><option value="">All</option>${setups.map((x) => `<option value="${x}" ${f.setup === x ? "selected" : ""}>${esc(s.setup_names[x])}</option>`).join("")}</select></label>
    <label><input type="checkbox" id="f-earn" ${f.hideEarnings ? "checked" : ""}> Hide stocks with earnings in the window</label>
    <label>Search <input id="f-q" value="${esc(f.q)}" style="width:90px"></label>
    <span class="note">showing ${rows.length} </span>
  </div>
  ${mode === "cards" ? `<div class="wall">${rows.map((c, i) => stockCard(c, i)).join("") || '<div class="card empty">No stocks match the filters.</div>'}</div>` : `<div class="card">${scanTable(side, rows)}</div>`}`;
  app.querySelectorAll("#seg-side button").forEach((b) => (b.onclick = () => { state.scannerSide = b.dataset.v; state.filters.setup = ""; route(); }));
  app.querySelectorAll("#seg-mode button").forEach((b) => (b.onclick = () => { state.scanMode = b.dataset.v; localStorageSet("scanMode", b.dataset.v); route(); }));
  $("#f-tier").onchange = (e) => { f.tier = e.target.value; route(); };
  $("#f-setup").onchange = (e) => { f.setup = e.target.value; route(); };
  $("#f-earn").onchange = (e) => { f.hideEarnings = e.target.checked; route(); };
  $("#f-q").oninput = (e) => { f.q = e.target.value; clearTimeout(state.qt); state.qt = setTimeout(() => { route(); const i = $("#f-q"); i.focus(); i.setSelectionRange(i.value.length, i.value.length); }, 250); };
  if (mode === "cards") wireCards(rows);
  loadMinicharts(rows);
}

function scanTable(side, rows) {
  const cols = [
    { key: "rank", label: "#", n: 1 },
    { key: "symbol", label: "Symbol", render: (c) => `<b>${c.symbol}</b> <span class="muted">${esc(c.sector_etf || "")}</span>` },
    { key: "spark", label: "Trend (40d)", render: (c) => `<span class="spark" data-mini-sm="${esc(c.symbol)}"></span>`, sort: (c) => c.technical.return_20d },
    { key: "setup_name", label: "Setup", render: (c) => `${esc(c.setup_name)}${c.other_setups.length ? ` <span class="chip">+${c.other_setups.length}</span>` : ""}` },
    { key: "score", label: "Score", n: 1, render: (c) => `<div class="scorecell"><div class="bar"><span style="width:${c.score}%"></span></div><span>${c.score.toFixed(0)}</span></div>` },
    { key: "tier", label: "Tier", render: (c) => tier(c.tier), sort: (c) => c.score },
    { key: "entry", label: "Entry", n: 1, render: (c) => num(c.plan.entry) + (c.plan.conditional_entry ? ' <span class="chip">cond.</span>' : ""), sort: (c) => c.plan.entry },
    { key: "stop", label: "Stop", n: 1, render: (c) => num(c.plan.stop), sort: (c) => c.plan.stop },
    { key: "t1", label: "T1", n: 1, render: (c) => num(c.plan.target1), sort: (c) => c.plan.target1 },
    { key: "rr", label: "Reward/risk", n: 1, render: (c) => `<span class="${c.plan.reward_risk >= 2 ? "up" : c.plan.reward_risk < 1.5 ? "down" : ""}">${num(c.plan.reward_risk, 1)}</span>`, sort: (c) => c.plan.reward_risk },
    { key: "risk", label: "Risk %", n: 1, render: (c) => pct(c.plan.risk_pct, 1, false), sort: (c) => c.plan.risk_pct },
    { key: "rs", label: "RS percentile", n: 1, render: (c) => num(c.technical.rs_percentile, 0), sort: (c) => c.technical.rs_percentile },
    { key: "earn", label: "Earnings", render: (c) => { const e = earningsOf(c); return e ? `<span class="${e.sessions_until <= 7 ? "warn" : ""}">${e.date} (${e.sessions_until})</span>` : '<span class="muted">—</span>'; }, sort: (c) => (earningsOf(c) || {}).sessions_until ?? 999 },
    { key: "ev", label: "Evidence +/−", n: 1, render: (c) => `<span class="up">${c.supporting.length}</span> / <span class="down">${c.contradicting.length}</span>`, sort: (c) => c.supporting.length - c.contradicting.length },
    { key: "pen", label: "Penalties", render: (c) => Object.keys(c.penalties).map((k) => `<span class="chip">${tr(k)}</span>`).join("") || '<span class="muted">—</span>', sort: (c) => c.penalty_total },
  ];
  return table("scan-" + side, cols, rows, { sortKey: "rank", sortDir: "asc", onClick: true, rowAttr: (c) => `data-sym="${c.symbol}"` });
}

function ring(score, size = 46) {
  const r = size / 2 - 4, circ = 2 * Math.PI * r, v = Math.max(0, Math.min(100, score || 0));
  return `<svg class="ring" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" aria-label="score ${v.toFixed(0)}">
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="var(--chip2)" stroke-width="3.5"/>
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="var(--fg)" stroke-width="3.5" stroke-linecap="round"
      stroke-dasharray="${(circ * v / 100).toFixed(1)} ${circ.toFixed(1)}" transform="rotate(-90 ${size / 2} ${size / 2})" style="filter:drop-shadow(0 0 4px rgba(255,255,255,.5))"/>
    <text x="50%" y="50%" text-anchor="middle" dominant-baseline="central">${v.toFixed(0)}</text></svg>`;
}

function stockCard(c, i) {
  const t = c.technical, deep = state.snap.deep_dive[c.symbol] || {}, e = earningsOf(c);
  const name = (deep.profile || {}).longName || c.sector_etf || "";
  const pens = Object.keys(c.penalties).map((k) => `<span class="chip">${tr(k)}</span>`).join("");
  return `<div><div class="scard" tabindex="0" role="button" data-i="${i}" aria-label="${esc(c.symbol)} ${esc(c.setup_name)}, score ${c.score.toFixed(0)}">
    <div class="sc-top z1"><span class="sc-label">${esc(tr(c.direction))} · <b>${String(c.rank).padStart(2, "0")}</b></span>${tier(c.tier)}</div>
    <div class="sc-id z2"><div><div class="sc-sym">${esc(c.symbol)}</div><div class="sc-name">${esc(name)}</div></div>
      <div class="sc-px"><div class="p">${num(t.close)}</div><div class="c ${cls(t.return_1d)}">${pct(t.return_1d, 2)}</div></div></div>
    <div class="sc-chart z1" data-mini="${esc(c.symbol)}"></div>
    <div class="sc-setup z1">${ring(c.score)}<div class="s">${esc(c.setup_name)}<div class="note">${esc(tr(t.trend))} · ${esc(tr(t.swing_structure))}</div></div></div>
    ${c.setup === "crash_rebound" ? `<div class="sc-bias"><b class="up">Counter-trend bounce</b> · buy within 0–2 days of the crash · stop 2 ATR · T1 1.5 ATR</div>` : c.analyst ? `<div class="sc-bias"><b class="${biasClass(c.analyst.bias)}">${esc(c.analyst.bias)}</b> · confidence ${esc(c.analyst.confidence)} — ${esc(c.analyst.headline.split(": ").slice(1).join(": "))}</div>` : ""}
    <div class="sc-stats z1"><div><span>Reward/risk</span><b class="${c.setup === "crash_rebound" ? "" : c.plan.reward_risk >= 2 ? "up" : c.plan.reward_risk < 1.5 ? "down" : ""}">${num(c.plan.reward_risk, 1)}</b></div>
      <div><span>Risk</span>${pct(c.plan.risk_pct, 1, false)}</div><div><span>Beta</span>${num(t.beta120, 2)}</div>
      <div><span>Earnings</span><b class="${e && e.sessions_until <= 7 ? "warn" : ""}">${e ? e.sessions_until + " days" : "—"}</b></div></div>
    <div class="sc-flags z1">${pens}</div></div></div>`;
}

// Cards open the focus view on click / Enter.
function wireCards(list) {
  app.querySelectorAll(".scard").forEach((card) => {
    const i = Number(card.dataset.i);
    card.addEventListener("click", () => openFocus(list, i, card));
    card.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openFocus(list, i, card); } });
  });
}

async function loadMinicharts(rows) {
  if (!rows.length) return;
  state.mini = state.mini || {};
  const key = state.session, cache = (state.mini[key] = state.mini[key] || {});
  const need = [...new Set(rows.map((c) => c.symbol))].filter((x) => !cache[x]);
  if (need.length) {
    try { Object.assign(cache, await api(`/api/minicharts?days=60&session=${key}&symbols=${need.join(",")}`)); } catch (_) { /* keep empty */ }
  }
  rows.forEach((c) => {
    if (!cache[c.symbol]) return;
    app.querySelectorAll(`[data-mini="${c.symbol}"]`).forEach((el) => { el.innerHTML = miniChart(cache[c.symbol], c.plan); });
    app.querySelectorAll(`[data-mini-sm="${c.symbol}"]`).forEach((el) => { el.innerHTML = miniChart(cache[c.symbol], c.plan, true); });
  });
}

function miniChart(d, plan, small = false) {
  const P = PAL(), W = small ? 160 : 300, H = small ? 44 : 132, R = small ? 0 : 26;
  if (small) d = { ...d, ...Object.fromEntries(["t", "o", "h", "l", "c", "ema9", "ema21", "ema50"].map((k) => [k, d[k].slice(-40)])) };
  const n = d.c.length;
  const lows = d.l.filter(isNum), highs = d.h.filter(isNum);
  let lo = Math.min(...lows), hi = Math.max(...highs);
  const span0 = hi - lo;
  for (const v of [plan.stop, plan.target1, plan.entry]) if (isNum(v) && v > lo - span0 * 0.6 && v < hi + span0 * 0.6) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  const step = (W - R) / n, x = (i) => i * step + step / 2, y = (v) => H - ((v - lo) / (hi - lo)) * H;
  let g = "";
  for (let i = 0; i < n; i++) {
    if (![d.o[i], d.h[i], d.l[i], d.c[i]].every(isNum)) continue;
    const up = d.c[i] >= d.o[i], col = up ? P.up : P.down, top = y(Math.max(d.o[i], d.c[i])), bot = y(Math.min(d.o[i], d.c[i]));
    g += `<line x1="${x(i).toFixed(1)}" x2="${x(i).toFixed(1)}" y1="${y(d.h[i]).toFixed(1)}" y2="${y(d.l[i]).toFixed(1)}" stroke="${col}" stroke-width="1"/>`;
    g += `<rect x="${(x(i) - step * 0.32).toFixed(1)}" y="${top.toFixed(1)}" width="${(step * 0.64).toFixed(1)}" height="${Math.max(1, bot - top).toFixed(1)}" fill="${col}" rx="0.6"/>`;
  }
  const poly = (arr, col, w, op = 1) => `<polyline fill="none" stroke="${col}" stroke-width="${w}" stroke-opacity="${op}" stroke-linejoin="round" points="${arr.map((v, i) => (isNum(v) ? `${x(i).toFixed(1)},${y(v).toFixed(1)}` : "")).filter(Boolean).join(" ")}"/>`;
  const level = (v, col, label, dash) => (isNum(v) && v >= lo && v <= hi ? `<line x1="0" x2="${W - R + 2}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}" stroke="${col}" stroke-width="1" stroke-dasharray="${dash}" stroke-opacity=".9"/><text x="${W - R + 5}" y="${(y(v) + 3.5).toFixed(1)}" fill="${col}" font-size="9.5" font-family="JetBrains Mono, monospace">${label}</text>` : "");
  if (small) return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${poly(d.ema21, P.ema21, 1)}${poly(d.ema9, P.ema9, 1)}${g}</svg>`;
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${poly(d.ema50, P.ema50, 1, 0.55)}${poly(d.ema21, P.ema21, 1.3)}${poly(d.ema9, P.ema9, 1.3)}${g}
    ${level(plan.target1, P.up, "T1", "4 3")}${level(plan.entry, P.neutral, plan.conditional_entry ? "TR" : "IN", "2 3")}${level(plan.stop, P.down, "SL", "4 3")}</svg>`;
}

// ------------------------------------------------------------------ focus view (expanded card)
function quickFacts(c, deep) {
  const f = (deep.fundamentals || {}).metrics || {}, o = deep.options || {}, p = deep.positioning || {}, e = earningsOf(c);
  const items = [
    ["Next earnings", e ? `${e.date} · ${e.sessions_until} days` : "—", e && e.sessions_until <= 7 ? "warn" : ""],
    ["Revenue YoY", pct(f.revenue_yoy, 0), cls(f.revenue_yoy)], ["EPS YoY", pct(f.eps_yoy, 0), cls(f.eps_yoy)],
    ["Operating margin", pct(f.operating_margin_ttm, 1, false), ""], ["Free cash flow", money(f.fcf_ttm), cls(f.fcf_ttm)], ["P/E", num(f.pe_ttm, 1), ""],
    ["ATM implied volatility", o.available ? pct(o.atm_iv, 0, false) : "n/a", ""], ["IV / HV20", o.available ? num(o.iv_hv_ratio, 2) : "n/a", o.iv_hv_ratio > 1.4 ? "warn" : ""],
    ["Options expected move", o.available ? "±" + pct(o.expected_move_pct, 1, false) : "n/a", ""], ["Put/call volume", o.available ? num(o.pcr_volume, 2) : "n/a", ""],
    ["Short % of float", pct(p.short_percent_float, 1, false), p.short_percent_float >= 0.15 ? "warn" : ""], ["Days to cover", num(p.days_to_cover, 1), ""],
    ["Beta (120d)", num(c.technical.beta120, 2), ""], ["60d volatility", pct(c.technical.hv60, 0, false), ""],
  ];
  return `<div class="facts">${items.map(([k, v, cl]) => `<div class="fact"><span>${k}</span><b class="${cl}">${v}</b></div>`).join("")}</div>`;
}

function focusHTML(list, i) {
  const c = list[i], deep = state.snap.deep_dive[c.symbol] || {};
  return `<div class="focus-body">${stockShell(c.symbol, c, deep, { focus: true, prefix: "f", big: 460 })}</div>`;
}

async function fillFocus(bd, list, i) {
  const panel = bd.querySelector(".focus-panel");
  disposeCharts(state.focusCharts);
  bd.dataset.i = i;
  panel.innerHTML = focusHTML(list, i);
  panel.setAttribute("aria-label", list[i].symbol);
  panel.querySelectorAll("[data-act]").forEach((b) => (b.onclick = (ev) => {
    ev.stopPropagation();
    const act = b.dataset.act;
    if (act === "close") closeFocus();
    else if (act === "full") { const sym = list[i].symbol; closeFocus(true); location.hash = `#/ticker/${sym}`; }
    else fillFocus(bd, list, (i + (act === "next" ? 1 : -1) + list.length) % list.length);
  }));
  const c = list[i];
  fillStock(c.symbol, c, state.snap.deep_dive[c.symbol] || {}, "f", state.focusCharts, 460, () => Number(bd.dataset.i) === i && bd.isConnected);
}

function openFocus(list, i, fromEl) {
  closeFocus(true);
  state.focusCharts = state.focusCharts || [];
  const bd = document.createElement("div");
  bd.className = "focus-backdrop";
  bd.innerHTML = `<div class="focus-panel" role="dialog" aria-modal="true"></div>`;
  document.body.appendChild(bd);
  document.body.style.overflow = "hidden";
  state.focusEl = bd;
  const panel = bd.querySelector(".focus-panel");
  panel.innerHTML = focusHTML(list, i);
  const body = panel.querySelector(".focus-body");
  if (fromEl && !reduceMotion()) {           // FLIP: grow the glass card into the panel
    const a = fromEl.getBoundingClientRect(), b = panel.getBoundingClientRect();
    panel.style.transform = `translate(${a.left - b.left}px, ${a.top - b.top}px) scale(${a.width / b.width}, ${a.height / b.height})`;
    body.style.opacity = "0";
    panel.getBoundingClientRect();
    requestAnimationFrame(() => { bd.classList.add("on"); panel.classList.add("anim"); panel.style.transform = ""; body.style.opacity = "1"; });
  } else bd.classList.add("on");
  bd.addEventListener("click", (ev) => { if (ev.target === bd) closeFocus(); });
  state.focusKeys = (ev) => {
    if (ev.key === "Escape") closeFocus();
    else if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") fillFocus(bd, list, (Number(bd.dataset.i) + (ev.key === "ArrowRight" ? 1 : -1) + list.length) % list.length);
  };
  document.addEventListener("keydown", state.focusKeys);
  setTimeout(() => { if (bd.isConnected) fillFocus(bd, list, i); }, fromEl && !reduceMotion() ? 520 : 0);
}

function closeFocus(instant = false) {
  const bd = state.focusEl;
  if (!bd) return;
  state.focusEl = null;
  document.removeEventListener("keydown", state.focusKeys);
  document.body.style.overflow = "";
  disposeCharts(state.focusCharts || []);
  if (instant || reduceMotion()) { bd.remove(); return; }
  bd.classList.remove("on");
  setTimeout(() => bd.remove(), 320);
}

// ------------------------------------------------------------------ stock page (focus view + full page share one layout)
// Top: big K-line (left) + Futu-style quote (right). Then the key points only: verdict, plan numbers, top
// bull / bear factors. Everything else sits in collapsed sections the user opens on demand.
const VPC = { va: "rgba(167,139,250,.42)", vah: "#22d3ee", poc: "#f472b6", val: "#fb923c" };   // value area / VAH / POC / VAL
const fmtCN = (v) => (!isNum(v) ? "—" : Math.abs(v) >= 1e12 ? (v / 1e12).toFixed(2) + "T" : Math.abs(v) >= 1e9 ? (v / 1e9).toFixed(2) + "B"
  : Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(2) + "M" : Math.abs(v) >= 1e4 ? (v / 1e3).toFixed(1) + "K" : num(v, 0));
const clip = (t, n = 120) => { t = String(t || ""); return t.length > n ? t.slice(0, n - 1) + "…" : t; };

function vpOverlay(chart, series, el, vp) {
  if (!vp || !isNum(vp.poc)) return;
  el.style.position = "relative";
  const box = document.createElement("div");
  box.className = "vp-strip";
  box.innerHTML = `<i class="va"></i><i class="vah"></i><i class="poc"></i><i class="val"></i>`;
  el.appendChild(box);
  const [va, vah, poc, val] = box.querySelectorAll("i");
  const place = () => {
    if (chart._gone) return;
    const y = (p) => series.priceToCoordinate(p);
    const yh = y(vp.vah), yl = y(vp.val), yp = y(vp.poc);
    const w = el.clientWidth - chart.priceScale("right").width();
    box.style.left = `${Math.max(0, w - 64)}px`;   // in the right-offset gap, clear of the price-line titles
    const set = (node, top, h) => { if (top === null || !isFinite(top)) { node.style.display = "none"; return; } node.style.display = ""; node.style.top = `${top}px`; node.style.height = `${h}px`; };
    if (yh !== null && yl !== null) set(va, Math.min(yh, yl), Math.max(2, Math.abs(yl - yh))); else va.style.display = "none";
    set(vah, yh - 1, 2); set(val, yl - 1, 2); set(poc, yp - 1.5, 3);
  };
  const line = (price, color, title) => series.createPriceLine({ price, color, lineWidth: 1, lineStyle: 3, lineVisible: false, axisLabelVisible: true, title });
  line(vp.vah, VPC.vah, "VAH"); line(vp.poc, VPC.poc, "POC"); line(vp.val, VPC.val, "VAL");
  const tick = () => requestAnimationFrame(place);
  chart.timeScale().subscribeVisibleLogicalRangeChange(tick);
  chart.subscribeCrosshairMove(tick);
  new ResizeObserver(tick).observe(el);
  setTimeout(place, 60);
}

function quoteHTML(d, deep, c, uni) {
  const q = d.quote || {}, prof = deep.profile || {}, pos = deep.positioning || {}, fm = (deep.fundamentals || {}).metrics || {}, o = deep.options || {};
  const pc = q.prev_close, col = (v) => (isNum(v) && isNum(pc) ? (v > pc ? "up" : v < pc ? "down" : "") : "");
  const shares = prof.sharesOutstanding || pos.shares_outstanding || fm.shares_outstanding;
  const flt = prof.floatShares || pos.float_shares;
  const cap = prof.marketCap || (uni || {}).market_cap || fm.market_cap || (isNum(shares) ? shares * q.close : null);
  const pe = isNum(prof.trailingPE) ? num(prof.trailingPE, 1) : isNum(fm.pe_ttm) ? (fm.pe_ttm > 0 ? num(fm.pe_ttm, 1) : "loss") : isNum(fm.eps_ttm) && fm.eps_ttm < 0 ? "loss" : "—";
  const fpe = isNum(prof.forwardPE) ? (prof.forwardPE > 0 ? num(prof.forwardPE, 1) : "loss") : "—";
  const dy = prof.trailingAnnualDividendYield;
  const sinceY = (q.hist_since || "").slice(0, 4);
  const e = (deep.catalysts || {}).next_earnings;
  const rows = [
    ["High", num(q.high), col(q.high)], ["Low", num(q.low), col(q.low)],
    ["Open", num(q.open), col(q.open)], ["Prev close", num(pc), "muted"],
    ["Volume", fmtCN(q.volume), ""], ["Dollar vol.", fmtCN(q.dollar_volume), ""],
    ["Market cap", fmtCN(cap), ""], ["Shares out", fmtCN(shares), ""],
    ["Float value", fmtCN(isNum(flt) ? flt * q.close : null), ""], ["Float", fmtCN(flt), ""],
    ["P/E TTM", pe, ""], ["Forward P/E", fpe, ""],
    ["P/B", num(prof.priceToBook, 2), ""], ["Dividend yield TTM", isNum(dy) && dy > 0 ? pct(dy, 2, false) : "—", ""],
    ["Vol. ratio", num(q.volume_ratio, 2), ""], ["Amplitude", pct(q.amplitude, 2, false), ""],
    ["52w high", num(q.high52), "up"], ["52w low", num(q.low52), "down"],
    [`High since ${sinceY || "start"}`, num(q.hist_high), "up"], [`Low since ${sinceY || "start"}`, num(q.hist_low), "down"],
    ["Avg price (est.)", num(q.avg_price), col(q.avg_price)], ["Turnover", isNum(flt) && flt > 0 ? pct(q.volume / flt, 2, false) : "—", ""],
    ["Beta", num(q.beta, 3), ""], ["IV", o.available ? pct(o.atm_iv, 2, false) : "—", ""],
    ["IV Rank", isNum(o.iv_rank) ? num(o.iv_rank, 0) : "—", ""], ["Short % float", pct(pos.short_percent_float, 1, false), pos.short_percent_float >= 0.15 ? "warn" : ""],
    ["ATR(14)", num(q.atr), ""], ["Next earnings", e ? `${e.date.slice(5)} (${e.sessions_until} days)` : "—", e && e.sessions_until <= 7 ? "warn" : ""],
  ];
  return `<div class="q-top"><div class="q-px ${cls(q.change)}">${num(q.close, q.close < 10 ? 3 : 2)}<span class="q-arrow">${q.change > 0 ? "↑" : q.change < 0 ? "↓" : ""}</span></div>
      <div class="q-ch ${cls(q.change)}">${isNum(q.change) ? (q.change > 0 ? "+" : "") + num(q.change, q.close < 10 ? 3 : 2) : ""} ${pct(q.change_pct, 2)}</div></div>
    <div class="q-time">Close ${esc(d.as_of)} 16:00 ET</div>
    <div class="q-grid">${rows.map(([k, v, cl]) => `<div><span>${k}</span><b class="${cl}">${v === "n/a" || v === "n/a%" ? "—" : v}</b></div>`).join("")}</div>
    ${isNum(o.iv_rank) ? "" : `<p class="note q-note">IV Rank needs 60 sessions of stored implied volatility (${o.iv_history_n ?? 0} days so far). Average price is estimated as (high + low + close) ÷ 3.</p>`}`;
}

function factorLists(a) {
  const sc = (a && a.scorecard) || [];
  const w = (x) => x.score * x.weight;
  const bull = sc.filter((x) => x.score >= 0.25).sort((x, y) => w(y) - w(x)).slice(0, 4);
  const bear = sc.filter((x) => x.score <= -0.25).sort((x, y) => w(x) - w(y)).slice(0, 4);
  const li = (x) => `<li title="${esc(x.text)}"><b>${esc(x.name)}</b><span class="${cls(x.score)}">${x.score > 0 ? "+" : ""}${x.score.toFixed(1)}</span><p>${esc(clip(x.text, 120))}</p></li>`;
  return { bull: bull.map(li).join("") || '<li class="muted">No clear bullish factors</li>', bear: bear.map(li).join("") || '<li class="muted">No clear bearish factors</li>' };
}

function keyHTML(a, c) {
  const f = factorLists(a);
  const p = c && c.plan;
  const pens = c ? Object.entries(c.penalties || {}).map(([k, v]) => `<span class="chip warn">${tr(k)} −${v}</span>`).join("") : "";
  const crash = c && c.setup === "crash_rebound";
  const b = state.btFull, ss = !crash && b && c && (b.selected_side || {})[c.direction];
  const k = ss && ss.stop.startsWith("atr") ? Number(ss.stop.slice(3)) : null, sg = c && c.direction === "long" ? 1 : -1;
  return `<section class="key">
    ${crash ? `<div class="key-crash"><b>Crash rebound · counter-trend</b> The trend read below is bearish by design - this is a short bounce trade after a crash,
      not a trend call. 5-year study: buying 0–2 days after such a crash beat random entries by ~+2.5% over 5 days; 2-ATR stop, quick 1.5-ATR target, 10-day time stop.</div>` : ""}
    ${a ? `<div class="key-head"><div class="key-bias ${biasClass(a.bias)}">${esc(a.bias)}<small>${a.bias_score > 0 ? "+" : ""}${a.bias_score.toFixed(0)} / 100 · confidence ${esc(a.confidence)}</small></div>
      <h2 class="key-hl">${esc(String(a.headline || "").replace(new RegExp("^" + a.bias + ":\\s*"), ""))}</h2></div>` : '<div class="key-head"><p class="note">Building the analyst call…</p></div>'}
    ${p ? `<div class="key-plan">
      <div><span>${p.conditional_entry ? "Trigger" : "Ref. entry"}</span><b>${num(p.entry)}</b></div>
      <div><span>Stop</span><b class="down">${num(p.stop)}</b><small>${pct(-Math.abs(p.entry - p.stop) / p.entry, 1)}</small></div>
      <div><span>T1</span><b class="up">${num(p.target1)}</b><small>${pct(Math.abs(p.target1 - p.entry) / p.entry * sg, 1)}</small></div>
      <div><span>Reward/risk</span><b>${num(p.reward_risk, 1)}R</b></div>
      <div><span>Max hold</span><b>${p.horizon_days} days</b></div>
      ${ss && k && isNum(p.atr) ? `<div class="alt"><span>Backtest's pick: ${k} ATR stop + trailing stop</span><b class="down">${num(p.entry - sg * k * p.atr)}</b><small>Out of sample ${num(ss.oos.avg_r)}R</small></div>` : ""}
    </div>` : ""}
    <div class="key-cols"><div class="bull"><h3>BULL CASE</h3><ol>${f.bull}</ol></div><div class="bear"><h3>BEAR CASE</h3><ol>${f.bear}</ol></div></div>
    ${c && c.news && c.news.items && c.news.items.length ? `<div class="key-news"><h3>Latest news <small>${c.news.n_recent} in ${c.news.days} days${(c.news.flags || []).length ? " · " + c.news.flags.join(", ") : ""}</small></h3>${newsList(c.news.items.slice(0, 3), false)}</div>` : ""}
    ${pens || (a && a.watch && a.watch.length) ? `<div class="key-foot">${pens}${a && a.watch ? a.watch.slice(0, 3).map((x) => `<span class="chip">Watch: ${esc(x)}</span>`).join("") : ""}</div>` : ""}
  </section>`;
}

function fold(title, teaser, body, open = false) {
  if (!body) return "";
  return `<details class="fold"${open ? " open" : ""}><summary><span class="ft">${title}</span><span class="fs">${teaser || ""}</span></summary><div class="fold-body">${body}</div></details>`;
}

function foldsHTML(sym, c, deep, a, extra = "", pre = "") {
  const o = deep.options || {}, pos = deep.positioning || {}, cat = deep.catalysts || {}, fm = (deep.fundamentals || {}).metrics || {}, es = deep.earnings_study || {}, ftd = deep.ftd || {};
  const e = cat.next_earnings;
  return `<div class="folds">
    ${pre}
    ${fold("Full analyst note", a ? `${a.scorecard.length} dimensions · ${(a.techniques || {}).count || 0} techniques · Scenarios` : "", `<div class="fold-analyst">${a ? analystPanel(a) : '<p class="note">Loading…</p>'}</div>`)}
    ${c ? fold("Risk plan · Why these levels", `Stop ${num(c.plan.stop)} · Target ${num(c.plan.target1)} / ${num(c.plan.target2)}`, planPanel(c)) : ""}
    ${c ? fold("Score breakdown · Setup history", `${num(c.score, 0)} (penalties ${num(c.penalty_total, 0)})`, candidatePanels(c, deep)) : ""}
    ${deep.earnings_study ? fold("Earnings reactions", es.available ? (es.implied && isNum(es.implied.move_pct) ? `Options implied ±${pct(es.implied.move_pct, 1, false)} vs history ±${pct(es.median_abs_move, 1, false)}` : `Median historical move ±${pct(es.median_abs_move, 1, false)}`) : "No data", earningsPanel(es)) : ""}
    ${deep.options ? fold("Options", o.available ? `IV ${pct(o.atm_iv, 0, false)} · expected move ±${pct(o.expected_move_pct, 1, false)} · put/call ${num(o.pcr_volume, 2)}` : "No options data", optionsPanel(o)) : ""}
    ${deep.positioning ? fold("Hold · Short", `Short ${pct(pos.short_percent_float, 1, false)} Float · cover ${num(pos.days_to_cover, 1)} days`, positioningPanel(pos)) : ""}
    ${deep.catalysts ? fold("Catalysts", e ? `Next earnings ${e.date} (${e.sessions_until} days)` : "No earnings date", catalystsPanel(cat)) : ""}
    ${deep.fundamentals ? fold("Fundamentals", deep.fundamentals.available ? `Revenue ${pct(fm.revenue_yoy, 0)} · EPS ${pct(fm.eps_yoy, 0)} · Operating margin ${pct(fm.operating_margin_ttm, 0, false)}` : "Unavailable", fundamentalsPanel(deep.fundamentals)) : ""}
    ${deep.ftd ? fold("SEC fails to deliver (FTD)", ftd.available ? (ftd.n ? `High ${fmtVol(ftd.max)} shares (${pct(ftd.max_vs_adv, 1, false)} of avg volume)` : "none") : "needs SEC_USER_AGENT", ftdPanel(ftd)) : ""}
    ${extra}
  </div>`;
}

function crashTeaser(ch) {
  const s = ch.summary || {};
  if (!s.n) return "No crash in the stored history";
  const d0 = s.d0 || {};
  return `${s.n} crashes since ${String(s.since).slice(0, 4)} · buying the next day: avg ${pct(d0.mean5, 1)} after 5 days, ${pct(d0.win5, 0, false)} up`;
}

function crashHistoryHTML(ch) {
  const s = ch.summary || {};
  if (!s.n) return '<p class="note">No crash (5-day drop ≥ 20%, 3-day drop ≥ 15% or ≥ 3 ATR below EMA21) in the stored history.</p>';
  const cell = (k) => s[k] ? `<div class="kpi"><span class="l">Buy ${k === "d0" ? "next day" : k === "d1" ? "1 day later" : "2 days later"}</span>
      <span class="v ${cls(s[k].mean5)}">${pct(s[k].mean5, 1)}</span><span class="note">avg 5-day return · ${pct(s[k].win5, 0, false)} up · 10-day ${pct(s[k].mean10, 1)}</span></div>` : "";
  return `<div class="grid g3 kpis">${cell("d0")}${cell("d1")}${cell("d2")}</div>
    ${table("crash-h", [{ key: "date", label: "Crash day" }, { key: "drop5", label: "5-day move", n: 1, render: (r) => `<span class="down">${pct(r.drop5, 1)}</span>` },
      { key: "below_ema21_atr", label: "ATR below EMA21", n: 1, render: (r) => num(r.below_ema21_atr, 1) },
      { key: "d0_ret5", label: "Next day → 5d", n: 1, render: (r) => `<span class="${cls(r.d0_ret5)}">${pct(r.d0_ret5, 1)}</span>` },
      { key: "d1_ret5", label: "+1 day → 5d", n: 1, render: (r) => `<span class="${cls(r.d1_ret5)}">${pct(r.d1_ret5, 1)}</span>` },
      { key: "d2_ret5", label: "+2 days → 5d", n: 1, render: (r) => `<span class="${cls(r.d2_ret5)}">${pct(r.d2_ret5, 1)}</span>` },
      { key: "d0_ret10", label: "Next day → 10d", n: 1, render: (r) => `<span class="${cls(r.d0_ret10)}">${pct(r.d0_ret10, 1)}</span>` },
      { key: "d0_mae10", label: "Worst dip after", n: 1, render: (r) => `<span class="down">${pct(r.d0_mae10, 1)}</span>` }], ch.events, { sortKey: "date" })}
    <p class="note">Crash = the first of: 5-day drop ≥ 20%, 3-day drop ≥ 15% or close ≥ 3 ATR below EMA21, at most one per 10 sessions. Returns from the next session's open.
      Each stock has only a handful of crashes, so its own "best day" is mostly noise: in the 5-year study a stock's own best entry day did <b>worse</b> out of sample than the universe-wide rule (buy 0–2 days after the crash).</p>`;
}

function stockShell(sym, c, deep, opts = {}) {
  const p = opts.prefix || "c", name = (deep.profile || {}).longName || "", g = c && c.group;
  return `<div class="sv">
    <div class="sv-head"><div>
        <div class="sv-sym">${esc(sym)}<span class="sv-name">${esc(name)}</span></div>
        <div class="sv-tags">${c ? `<span class="chip ${c.direction}">${esc(tr(c.direction))}</span><span class="chip">${esc(c.setup_name)}</span>${tier(c.tier)}<span class="chip">Score ${num(c.score, 0)}</span>` : '<span class="chip">Not a candidate this session</span>'}
          ${g ? `<span class="chip">${esc(g.group)} · group rank ${g.rank} / ${g.of}</span>` : ""}</div></div>
      ${opts.focus ? `<div class="focus-tools"><button data-act="prev" aria-label="Previous" title="Previous (←)">‹</button><button data-act="next" aria-label="Next" title="Next (→)">›</button>
        <button class="wide" data-act="full">Full page ↗</button><button data-act="close" aria-label="Close" title="Close (Esc)">✕</button></div>` : ""}</div>
    <div class="sv-top">
      <div class="sv-chart">${chartBlock(p, opts.big || 480)}
        <div class="vp-legend"><span><i style="background:${VPC.va}"></i>Value area VA (70% of volume)</span><span><i style="background:${VPC.vah}"></i>VAH value-area high</span>
          <span><i style="background:${VPC.poc}"></i>POC most traded price</span><span><i style="background:${VPC.val}"></i>VAL value-area low</span><span id="${p}-vp-txt" class="muted"></span></div></div>
      <aside class="sv-quote" id="${p}-quote"><p class="note">Loading quote…</p></aside>
    </div>
    <div id="${p}-key">${keyHTML(c && c.analyst, c)}</div>
    <div id="${p}-folds">${foldsHTML(sym, c, deep, c && c.analyst, (opts.extra || ""),
      fold("News", `<span id="${p}-news-teaser">${c && c.news ? `${c.news.n_recent} headlines in ${c.news.days} days` : "loading…"}</span>`, `<div id="${p}-news" class="note">Loading…</div>`)
      + fold("Crash history · buy the plunge", `<span id="${p}-crash-teaser">loading…</span>`, `<div id="${p}-crash" class="note">Loading…</div>`, !!(c && c.setup === "crash_rebound")))}</div>
  </div>`;
}

async function fillStock(sym, c, deep, p, bucket, big, isCurrent = () => true) {
  const uni = state.snap.universe.find((u) => u.symbol === sym);
  if (!(c && c.analyst)) {
    api(`/api/analyst/${encodeURIComponent(sym)}?session=${state.session}&live=true`).then((r) => {
      if (!isCurrent() || !r || !r.note) return;
      const k = $(`#${p}-key`); if (k) k.innerHTML = keyHTML(r.note, c);
      const fa = document.querySelector(`#${p}-folds .fold-analyst`); if (fa) fa.innerHTML = analystPanel(r.note);
    }).catch(() => { const k = $(`#${p}-key .key-head`); if (k) k.innerHTML = '<p class="note">Could not build the analyst call.</p>'; });
  }
  api(`/api/news/${encodeURIComponent(sym)}/live`).then((r) => {
    if (!isCurrent()) return;
    const el = $(`#${p}-news`), t = $(`#${p}-news-teaser`);
    if (el) el.innerHTML = newsList(r.items, false, "No headlines in the last three weeks.");
    if (t) t.textContent = `${r.summary.n_recent} in ${r.summary.days} days · ${r.items.length} in 3 weeks` + ((r.summary.flags || []).length ? ` · ${r.summary.flags.join(", ")}` : "");
    if (c && c.setup === "crash_rebound") {
      const ce = $(`#${p}-crash`); const recent = r.items.filter((x) => x.published && Date.now() - Date.parse(x.published) < 6 * 864e5);
      if (ce && !ce.querySelector(".why")) ce.insertAdjacentHTML("afterbegin", `<div class="why"><h3>Why did it crash? <small>headlines from the last 6 days</small></h3>${newsList(recent.slice(0, 8), false, "No headlines found around the crash - possibly a market-wide or sector move.")}</div>`);
    }
  }).catch(() => { const el = $(`#${p}-news`); if (el) el.innerHTML = c && c.news ? newsList(c.news.items, false) : '<p class="note">News unavailable.</p>'; });
  try {
    const d = await api(`/api/chart/${encodeURIComponent(sym)}?days=180&session=${state.session}`);
    if (!isCurrent()) return null;
    renderTickerCharts(d, c, p, bucket, big, 74);
    $(`#${p}-quote`).innerHTML = quoteHTML(d, deep, c, uni);
    const vp = d.volume_profile, t = $(`#${p}-vp-txt`);
    if (t) t.textContent = vp ? `Last ${vp.window}d: VA ${num(vp.val)}–${num(vp.vah)} · POC ${num(vp.poc)}` : "";
    const ch = d.crash_history, ce = $(`#${p}-crash`), ct = $(`#${p}-crash-teaser`);
    if (ce && ch) { const why = ce.querySelector(".why"); ce.innerHTML = crashHistoryHTML(ch); if (why) ce.prepend(why); if (ct) ct.textContent = crashTeaser(ch); }
    return d;
  } catch (err) {
    const el = $(`#${p}-price`); if (el) el.innerHTML = `<div class="empty">Could not load the chart: ${esc(err.message)}</div>`;
    return null;
  }
}

// ------------------------------------------------------------------ views: ticker
const levelLabel = (t) => String(t || "").replace(/(\d+)D (high|low)/g, "$1-day $2");
async function ticker(sym) {
  sym = (sym || "").toUpperCase();
  const s = state.snap;
  const cands = [...s.candidates.long, ...s.candidates.short].filter((c) => c.symbol === sym);
  const deep = s.deep_dive[sym] || {};
  const uni = s.universe.find((u) => u.symbol === sym);
  const c = cands[0];
  const extra = fold("Technical snapshot · Support / resistance", uni ? `Trend ${esc(tr(uni.trend))} · RS # ${num(uni.rs_pct, 0)} percentile` : "", `<div id="t-snap" class="kv"></div><h3>Support / resistance (60 sessions)</h3><div id="t-levels" class="note">…</div>`)
    + fold("Candidate history", "past appearances in the candidate lists", `<div id="t-hist" class="note">…</div>`);
  app.innerHTML = stockShell(sym, c, deep, { prefix: "c", big: 520, extra });
  const d = await fillStock(sym, c, deep, "c", undefined, 520);
  if (d) {
    const L = d.last;
    $("#t-snap").innerHTML = [["Close", num(L.close)], ["ATR(14)", `${num(L.atr)} (${num(L.atr_pct, 1)}%)`], ["RSI6 / RSI14", `${num(L.rsi6, 0)} / ${num(L.rsi14, 0)}`],
      ["Trend", esc(L.trend)], ["Swing structure", esc(L.swing_structure)], ["Structure state", esc(L.structure)], ["Swing condition", esc(L.swing_condition)],
      ["Volume / MAVOL20", num(L.volume_ratio, 2) + "×"], ["Extension (ATR from EMA21)", num(L.extension_atr, 2)], ["HV20", pct(L.hv20, 0, false)],
      ["Rel. 20D / 60D vs SPY", `${pct(L.rel_20d)} / ${pct(L.rel_60d)}`], ["From 52w high", pct(L.from_52w_high)]]
      .map(([k, v]) => `<div><span>${k}</span><span class="num">${v}</span></div>`).join("");
    $("#t-levels").innerHTML = ["resistance", "support"].map((side) => `<div><b class="${side === "support" ? "up" : "down"}">${side}</b>: ${d.levels[side].map((l) => `<span class="mono">${num(l.price)}</span> <span class="muted">(${esc(l.label)}, ${num(l.distance_atr, 1)} ATR)</span>`).join(" · ") || "none in range"}</div>`).join("");
  }
  try {
    const h = await api(`/api/candidate-history/${encodeURIComponent(sym)}`);
    $("#t-hist").innerHTML = h.history.length ? table("chist", [
      { key: "session_date", label: "Session" }, { key: "direction", label: "Side" }, { key: "setup", label: "Setup", render: (x) => esc(s.setup_names[x.setup] || x.setup) },
      { key: "score", label: "Score", n: 1, render: (x) => num(x.score, 0) }, { key: "tier", label: "Tier", render: (x) => tier(x.tier) }], h.history, { sortKey: "session_date" }) : "Never listed as a candidate.";
  } catch (_) { $("#t-hist").textContent = "n/a"; }
}

// ------------------------------------------------------------------ analyst note
function biasClass(label) { return /bullish/i.test(label || "") ? "up" : /bearish/i.test(label || "") ? "down" : ""; }
function analystPanel(a, opts = {}) {
  if (!a) return "";
  const rows = a.scorecard.map((x) => {
    const w = Math.abs(x.score) / 2 * 50, left = x.score < 0 ? 50 - w : 50, col = x.score >= 0 ? "var(--up)" : "var(--down)";
    return `<div class="sc-row"><span class="nm">${esc(x.name)} <span class="muted">${x.weight}</span></span>
      <div class="zbar"><div class="mid"></div><div class="fill" style="left:${left}%;width:${w}%;background:${col}"></div></div>
      <span class="n ${cls(x.score)}">${x.score > 0 ? "+" : ""}${x.score.toFixed(1)}</span><span class="tx">${esc(x.text)}</span></div>`;
  }).join("");
  const scen = a.scenarios.map((x) => `<div><h4 class="${x.name.startsWith("Bull") ? "up" : x.name.startsWith("Bear") ? "down" : ""}">${esc(x.name)}</h4>
    <p><b>Condition</b> ${esc(x.trigger)}</p><p><b>Target</b> ${esc(x.target)}</p><p><b>Invalidation</b> ${esc(x.invalidation)}</p></div>`).join("");
  const secs = a.sections.map((x) => {
    const sig = x.signals ? x.signals : null;
    const nb = sig ? sig.filter((y) => y.signal > 0).length : 0, ns = sig ? sig.filter((y) => y.signal < 0).length : 0;
    return `<div class="sec"><h4>${esc(x.title)}${sig ? `<small><span class="up">+${nb}</span> / <span class="down">−${ns}</span> / ${sig.length}</small>` : ""}</h4>
      ${x.body.map((t, i) => `<p>${sig ? `<span class="sig ${sig[i] && sig[i].signal > 0 ? "up" : sig[i] && sig[i].signal < 0 ? "down" : "muted"}">${sig[i] && sig[i].signal > 0 ? "+" : sig[i] && sig[i].signal < 0 ? "−" : "·"}</span>` : ""}${esc(t)}</p>`).join("")}</div>`;
  }).join("");
  const lv = a.levels || {};
  const levels = [...(lv.resistance || []).map((x) => ["Resistance", x]), ...(lv.support || []).map((x) => ["Support", x])]
    .map(([k, x]) => `<div><span>${k} ${num(x.price)}</span><span class="num">${esc(x.label)} · ${pct(x.pct)}</span></div>`).join("");
  const vp = a.volume_profile;
  return `<div class="note-block">
    <div class="eyebrow">Analyst note / <b>${esc(a.symbol)}</b> · as of ${esc(a.as_of)} · combining ${a.scorecard.length} dimensions, ${(a.techniques || {}).count || 0} techniques</div>
    <div class="note-head"><div class="hl">${esc(a.headline)}</div>
      <div class="meta">Overall<b class="${biasClass(a.bias)}">${esc(a.bias)}</b>${a.bias_score > 0 ? "+" : ""}${a.bias_score.toFixed(0)} / 100 · confidence ${esc(a.confidence)}</div></div>
    <p class="note-summary">${esc(a.summary)}</p>
    <div class="eyebrow">Scorecard (−2 to +2; the number after each name is its weight)</div><div class="scorecard">${rows}</div>
    <div class="eyebrow" style="margin-top:22px">Scenarios</div><div class="scen">${scen}</div>
    <div class="grid g2" style="margin-top:22px">
      <div><div class="eyebrow">Watch next</div><ul class="list">${a.watch.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
        <div class="eyebrow" style="margin-top:18px">Main risks</div><ul class="list ev-neg">${a.risks.map((x) => `<li>${esc(x)}</li>`).join("") || "<li>No specific risks listed</li>"}</ul></div>
      <div><div class="eyebrow">Key levels</div><div class="kv" style="grid-template-columns:1fr">${levels}
        <div><span>EMA21 / EMA50 / EMA200</span><span class="num">${num(lv.ema21)} / ${num(lv.ema50)} / ${num(lv.ema200)}</span></div>
        <div><span>52w high / low</span><span class="num">${num(lv.high52)} / ${num(lv.low52)}</span></div>
        ${lv.fib ? `<div><span>Fibonacci retracement (${esc(lv.fib.direction)})</span><span class="num">38.2% ${num(lv.fib.levels["0.382"])} · 50% ${num(lv.fib.levels["0.5"])} · 61.8% ${num(lv.fib.levels["0.618"])}</span></div>` : ""}
        ${vp ? `<div><span>Volume POC / value area</span><span class="num">${num(vp.poc)} / ${num(vp.val)}–${num(vp.vah)}</span></div>` : ""}</div>
        ${vp ? volumeProfileSVG(vp, a.close) : ""}</div></div>
    ${opts.compact ? "" : `<div class="eyebrow" style="margin-top:24px">Detailed analysis</div><div class="sections">${secs}</div>`}
    <p class="note">Generated by fixed rules from the data above - descriptive research, not investment advice.</p></div>`;
}

// Horizontal volume-by-price histogram with POC, value area, vacuum zones and the current price.
function volumeProfileSVG(vp, close) {
  const bins = vp.bins || [];
  if (!bins.length) return "";
  const W = 320, H = 190, maxV = Math.max(...bins.map((b) => b.volume)), lo = bins[0].price, hi = bins[bins.length - 1].price;
  const y = (p) => H - 6 - ((p - lo) / (hi - lo || 1)) * (H - 12), bh = Math.max(2, (H - 12) / bins.length - 1);
  let g = (vp.vacuum || []).map((z) => `<rect x="0" y="${y(z.high)}" width="${W}" height="${Math.max(1, y(z.low) - y(z.high))}" fill="rgba(232,179,75,.08)"/>`).join("");
  g += `<rect x="0" y="${y(vp.vah)}" width="${W}" height="${y(vp.val) - y(vp.vah)}" fill="rgba(255,255,255,.04)"/>`;
  g += bins.map((b) => `<rect x="0" y="${(y(b.price) - bh / 2).toFixed(1)}" width="${((b.volume / maxV) * (W - 70)).toFixed(1)}" height="${bh.toFixed(1)}" fill="${b.price <= close ? "rgba(31,191,117,.55)" : "rgba(239,67,81,.45)"}"/>`).join("");
  const line = (p, col, label) => `<line x1="0" x2="${W}" y1="${y(p)}" y2="${y(p)}" stroke="${col}" stroke-dasharray="3 3"/><text x="${W - 2}" y="${y(p) - 3}" text-anchor="end" fill="${col}" font-size="10" font-family="JetBrains Mono, monospace">${label} ${num(p)}</text>`;
  g += line(vp.poc, cssVar("--warn"), "POC") + line(close, cssVar("--fg"), "Price");
  return `<div class="eyebrow" style="margin-top:16px">Volume profile (last ${vp.window} days; green = volume below price, amber band = vacuum)</div><svg viewBox="0 0 ${W} ${H}" style="width:100%;height:${H}px">${g}</svg>`;
}

async function loadAnalyst(sym, el, compact = false) {
  el.innerHTML = '<div class="note-block"><div class="eyebrow">Analyst note</div><p class="note">Combining technicals, volume profile, fundamentals, options and catalysts…</p></div>';
  try {
    const r = await api(`/api/analyst/${encodeURIComponent(sym)}?session=${state.session}&live=true`);
    if (el.isConnected) el.innerHTML = analystPanel(r.note, { compact });
    return r;
  } catch (err) { if (el.isConnected) el.innerHTML = `<div class="note-block"><p class="note">Could not build the analyst note: ${esc(err.message)}</p></div>`; }
  return null;
}

function candidatePanels(c, deep) {
  const comps = Object.entries(c.components).map(([k, v]) => `<div class="comp"><div>${tr(k)} <span class="muted">· ${state.snap.settings.weights[k]}</span></div><div class="bar"><span style="width:${v}%"></span></div><div class="n mono">${v.toFixed(0)}</div></div>`).join("");
  const pens = Object.entries(c.penalties).map(([k, v]) => `<span class="chip">${k.replace(/_/g, " ")} −${v}</span>`).join("") || '<span class="muted">none</span>';
  const h = c.history || {};
  const hrow = (st, label) => st && st.n ? `<tr><td>${label}</td><td class="n">${st.n}${st.small_sample ? " ⚠" : ""}</td><td class="n">${pct(st.win_rate, 0, false)}</td><td class="n ${cls(st.avg_r)}">${num(st.avg_r)}</td><td class="n">${pct(st.hit_target_rate, 0, false)}</td><td class="n">${pct(st.stop_rate, 0, false)}</td><td class="n">${num(st.avg_days, 1)}</td></tr>` : `<tr><td>${label}</td><td colspan="6" class="muted">no history</td></tr>`;
  const g = c.group;
  return `<div class="card"><h2>Research thesis <small>${esc(c.direction)} · ${esc(c.setup_name)}</small></h2><p>${esc(c.thesis)}</p>
    ${g ? `<p class="note">Industry group: <b>${esc(g.group)}</b>, RS rank ${g.rank} / ${g.of} ${g.rank_change ? ` (${g.rank_change > 0 ? "up" : "down"} ${Math.abs(g.rank_change)} in 5 days)` : ""}.</p>` : ""}
    <div class="grid g2"><div><h3 class="up">Supporting evidence</h3><ul class="list ev-pos">${c.supporting.map((x) => `<li>${esc(x)}</li>`).join("") || "<li class=muted>none</li>"}</ul></div>
    <div><h3 class="down">Contradicting evidence / risks</h3><ul class="list ev-neg">${c.contradicting.map((x) => `<li>${esc(x)}</li>`).join("") || "<li class=muted>none</li>"}</ul></div></div></div>
    <div class="card"><h2>Score breakdown <small>raw ${num(c.raw_score, 1)} − penalties ${num(c.penalty_total, 0)}${c.setup_bonus ? ` ${c.setup_bonus > 0 ? "+" : "−"} setup weighting ${num(Math.abs(c.setup_bonus), 1)}` : ""} = ${num(c.score, 1)}</small></h2>${comps}<div style="margin-top:8px">Penalties: ${pens}</div>
    ${c.setup_bonus ? `<p class="note">Setup weighting comes from the backtest: each setup gets 150 points per R of its average result (capped −10 to +20)${(state.snap.setup_bonus || {}).source ? `, source: ${esc(state.snap.setup_bonus.source)}` : ""}.</p>` : ""}</div>
    <div class="card"><h2>Historical simulation of this setup <small>universe-wide, in-sample</small></h2><div class="tw"><table><thead><tr><th>Sample</th><th class="n">N</th><th class="n">Win</th><th class="n">Avg R</th><th class="n">T1 hit</th><th class="n">Stopped</th><th class="n">Days</th></tr></thead>
    <tbody>${hrow(h.setup, "All regimes")}${hrow(h.setup_in_regime, `In ${esc(h.regime)} regimes`)}</tbody></table></div>
    <p class="note">Next-open entry, same stop / T1 / time-stop rules, slippage included. Survivorship and in-sample bias apply; ⚠ = fewer than 30 trades.</p></div>`;
}

function planPanel(c) {
  const p = c.plan;
  const budget = c.direction === "long" ? state.snap.regime.long_risk_budget : state.snap.regime.short_risk_budget;
  return `<div class="card"><h2>Risk plan <small>${p.horizon_days}-session time stop</small></h2><div class="kv">
    <div><span>${p.conditional_entry ? "Trigger (conditional entry)" : "Reference entry"}</span><span class="num">${num(p.entry)}</span></div>
    <div><span>Stop</span><span class="num down">${num(p.stop)} (${num(p.risk_atr, 1)} ATR)</span></div>
    <div><span>T1</span><span class="num up">${num(p.target1)}</span></div><div><span>T2</span><span class="num up">${num(p.target2)}</span></div>
    <div><span>Reward/risk (T1)</span><span class="num">${num(p.reward_risk, 2)}</span></div><div><span>Risk / share</span><span class="num">${pct(p.risk_pct, 2, false)}</span></div>
    <div><span>Structural stop</span><span class="num ${p.wide_stop ? "warn" : ""}">${num(p.structural_stop_atr, 1)} ATR${p.wide_stop ? " (capped)" : ""}</span></div>
    <div><span>${p.horizon_days}-day 1σ move (HV20)</span><span class="num">±${num(p.expected_move_hv)}</span></div></div>
    <p class="note">Position size is your call. Example: risking 0.5% of a $100,000 account = $500 ÷ ${num(Math.abs(p.entry - p.stop))} ≈ ${isNum(p.entry - p.stop) && p.entry !== p.stop ? Math.floor(500 / Math.abs(p.entry - p.stop)) : "n/a"} shares (then × regime budget ×${budget}).</p>
    <div class="qa">Add alert: ${(() => {
      const L = c.direction === "long", items = [[L ? "below" : "above", p.stop, "Stop hit"], [L ? "above" : "below", p.target1, "Reach T1"]];
      if (p.conditional_entry) items.unshift([L ? "above" : "below", p.entry, "Entry trigger"]);
      return items.map(([k, lv, n]) => `<button data-qa="${n}" data-sym="${c.symbol}" data-kind="${k}" data-level="${lv}">${n} ${num(lv)}</button>`).join("");
    })()}</div>
    ${planBasis(c)}</div>`;
}

function planBasis(c) {
  const p = c.plan, long = c.direction === "long", b = state.btFull;
  let alt = "";
  const ss = c.setup !== "crash_rebound" && b && (b.selected_side || {})[c.direction];
  if (ss && isNum(p.atr)) {
    const k = ss.stop.startsWith("atr") ? Number(ss.stop.slice(3)) : null, sg = long ? 1 : -1;
    const bl = (b.baseline || {})[`${c.direction}|${ss.stop}|${ss.target}|${ss.horizon}`];
    const tgt = ss.target === "trail" ? `no fixed target, trailing stop (${long ? "highest high in the trade − 3 ATR" : "lowest low in the trade + 3 ATR"}, ~ ${num(3 * p.atr)})`
      : ss.target.startsWith("r") ? `Target ${ss.target.slice(1)}R ≈ ${k ? num(p.entry + sg * Number(ss.target.slice(1)) * k * p.atr) : "n/a"}` : esc(b.target_names[ss.target]);
    alt = `<p class="${ss.oos.avg_r > 0 ? "" : "down"}"><b>Backtest's preferred plan (${long ? "Long" : "Short"})</b>: ${k ? `Stop ${num(p.entry - sg * k * p.atr)} (${k} ATR)` : esc(b.stop_names[ss.stop])}; ${tgt}; ${ss.horizon}-day time stop.
      Five-year backtest, out of sample per trade ${num(ss.oos.avg_r)}R (live plan ${num(b.live.oos.avg_r)}R)${bl ? `; random entries with the same rule ${num(bl.baseline_oos.avg_r)}R` : ""}.${ss.oos.avg_r > 0 ? "" : "Short setups still lose overall - use as an avoid / hedge list only."}
      <a href="#/research">Details on the Research page</a></p>`;
  }
  return `<div class="plan-basis"><div class="eyebrow">Why these levels</div>
    ${p.stop_basis ? `<p><b>Stop</b>: ${esc(p.stop_basis)}</p>` : ""}${p.target_basis ? `<p><b>Targets</b>: ${esc(p.target_basis)}</p>` : ""}
    <p class="note">Basis: ${esc(p.method || "price structure + ATR noise floor + historical resistance / support")}.</p>${alt}</div>`;
}

function fundamentalsPanel(f) {
  if (!f.available) return `<div class="card"><h2>Fundamentals</h2><p class="note">Not available — ${esc(f.reason)}</p></div>`;
  const m = f.metrics;
  const kv = [["Quarter end / published", `${m.latest_quarter_end} / ${m.latest_filed}`], ["Revenue YoY", pct(m.revenue_yoy, 0)], ["EPS YoY", pct(m.eps_yoy, 0)],
    ["Gross margin (TTM)", pct(m.gross_margin_ttm, 1, false)], ["Operating margin (TTM)", pct(m.operating_margin_ttm, 1, false)],
    ["Op. margin Q vs year-ago", `${pct(m.operating_margin_q, 1, false)} vs ${pct(m.operating_margin_q_year_ago, 1, false)}`],
    ["Free cash flow (TTM)", money(m.fcf_ttm)], ["FCF margin", pct(m.fcf_margin_ttm, 1, false)], ["Cash + ST investments", money(m.cash_and_st_investments)],
    ["Total debt", money(m.total_debt)], ["Net debt", money(m.net_debt)], ["Debt / equity", num(m.debt_to_equity, 2)], ["Current ratio", num(m.current_ratio, 2)],
    ["Interest coverage", isNum(m.interest_coverage) ? num(m.interest_coverage, 1) + "×" : "n/a"], ["Shares outstanding", money(m.shares_outstanding).replace("$", "")],
    ["Share count YoY", pct(m.shares_yoy, 1)], ["Market cap", money(m.market_cap)], ["P/E (TTM)", num(m.pe_ttm, 1)], ["P/S (TTM)", num(m.ps_ttm, 1)],
    ["FCF yield", pct(m.fcf_yield, 1, false)], ["EV / EBIT", num(m.ev_to_ebit, 1)]];
  return `<div class="card"><h2>Fundamentals <small>${esc(f.source)}</small></h2><div class="kv">${kv.map(([k, v]) => `<div><span>${k}</span><span class="num">${v}</span></div>`).join("")}</div>
    <h3>Flags</h3><ul class="list">${f.flags.map((x) => `<li class="${x.kind === "risk" ? (x.severity === "high" ? "down" : "warn") : "up"}">${esc(x.text)}</li>`).join("") || "<li class=muted>none</li>"}</ul>
    <h3>Quarters</h3>${table("fq", [{ key: "end", label: "Quarter end" }, { key: "filed", label: "Published" }, { key: "revenue", label: "Revenue", n: 1, render: (q) => money(q.revenue) },
      { key: "gross_profit", label: "Gross profit", n: 1, render: (q) => money(q.gross_profit) }, { key: "operating_income", label: "Op. income", n: 1, render: (q) => money(q.operating_income) },
      { key: "net_income", label: "Net income", n: 1, render: (q) => money(q.net_income) }, { key: "eps_diluted", label: "EPS", n: 1, render: (q) => num(q.eps_diluted) }], f.quarters, { sortKey: "end" })}
    <p class="note">Point-in-time: only values published on or before the session are used. Guidance: ${esc(f.guidance || "")}</p></div>`;
}

function optionsPanel(o) {
  if (!o.available) return `<div class="card"><h2>Options</h2><p class="note">Not available — ${esc(o.reason)}</p></div>`;
  const maxOI = Math.max(1, ...o.top_call_oi_strikes.map((x) => x.open_interest), ...o.top_put_oi_strikes.map((x) => x.open_interest));
  const oiRows = (arr, color) => arr.map((x) => `<div class="comp" style="grid-template-columns:70px 1fr 70px"><div class="mono">${num(x.strike, x.strike % 1 ? 1 : 0)}</div><div class="bar"><span style="width:${(x.open_interest / maxOI) * 100}%;background:${color}"></span></div><div class="n mono">${(x.open_interest / 1000).toFixed(1)}k</div></div>`).join("");
  return `<div class="card"><h2>Options <small>${esc(o.source)}</small></h2><div class="kv">
    <div><span>ATM IV (${o.atm_iv_expiry})</span><span class="num">${pct(o.atm_iv, 1, false)}</span></div><div><span>Next expiry ATM IV</span><span class="num">${pct(o.atm_iv_next, 1, false)}</span></div>
    <div><span>IV / HV20</span><span class="num ${o.iv_hv_ratio > 1.4 ? "warn" : ""}">${num(o.iv_hv_ratio, 2)}</span></div><div><span>IV rank</span><span class="num">${isNum(o.iv_rank) ? num(o.iv_rank, 0) : "n/a"}</span></div>
    <div><span>Expected move → ${o.expected_move_expiry}</span><span class="num">±${num(o.expected_move)} (${pct(o.expected_move_pct, 1, false)})</span></div><div><span>Max pain (front)</span><span class="num">${num(o.max_pain_front)}</span></div>
    <div><span>Call / put volume</span><span class="num">${(o.call_volume / 1000).toFixed(1)}k / ${(o.put_volume / 1000).toFixed(1)}k</span></div><div><span>P/C volume · OI</span><span class="num">${num(o.pcr_volume)} · ${num(o.pcr_oi)}</span></div>
    <div><span>Call / put OI</span><span class="num">${(o.call_oi / 1000).toFixed(1)}k / ${(o.put_oi / 1000).toFixed(1)}k</span></div><div><span>GEX estimate (inferred)</span><span class="num">${money(o.gex_estimate)}</span></div></div>
    ${o.iv_rank_note ? `<p class="note">${esc(o.iv_rank_note)}</p>` : ""}
    <h3>Largest open interest — calls / puts</h3>${oiRows(o.top_call_oi_strikes, "var(--up)")}${oiRows(o.top_put_oi_strikes, "var(--down)")}
    ${o.unusual.length ? `<h3>Volume &gt; 2× prior-day OI (inferred opening activity)</h3>${table("unu", [{ key: "side", label: "Side" }, { key: "expiry", label: "Expiry" }, { key: "strike", label: "Strike", n: 1 }, { key: "volume", label: "Vol", n: 1, render: (u) => u.volume.toLocaleString() }, { key: "open_interest", label: "OI", n: 1, render: (u) => u.open_interest.toLocaleString() }, { key: "premium", label: "Premium", n: 1, render: (u) => money(u.premium) }], o.unusual, { sortKey: "premium" })}` : ""}
    <ul class="list">${o.interpretation.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
    <p class="note">${o.limitations.map(esc).join(" ")} ${esc(o.gex_note)}</p></div>`;
}

function positioningPanel(p) {
  const ins = p.insider_90d || {}, inst = p.institutional || {}, sv = p.short_volume || {}, ox = p.off_exchange || {};
  return `<div class="card"><h2>Positioning & ownership <small>delayed disclosures</small></h2><div class="kv">
    <div><span>Short % float</span><span class="num">${pct(p.short_percent_float, 1, false)}</span></div><div><span>Days to cover</span><span class="num">${num(p.days_to_cover, 1)}</span></div>
    <div><span>Short interest date</span><span class="num">${esc(p.short_interest_date || "n/a")} (${p.short_interest_age_days ?? "?"}d old)</span></div><div><span>SI change vs prior</span><span class="num">${pct(p.short_interest_change, 0)}</span></div>
    <div><span>Float</span><span class="num">${money(p.float_shares).replace("$", "")}</span></div><div><span>Shares outstanding</span><span class="num">${money(p.shares_outstanding).replace("$", "")}</span></div>
    <div><span>Held by institutions</span><span class="num">${pct(p.held_pct_institutions, 0, false)}</span></div><div><span>Held by insiders</span><span class="num">${pct(p.held_pct_insiders, 1, false)}</span></div>
    <div><span>Insider buys / sales (90d)</span><span class="num">${ins.available ? `${ins.n_buys} / ${ins.n_sales}` : "n/a"}</span></div><div><span>Net insider value</span><span class="num ${cls(ins.net_value)}">${money(ins.net_value)}</span></div>
    <div><span>Short-sale volume share</span><span class="num">${sv.available ? pct(sv.latest.ratio, 0, false) + " (10d avg " + pct(sv.avg_ratio, 0, false) + ")" : "n/a"}</span></div>
    <div><span>Off-exchange share</span><span class="num">${ox.available && isNum(ox.off_exchange_share) ? pct(ox.off_exchange_share, 0, false) + " (wk " + ox.week_start + ")" : "n/a"}</span></div></div>
    <ul class="list">${(p.notes || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
    ${inst.available ? `<h3>Top institutional holders (13F)</h3>${table("inst", [{ key: "holder", label: "Holder" }, { key: "pct_held", label: "% held", n: 1, render: (r) => pct(r.pct_held, 2, false) }, { key: "pct_change", label: "Δ", n: 1, render: (r) => `<span class="${cls(r.pct_change)}">${pct(r.pct_change, 1)}</span>` }, { key: "date_reported", label: "As of" }], inst.top_holders.slice(0, 6), { sortKey: "pct_held" })}<p class="note">${esc(inst.note)}</p>` : ""}
    <p class="note">Block trades: ${esc((p.block_trades || {}).reason || "n/a")}. ${esc(sv.note || "")} ${esc(ox.note || "")}</p></div>`;
}

function catalystsPanel(cat) {
  const n = cat.next_earnings;
  return `<div class="card"><h2>Catalysts</h2>
    ${n ? `<p><b>Next earnings:</b> <span class="mono">${n.date}</span> ${esc(n.timing || "(time not confirmed)")} · reaction session ${n.reaction_session} · <b class="${n.sessions_until <= 7 ? "warn" : ""}">${n.sessions_until} session(s)</b> <span class="chip">${n.status}</span></p>` : '<p class="note">No upcoming earnings date listed.</p>'}
    ${cat.earnings_history && cat.earnings_history.length ? `<p class="note">Last ${cat.earnings_history.length} reports: ${cat.earnings_history.map((h) => `${h.date} ${isNum(h.surprise_pct) ? (h.surprise_pct > 0 ? "+" : "") + h.surprise_pct.toFixed(1) + "%" : "n/a"}`).join(" · ")} (EPS surprise)</p>` : ""}
    ${table("cev", [{ key: "event_date", label: "Date" }, { key: "event_type", label: "Event", wrap: 1, render: (e) => e.url ? `<a href="${esc(e.url)}" target="_blank" rel="noopener">${esc(e.description || e.event_type)}</a>` : esc(e.description || e.event_type) },
      { key: "sentiment", label: "Typical implication", render: (e) => `<span class="${e.sentiment === "Positive" ? "up" : e.sentiment === "Negative" ? "down" : "muted"}">${e.sentiment}</span>` }, { key: "status", label: "Status" }, { key: "source", label: "Source" }], cat.events || [], { sortKey: "event_date" })}
    <p class="note">${(cat.limitations || []).map(esc).join(" ")}</p></div>`;
}

// ------------------------------------------------------------------ views: universe
function universe() {
  const s = state.snap, q = state.filters.q.toUpperCase();
  const rows = s.universe.filter((u) => !q || u.symbol.includes(q));
  const uf = s.universe_filter || {};
  app.innerHTML = `<div class="wall-head"><div><div class="eyebrow">High-beta / high-volatility filter / <b>kept ${s.universe.length} </b></div>
    <h1 class="display">Universe <em>high beta</em></h1>
    <p>AI, semiconductor, high-growth and high-volatility names only. Daily filter: 120-day Beta ≥ ${uf.rules ? uf.rules.min_beta : "1.3"}, 
    60d volatility ≥ ${uf.rules ? pct(uf.rules.min_hv60, 0, false) : "35%"}, market cap ≥ ${uf.rules ? money(uf.rules.min_market_cap) : "$2B"} (leveraged ETFs exempt), 
    Dollar volume ≥ ${uf.rules ? money(uf.rules.min_avg_dollar_volume) : "$100M"}. Excluded names are on the Data page.</p></div></div>
    <div class="card"><div class="filters"><label>Search <input id="u-q" value="${esc(state.filters.q)}" style="width:90px"></label><span class="note">click any row for the chart</span></div>
    ${table("uni", [
      { key: "symbol", label: "Symbol", render: (u) => `<b>${u.symbol}</b>${u.leveraged ? ' <span class="chip">Leveraged ETF</span>' : ""}` }, { key: "sector_etf", label: "Sector" },
      { key: "close", label: "Close", n: 1, render: (u) => num(u.close) },
      { key: "ret_1d", label: "1d", n: 1, render: (u) => `<span class="${cls(u.ret_1d)}">${pct(u.ret_1d)}</span>` },
      { key: "ret_5d", label: "5d", n: 1, render: (u) => `<span class="${cls(u.ret_5d)}">${pct(u.ret_5d)}</span>` },
      { key: "ret_20d", label: "20d", n: 1, render: (u) => `<span class="${cls(u.ret_20d)}">${pct(u.ret_20d)}</span>` },
      { key: "beta", label: "Beta", n: 1, render: (u) => num(u.beta, 2) },
      { key: "hv60", label: "60d vol", n: 1, render: (u) => pct(u.hv60, 0, false) },
      { key: "market_cap", label: "Market cap", n: 1, render: (u) => money(u.market_cap) },
      { key: "rel_20d", label: "Rel 20d", n: 1, render: (u) => `<span class="${cls(u.rel_20d)}">${pct(u.rel_20d)}</span>` },
      { key: "rs_pct", label: "RS percentile", n: 1, render: (u) => num(u.rs_pct, 0) }, { key: "rsi14", label: "RSI14", n: 1, render: (u) => num(u.rsi14, 0) },
      { key: "volume_ratio", label: "Vol. ratio", n: 1, render: (u) => num(u.volume_ratio, 2) }, { key: "extension_atr", label: "Ext. ATR", n: 1, render: (u) => num(u.extension_atr, 1) },
      { key: "trend", label: "Trend", render: (u) => esc(tr(u.trend)) }, { key: "structure", label: "Structure", render: (u) => esc(tr(u.structure)) },
      { key: "setups", label: "Setup", render: (u) => u.setups.map((x) => `<span class="chip ${s.setup_directions[x]}">${esc(s.setup_names[x])}</span>`).join("") || '<span class="muted">—</span>', sort: (u) => u.setups.length },
      { key: "dollar_volume20", label: "Dollar volume", n: 1, render: (u) => money(u.dollar_volume20) },
    ], rows, { sortKey: "beta", rowAttr: (u) => `data-sym="${u.symbol}"`, onClick: true })}</div>`;
  $("#u-q").oninput = (e) => { state.filters.q = e.target.value; clearTimeout(state.qt); state.qt = setTimeout(() => { route(); const i = $("#u-q"); i.focus(); i.setSelectionRange(i.value.length, i.value.length); }, 250); };
}

// ------------------------------------------------------------------ views: catalysts
function catalysts() {
  const s = state.snap;
  const filings = [], analyst = [];
  for (const [sym, d] of Object.entries(s.deep_dive)) {
    for (const f of (d.catalysts || {}).filings || []) filings.push({ symbol: sym, ...f });
    for (const a of (d.catalysts || {}).analyst || []) analyst.push({ symbol: sym, ...a });
  }
  const sent = (v) => `<span class="${v === "Positive" ? "up" : v === "Negative" ? "down" : "muted"}">${tr(v)}</span>`;
  app.innerHTML = `<div class="wall-head"><div><div class="eyebrow">Earnings · SEC filings · Analysts · macro</div><h1 class="display">Catalysts <em>calendar</em></h1></div></div>
  <div class="grid g2">
    <div class="card"><h2>Upcoming earnings <small>listed names</small></h2>${table("ue", [
      { key: "symbol", label: "Symbol", render: (e) => `<b>${e.symbol}</b>` }, { key: "date", label: "Date" }, { key: "timing", label: "Timing", render: (e) => esc(tr(e.timing) || "unconfirmed") },
      { key: "reaction_session", label: "Reaction session" }, { key: "sessions_until", label: "Sessions away", n: 1 }, { key: "eps_estimate", label: "EPS est.", n: 1, render: (e) => num(e.eps_estimate) },
      { key: "status", label: "Status", render: (e) => tr(e.status) }],
      s.upcoming_earnings || [], { sortKey: "sessions_until", sortDir: "asc", rowAttr: (e) => `data-sym="${e.symbol}"`, onClick: true })}</div>
    <div class="card"><h2>Macro & market-structure calendar <small>next 30 days</small></h2>${table("mac", [
      { key: "event_date", label: "Date" }, { key: "event_type", label: "Event", wrap: 1 }, { key: "status", label: "Status" }, { key: "source", label: "Source", wrap: 1 }],
      s.macro_events || [], { sortKey: "event_date", sortDir: "asc" })}
      <p class="note">Holidays: ${(s.holidays || []).map((h) => `${h.date} ${esc(h.name)}`).join(" · ") || "none in the next 45 days"}. Add CPI / PCE dates in config/macro_events.yaml.</p></div>
  </div>
  <div class="card" style="margin-top:12px"><h2>SEC filings <small>shortlisted names, last 60 days</small></h2>${table("fil", [
    { key: "symbol", label: "Symbol", render: (f) => `<b>${f.symbol}</b>` }, { key: "event_date", label: "Filed" }, { key: "published_at", label: "Accepted (UTC)" }, { key: "form", label: "Form" },
    { key: "event_type", label: "Type", wrap: 1, render: (f) => f.url ? `<a href="${esc(f.url)}" target="_blank" rel="noopener">${esc(f.event_type)}</a>` : esc(f.event_type) },
    { key: "sentiment", label: "Typical implication", render: (f) => sent(f.sentiment) }, { key: "dilution_risk", label: "Dilution", render: (f) => f.dilution_risk ? '<span class="down">yes</span>' : "" }],
    filings, { sortKey: "event_date" })}${filings.length ? "" : '<p class="note">No SEC filings loaded - set SEC_USER_AGENT (see banner) to enable EDGAR.</p>'}</div>
  <div class="card" style="margin-top:12px"><h2>Analyst actions <small>last 30 days</small></h2>${table("ana", [
    { key: "symbol", label: "Symbol", render: (a) => `<b>${a.symbol}</b>` }, { key: "date", label: "Date" }, { key: "description", label: "Action", wrap: 1 }, { key: "sentiment", label: "Sentiment", render: (a) => sent(a.sentiment) }],
    analyst, { sortKey: "date" })}</div>`;
}

// ------------------------------------------------------------------ market extras: cross-asset, A/D breadth, industry groups
function sparkLine(vals, w = 120, h = 28) {
  const v = (vals || []).filter(isNum);
  if (v.length < 2) return "";
  const lo = Math.min(...v), hi = Math.max(...v), r = hi - lo || 1;
  const pts = v.map((x, i) => `${(i / (v.length - 1) * w).toFixed(1)},${(h - 2 - (x - lo) / r * (h - 4)).toFixed(1)}`).join(" ");
  return `<svg class="sparkline" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}"><polyline points="${pts}" fill="none" stroke="var(${v[v.length - 1] >= v[0] ? "--up" : "--down"})" stroke-width="1.4"/></svg>`;
}

function marketExtras(s) {
  const m = s.macro || [], ad = s.advance_decline || {}, groups = s.industry_groups || [];
  if (!m.length && !ad.available && !groups.length) return "";
  return `<div class="grid g2" style="margin-top:12px">
    <div class="card"><h2>Cross-asset <small>dollar · oil · gold · Bitcoin (context only, not in the regime score)</small></h2>${table("mac", [
      { key: "name", label: "Asset", render: (x) => `<b>${esc(x.name)}</b> <span class="muted mono">${esc(x.symbol)}</span>` },
      { key: "spark", label: "60d", render: (x) => sparkLine(x.spark) },
      { key: "close", label: "Latest", n: 1, render: (x) => (x.available ? num(x.close, x.close > 1000 ? 0 : 2) : '<span class="muted">No data</span>') },
      { key: "ret_5d", label: "5D", n: 1, render: (x) => `<span class="${cls(x.ret_5d)}">${pct(x.ret_5d)}</span>` },
      { key: "ret_20d", label: "20D", n: 1, render: (x) => `<span class="${cls(x.ret_20d)}">${pct(x.ret_20d)}</span>` },
      { key: "z20", label: "20d z", n: 1, render: (x) => `<span class="${isNum(x.z20) && Math.abs(x.z20) >= 1.5 ? "warn" : ""}">${num(x.z20, 1)}</span>` },
      { key: "corr_qqq_60d", label: "Corr. QQQ", n: 1, render: (x) => num(x.corr_qqq_60d, 2) }], m)}
      <ul class="list">${m.filter((x) => x.comment).map((x) => `<li><b>${esc(x.name)}</b>: ${esc(x.comment)}</li>`).join("")}</ul></div>
    <div class="card"><h2>Advance / decline <small>${esc(ad.scope || "")}</small></h2>
      ${ad.available ? `<div class="grid g4 kpis"><div class="kpi"><span class="l">Advancers</span><span class="v up">${ad.advancers}</span></div>
        <div class="kpi"><span class="l">Decliners</span><span class="v down">${ad.decliners}</span></div>
        <div class="kpi"><span class="l">McClellan</span><span class="v ${cls(ad.mcclellan)}">${num(ad.mcclellan, 0)}</span></div>
        <div class="kpi"><span class="l">Up-volume share</span><span class="v">${pct(ad.up_volume_share, 0, false)}</span></div></div>
        <div class="lg">A/D line (cumulative advances minus declines) and McClellan oscillator · 10-day net advances ${ad.net_10d > 0 ? "+" : ""}${ad.net_10d}</div>
        <div id="ad-chart" class="chart" style="height:150px"></div><div id="mcc-chart" class="chart small"></div>
        <p class="note">${esc(ad.comment || "")} There is no reliable free source for NYSE-wide advance/decline data, so this uses the universe - participation among high-beta names.</p>` : '<p class="note">Not enough data.</p>'}</div>
  </div>
  ${groups.length ? `<div class="card" style="margin-top:12px"><h2>Industry-group relative strength <small>${groups.length}  · ranked by members' median RS, arrows = rank change over 5 days</small></h2>${table("grp", [
    { key: "rank", label: "#", n: 1 },
    { key: "group", label: "Group", render: (g) => `<b>${esc(g.group)}</b>` },
    { key: "rank_change", label: "5d change", n: 1, render: (g) => (g.rank_change ? `<span class="${cls(g.rank_change)}">${g.rank_change > 0 ? "▲" : "▼"}${Math.abs(g.rank_change)}</span>` : '<span class="muted">—</span>') },
    { key: "n", label: "Members", n: 1 },
    { key: "pct", label: "RS percentile", n: 1, render: (g) => num(g.pct, 0) },
    { key: "ret_5d", label: "5d (median)", n: 1, render: (g) => `<span class="${cls(g.ret_5d)}">${pct(g.ret_5d)}</span>` },
    { key: "ret_20d", label: "20d (median)", n: 1, render: (g) => `<span class="${cls(g.ret_20d)}">${pct(g.ret_20d)}</span>` },
    { key: "breadth", label: "Above EMA50", n: 1, render: (g) => pct(g.breadth, 0, false) },
    { key: "leaders", label: "Leaders", wrap: 1, render: (g) => g.leaders.map((x) => `<a href="#/ticker/${x}">${x}</a>`).join(" ") }],
    groups, { sortKey: "rank", sortDir: "asc" })}<p class="note">Industries come from Yahoo company profiles (refreshed every 120 days); single-member industries are not listed. RS score = 0.5 × vs SPY 20d + 0.3 × 60d + 0.2 × 5d. Leaders of strong groups tend to keep leading.</p></div>` : ""}`;
}

function drawAD(s) {
  const ad = s.advance_decline;
  if (!ad || !ad.available || !$("#ad-chart")) return;
  const P = PAL(), d = ad.series.dates;
  const c1 = makeChart($("#ad-chart"), 150, { timeVisible: false });
  c1.addLineSeries({ color: P.neutral, lineWidth: 2, priceLineVisible: false }).setData(d.map((t, i) => ({ time: t, value: ad.series.ad_line[i] })));
  const c2 = makeChart($("#mcc-chart"), 90);
  c2.addHistogramSeries({ priceLineVisible: false }).setData(d.map((t, i) => ({ time: t, value: ad.series.mcclellan[i], color: ad.series.mcclellan[i] >= 0 ? P.up : P.down })));
  c1.timeScale().fitContent(); c2.timeScale().fitContent(); syncCharts([c1, c2]);
}

// ------------------------------------------------------------------ ticker extras: earnings reactions, FTD
function earningsPanel(e) {
  if (!e) return "";
  if (!e.available) return `<div class="card"><h2>Earnings reactions</h2><p class="note">${esc(e.reason || "No data.")}</p></div>`;
  const imp = e.implied;
  return `<div class="card"><h2>Earnings reactions <small>last ${e.n} × · options-implied vs history</small></h2>
    <div class="grid g4 kpis">
      <div class="kpi"><span class="l">Options implied</span><span class="v">${imp && isNum(imp.move_pct) ? "±" + pct(imp.move_pct, 1, false) : "—"}</span><span class="note">${imp ? `${imp.expiry} straddle ${num(imp.straddle)}` : "no expiry covers the report"}</span></div>
      <div class="kpi"><span class="l">Median historical move</span><span class="v">±${pct(e.median_abs_move, 1, false)}</span><span class="note">avg ±${pct(e.mean_abs_move, 1, false)}</span></div>
      <div class="kpi"><span class="l">Implied ÷ history</span><span class="v ${isNum(e.implied_vs_history) ? (e.implied_vs_history > 1.15 ? "warn" : e.implied_vs_history < 0.85 ? "up" : "") : "muted"}">${isNum(e.implied_vs_history) ? num(e.implied_vs_history, 2) + "×" : "—"}</span><span class="note">&gt; 1 options rich</span></div>
      <div class="kpi"><span class="l">Up share</span><span class="v">${pct(e.up_share, 0, false)}</span><span class="note">max ${pct(e.max_up, 0)}  / ${pct(e.max_down, 0)}</span></div></div>
    <p>${esc(e.note || "")}</p>${e.drift_note ? `<p>${esc(e.drift_note)}</p>` : ""}
    <p class="note">Gaps that faded by the close ${pct(e.gap_fade_share, 0, false)}; the next 10 days continued the first-day direction ${pct(e.drift10_same_dir, 0, false)} (avg ${pct(e.drift10_signed_mean, 1)}).</p>
    ${table("earn", [{ key: "date", label: "Reported" }, { key: "timing", label: "Timing", render: (x) => ({ "before open": "Before open", "after close": "After close", estimated: "est." }[x.timing] || x.timing) },
      { key: "surprise_pct", label: "EPS surprise", n: 1, render: (x) => (isNum(x.surprise_pct) ? `<span class="${cls(x.surprise_pct)}">${num(x.surprise_pct, 1)}%</span>` : "—") },
      { key: "gap", label: "Gap", n: 1, render: (x) => `<span class="${cls(x.gap)}">${pct(x.gap)}</span>` },
      { key: "move", label: "Day", n: 1, render: (x) => `<span class="${cls(x.move)}">${pct(x.move)}</span>` },
      { key: "drift5", label: "Next 5d", n: 1, render: (x) => `<span class="${cls(x.drift5)}">${pct(x.drift5)}</span>` },
      { key: "drift10", label: "Next 10d", n: 1, render: (x) => `<span class="${cls(x.drift10)}">${pct(x.drift10)}</span>` }], e.events, { sortKey: "date" })}
    <p class="note">Only ${e.n} events - descriptive statistics. "Est." = no release time from the provider; the session with the bigger gap is used.</p></div>`;
}

function ftdPanel(f) {
  if (!f) return "";
  if (!f.available) return `<div class="card"><h2>SEC fails to deliver (FTD)</h2><p class="note">${esc(f.reason)}</p></div>`;
  return `<div class="card"><h2>SEC fails to deliver (FTD) <small>${esc(f.period)}</small></h2>
    ${f.n ? `<div class="grid g4 kpis"><div class="kpi"><span class="l">High</span><span class="v">${fmtVol(f.max)}</span><span class="note">${f.max_date}</span></div>
      <div class="kpi"><span class="l">Peak ÷ avg volume</span><span class="v ${f.max_vs_adv >= 0.1 ? "warn" : ""}">${pct(f.max_vs_adv, 1, false)}</span></div>
      <div class="kpi"><span class="l">Latest</span><span class="v">${fmtVol(f.latest)}</span><span class="note">${f.latest_date}</span></div>
      <div class="kpi"><span class="l">Days listed</span><span class="v">${f.n}</span><span class="note">≥ 0.5% avg volume: ${f.sessions_over_half_pct_adv ?? "—"}</span></div></div>
      <div class="ftd-bars">${(() => { const mx = Math.max(...f.series.map((x) => x.fails)); return f.series.map((x) => `<i style="height:${Math.max(2, x.fails / mx * 100)}%" title="${x.date}: ${x.fails.toLocaleString()} shares"></i>`).join(""); })()}</div>` : ""}
    <p>${esc(f.note)}</p><p class="note">The SEC publishes this twice a month, usually two to four weeks late; figures are the cumulative undelivered balance, not new fails. ETF creation/redemption timing also creates FTDs, so on its own it means little.</p></div>`;
}

// ------------------------------------------------------------------ portfolio page: sizing, correlation, heat, journal, alerts
const PF_KEY = "swing.portfolio.v1";
function pfSettings() {
  const d = (state.snap.portfolio || {});
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(PF_KEY) || "{}"); } catch (_) { saved = {}; }
  return { account: saved.account ?? d.account_size ?? 100000, risk: saved.risk ?? (d.risk_per_trade ?? 0.005) * 100,
    maxPos: saved.maxPos ?? (d.max_position_pct ?? 0.25) * 100, maxHeat: saved.maxHeat ?? (d.max_heat ?? 0.04) * 100, picked: saved.picked || null };
}
function pfSave(v) { try { localStorage.setItem(PF_KEY, JSON.stringify(v)); } catch (_) { /* storage unavailable */ } }

function pfSized(v) {
  const pf = state.snap.portfolio || {}, budgets = pf.budgets || { long: 1, short: 1 };
  return (pf.positions || []).map((p) => {
    const per = Math.abs(p.entry - p.stop), risk = v.account * v.risk / 100 * (budgets[p.direction] || 1);
    let shares = per > 0 ? Math.floor(risk / per) : 0;
    const cap = Math.floor(v.account * v.maxPos / 100 / p.entry), capped = shares > cap;
   shares = Math.min(shares, cap);
    return { ...p, shares, dollar_risk: shares * per, value: shares * p.entry, value_pct: shares * p.entry / v.account, capped };
  });
}

function corrHeat(pf, picked) {
  if (!pf.labels) return '<p class="note">Fewer than two candidates - no correlation matrix.</p>';
  const idx = pf.labels.map((l, i) => [l, i]).filter(([l]) => !picked || picked.has(l.replace(" (S)", "")));
  const bg = (v) => `background:color-mix(in srgb, var(${v >= 0 ? "--down" : "--up"}) ${Math.round(Math.min(1, Math.abs(v)) * 70)}%, transparent)`;
  return `<div class="tw"><table class="heat corr"><thead><tr><th></th>${idx.map(([l]) => `<th class="n">${esc(l)}</th>`).join("")}</tr></thead><tbody>
    ${idx.map(([l, i]) => `<tr><th>${esc(l)}</th>${idx.map(([, j]) => `<td class="n" style="${i === j ? "" : bg(pf.correlation[i][j])}">${i === j ? "—" : num(pf.correlation[i][j], 2)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <p class="note">Position correlation = 60-day correlation of daily returns, with shorts sign-flipped: deeper red = they win and lose together (risk stacks), green = they hedge each other.</p>`;
}

async function portfolio() {
  const s = state.snap, pf = s.portfolio || {};
  const v = pfSettings();
  const draw = () => {
    const rows = pfSized(v);
    const picked = new Set(v.picked || rows.map((r) => r.symbol));
    const sel = rows.filter((r) => picked.has(r.symbol));
    const heat = sel.reduce((a, r) => a + r.dollar_risk, 0), heatPct = heat / v.account * 100;
    const lab = sel.map((r) => r.symbol + (r.direction === "long" ? "" : " (S)"));
    const ix = lab.map((l) => (pf.labels || []).indexOf(l)).filter((i) => i >= 0);
    let sum = 0; ix.forEach((i) => ix.forEach((j) => { sum += pf.correlation[i][j]; }));
    const eff = ix.length > 1 && sum > 0 ? ix.length * ix.length / sum : null;
    $("#pf-body").innerHTML = `<div class="grid g4 kpis">
        <div class="kpi"><span class="l">Selected</span><span class="v">${sel.length}</span><span class="note">Long ${sel.filter((r) => r.direction === "long").length} / short ${sel.filter((r) => r.direction === "short").length}</span></div>
        <div class="kpi"><span class="l">Total heat (all stops hit)</span><span class="v ${heatPct > v.maxHeat ? "down" : "up"}">${num(heatPct, 2)}%</span><span class="note">$${Math.round(heat).toLocaleString()} / cap ${num(v.maxHeat, 1)}%</span></div>
        <div class="kpi"><span class="l">Independent bets (est.)</span><span class="v">${eff ? num(eff, 1) : "—"}</span><span class="note">n² ÷ Σ correlation</span></div>
        <div class="kpi"><span class="l">Gross / net exposure</span><span class="v">${pct(sel.reduce((a, r) => a + r.value, 0) / v.account, 0, false)}</span><span class="note">net ${pct(sel.reduce((a, r) => a + r.value * (r.direction === "long" ? 1 : -1), 0) / v.account, 0)}</span></div></div>
      <div class="heatbar"><i style="width:${Math.min(100, heatPct / Math.max(v.maxHeat, 0.01) * 100)}%" class="${heatPct > v.maxHeat ? "over" : ""}"></i></div>
      ${table("pf", [{ key: "pick", label: "Pick", render: (r) => `<input type="checkbox" data-pick="${r.symbol}" ${picked.has(r.symbol) ? "checked" : ""}>` },
        { key: "symbol", label: "Symbol", render: (r) => `<a href="#/ticker/${r.symbol}"><b>${r.symbol}</b></a> <span class="chip ${r.direction}">${r.direction === "long" ? "L" : "S"}</span>` },
        { key: "score", label: "Score", n: 1, render: (r) => num(r.score, 0) }, { key: "entry", label: "Entry", n: 1, render: (r) => num(r.entry) },
        { key: "stop", label: "Stop", n: 1, render: (r) => `<span class="down">${num(r.stop)}</span>` },
        { key: "per", label: "Risk / share", n: 1, sort: (r) => Math.abs(r.entry - r.stop), render: (r) => num(Math.abs(r.entry - r.stop)) },
        { key: "shares", label: "Shares", n: 1, render: (r) => `${r.shares.toLocaleString()}${r.capped ? ' <span class="warn" title="Reduced to the single-position cap">⚑</span>' : ""}` },
        { key: "value", label: "Position value", n: 1, render: (r) => "$" + Math.round(r.value).toLocaleString() },
        { key: "value_pct", label: "% of account", n: 1, render: (r) => pct(r.value_pct, 1, false) },
        { key: "dollar_risk", label: "Dollar risk", n: 1, render: (r) => "$" + Math.round(r.dollar_risk).toLocaleString() }], rows, { sortKey: "score" })}
      <h3>Position correlation (selected)</h3>${corrHeat(pf, picked)}
      <ul class="list">${(pf.clusters || []).map((c) => `<li class="warn">Highly correlated (≥ 0.7), likely to stop out together: ${c.map(esc).join(", ")}</li>`).join("")}</ul>`;
    app.querySelectorAll("[data-pick]").forEach((cb) => cb.addEventListener("change", () => {
      const cur = new Set(v.picked || rows.map((r) => r.symbol));
      cb.checked ? cur.add(cb.dataset.pick) : cur.delete(cb.dataset.pick);
      v.picked = [...cur]; pfSave(v); draw();
    }));
  };
  app.innerHTML = `<div class="wall-head"><div><div class="eyebrow">positions · correlation · total heat · Trade journal · Price alerts</div><h1 class="display">Portfolio <em>risk</em></h1>
    <p>Today's candidates as a portfolio: each position is sized from its stop distance (fixed fractional risk), then checked for whether they would all lose together. Nothing is ever traded; you enter journal records yourself.</p></div></div>
    <section class="note-block"><div class="pf-inputs">
      <label>Account size (US$)<input id="pf-acc" type="number" min="1000" step="1000" value="${v.account}"></label>
      <label>Risk per trade (%)<input id="pf-risk" type="number" min="0.05" max="5" step="0.05" value="${v.risk}"></label>
      <label>Max per position (%)<input id="pf-max" type="number" min="1" max="100" step="1" value="${v.maxPos}"></label>
      <label>Max total heat (%)<input id="pf-heat" type="number" min="0.5" max="20" step="0.5" value="${v.maxHeat}"></label>
      <span class="note">Shares = account × risk per trade × regime budget (long ×${(pf.budgets || {}).long ?? 1}, short ×${(pf.budgets || {}).short ?? 1}) ÷ |entry − stop|. Settings are saved in this browser only.</span></div>
      <div id="pf-body"></div></section>
    <section class="note-block"><div class="note-head"><div><div class="eyebrow">Price alerts</div><div class="hl">Price alerts</div></div></div>
      <form id="al-form" class="pf-inputs"><label>Symbol<input name="symbol" required placeholder="NVDA"></label>
        <label>Condition<select name="kind"><option value="above">Trades above</option><option value="below">Trades below</option><option value="close_above">Closes above</option><option value="close_below">Closes below</option></select></label>
        <label>Level<input name="level" type="number" step="0.01" required></label><label class="wide">Note<input name="note" placeholder="e.g. add on breakout"></label>
        <button type="submit">Add alert</button></form>
      <div id="al-list" class="note">Loading…</div>
      <p class="note">Alerts are checked against stored daily bars after every daily run and when the dashboard opens ("trades above / below" = the day's high / low touched it). Not real time; on a Mac a triggered alert shows a system notification.</p></section>
    <section class="note-block"><div class="note-head"><div><div class="eyebrow">Trade journal · review</div><div class="hl">Trade journal</div></div></div>
      <form id="jn-form" class="pf-inputs"><label>Symbol<input name="symbol" required></label>
        <label>Side<select name="direction"><option value="long">Long</option><option value="short">Short</option></select></label>
        <label>Setup<select name="setup"><option value="">—</option>${Object.entries(s.setup_names).map(([k, n]) => `<option value="${k}">${esc(n)}</option>`).join("")}</select></label>
        <label>Entry date<input name="entry_date" type="date"></label><label>Entry price<input name="entry_price" type="number" step="0.01" required></label>
        <label>Initial stop<input name="stop" type="number" step="0.01" required></label><label>Target<input name="target" type="number" step="0.01"></label>
        <label>Shares<input name="shares" type="number" step="1"></label><label>Exit date<input name="exit_date" type="date"></label><label>Exit price<input name="exit_price" type="number" step="0.01"></label>
        <label class="wide">Note<input name="notes" placeholder="why in / out, did you follow the plan?"></label><button type="submit">Log trade</button></form>
      <div id="jn-list" class="note">Loading…</div></section>`;
  const bind = (id, key) => $(id).addEventListener("input", (e) => { const x = Number(e.target.value); if (isNum(x) && x > 0) { v[key] = x; pfSave(v); draw(); } });
  bind("#pf-acc", "account"); bind("#pf-risk", "risk"); bind("#pf-max", "maxPos"); bind("#pf-heat", "maxHeat");
  draw();
  const formData = (f) => Object.fromEntries([...new FormData(f).entries()].filter(([, x]) => x !== ""));
  const ro = !!window.__PREVIEW__;
  const loadAlerts = async () => {
    try {
      const a = await api("/api/alerts");
      $("#al-list").innerHTML = table("al", [{ key: "symbol", label: "Symbol", render: (x) => `<a href="#/ticker/${x.symbol}"><b>${x.symbol}</b></a>` },
        { key: "kind_name", label: "Condition" }, { key: "level", label: "Level", n: 1, render: (x) => num(x.level) },
        { key: "last", label: "Last close", n: 1, render: (x) => num(x.last) }, { key: "distance", label: "Distance", n: 1, render: (x) => pct(x.distance) },
        { key: "active", label: "Status", render: (x) => (x.active ? '<span class="muted">waiting</span>' : `<span class="warn">Triggered ${x.triggered_session} (${num(x.triggered_price)})</span>`) },
        { key: "note", label: "Note", wrap: 1 }, { key: "act", label: "", render: (x) => `${x.active ? "" : `<button data-rearm="${x.id}">Re-arm</button>`} <button data-del-al="${x.id}">Delete</button>` }],
        a.alerts, { sortKey: "active" });
      app.querySelectorAll("[data-del-al]").forEach((b) => b.addEventListener("click", async () => { await api(`/api/alerts/${b.dataset.delAl}`, { method: "DELETE" }); loadAlerts(); }));
      app.querySelectorAll("[data-rearm]").forEach((b) => b.addEventListener("click", async () => { await api(`/api/alerts/${b.dataset.rearm}/rearm`, { method: "POST" }); loadAlerts(); }));
    } catch (err) { $("#al-list").textContent = err.message; }
  };
  const loadJournal = async () => {
    try {
      const j = await api("/api/journal"), st = j.stats || {};
      $("#jn-list").innerHTML = `<div class="grid g4 kpis"><div class="kpi"><span class="l">Closed</span><span class="v">${st.n_closed || 0}</span><span class="note">open ${st.n_open || 0}</span></div>
        <div class="kpi"><span class="l">Win rate</span><span class="v">${pct(st.win_rate, 0, false)}</span><span class="note">Profit factor ${num(st.profit_factor)}</span></div>
        <div class="kpi"><span class="l">avg R</span><span class="v ${cls(st.avg_r)}">${num(st.avg_r)}</span><span class="note">avg win ${num(st.avg_win_r)}R / loss ${num(st.avg_loss_r)}R</span></div>
        <div class="kpi"><span class="l">Total P&L</span><span class="v ${cls(st.total_pnl)}">${isNum(st.total_pnl) ? "$" + Math.round(st.total_pnl).toLocaleString() : "—"}</span><span class="note">total ${num(st.total_r, 1)}R</span></div></div>
        ${(st.by_setup || []).length ? `<h3>By setup</h3>${table("jn-set", [{ key: "setup", label: "Setup", render: (x) => esc(s.setup_names[x.setup] || x.setup) }, { key: "n", label: "Trades", n: 1 },
          { key: "win_rate", label: "Win rate", n: 1, render: (x) => pct(x.win_rate, 0, false) }, { key: "avg_r", label: "Avg R", n: 1, render: (x) => `<span class="${cls(x.avg_r)}">${num(x.avg_r)}</span>` }], st.by_setup)}` : ""}
        ${table("jn", [{ key: "symbol", label: "Symbol", render: (t) => `<a href="#/ticker/${t.symbol}"><b>${t.symbol}</b></a> <span class="chip ${t.direction}">${t.direction === "long" ? "L" : "S"}</span>` },
          { key: "setup", label: "Setup", render: (t) => esc(s.setup_names[t.setup] || t.setup || "") }, { key: "entry_date", label: "Entry date" },
          { key: "entry_price", label: "Entry", n: 1, render: (t) => num(t.entry_price) }, { key: "stop", label: "Stop", n: 1, render: (t) => num(t.stop) },
          { key: "exit_price", label: "Exit / last", n: 1, render: (t) => (t.status === "closed" ? num(t.exit_price) : `<span class="muted">${num(t.mark)}</span>`) },
          { key: "r", label: "R", n: 1, sort: (t) => t.r ?? t.r_open, render: (t) => (t.status === "closed" ? `<span class="${cls(t.r)}">${num(t.r)}</span>` : `<span class="muted">${num(t.r_open)} (open)</span>`) },
          { key: "pnl", label: "P&L", n: 1, sort: (t) => t.pnl ?? t.pnl_open, render: (t) => { const x = t.status === "closed" ? t.pnl : t.pnl_open; return isNum(x) ? `<span class="${cls(x)}">$${Math.round(x).toLocaleString()}</span>` : "—"; } },
          { key: "flag", label: "", render: (t) => (t.flag ? `<span class="warn">${esc(t.flag)}</span>` : "") },
          { key: "notes", label: "Note", wrap: 1 },
          { key: "act", label: "", render: (t) => `${t.status !== "closed" ? `<button data-close="${t.id}">Close out</button>` : ""} <button data-del-jn="${t.id}">Delete</button>` }], j.trades, { sortKey: "entry_date" })}
        <p class="note">R = direction × (exit − entry) ÷ |entry − initial stop|. Open trades are marked to the latest close.</p>`;
      app.querySelectorAll("[data-del-jn]").forEach((b) => b.addEventListener("click", async () => { if (confirm("Delete this record?")) { await api(`/api/journal/${b.dataset.delJn}`, { method: "DELETE" }); loadJournal(); } }));
      app.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", async () => {
        const px = Number(prompt("Exit price?")); if (!isNum(px) || px <= 0) return;
        await api(`/api/journal/${b.dataset.close}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ exit_price: px, exit_date: new Date().toISOString().slice(0, 10), status: "closed" }) });
        loadJournal();
      }));
    } catch (err) { $("#jn-list").textContent = err.message; }
  };
  const post = (url, f, after) => f.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (ro) { alert("Cannot save in the preview."); return; }
    try { await api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(formData(f)) }); f.reset(); after(); }
    catch (err) { alert(err.message); }
  });
  post("/api/alerts", $("#al-form"), loadAlerts);
  post("/api/journal", $("#jn-form"), loadJournal);
  loadAlerts(); loadJournal();
}

async function quickAlert(symbol, kind, level, note) {
  if (window.__PREVIEW__) { alert("Alerts cannot be saved in the preview."); return; }
  try {
    await api("/api/alerts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ symbol, kind, level, note }) });
    alert(`Alert added: ${symbol} ${note} (${num(level)}). Manage alerts on the Portfolio page.`);
  } catch (err) { alert(err.message); }
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-qa]");
  if (b) quickAlert(b.dataset.sym, b.dataset.kind, Number(b.dataset.level), b.dataset.qa);
});

// ------------------------------------------------------------------ news (free RSS: Yahoo Finance, Nasdaq, Google News)
function ago(iso) {
  if (!iso) return "";
  const m = (Date.now() - Date.parse(iso)) / 60000;
  return m < 60 ? `${Math.max(1, Math.round(m))}m` : m < 1440 ? `${Math.round(m / 60)}h` : `${Math.round(m / 1440)}d`;
}
function newsRow(x, showSym = true) {
  const dot = x.sentiment === "positive" ? "up" : x.sentiment === "negative" ? "down" : "muted";
  return `<li class="nw"><span class="nw-t mono">${ago(x.published)}</span>
    ${showSym && x.symbol && x.symbol !== "_MARKET" ? `<a class="nw-s" href="#/ticker/${esc(x.symbol)}">${esc(x.symbol)}</a>` : x.topic ? `<span class="nw-s muted">${esc(x.topic)}</span>` : ""}
    <span class="nw-d ${dot}">●</span>
    <a class="nw-h" href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.title)}</a>
    <span class="nw-p">${esc(x.publisher || x.feed || "")}</span>
    ${(x.tags || []).map((t) => `<span class="chip nw-tag">${esc(t)}</span>`).join("")}</li>`;
}
function newsList(items, showSym = true, empty = "No headlines.") {
  return items && items.length ? `<ul class="nwl">${items.map((x) => newsRow(x, showSym)).join("")}</ul>` : `<p class="note">${empty}</p>`;
}

async function news() {
  const s = state.snap, cands = [...s.candidates.long, ...(s.candidates.short || [])];
  const nd = s.news_digest || {};
  app.innerHTML = `<div class="wall-head"><div><div class="eyebrow">Yahoo Finance · Nasdaq · Google News · ${nd.headlines ? `${nd.headlines.toLocaleString()} headlines for ${nd.with_news} stocks at the last run` : "fetched automatically"}</div>
    <h1 class="display">News <em>flow</em></h1>
    <p>Every stock in the universe and the market, fetched after each run and every hour by the background job while your Mac is on. Tags and tone are keyword rules - a reading aid, not a signal.</p></div>
    <div><button id="nw-refresh" class="primary">Refresh now</button> <span id="nw-status" class="note"></span></div></div>
    <section class="note-block"><div class="note-head"><div><div class="eyebrow">Market</div><div class="hl">What the market is reading</div></div></div><div id="nw-market" class="note">Loading…</div></section>
    <section class="note-block"><div class="note-head"><div><div class="eyebrow">Today's candidates</div><div class="hl">News on the names you are looking at</div></div></div>
      <div class="nw-cands">${cands.filter((c) => c.news && c.news.items && c.news.items.length).map((c) => `<div class="nw-c"><div class="nw-ch"><a href="#/ticker/${c.symbol}"><b>${c.symbol}</b></a>
        <span class="muted">${esc(c.setup_name)}</span>${(c.news.flags || []).map((f) => `<span class="chip warn">${esc(f)}</span>`).join("")}
        <span class="note">${c.news.n_recent} in ${c.news.days} days · <span class="up">${c.news.positive}+</span> / <span class="down">${c.news.negative}−</span></span></div>
        ${newsList(c.news.items.slice(0, 4), false)}</div>`).join("") || '<p class="note">No candidate news stored for this session.</p>'}</div></section>
    <section class="note-block"><div class="note-head"><div><div class="eyebrow">Universe</div><div class="hl">All headlines</div></div></div>
      <div class="filters"><label>Tag <select id="nw-tag"><option value="">All</option>${["earnings", "guidance", "analyst", "M&A", "offering", "legal", "FDA", "contract", "product", "insider", "macro", "big move"].map((t) => `<option>${t}</option>`).join("")}</select></label>
        <label>Tone <select id="nw-tone"><option value="">All</option><option value="positive">Positive</option><option value="negative">Negative</option></select></label>
        <label><input type="checkbox" id="nw-cand"> Candidates only</label>
        <label>Search <input id="nw-q" style="width:140px"></label><span id="nw-count" class="note"></span></div>
      <div id="nw-all" class="note">Loading…</div></section>`;
  const candSet = new Set(cands.map((c) => c.symbol));
  let all = [];
  const draw = () => {
    const tag = $("#nw-tag").value, tone = $("#nw-tone").value, only = $("#nw-cand").checked, q = $("#nw-q").value.trim().toLowerCase();
    const rows = all.filter((x) => (!tag || (x.tags || []).includes(tag)) && (!tone || x.sentiment === tone) && (!only || candSet.has(x.symbol))
      && (!q || x.title.toLowerCase().includes(q) || (x.symbol || "").toLowerCase() === q));
    $("#nw-count").textContent = `${rows.length.toLocaleString()} headlines`;
    $("#nw-all").innerHTML = newsList(rows.slice(0, 400));
  };
  const load = async () => {
    try { $("#nw-market").innerHTML = newsList((await api("/api/news?symbol=_MARKET&days=3&limit=60")).items); } catch (err) { $("#nw-market").textContent = err.message; }
    try { all = (await api("/api/news?days=7&limit=1500")).items.filter((x) => x.symbol !== "_MARKET"); draw(); } catch (err) { $("#nw-all").textContent = err.message; }
  };
  ["#nw-tag", "#nw-tone", "#nw-cand"].forEach((id) => $(id).addEventListener("change", draw));
  $("#nw-q").addEventListener("input", draw);
  const btn = $("#nw-refresh");
  if (window.__PREVIEW__) { btn.disabled = true; btn.title = "Not available in the preview"; }
  btn.addEventListener("click", async () => {
    try { await api("/api/news/refresh", { method: "POST" }); } catch (err) { $("#nw-status").textContent = err.message; return; }
    const poll = async () => {
      const r = (await api("/api/news?limit=1")).refresh;
      $("#nw-status").textContent = r.running ? "Refreshing all headlines…" : (r.message || "Done");
      if (r.running) setTimeout(poll, 3000); else load();
    };
    poll();
  });
  load();
}

// ------------------------------------------------------------------ crash-rebound / spike-fade study (python crash_study.py)
const CS_ENTRIES = ["d0", "d1", "d2", "d3", "d4", "d5", "d7", "d10", "green", "hold_hi"];
const csEntryName = (k) => ({ green: "1st up day", hold_hi: "1st close > prior high" }[k] || (k === "d0" ? "next day" : `+${k.slice(1)} days`));

function crashHeat(r, names, part = "oos") {
  const base = ((r.baseline || {}).ret5 || {})[part] || {};
  const cell = {};
  for (const g of r.grid || []) cell[g.event + "|" + g.entry] = g[`ret5_${part}`];
  const ex = (st) => (st && st.n >= 30 && isNum(st.mean) ? st.mean - (base.mean || 0) : null);
  const vals = Object.values(cell).map(ex).filter(isNum), m = Math.max(0.01, ...vals.map(Math.abs));
  const bg = (x) => (isNum(x) ? `background:color-mix(in srgb, var(${x > 0 ? "--up" : "--down"}) ${Math.round(8 + 52 * Math.min(1, Math.abs(x) / m))}%, transparent)` : "");
  return `<div class="tw"><table class="heat"><thead><tr><th>Event ↓ / entry →</th>${CS_ENTRIES.map((k) => `<th class="n">${csEntryName(k)}</th>`).join("")}</tr></thead><tbody>
    ${Object.keys(names).map((e) => `<tr><th>${esc(names[e])}</th>${CS_ENTRIES.map((k) => { const st = cell[e + "|" + k], x = ex(st);
      return `<td class="n" style="${bg(x)}" title="${st ? st.n : 0} trades · win ${pct(st && st.win, 0, false)} · raw ${pct(st && st.mean, 2)}">${isNum(x) ? pct(x, 1) : "—"}</td>`; }).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function crashBest(rows, id, trade = false) {
  return table(id, [{ key: "event", label: "Event", render: (x) => esc(x.ev_name) }, { key: "entry", label: "Entry", render: (x) => csEntryName(x.entry) },
    ...(trade ? [{ key: "target", label: "Target", render: (x) => (x.target === "ema21" ? "EMA21" : x.target.replace("atr", "") + " ATR") }] : []),
    { key: "is", label: trade ? "In-sample R" : "In-sample excess 5d", n: 1, sort: (x) => (trade ? x.is.mean : x.excess_is), render: (x) => (trade ? rv(x.is.mean) : `<span class="${cls(x.excess_is)}">${pct(x.excess_is, 1)}</span>`) },
    { key: "oos", label: trade ? "Out-of-sample R" : "Out-of-sample excess 5d", n: 1, sort: (x) => (trade ? x.oos.mean : x.excess_oos), render: (x) => (trade ? rv(x.oos.mean) : `<span class="${cls(x.excess_oos)}">${pct(x.excess_oos, 1)}</span>`) },
    { key: "win", label: "OOS win rate", n: 1, sort: (x) => x.oos.win, render: (x) => pct(x.oos.win, 0, false) },
    { key: "n", label: "OOS trades", n: 1, sort: (x) => x.oos.n, render: (x) => (x.oos.n || 0).toLocaleString() }], rows);
}

function setupWeighting(s) {
  const sb = s.setup_bonus || {}, bt = (state.btFull || {}).by_setup || {};
  const rows = Object.entries(sb.bonus || {}).map(([k, v]) => ({ k, name: s.setup_names[k] || k, bonus: v, ...(bt[k] || {}) }));
  if (!rows.length) return "";
  return `<section class="note-block"><div class="note-head"><div><div class="eyebrow">Score weighting · ${esc(sb.source || "")}</div>
      <div class="hl">Which setups get extra points - decided by the backtest</div></div></div>
    ${table("sw", [{ key: "name", label: "Setup" }, { key: "n", label: "Backtest trades", n: 1, render: (r) => (r.n || 0).toLocaleString() },
      { key: "avg_r", label: "Average R", n: 1, render: (r) => rv(r.avg_r) }, { key: "win_rate", label: "Win rate", n: 1, render: (r) => pct(r.win_rate, 0, false) },
      { key: "bonus", label: "Score points", n: 1, render: (r) => `<b class="${cls(r.bonus)}">${r.bonus > 0 ? "+" : ""}${num(r.bonus, 1)}</b>` }], rows, { sortKey: "bonus" })}
    <p class="note">Points = 150 × the setup's average R under the live plan (capped −10 to +20), refreshed with the monthly backtest. Every setup can still make the list - the
      weighting only tilts the ranking toward what has worked. A fuller model (setup + indicator factors, fitted in sample) did not rank trades reliably out of sample, so it is not used.</p></section>`;
}

function crashSection(cs) {
  if (!cs || !cs.available || !cs.crash || !cs.crash.available) return "";
  const r = cs.crash, sp = cs.spike || {}, names = cs.event_names, snames = cs.spike_names;
  const top = (r.best_ret5 || [])[0], topT = (r.best_trade || [])[0], ps = r.per_stock || {};
  const named = (rows, nm) => (rows || []).map((x) => ({ ...x, ev_name: nm[x.event] || x.event }));
  const spTop = (sp.best_trade || [])[0], spEx = (sp.best_ret5 || [])[0];
  const cond = r.conditions ? Object.entries(r.conditions.by).map(([k, v]) => ({ k, ...v })) : [];
  return `<section class="note-block"><div class="note-head"><div><div class="eyebrow">${cs.n_stocks} stocks · ${(r.n_events || 0).toLocaleString()} crash events · in sample to ${r.cutoff}</div>
      <div class="hl">Buy the plunge: when does buying after a crash pay?</div></div>
      <div class="meta">Best rule, out of sample<b class="${cls(top && top.excess_oos)}">${top ? pct(top.excess_oos, 1) : "—"}</b>vs random entry, 5 days</div></div>
    <ul class="list verdict">
      ${top ? `<li><b class="tag up">EDGE</b>Best rule: <b>${esc(names[top.event])}</b>, buy ${csEntryName(top.entry)}: +${pct(top.excess_is, 1, false)} vs random entry over 5 days in sample, ${pct(top.excess_oos, 1)} out of sample (win rate ${pct(top.oos.win, 0, false)}).</li>` : ""}
      ${topT ? `<li><b class="tag up">TRADE</b>As a trade (stop 2 ATR, target ${topT.target === "ema21" ? "EMA21" : topT.target.replace("atr", "") + " ATR"}, 10 days): <b>${esc(names[topT.event])}</b>, buy ${csEntryName(topT.entry)} → ${num(topT.is.mean)}R in sample, ${num(topT.oos.mean)}R out of sample, win rate ${pct(topT.oos.win, 0, false)}.</li>` : ""}
      <li><b class="tag up">TIMING</b>The edge sits in the first three sessions after the crash (next day to +2 days) and is mostly gone by day 4–7. Waiting for a "confirmation" (first up day, close above the prior high) gives most of it away.</li>
      ${ps.n_compared ? `<li><b class="tag down">PER STOCK</b>Each stock's own best entry day (in sample) earned ${pct(ps.oos_own_best_mean, 1)} out of sample vs ${pct(ps.oos_pooled_best_mean, 1)} for the universe-wide best day (${csEntryName(ps.pooled_best)}), over ${ps.n_compared} stocks: stock-specific timing is noise - the terminal uses the universal rule.</li>` : ""}
      ${spEx ? `<li><b class="tag down">SHORT SPIKES</b>Shorting after surges: best in-sample rule ${pct(spEx.excess_is, 1)} vs random shorts (${pct(spEx.excess_oos, 1)} out of sample), but random shorts lose ${pct(-(((sp.baseline || {}).ret5 || {}).oos || {}).mean, 1, false)} per 5 days here, so the absolute result is ~zero; as trades ${spTop ? `the best is ${num(spTop.oos.mean)}R out of sample` : "all lose"}. Not used.</li>` : ""}
    </ul>
    <h3>Excess 5-day return vs random entry, by crash type and entry day <small>out of sample</small></h3>${crashHeat(r, names, "oos")}
    <p class="note">Each cell = average 5-day return after entering at the next open, minus the random-entry average over the same period. Hover for trades, win rate and raw return. Green = beat random entry.</p>
    <h3>Best rules (5-day excess return)</h3>${crashBest(named(r.best_ret5, names).slice(0, 8), "cs-b5")}
    <h3>Best trades (stop 2 ATR, close target, 10 days)</h3>${crashBest(named(r.best_trade, names).slice(0, 8), "cs-bt", true)}
    ${cond.length ? `<h3>When does it work best? <small>${esc(names[r.conditions.event])}, buy ${csEntryName(r.conditions.entry)} · average 5-day return</small></h3>${table("cs-cond", [{ key: "k", label: "Condition" },
      { key: "is", label: "In sample", n: 1, sort: (x) => x.is.mean, render: (x) => `<span class="${cls(x.is.mean)}">${pct(x.is.mean, 1)}</span> <span class="muted">(${x.is.n || 0})</span>` },
      { key: "oos", label: "Out of sample", n: 1, sort: (x) => x.oos.mean, render: (x) => `<span class="${cls(x.oos.mean)}">${pct(x.oos.mean, 1)}</span> <span class="muted">(${x.oos.n || 0})</span>` }], cond)}` : ""}
    <h3>Short after a surge? <small>excess vs random shorts, out of sample</small></h3>${crashHeat(sp, snames, "oos")}
    <p class="note">Survivorship bias applies (today's universe); events of one type within 10 sessions are merged; daily bars only. The live "Crash rebound" setup uses the strongest, most stable rules: 5-day drop ≥ 20%, 3-day drop ≥ 15% or ≥ 3 ATR below EMA21, buy within 0–2 days, stop 2 ATR, T1 1.5 ATR, 10-day time stop, and gets a +${num(((state.snap.settings || {}).scanner || {}).setup_bonus ? state.snap.settings.scanner.setup_bonus.crash_rebound : 15, 0)} score bonus.</p>
  </section>`;
}

// ------------------------------------------------------------------ SL / TP validation backtest (python backtest.py)
const BT_STOP_ORDER = ["structural", "atr1.0", "atr1.5", "atr2.0", "atr2.5", "atr3.0", "swing", "chip"];
const BT_TGT_ORDER = ["resistance", "atr1", "atr1.5", "atr2", "atr3", "atr4", "r1", "r1.5", "r2", "r3", "chip", "trail"];
const REGIMES = ["Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish"];
state.bt = { side: "long", hz: 10, period: "oos" };
const rv = (v, d = 2) => (isNum(v) ? `<span class="${cls(v)}">${v > 0 ? "+" : ""}${num(v, d)}R</span>` : '<span class="muted">—</span>');
const tval = (v) => (isNum(v) ? `<span class="${Math.abs(v) >= 2 ? (v > 0 ? "up" : "down") : "muted"}">${num(v, 1)}</span>` : "—");

function slTpExplainer(b) {
  const live = b && b.available ? b.live : null, d = (b && b.diagnostics) || {};
  const mae = d.winner_mae_atr || {}, mfe = d.mfe_r || {}, rr = d.rr_calibration || [];
  const edge = !!live && live.oos.avg_r > 0.05 && live.oos.t_stat > 2, tight = (d.stopped_then_target || 0) > 0.2;
  const below = rr.filter((r) => r.target_rate < r.breakeven).length;
  const verdictLine = (txt, ok) => `<li><b class="tag ${ok ? "up" : "down"}">${ok ? "PASS" : "ISSUE"}</b>${txt}</li>`;
  return `<section class="note-block bt-explain"><div class="note-head"><div><div class="eyebrow">Stop · Target · Time stop</div>
      <div class="hl">SL/TP: how they are set, what is used and what is not</div></div></div>
    <div class="grid g3 ruled">
      <div><h3>Stop (SL)</h3><p>Set from <b>price structure</b>: every setup has a level where its premise fails - for a breakout, the lower of the breakout level (prior 20-day high) and the breakout-day low;
        for a pullback, the lower of the 5-day low and EMA21 - then a <b>0.25 ATR</b> buffer.</p>
        <p>Then the distance is bounded by <b>ATR (14-day average true range)</b>: under 1 ATR it is widened to 1 ATR (so daily noise does not stop it out), over 3 ATR it is tightened to 3 ATR and flagged "wide stop".</p></div>
      <div><h3>Target (TP)</h3><p><b>T1</b> = the nearest prior high at least 0.75 ATR above (20-day, 60-day or 52-week high, i.e. resistance); if price is at new highs with no resistance above, entry + 2 ATR.</p>
        <p><b>T2</b> = the further of T1 + 1 ATR and 2× risk. Short setups mirror all of this (prior lows as support).</p></div>
      <div><h3>Time stop</h3><p>After <b>7 sessions</b>, exit at the close if neither stop nor target was hit: swing setups usually lose their push within one to two weeks.</p>
        <p class="note">The volume profile is <b>not</b> used in the live SL/TP, but the backtest tests a volume-node / value-area-low stop and a volume-node / vacuum-top target as alternatives (see below).
        Options are <b>not</b> used: there is no free historical implied-volatility data to backtest, so the options expected move is context on the stock page only.</p></div>
    </div>
    ${live ? `<h3>What the backtest says about this SL/TP</h3><ul class="list verdict">
      ${verdictLine(`Overall ${live.all.n.toLocaleString()} trades: win rate ${pct(live.all.win_rate, 0, false)}, avg ${rv(live.all.avg_r)}; out of sample ${rv(live.oos.avg_r)} (t ${tval(live.oos.t_stat)}) - ${edge ? "<b>a statistically significant edge</b>" : "<b>no reliable edge</b>"}.`, edge)}
      ${verdictLine(`Stop${tight ? " may be too tight" : " not too tight"}: stopped trades that still reached the target ${pct(d.stopped_then_target, 0, false)}; winners took a median heat of ${num(mae.p50)} ATR, 90% under ${num(mae.p90)} ATR; the average stop is ${num(d.avg_risk_atr)} ATR.`, !tight)}
      ${verdictLine(`Target${below === rr.length && rr.length ? " too far: every planned-R:R bucket hits below break-even" : `: ${rr.length - below} / ${rr.length} planned-R:R buckets hit above break-even`}; median best open profit ${num(mfe.p50)}R, ${pct(mfe.ge2, 0, false)} reached 2R.`, below < rr.length)}
      ${live.short && live.short.n ? verdictLine(`Long ${rv(live.long.avg_r)}, short ${rv(live.short.avg_r)}: short setups lose money in this high-beta universe.`, live.short.avg_r > 0) : verdictLine(`Long only: short setups were removed after losing money in this high-beta universe (see the heat map's Short tab).`, true)}
    </ul>` : ""}
  </section>`;
}

function btHeat(b) {
  const v = state.bt, cell = {};
  for (const g of b.grid) if (g.side === v.side && g.horizon === v.hz) cell[g.stop + "|" + g.target] = g[v.period];
  const vals = Object.values(cell).map((x) => x && x.avg_r).filter(isNum), m = Math.max(0.05, ...vals.map(Math.abs));
  const bg = (x) => (isNum(x) ? `background:color-mix(in srgb, var(${x > 0 ? "--up" : "--down"}) ${Math.round(8 + 52 * Math.min(1, Math.abs(x) / m))}%, transparent)` : "");
  const head = BT_TGT_ORDER.map((t) => `<th class="n">${esc(b.target_names[t])}</th>`).join("");
  const rows = BT_STOP_ORDER.map((s) => `<tr><th>${esc(b.stop_names[s])}</th>${BT_TGT_ORDER.map((t) => {
    const c = cell[s + "|" + t] || {}, live = s === "structural" && t === "resistance" && v.hz === 7;
    return `<td class="n ${live ? "live" : ""}" style="${bg(c.avg_r)}" title="${c.n || 0}  · Win rate ${pct(c.win_rate, 0, false)} · t ${num(c.t_stat, 1)}">${isNum(c.avg_r) ? (c.avg_r > 0 ? "+" : "") + num(c.avg_r) : "—"}</td>`;
  }).join("")}</tr>`).join("");
  const seg = (k, opts) => `<div class="seg" data-bt="${k}">${opts.map(([val, lab]) => `<button data-v="${val}" class="${String(v[k]) === String(val) ? "active" : ""}">${lab}</button>`).join("")}</div>`;
  return `<div class="heat-ctl">${seg("side", [["long", "Long"], ["short", "Short"]])} ${seg("hz", [[5, "5d"], [7, "7d"], [10, "10d"]])} ${seg("period", [["is", "In sample"], ["oos", "Out of sample"]])}</div>
    <div class="tw"><table class="heat"><thead><tr><th>Stop ↓ / target →</th>${head}</tr></thead><tbody>${rows}</tbody></table></div>
    <p class="note">Each cell = average R per trade (1R = stop distance). Hover for trades, win rate and t-stat. The outlined cell is the live plan (structural stop + resistance target + 7 days).</p>`;
}

function btBestTable(b, side) {
  const base = b.baseline || {};
  const rows = (b.best_is[side] || []).map((g, i) => {
    const bl = base[`${side}|${g.stop}|${g.target}|${g.horizon}`];
    return { rank: i + 1, ...g, bl_oos: bl ? bl.baseline_oos.avg_r : null, edge: bl ? bl.edge_oos : null };
  });
  const lv = b.grid.find((g) => g.side === side && g.stop === "structural" && g.target === "resistance" && g.horizon === 7);
  if (lv) rows.push({ rank: "Live", ...lv, bl_oos: null, edge: null });
  return table("bt-best-" + side, [
    { key: "rank", label: "#", render: (r) => (r.rank === "Live" ? '<b class="warn">Live</b>' : r.rank) },
    { key: "stop", label: "Stop", render: (r) => esc(b.stop_names[r.stop]) }, { key: "target", label: "Target", render: (r) => esc(b.target_names[r.target]) },
    { key: "horizon", label: "Hold", n: 1, render: (r) => r.horizon + " days" },
    { key: "is_r", label: "In sample R", n: 1, sort: (r) => r.is.avg_r, render: (r) => rv(r.is.avg_r) },
    { key: "oos_r", label: "Out of sample R", n: 1, sort: (r) => r.oos.avg_r, render: (r) => rv(r.oos.avg_r) },
    { key: "oos_w", label: "OOS win rate", n: 1, sort: (r) => r.oos.win_rate, render: (r) => pct(r.oos.win_rate, 0, false) },
    { key: "oos_pf", label: "Profit factor", n: 1, sort: (r) => r.oos.profit_factor, render: (r) => num(r.oos.profit_factor) },
    { key: "oos_t", label: "t-stat", n: 1, sort: (r) => r.oos.t_stat, render: (r) => tval(r.oos.t_stat) },
    { key: "oos_n", label: "Trades", n: 1, sort: (r) => r.oos.n, render: (r) => (r.oos.n || 0).toLocaleString() },
    { key: "bl_oos", label: "Random entry (OOS)", n: 1, render: (r) => rv(r.bl_oos) },
    { key: "edge", label: "Setup edge", n: 1, render: (r) => rv(r.edge) }], rows);
}

function quarterBars(q, title) {
  const ks = Object.keys(q || {}).sort(), vals = ks.map((k) => q[k]);
  if (!ks.length) return "";
  const W = 640, H = 130, m = Math.max(0.1, ...vals.filter(isNum).map(Math.abs)), bw = W / ks.length;
  const y0 = H / 2, sc = (H / 2 - 14) / m;
  const bars = ks.map((k, i) => {
    const v = isNum(q[k]) ? q[k] : 0, h = Math.abs(v) * sc;
    return `<rect x="${i * bw + 2}" y="${v > 0 ? y0 - h : y0}" width="${Math.max(1, bw - 4)}" height="${Math.max(0.5, h)}" fill="var(${v > 0 ? "--up" : "--down"})"><title>${k}: ${num(v)}R</title></rect>`
      + (i % 4 === 0 ? `<text x="${i * bw + 2}" y="${H - 1}" class="ax">${k.slice(0, 4)}</text>` : "");
  }).join("");
  return `<div class="qbars"><div class="eyebrow">${esc(title)}</div><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="${esc(title)}">
    <line x1="0" x2="${W}" y1="${y0}" y2="${y0}" stroke="var(--line3)"/>${bars}</svg></div>`;
}

function icTable(f, id) {
  if (!f) return "";
  const rows = f.factors.slice(0, 14);
  const bar = (v) => `<span class="icbar ${v > 0 ? "pos" : "neg"}"><i style="width:${Math.min(100, Math.abs(v) / 0.12 * 100)}%"></i></span>`;
  return table(id, [{ key: "name", label: "Factor" },
    { key: "ic_is", label: "In sample IC", n: 1, render: (r) => `${bar(r.ic_is)} <span class="${cls(r.ic_is)}">${num(r.ic_is, 3)}</span>` },
    { key: "ic_oos", label: "Out of sample IC", n: 1, render: (r) => `${bar(r.ic_oos)} <span class="${cls(r.ic_oos)}">${num(r.ic_oos, 3)}</span>` },
    { key: "stable", label: "Stable", render: (r) => (r.stable ? '<span class="up">yes</span>' : '<span class="muted">no</span>') }], rows);
}

function calibNote(f) {
  const q = ((f || {}).calibrated || {}).oos_quintiles;
  if (!q || q.length < 5) return "";
  const r = q.map((x) => x.avg_r), mono = r.every((v, i) => i === 0 || v >= r[i - 1] - 0.01), spread = r[4] - r[0];
  return `A score refitted with in-sample IC weights, in out-of-sample quintiles from low to high: ${r.map((v) => num(v)).join("  /  ")} R.`
    + (mono && spread > 0.05 ? "Broadly rising: these factors help, so re-weighting the score is worth considering." : "Not monotonic: the refitted score does not hold up out of sample, so the score weights stay as they are.");
}

function backtestSection(b) {
  if (!b || !b.available) {
    return `<section class="note-block"><div class="hl">SL/TP backtest</div><p class="note">${esc((b && b.reason) || "No backtest results.")}</p>
      <button id="bt-run" class="wide">Re-run backtest</button> <span id="bt-status" class="note"></span></section>`;
  }
  const live = b.live, L = (b.selected_side || {}).long, S = (b.selected_side || {}).short, sc = b.score_calibration || {}, d = b.diagnostics || {};
  const blL = L && (b.baseline || {})[`long|${L.stop}|${L.target}|${L.horizon}`];
  const fx = b.factors || {};
  return `<section class="note-block bt"><div class="note-head"><div><div class="eyebrow">${b.first} → ${b.last} · ${b.n_symbols}  · ${b.n_signals.toLocaleString()} signals · in sample to ${b.cutoff}, out of sample after</div>
      <div class="hl">SL/TP backtest: are the live stops and targets any good?</div></div>
      <div class="meta">Live plan, out of sample<b class="${cls(live.oos.avg_r)}">${num(live.oos.avg_r)}R</b>t ${num(live.oos.t_stat, 1)}</div></div>
    <ul class="list verdict">${(b.verdict || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
    <div class="grid g4 kpis">
      <div class="kpi"><span class="l">Live plan win rate</span><span class="v">${pct(live.all.win_rate, 0, false)}</span><span class="note">avg ${rv(live.all.avg_r)}</span></div>
      <div class="kpi"><span class="l">Best long (out of sample)</span><span class="v ${cls(L && L.oos.avg_r)}">${L ? num(L.oos.avg_r) + "R" : "—"}</span><span class="note">${L ? esc(b.stop_names[L.stop] + " + " + b.target_names[L.target] + " + " + L.horizon + " days") : ""}</span></div>
      <div class="kpi"><span class="l">Random entry, same rule</span><span class="v ${cls(blL && blL.baseline_oos.avg_r)}">${blL ? num(blL.baseline_oos.avg_r) + "R" : "—"}</span><span class="note">Setup edge ${blL ? rv(blL.edge_oos) : "—"}</span></div>
      <div class="kpi"><span class="l">Score vs outcome</span><span class="v ${Math.abs(sc.spearman || 0) < 0.03 ? "muted" : cls(sc.spearman)}">${num(sc.spearman, 2)}</span><span class="note">rank correlation (0 = no predictive power)</span></div>
    </div>
    <h3>Stop × target heat map</h3><div id="bt-heat">${btHeat(b)}</div>
    <h3>Long: best 8 in sample, then out of sample</h3>${btBestTable(b, "long")}
    <h3>Short: best 8 in sample, then out of sample</h3>${btBestTable(b, "short")}
    ${S && S.oos.avg_r <= 0 ? `<p class="note down">The best short combination is still ${num(S.oos.avg_r)}R; only ${pct(S.positive_quarters, 0, false)} of quarters positive: in this high-beta universe, short setups are only useful as an avoid / hedge list for now.</p>` : ""}
    <p class="note">Method: combinations are ranked on the in-sample period only (the earlier 60% of days); the out-of-sample numbers (the later 40%) never took part in the choice, so they are the real test. "Random entry" = the same stocks entered every 5th session regardless of setup, with the same exit rules;
      if random entries also make money, the profit comes from market drift and the exit rule, not the setups.</p>
    <div class="grid g2">
      <div><h3>Planned R:R vs actual hit rate</h3>${table("bt-rr", [{ key: "bucket", label: "Planned R:R" }, { key: "n", label: "Trades", n: 1 },
        { key: "target_rate", label: "Actual hit", n: 1, render: (r) => pct(r.target_rate, 1, false) },
        { key: "breakeven", label: "Break-even", n: 1, render: (r) => pct(r.breakeven, 0, false) },
        { key: "ok", label: "", render: (r) => (r.target_rate >= r.breakeven ? '<span class="up">✓</span>' : '<span class="down">✗</span>') },
        { key: "avg_r", label: "Avg R", n: 1, render: (r) => rv(r.avg_r) }], d.rr_calibration || [])}
        <p class="note">The higher the planned R:R, the further the target and the faster the hit rate drops - a high R:R does not make a better trade.</p></div>
      <div><h3>Heat and run-up</h3><div class="kv">
        <div><span>Stopped, then reached the target</span><span class="num">${pct(d.stopped_then_target, 0, false)}</span></div>
        <div><span>Heat taken by winners (median / 75% / 90%)</span><span class="num">${num((d.winner_mae_atr || {}).p50)}  / ${num((d.winner_mae_atr || {}).p75)}  / ${num((d.winner_mae_atr || {}).p90)} ATR</span></div>
        <div><span>Average stop distance</span><span class="num">${num(d.avg_risk_atr)} ATR</span></div><div><span>Average target distance</span><span class="num">${num(d.avg_reward_atr)} ATR</span></div>
        <div><span>Median best open profit (MFE)</span><span class="num">${num((d.mfe_r || {}).p50)}R</span></div>
        <div><span>Reached 1R / 2R / 3R</span><span class="num">${pct((d.mfe_r || {}).ge1, 0, false)}  / ${pct((d.mfe_r || {}).ge2, 0, false)}  / ${pct((d.mfe_r || {}).ge3, 0, false)}</span></div>
        <div><span>Time-stop exits in profit</span><span class="num">${pct(d.time_exits_positive, 0, false)}</span></div></div></div>
    </div>
    <div class="grid g2">
      <div><h3>Live score quintiles</h3>${table("bt-sc", [{ key: "bucket", label: "Score bucket" }, { key: "score_range", label: "Score" }, { key: "n", label: "Trades", n: 1 },
        { key: "win_rate", label: "Win rate", n: 1, render: (r) => pct(r.win_rate, 0, false) }, { key: "avg_r", label: "Avg R", n: 1, render: (r) => rv(r.avg_r) }], sc.quintiles || [])}
        <h3>Factor predictive power (long, rank correlation with each trade's R)</h3>${icTable(fx.long, "bt-ic-l")}
        <p class="note">IC is the rank correlation between a factor and the trade outcome: |IC| 0.02–0.05 is weak; above 0.05 is useful for swing trading. "Stable" = same sign in and out of sample.
          ${calibNote(fx.long)}</p></div>
      <div><h3>Factor predictive power (short)</h3>${icTable(fx.short, "bt-ic-s")}
        <h3>By setup (live plan)</h3>${table("bt-set", [{ key: "name", label: "Setup" }, { key: "n", label: "Trades", n: 1 },
          { key: "win_rate", label: "Win rate", n: 1, render: (r) => pct(r.win_rate, 0, false) }, { key: "avg_r", label: "Avg R", n: 1, render: (r) => rv(r.avg_r) },
          { key: "t_stat", label: "t-stat", n: 1, render: (r) => tval(r.t_stat) },
          { key: "best", label: "Best for this setup in sample → Out of sample", wrap: 1, render: (r) => { const x = ((b.best_by_setup[r.key] || {}).best || [])[0]; return x ? `${esc(b.stop_names[x.stop])} + ${esc(b.target_names[x.target])} + ${x.horizon} days: ${rv(x.is.avg_r)} → ${rv(x.oos.avg_r)}` : "—"; } }],
          Object.entries(b.by_setup).map(([k, v]) => ({ key: k, ...v })), { sortKey: "avg_r" })}</div>
    </div>
    <h3>By regime (live plan, average R)</h3>${table("bt-reg", [{ key: "side", label: "Side" }, ...REGIMES.map((r) => ({ key: r, label: tr(r), n: 1, sort: (x) => (x[r] || {}).avg_r,
      render: (x) => (x[r] && x[r].n ? `${rv(x[r].avg_r)} <span class="muted">(${x[r].n})</span>` : '<span class="muted">—</span>') }))],
      [["long", "Long"], ["short", "Short"]].map(([k, lab]) => ({ side: lab, ...Object.fromEntries(REGIMES.map((r) => [r, (b.by_regime_side || {})[`${r}|${k}`]])) })))}
    <div class="grid g2">${quarterBars(Object.fromEntries(Object.entries(b.by_quarter || {}).map(([k, v]) => [k, v.avg_r])), "Quarterly average R: live plan")}
      ${L ? quarterBars(L.quarters, `Quarterly average R: best long (${b.stop_names[L.stop]} + ${b.target_names[L.target]} + ${L.horizon} days)`) : ""}</div>
    <p class="note">${esc(b.survivorship_note || "")} Other limits: daily bars only (a bar touching both stop and target counts as a stop), slippage ${b.slippage_bps} bps per side, no commissions or borrow costs, correlated signals on the same day overlap.
      Generated ${esc(b.generated_at || "")}, took ${b.elapsed_seconds} s. <button id="bt-run">Re-run backtest</button> <span id="bt-status"></span></p>
  </section>`;
}

function wireBacktest(b) {
  document.querySelectorAll(".seg[data-bt] button").forEach((btn) => btn.addEventListener("click", () => {
    const k = btn.closest(".seg").dataset.bt;
    state.bt[k] = k === "hz" ? Number(btn.dataset.v) : btn.dataset.v;
    $("#bt-heat").innerHTML = btHeat(b);
    wireBacktest(b);
  }));
  const run = $("#bt-run");
  if (run && !run.dataset.wired) {
    run.dataset.wired = 1;
    if (window.__PREVIEW__) { run.disabled = true; run.title = "Not available in the preview"; }
    run.addEventListener("click", async () => {
      try { await api("/api/backtest/run", { method: "POST" }); } catch (err) { $("#bt-status").textContent = err.message; return; }
      const poll = async () => {
        const st = await api("/api/backtest/status");
        $("#bt-status").textContent = st.running ? `Running… ${(st.log || []).slice(-1)[0] || ""}` : st.status === "success" ? "Done, reloading…" : `Failed: ${(st.log || []).slice(-1)[0] || ""}`;
        if (st.running) setTimeout(poll, 3000); else if (st.status === "success") route();
      };
      poll();
    });
  }
}

// ------------------------------------------------------------------ views: research
async function research() {
  const s = state.snap, bt = s.backtest;
  let full = null;
  try { full = await api("/api/backtest"); } catch (err) { full = { available: false, reason: err.message }; }
  let crash = null;
  try { crash = await api("/api/crash-study"); } catch (_) { crash = null; }
  const rows = Object.entries(bt.by_setup || {}).map(([k, v]) => ({ key: k, name: s.setup_names[k] || tr(k), ...v })).filter((r) => r.n);
  const regimes = ["Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish"];
  const reg = Object.entries(bt.by_setup_regime || {}).map(([k, m]) => ({ name: s.setup_names[k] || k, ...Object.fromEntries(regimes.map((r) => [r, m[r]])) }));
  app.innerHTML = `<div class="wall-head"><div><div class="eyebrow">${bt.first_signal || ""} → ${bt.last_signal || ""} / <b>${bt.n_trades} simulated trades</b></div><h1 class="display">Research <em>&amp; backtest</em></h1>
    <p>How stops and targets are set, how accurate they were over five years, and what works better; then the daily report's setup simulation and the methodology.</p></div></div>
    ${setupWeighting(s)}
    ${crashSection(crash)}
    ${slTpExplainer(full)}
    ${backtestSection(full)}
    <div class="card" style="margin-top:14px"><h2>Setup simulation <small>daily report · ${bt.horizon_days}-day time stop · slippage ${bt.slippage_bps} bps</small></h2>
    ${table("bt", [{ key: "name", label: "Setup" }, { key: "n", label: "N", n: 1 }, { key: "win_rate", label: "Win rate", n: 1, render: (r) => pct(r.win_rate, 0, false) },
      { key: "avg_r", label: "Avg R", n: 1, render: (r) => `<span class="${cls(r.avg_r)}">${num(r.avg_r)}</span>` }, { key: "median_r", label: "Median R", n: 1, render: (r) => num(r.median_r) },
      { key: "profit_factor", label: "Profit factor", n: 1, render: (r) => num(r.profit_factor) }, { key: "hit_target_rate", label: "Hit T1", n: 1, render: (r) => pct(r.hit_target_rate, 0, false) },
      { key: "stop_rate", label: "Stop", n: 1, render: (r) => pct(r.stop_rate, 0, false) }, { key: "avg_days", label: "Avg days", n: 1, render: (r) => num(r.avg_days, 1) },
      { key: "avg_mfe", label: "Avg MFE", n: 1, render: (r) => pct(r.avg_mfe) }, { key: "avg_mae", label: "Avg MAE", n: 1, render: (r) => pct(r.avg_mae) },
      { key: "fwd_5d_mean", label: "5d fwd (signed)", n: 1, render: (r) => pct(r.fwd_5d_mean, 2) }, { key: "fwd_10d_mean", label: "10d fwd (signed)", n: 1, render: (r) => pct(r.fwd_10d_mean, 2) }], rows, { sortKey: "avg_r" })}
    <p class="note">Baseline (all stock-days, long): average 5-day return ${pct((bt.baseline || {}).fwd_5d_mean, 2)}, up probability ${pct((bt.baseline || {}).fwd_5d_positive, 0, false)} (n ${(bt.baseline || {}).n}).
    "Signed" = close-to-close return in the trade direction. Biases: survivorship (today's universe), in-sample, overlapping correlated signals, no commissions or borrow costs, daily bars only.
    These numbers are a sanity check of the rules, not expected returns.</p></div>
    <div class="card" style="margin-top:14px"><h2>Average by regime on the signal day R</h2>${table("btr", [{ key: "name", label: "Setup" }, ...regimes.map((r) => ({ key: r, label: tr(r), n: 1, sort: (x) => (x[r] || {}).avg_r, render: (x) => x[r] && x[r].n ? `<span class="${cls(x[r].avg_r)}">${num(x[r].avg_r)}</span> <span class="muted">(${x[r].n})</span>` : '<span class="muted">—</span>' }))], reg)}</div>
    <div class="card" style="margin-top:14px"><h2>Methodology</h2><div id="meth" class="note">Loading…</div></div>`;
  try {
    const m = await api("/api/methodology");
    $("#meth").innerHTML = [["Market regime", m.regime], ["Setups", m.setups], ["Risk plan", m.plan], ["Scoring", m.scoring], ["Historical simulation", m.backtest]]
      .map(([t, d]) => `<h3>${t}</h3><pre class="doc">${esc((d || "").trim())}</pre>`).join("")
      + `<h3>Weights · penalties · tiers (config/terminal.yaml)</h3><pre class="doc">${esc(JSON.stringify({ weights: m.weights, penalties: m.penalties, tiers: m.tiers }, null, 2))}</pre>`;
  } catch (err) { $("#meth").textContent = err.message; }
  if (full && full.available) wireBacktest(full); else wireBacktest(null);
}

// ------------------------------------------------------------------ views: data quality
async function quality() {
  const s = state.snap, q = s.quality, u = q.update || {}, uf = s.universe_filter || {};
  app.innerHTML = `<div class="wall-head"><div><div class="eyebrow">Data checks · filter · run log</div><h1 class="display">Data <em>quality</em></h1></div></div>
    <div class="grid g4">
    <div class="card kpi"><span class="l">Data OK</span><span class="v up">${q.counts.ok}</span></div>
    <div class="card kpi"><span class="l">Warnings</span><span class="v warn">${q.counts.warning}</span></div>
    <div class="card kpi"><span class="l">Excluded (data)</span><span class="v down">${q.counts.excluded}</span></div>
    <div class="card kpi"><span class="l">Kept after filter</span><span class="v">${(uf.kept || []).length || s.universe.length}</span></div></div>
    <div class="card" style="margin-top:14px"><h2>Excluded by the high-beta / high-volatility filter <small>${(uf.excluded || []).length} </small></h2>
    ${table("uf", [{ key: "symbol", label: "Symbol", render: (x) => `<b>${x.symbol}</b>` }, { key: "beta", label: "Beta", n: 1, render: (x) => num(x.beta, 2) },
      { key: "hv60", label: "60d vol", n: 1, render: (x) => pct(x.hv60, 0, false) }, { key: "market_cap", label: "Market cap", n: 1, render: (x) => money(x.market_cap) },
      { key: "dollar_volume20", label: "Dollar volume", n: 1, render: (x) => money(x.dollar_volume20) }, { key: "reason", label: "Reason", wrap: 1 }], uf.excluded || [], { sortKey: "beta" })}</div>
    <div class="grid g2" style="margin-top:14px">
    <div class="card"><h2>Symbol checks <small>stale · missing days · OHLC · Volume · frozen quotes · spikes</small></h2>${table("dq", [
      { key: "symbol", label: "Symbol", render: (r) => `<b>${r.symbol}</b>` }, { key: "status", label: "Status", render: (r) => `<span class="${r.status === "ok" ? "up" : r.status === "warning" ? "warn" : "down"}">${tr(r.status)}</span>` },
      { key: "last_date", label: "Last bar" }, { key: "sessions", label: "Sessions", n: 1 }, { key: "missing_sessions", label: "Missing (1y)", n: 1 }, { key: "issues", label: "ISSUE", wrap: 1, render: (r) => esc(r.issues.join(" ")), sort: (r) => r.issues.length }],
      q.symbols, { sortKey: "issues" })}</div>
    <div class="card"><h2>Update summary</h2><div class="kv"><div><span>Incremental refresh</span><span class="num">${(u.incremental || []).length}</span></div><div><span>Full download</span><span class="num">${(u.full || []).length}</span></div>
      <div><span>Re-adjusted (corporate action)</span><span class="num">${(u.readjusted || []).join(", ") || 0}</span></div><div><span>Failed</span><span class="num">${(u.failed || []).join(", ") || 0}</span></div>
      <div><span>Rows written</span><span class="num">${u.rows_written ?? "n/a"}</span></div><div><span>Discarded incomplete bars</span><span class="num">${u.discarded_future_bars ?? 0}</span></div></div>
      <h3>Run log</h3><div id="runs" class="note">Loading…</div>
      <h3>Reports</h3><ul class="list" id="reports"></ul></div></div>`;
  try {
    const r = await api("/api/runs");
    $("#runs").innerHTML = table("runs", [{ key: "run_id", label: "Run", n: 1 }, { key: "session_date", label: "Session" }, { key: "status", label: "Status", render: (x) => `<span class="${x.status.startsWith("success") ? "up" : x.status === "failed" ? "down" : "warn"}">${x.status}</span>` },
      { key: "regime_label", label: "Regime" }, { key: "n_candidates", label: "Cands", n: 1 }, { key: "finished_at", label: "Finished" }], r.runs, { sortKey: "run_id" });
    const { sessions } = await api("/api/sessions");
    $("#reports").innerHTML = sessions.map((x) => `<li><a href="/reports/${x}/report.html" target="_blank">${x} report</a></li>`).join("");
  } catch (err) { $("#runs").textContent = err.message; }
}

boot();

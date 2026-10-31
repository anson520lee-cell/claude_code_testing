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
const tier = (t) => `<span class="tier tier-${esc(t)}">${esc(t === "Watch" ? "觀察" : t)}</span>`;
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
  return `<div class="tw"><table id="${id}"><thead><tr>${head}</tr></thead><tbody>${body || `<tr><td colspan="${columns.length}" class="empty">沒有資料。</td></tr>`}</tbody></table></div>`;
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
        ? `<div class="card empty"><h2>正在下載市場數據…</h2><p>首次運行會下載股票池約兩年的歷史並進行分析（約 2 至 4 分鐘），完成後此頁會自動打開報告。</p></div>`
        : `<div class="card empty"><h2>暫時沒有數據</h2><p>請按上方的「更新」按鈕，或在終端機執行 <code>python daily_run.py</code>。首次運行會下載約兩年的歷史數據（需時數分鐘）。</p></div>`;
      return;
    }
    state.session = sessions[0];
    sel.onchange = () => loadSession(sel.value);
    await loadSession(state.session);
  } catch (err) {
    app.innerHTML = `<div class="card empty">無法載入：${esc(err.message)}</div>`;
  }
}
async function loadSession(session) {
  app.innerHTML = `<div class="loading">正在載入 ${esc(session)}…</div>`;
  state.snap = await api(`/api/snapshot?session=${encodeURIComponent(session)}`);
  state.session = session;
  const reg = state.snap.regime;
  const b = $("#regime-badge");
  b.className = `badge ${regimeClass(reg.label)}`;
  b.textContent = `${tr(reg.label)} ${isNum(reg.composite) ? (reg.composite > 0 ? "+" : "") + reg.composite.toFixed(0) : ""}`;
  $("#foot-session").textContent = `交易日 ${session} · 生成於 ${state.snap.generated_at}`;
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
  if (window.__PREVIEW__) { const b = $("#banner"); b.classList.remove("hidden"); b.textContent = "這是預覽版，只能瀏覽 2026-10-30 的分析結果，不能更新數據。在 Mac 上用 Dock 的 Swing Terminal 才可以更新。"; return; }
  if (!confirm("現在運行每日分析？會下載市場數據，需時數分鐘。")) return;
  try { await api("/api/run", { method: "POST" }); watchRun(false); } catch (err) { alert(err.message); }
};
// Poll a running pipeline. ``quiet``: started elsewhere (Dock launcher / schedule) - no alert, just refresh.
async function watchRun(quiet) {
  const btn = $("#run-btn"), banner = $("#banner");
  let st;
  try { st = await api("/api/run/status"); } catch (_) { setTimeout(() => watchRun(quiet), 5000); return; }
  btn.disabled = st.running;
  btn.textContent = st.running ? "更新中…" : "更新";
  if (st.running) {
    banner.classList.remove("hidden");
    banner.textContent = `正在更新數據：${st.log.length ? st.log[st.log.length - 1] : "啟動中…"}`;
    setTimeout(() => watchRun(quiet), 3000);
    return;
  }
  if (!quiet && st.status) alert(`運行完成：${tr(st.status)}${st.message ? "－" + st.message : ""}`);
  const { sessions } = await api("/api/sessions");
  const sel = $("#session");
  sel.innerHTML = sessions.map((x) => `<option>${x}</option>`).join("");
  sel.onchange = () => loadSession(sel.value);
  if (sessions.length) loadSession(sessions[0]);
  else { banner.classList.remove("hidden"); banner.textContent = `運行${tr(st.status) || "結束"}：${st.message || ""}`; }
}

// ------------------------------------------------------------------ router
window.addEventListener("hashchange", route);
function route() {
  if (!state.snap) return;
  closeFocus(true);
  disposeCharts(state.charts);
  const [, view = "overview", arg] = (location.hash || "#/overview").split("/");
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === view));
  const views = { overview, scanner, universe, catalysts, portfolio, research, quality, ticker };
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

function renderTickerCharts(d, c, p = "c", bucket = undefined, big = 420) {
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
  if (c) {
    const pl = (v, color, title, style = 0) => candles.createPriceLine({ price: v, color, lineWidth: 1, lineStyle: style, axisLabelVisible: true, title });
    const pp = c.plan;
    pl(pp.entry, P.neutral, pp.conditional_entry ? "trigger" : "ref", 2); pl(pp.stop, P.down, "stop"); pl(pp.target1, P.up, "T1"); pl(pp.target2, P.up, "T2", 2);
  }
  const upAt = new Map(d.candles.map((k) => [k.time, k.close >= k.open]));
  const vol = makeChart($(`#${p}-vol`), 104, { ...opt, timeVisible: false });
  vol.addHistogramSeries({ priceFormat: { type: "volume" }, priceLineVisible: false, lastValueVisible: false })
    .setData(d.volume.map((x) => ({ time: x.time, value: x.value, color: upAt.get(x.time) ? P.up : P.down })));
  line(vol, d.mavol20, P.neutral, 1.2);
  const macd = makeChart($(`#${p}-macd`), 104, { ...opt, timeVisible: false });
  macd.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false }).setData(d.macd_hist.map((x) => ({ ...x, color: x.value >= 0 ? P.up : P.down })));
  line(macd, d.macd, P.dif, 1.3); line(macd, d.macd_signal, P.dea, 1.3);
  const rsi = makeChart($(`#${p}-rsi`), 104, opt);
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
  const tops = (side) => s.candidates[side].slice(0, 6).map((c) => `<tr class="click" data-sym="${c.symbol}"><td><b>${c.symbol}</b></td><td><span class="spark" data-mini-sm="${esc(c.symbol)}"></span></td><td>${esc(c.setup_name)}</td><td class="n">${c.score.toFixed(0)}</td><td>${tier(c.tier)}</td><td class="n">${num(c.plan.reward_risk, 1)}R</td></tr>`).join("") || `<tr><td class="empty">沒有</td></tr>`;
  app.innerHTML = `
  <div class="grid g-side">
    <div class="grid">
      <div class="card"><h2>市場狀態 <small>${esc(s.session.date)} · ${esc(tr(s.session.weekday))}${s.session.early_close ? " · 提早收市" : ""}</small></h2>
        <div class="grid g4">
          <div class="kpi"><span class="l">市場狀態</span><span class="v"><span class="badge ${regimeClass(r.label)}" style="font-size:18px">${esc(tr(r.label))}</span></span></div>
          <div class="kpi"><span class="l">綜合分數</span><span class="v num ${cls(r.composite)}">${isNum(r.composite) ? r.composite.toFixed(0) : "n/a"}</span><span class="note">${isNum(r.composite_5d_change) ? (Number(fixed(r.composite_5d_change, 0)) > 0 ? "+" : "") + fixed(r.composite_5d_change, 0) + "（較 5 日前）" : ""}</span></div>
          <div class="kpi"><span class="l">倉位比例（做多 / 做空）</span><span class="v num">×${r.long_risk_budget} / ×${r.short_risk_budget}</span></div>
          <div class="kpi"><span class="l">VIX · VIX/VIX3M</span><span class="v num">${num(v.vix, 1)} <span class="muted" style="font-size:14px">${num(v.vix_term, 2)}</span></span></div>
        </div>
        <div style="margin:14px 0 4px"><div class="gauge"><div class="needle" style="left:calc(${needle}% - 1px)"></div></div>
          <div class="gauge-labels"><span>強勢看空</span><span>看空</span><span>中性</span><span>看多</span><span>強勢看多</span></div></div>
        <p>${esc(r.guidance)}</p>
        ${comps}
        <p class="note">綜合分數 = Σ 權重 × 分數 / 2（−100 至 +100）。計算方法見<a href="#/research">「研究」頁</a>。</p>
      </div>
      <div class="card"><h2>市場狀態走勢 <small>綜合分數，最近 ${r.history.length} 個交易日</small></h2><div id="regime-chart" class="chart mid"></div></div>
      <div class="card"><h2>指數 · 波動率 · 信貸</h2>${table("idx", [
        { key: "symbol", label: "代號", render: (x) => `<b>${esc(x.symbol)}</b>` },
        { key: "close", label: "收市", n: 1, render: (x) => num(x.close) },
        { key: "ret_1d", label: "1 日", n: 1, render: (x) => `<span class="${cls(x.ret_1d)}">${pct(x.ret_1d)}</span>` },
        { key: "ret_5d", label: "5 日", n: 1, render: (x) => `<span class="${cls(x.ret_5d)}">${pct(x.ret_5d)}</span>` },
        { key: "ret_20d", label: "20 日", n: 1, render: (x) => `<span class="${cls(x.ret_20d)}">${pct(x.ret_20d)}</span>` },
        { key: "ret_ytd", label: "年初至今", n: 1, render: (x) => `<span class="${cls(x.ret_ytd)}">${pct(x.ret_ytd)}</span>` },
        { key: "vs_ema50", label: "相對 EMA50", n: 1, render: (x) => pct(x.vs_ema50) },
        { key: "vs_ema200", label: "相對 EMA200", n: 1, render: (x) => pct(x.vs_ema200) },
        { key: "from_52w_high", label: "距 52 週高位", n: 1, render: (x) => pct(x.from_52w_high) },
        { key: "ema_stack", label: "均線排列", render: (x) => `<span class="${x.ema_stack === "bullish" ? "up" : x.ema_stack === "bearish" ? "down" : "muted"}">${tr(x.ema_stack)}</span>` },
      ], s.indices)}</div>
    </div>
    <div class="grid" style="align-content:start">
      <div class="card"><h2>市場寬度 <small>股票池（${num(b.breadth_n, 0)} 檔）</small></h2>
        <div class="kv"><div><span>高於 EMA50</span><span class="num">${num(b.pct_above_ema50, 0)}%</span></div>
        <div><span>高於 EMA200</span><span class="num">${num(b.pct_above_ema200, 0)}%</span></div>
        <div><span>創 20 日新高</span><span class="num up">${num(b.pct_new_high20, 0)}%</span></div>
        <div><span>創 20 日新低</span><span class="num down">${num(b.pct_new_low20, 0)}%</span></div>
        <div><span>美國 10 年期</span><span class="num">${num(rt.us10y, 2)}%</span></div>
        <div><span>10 年期 20 日變化</span><span class="num ${cls(-(rt.us10y_chg20_bp || 0))}">${num(rt.us10y_chg20_bp, 0)}bp</span></div>
        <div><span>2 年／10 年息差</span><span class="num">${isNum(rt.curve_2s10s) ? (rt.curve_2s10s * 100).toFixed(0) + "bp" : "n/a"}</span></div>
        <div><span>VIX 5 日變化</span><span class="num ${cls(-(v.vix_chg5 || 0))}">${pct(v.vix_chg5, 0)}</span></div></div></div>
      <div class="card"><h2>做多候選前列</h2><div class="tw"><table><tbody>${tops("long")}</tbody></table></div></div>
      <div class="card"><h2>做空候選前列</h2><div class="tw"><table><tbody>${tops("short")}</tbody></table></div></div>
      <div class="card"><h2>即將發生的事件</h2><ul class="list">${(s.macro_events || []).slice(0, 6).map((e) => `<li><span class="mono">${e.event_date}</span> ${esc(e.event_type)} <span class="chip">${tr(e.status)}</span></li>`).join("") || "<li>未來 30 日沒有</li>"}
        ${(s.upcoming_earnings || []).filter((e) => e.sessions_until <= 10).slice(0, 8).map((e) => `<li><span class="mono">${e.date}</span> <a href="#/ticker/${e.symbol}">${e.symbol}</a> 財報 ${esc(tr(e.timing) || "（時間未確認）")} <span class="chip">${tr(e.status)}</span></li>`).join("")}</ul></div>
    </div>
  </div>
  <div class="grid g2" style="margin-top:12px">
    <div class="card"><h2>板塊輪動 <small>相對 SPY：RS 趨勢（橫軸）對 RS 動量（縱軸），尾巴為近 10 日軌跡</small></h2><svg id="rrg" class="rrg"></svg></div>
    <div class="card"><h2>板塊及行業 ETF</h2>${table("sec", [
      { key: "rank_20d", label: "#", n: 1 },
      { key: "symbol", label: "ETF", render: (x) => `<b>${x.symbol}</b> <span class="muted">${esc(tr(x.name))}</span>` },
      { key: "ret_5d", label: "5 日", n: 1, render: (x) => `<span class="${cls(x.ret_5d)}">${pct(x.ret_5d)}</span>` },
      { key: "ret_20d", label: "20 日", n: 1, render: (x) => `<span class="${cls(x.ret_20d)}">${pct(x.ret_20d)}</span>` },
      { key: "rel_20d", label: "相對 20 日", n: 1, render: (x) => `<span class="${cls(x.rel_20d)}">${pct(x.rel_20d)}</span>` },
      { key: "rel_60d", label: "相對 60 日", n: 1, render: (x) => `<span class="${cls(x.rel_60d)}">${pct(x.rel_60d)}</span>` },
      { key: "quadrant", label: "象限", render: (x) => `<span class="${{ Leading: "up", Lagging: "down", Improving: "", Weakening: "warn" }[x.quadrant] || "muted"}">${tr(x.quadrant)}</span>` },
    ], s.sectors, { sortKey: "rank_20d", sortDir: "asc" })}</div>
  </div>
  ${marketExtras(s)}`;
  const el = $("#regime-chart");
  const ch = makeChart(el, 220);
  const line = ch.addBaselineSeries({ baseValue: { type: "price", price: 0 }, topLineColor: cssVar("--up"), bottomLineColor: cssVar("--down"),
    topFillColor1: "rgba(38,166,154,0.25)", topFillColor2: "rgba(38,166,154,0.02)", bottomFillColor1: "rgba(239,83,80,0.02)", bottomFillColor2: "rgba(239,83,80,0.25)", lineWidth: 2 });
  line.setData(r.history.filter((h) => isNum(h.composite)).map((h) => ({ time: h.date, value: h.composite })));
  for (const [p, t] of [[45, "強勢看多"], [15, "看多"], [-15, "看空"], [-45, "強勢看空"]])
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
    out += `<circle cx="${X(p.rs_trend)}" cy="${Y(p.rs_momentum)}" r="4.5" fill="${color}"><title>${p.symbol} ${tr(p.name)}：RS 趨勢 ${pct(p.rs_trend)}，RS 動量 ${pct(p.rs_momentum)}</title></circle>`;
    out += `<text x="${X(p.rs_trend) + 6}" y="${Y(p.rs_momentum) - 6}" fill="${cssVar("--fg")}" font-size="11">${p.symbol}</text>`;
  }
  out += `<text x="${W / 2}" y="${H - 6}" text-anchor="middle" fill="${cssVar("--dim")}" font-size="10">RS 趨勢（RS ÷ RS 50 日均線 − 1）→</text>`;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = out;
}

// ------------------------------------------------------------------ views: scanner
function earningsOf(c) { return ((state.snap.deep_dive[c.symbol] || {}).catalysts || {}).next_earnings; }
function scanner() {
  const s = state.snap, f = state.filters, side = state.scannerSide, mode = state.scanMode || "cards";
  const all = s.candidates[side];
  const setups = [...new Set(all.map((c) => c.setup))];
  const rows = all.filter((c) => (!f.tier || c.tier === f.tier) && (!f.setup || c.setup === f.setup)
    && (!f.hideEarnings || !c.penalties.earnings_in_window) && (!f.q || c.symbol.includes(f.q.toUpperCase())));
  state.wallList = rows;
  app.innerHTML = `
  <div class="wall-head"><div><div class="eyebrow">交易日 ${esc(s.session.date)} / <b>市場狀態 ${esc(tr(s.regime.label))}</b></div>
    <h1 class="display">波段候選 <em>swing</em></h1>
    <p>按研究優先次序排列，不是交易信號。每張卡片和表格每一行都有近期 K 線預覽，點擊可打開該股票的完整圖表和研究資料。</p></div>
    <div class="filters" style="margin:0">
      <div class="seg" id="seg-side"><button data-v="long" class="${side === "long" ? "active" : ""}">做多 ${s.candidates.long.length}</button><button data-v="short" class="${side === "short" ? "active" : ""}">做空 ${s.candidates.short.length}</button></div>
      <div class="seg" id="seg-mode"><button data-v="cards" class="${mode === "cards" ? "active" : ""}">卡片</button><button data-v="table" class="${mode === "table" ? "active" : ""}">表格</button></div>
    </div></div>
  <div class="filters">
    <label>級別 <select id="f-tier"><option value="">全部</option>${["A", "B", "C", "Watch"].map((t) => `<option value="${t}" ${f.tier === t ? "selected" : ""}>${t === "Watch" ? "觀察" : t}</option>`).join("")}</select></label>
    <label>形態 <select id="f-setup"><option value="">全部</option>${setups.map((x) => `<option value="${x}" ${f.setup === x ? "selected" : ""}>${esc(s.setup_names[x])}</option>`).join("")}</select></label>
    <label><input type="checkbox" id="f-earn" ${f.hideEarnings ? "checked" : ""}> 隱藏持倉期內有財報的股票</label>
    <label>搜尋 <input id="f-q" value="${esc(f.q)}" style="width:90px"></label>
    <span class="note">顯示 ${rows.length} 檔</span>
  </div>
  ${mode === "cards" ? `<div class="wall">${rows.map((c, i) => stockCard(c, i)).join("") || '<div class="card empty">沒有符合篩選條件的股票。</div>'}</div>` : `<div class="card">${scanTable(side, rows)}</div>`}`;
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
    { key: "symbol", label: "代號", render: (c) => `<b>${c.symbol}</b> <span class="muted">${esc(c.sector_etf || "")}</span>` },
    { key: "spark", label: "走勢（40 日）", render: (c) => `<span class="spark" data-mini-sm="${esc(c.symbol)}"></span>`, sort: (c) => c.technical.return_20d },
    { key: "setup_name", label: "形態", render: (c) => `${esc(c.setup_name)}${c.other_setups.length ? ` <span class="chip">+${c.other_setups.length}</span>` : ""}` },
    { key: "score", label: "分數", n: 1, render: (c) => `<div class="scorecell"><div class="bar"><span style="width:${c.score}%"></span></div><span>${c.score.toFixed(0)}</span></div>` },
    { key: "tier", label: "級別", render: (c) => tier(c.tier), sort: (c) => c.score },
    { key: "entry", label: "進場", n: 1, render: (c) => num(c.plan.entry) + (c.plan.conditional_entry ? ' <span class="chip">cond.</span>' : ""), sort: (c) => c.plan.entry },
    { key: "stop", label: "止損", n: 1, render: (c) => num(c.plan.stop), sort: (c) => c.plan.stop },
    { key: "t1", label: "目標一", n: 1, render: (c) => num(c.plan.target1), sort: (c) => c.plan.target1 },
    { key: "rr", label: "風險報酬", n: 1, render: (c) => `<span class="${c.plan.reward_risk >= 2 ? "up" : c.plan.reward_risk < 1.5 ? "down" : ""}">${num(c.plan.reward_risk, 1)}</span>`, sort: (c) => c.plan.reward_risk },
    { key: "risk", label: "風險 %", n: 1, render: (c) => pct(c.plan.risk_pct, 1, false), sort: (c) => c.plan.risk_pct },
    { key: "rs", label: "RS 百分位", n: 1, render: (c) => num(c.technical.rs_percentile, 0), sort: (c) => c.technical.rs_percentile },
    { key: "earn", label: "財報", render: (c) => { const e = earningsOf(c); return e ? `<span class="${e.sessions_until <= 7 ? "warn" : ""}">${e.date} (${e.sessions_until})</span>` : '<span class="muted">—</span>'; }, sort: (c) => (earningsOf(c) || {}).sessions_until ?? 999 },
    { key: "ev", label: "證據 +/−", n: 1, render: (c) => `<span class="up">${c.supporting.length}</span> / <span class="down">${c.contradicting.length}</span>`, sort: (c) => c.supporting.length - c.contradicting.length },
    { key: "pen", label: "扣分", render: (c) => Object.keys(c.penalties).map((k) => `<span class="chip">${tr(k)}</span>`).join("") || '<span class="muted">—</span>', sort: (c) => c.penalty_total },
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
  return `<div><div class="scard" tabindex="0" role="button" data-i="${i}" aria-label="${esc(c.symbol)} ${esc(c.setup_name)}，分數 ${c.score.toFixed(0)}">
    <div class="sc-top z1"><span class="sc-label">${esc(tr(c.direction))} · <b>${String(c.rank).padStart(2, "0")}</b></span>${tier(c.tier)}</div>
    <div class="sc-id z2"><div><div class="sc-sym">${esc(c.symbol)}</div><div class="sc-name">${esc(name)}</div></div>
      <div class="sc-px"><div class="p">${num(t.close)}</div><div class="c ${cls(t.return_1d)}">${pct(t.return_1d, 2)}</div></div></div>
    <div class="sc-chart z1" data-mini="${esc(c.symbol)}"></div>
    <div class="sc-setup z1">${ring(c.score)}<div class="s">${esc(c.setup_name)}<div class="note">${esc(tr(t.trend))} · ${esc(tr(t.swing_structure))}</div></div></div>
    ${c.analyst ? `<div class="sc-bias"><b class="${biasClass(c.analyst.bias)}">${esc(c.analyst.bias)}</b> · 信心${esc(c.analyst.confidence)} — ${esc(c.analyst.headline.split("：").slice(1).join("："))}</div>` : ""}
    <div class="sc-stats z1"><div><span>風險報酬</span><b class="${c.plan.reward_risk >= 2 ? "up" : c.plan.reward_risk < 1.5 ? "down" : ""}">${num(c.plan.reward_risk, 1)}</b></div>
      <div><span>風險</span>${pct(c.plan.risk_pct, 1, false)}</div><div><span>Beta</span>${num(t.beta120, 2)}</div>
      <div><span>財報</span><b class="${e && e.sessions_until <= 7 ? "warn" : ""}">${e ? e.sessions_until + " 日" : "—"}</b></div></div>
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
    ["下次財報", e ? `${e.date} · ${e.sessions_until} 日` : "—", e && e.sessions_until <= 7 ? "warn" : ""],
    ["營收按年", pct(f.revenue_yoy, 0), cls(f.revenue_yoy)], ["EPS 按年", pct(f.eps_yoy, 0), cls(f.eps_yoy)],
    ["經營利潤率", pct(f.operating_margin_ttm, 1, false), ""], ["自由現金流", money(f.fcf_ttm), cls(f.fcf_ttm)], ["市盈率", num(f.pe_ttm, 1), ""],
    ["平價隱含波動率", o.available ? pct(o.atm_iv, 0, false) : "n/a", ""], ["IV / HV20", o.available ? num(o.iv_hv_ratio, 2) : "n/a", o.iv_hv_ratio > 1.4 ? "warn" : ""],
    ["期權預期波幅", o.available ? "±" + pct(o.expected_move_pct, 1, false) : "n/a", ""], ["認沽／認購成交比", o.available ? num(o.pcr_volume, 2) : "n/a", ""],
    ["做空佔流通股", pct(p.short_percent_float, 1, false), p.short_percent_float >= 0.15 ? "warn" : ""], ["回補天數", num(p.days_to_cover, 1), ""],
    ["Beta（120 日）", num(c.technical.beta120, 2), ""], ["60 日波動率", pct(c.technical.hv60, 0, false), ""],
  ];
  return `<div class="facts">${items.map(([k, v, cl]) => `<div class="fact"><span>${k}</span><b class="${cl}">${v}</b></div>`).join("")}</div>`;
}

function focusHTML(list, i) {
  const c = list[i], t = c.technical, deep = state.snap.deep_dive[c.symbol] || {};
  const name = (deep.profile || {}).longName || "";
  return `<div class="focus-body">
    <div class="focus-head">
      <div><div class="eyebrow">${esc(tr(c.direction))} · ${String(c.rank).padStart(2, "0")} / <b>${esc(c.setup_name)}</b></div>
        <div class="display sym">${esc(c.symbol)} <em style="font-size:.42em;color:var(--muted)">${esc(name)}</em></div></div>
      <div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap">
        <div style="text-align:right"><div class="px">${num(t.close)}</div><div class="${cls(t.return_1d)}" style="font-weight:650">今日 ${pct(t.return_1d, 2)}</div></div>
        ${ring(c.score, 58)}${tier(c.tier)}
        <div class="focus-tools"><button data-act="prev" aria-label="上一檔" title="上一檔（←）">‹</button><button data-act="next" aria-label="下一檔" title="下一檔（→）">›</button>
          <button class="wide" data-act="full">完整研究頁 ↗</button><button data-act="close" aria-label="關閉" title="關閉（Esc）">✕</button></div>
      </div></div>
    ${c.analyst ? analystPanel(c.analyst, { compact: true }) : '<div id="f-analyst"></div>'}
    <div class="card" style="margin-top:22px">${chartBlock("f", 380)}</div>
    <div style="margin:14px 0">${quickFacts(c, deep)}</div>
    ${c.analyst ? `<div class="eyebrow" style="margin-top:22px">詳細分析</div><div class="sections">${c.analyst.sections.map((x) => {
      const sig = x.signals; return `<div class="sec"><h4>${esc(x.title)}</h4>${x.body.map((t, i) => `<p>${sig ? `<span class="sig ${sig[i] && sig[i].signal > 0 ? "up" : sig[i] && sig[i].signal < 0 ? "down" : "muted"}">${sig[i] && sig[i].signal > 0 ? "+" : sig[i] && sig[i].signal < 0 ? "−" : "·"}</span>` : ""}${esc(t)}</p>`).join("")}</div>`; }).join("")}</div>` : ""}
    <div class="grid g-side"><div class="grid" style="align-content:start">${candidatePanels(c, deep)}</div>
      <div class="grid" style="align-content:start">${planPanel(c)}${earningsPanel(deep.earnings_study)}${deep.catalysts ? catalystsPanel(deep.catalysts) : ""}${ftdPanel(deep.ftd)}</div></div>
  </div>`;
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
  if (!list[i].analyst) { const el = panel.querySelector("#f-analyst"); if (el) loadAnalyst(list[i].symbol, el); }
  try {
    const d = await api(`/api/chart/${encodeURIComponent(list[i].symbol)}?days=160&session=${state.session}`);
    if (Number(bd.dataset.i) === i && bd.isConnected) renderTickerCharts(d, list[i], "f", state.focusCharts, 380);
  } catch (err) { const el = panel.querySelector("#f-price"); if (el) el.innerHTML = `<div class="empty">無法載入圖表：${esc(err.message)}</div>`; }
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

// ------------------------------------------------------------------ views: ticker
const levelLabel = (t) => String(t || "").replace(/swing high/g, "波段高點").replace(/swing low/g, "波段低點")
  .replace(/(\d+)D high/g, "$1 日高位").replace(/(\d+)D low/g, "$1 日低位");
async function ticker(sym) {
  sym = (sym || "").toUpperCase();
  const s = state.snap;
  const cands = [...s.candidates.long, ...s.candidates.short].filter((c) => c.symbol === sym);
  const deep = s.deep_dive[sym] || {};
  const uni = s.universe.find((u) => u.symbol === sym);
  const c = cands[0];
  app.innerHTML = `<div class="grid g-side">
    <div class="grid" style="align-content:start">
      <div class="card"><div class="focus-head"><div><div class="eyebrow">${c ? `${esc(tr(c.direction))} · <b>${esc(c.setup_name)}</b>` : "圖表"}</div>
        <div class="display" style="font-size:clamp(40px,4.4vw,64px)">${esc(sym)}</div><div class="note">${esc((deep.profile || {}).longName || "")}</div></div><small class="note" id="t-asof"></small></div>
        <div id="t-analyst"></div>
        ${chartBlock("c")}
      </div>
      ${c ? candidatePanels(c, deep) : `<div class="card"><h2>本交易日不是候選股</h2><p class="note">${uni ? `目前形態：${uni.setups.length ? uni.setups.map((x) => esc(s.setup_names[x])).join("、") : "沒有"}。趨勢「${esc(tr(uni.trend))}」，結構「${esc(tr(uni.structure))}」，相對強弱第 ${num(uni.rs_pct, 0)} 百分位。` : "不在篩選後的股票池內；如有已儲存的價格，只顯示圖表。"}</p></div>`}
      ${deep.fundamentals ? fundamentalsPanel(deep.fundamentals) : ""}
    </div>
    <div class="grid" style="align-content:start">
      ${c ? planPanel(c) : ""}
      <div class="card"><h2>技術概況</h2><div id="t-snap" class="kv"></div><h3>支持／阻力（60 個交易日）</h3><div id="t-levels" class="note">…</div></div>
      ${deep.options ? optionsPanel(deep.options) : ""}
      ${deep.positioning ? positioningPanel(deep.positioning) : ""}
      ${earningsPanel(deep.earnings_study)}
      ${deep.catalysts ? catalystsPanel(deep.catalysts) : ""}
      ${ftdPanel(deep.ftd)}
      <div class="card"><h2>候選紀錄</h2><div id="t-hist" class="note">…</div></div>
    </div></div>`;
  if (c && c.analyst) $("#t-analyst").innerHTML = analystPanel(c.analyst); else loadAnalyst(sym, $("#t-analyst"));
  try {
    const d = await api(`/api/chart/${encodeURIComponent(sym)}?days=200&session=${state.session}`);
    renderTickerCharts(d, c);
    $("#t-asof").textContent = `as of ${d.as_of}`;
    const L = d.last;
    $("#t-snap").innerHTML = [["收市", num(L.close)], ["ATR(14)", `${num(L.atr)}（${num(L.atr_pct, 1)}%）`], ["RSI6 / RSI14", `${num(L.rsi6, 0)} / ${num(L.rsi14, 0)}`],
      ["趨勢", esc(tr(L.trend))], ["波段結構", esc(tr(L.swing_structure))], ["結構狀態", esc(tr(L.structure))], ["波段狀況", esc(tr(L.swing_condition))],
      ["量比（成交量 / MAVOL20）", num(L.volume_ratio, 2) + "×"], ["距 EMA21（ATR）", num(L.extension_atr, 2)], ["20 日波動率", pct(L.hv20, 0, false)],
      ["相對 SPY 20 日 / 60 日", `${pct(L.rel_20d)} / ${pct(L.rel_60d)}`], ["距 52 週高位", pct(L.from_52w_high)]]
      .map(([k, v]) => `<div><span>${k}</span><span class="num">${v}</span></div>`).join("");
    $("#t-levels").innerHTML = ["resistance", "support"].map((side) => `<div><b class="${side === "support" ? "up" : "down"}">${side === "support" ? "支持" : "阻力"}</b>：${d.levels[side].map((l) => `<span class="mono">${num(l.price)}</span> <span class="muted">（${esc(levelLabel(l.label))}，${num(l.distance_atr, 1)} ATR）</span>`).join(" · ") || "範圍內沒有"}</div>`).join("");
  } catch (err) {
    $("#c-price").innerHTML = `<div class="empty">無法載入圖表：${esc(err.message)}</div>`;
  }
  try {
    const h = await api(`/api/candidate-history/${encodeURIComponent(sym)}`);
    $("#t-hist").innerHTML = h.history.length ? table("chist", [
      { key: "session_date", label: "交易日" }, { key: "direction", label: "方向", render: (x) => tr(x.direction) }, { key: "setup", label: "形態", render: (x) => esc(s.setup_names[x.setup] || x.setup) },
      { key: "score", label: "分數", n: 1, render: (x) => num(x.score, 0) }, { key: "tier", label: "級別", render: (x) => tier(x.tier) }], h.history, { sortKey: "session_date" }) : "從未列為候選股。";
  } catch (_) { $("#t-hist").textContent = "n/a"; }
}

// ------------------------------------------------------------------ analyst note
function biasClass(label) { return /偏多/.test(label || "") ? "up" : /偏空/.test(label || "") ? "down" : ""; }
function analystPanel(a, opts = {}) {
  if (!a) return "";
  const rows = a.scorecard.map((x) => {
    const w = Math.abs(x.score) / 2 * 50, left = x.score < 0 ? 50 - w : 50, col = x.score >= 0 ? "var(--up)" : "var(--down)";
    return `<div class="sc-row"><span class="nm">${esc(x.name)} <span class="muted">${x.weight}</span></span>
      <div class="zbar"><div class="mid"></div><div class="fill" style="left:${left}%;width:${w}%;background:${col}"></div></div>
      <span class="n ${cls(x.score)}">${x.score > 0 ? "+" : ""}${x.score.toFixed(1)}</span><span class="tx">${esc(x.text)}</span></div>`;
  }).join("");
  const scen = a.scenarios.map((x) => `<div><h4 class="${x.name.startsWith("多") ? "up" : x.name.startsWith("空") ? "down" : ""}">${esc(x.name)}</h4>
    <p><b>條件</b> ${esc(x.trigger)}</p><p><b>目標</b> ${esc(x.target)}</p><p><b>失效</b> ${esc(x.invalidation)}</p></div>`).join("");
  const secs = a.sections.map((x) => {
    const sig = x.signals ? x.signals : null;
    const nb = sig ? sig.filter((y) => y.signal > 0).length : 0, ns = sig ? sig.filter((y) => y.signal < 0).length : 0;
    return `<div class="sec"><h4>${esc(x.title)}${sig ? `<small><span class="up">+${nb}</span> / <span class="down">−${ns}</span> / ${sig.length}</small>` : ""}</h4>
      ${x.body.map((t, i) => `<p>${sig ? `<span class="sig ${sig[i] && sig[i].signal > 0 ? "up" : sig[i] && sig[i].signal < 0 ? "down" : "muted"}">${sig[i] && sig[i].signal > 0 ? "+" : sig[i] && sig[i].signal < 0 ? "−" : "·"}</span>` : ""}${esc(t)}</p>`).join("")}</div>`;
  }).join("");
  const lv = a.levels || {};
  const levels = [...(lv.resistance || []).map((x) => ["阻力", x]), ...(lv.support || []).map((x) => ["支持", x])]
    .map(([k, x]) => `<div><span>${k} ${num(x.price)}</span><span class="num">${esc(x.label)} · ${pct(x.pct)}</span></div>`).join("");
  const vp = a.volume_profile;
  return `<div class="note-block">
    <div class="eyebrow">分析師報告 / <b>${esc(a.symbol)}</b> · 截至 ${esc(a.as_of)} · 綜合 ${a.scorecard.length} 個面向、${(a.techniques || {}).count || 0} 項技術分析</div>
    <div class="note-head"><div class="hl">${esc(a.headline)}</div>
      <div class="meta">整體判斷<b class="${biasClass(a.bias)}">${esc(a.bias)}</b>${a.bias_score > 0 ? "+" : ""}${a.bias_score.toFixed(0)} / 100 · 信心${esc(a.confidence)}</div></div>
    <p class="note-summary">${esc(a.summary)}</p>
    <div class="eyebrow">評分卡（−2 至 +2，右邊數字為權重）</div><div class="scorecard">${rows}</div>
    <div class="eyebrow" style="margin-top:22px">情景分析</div><div class="scen">${scen}</div>
    <div class="grid g2" style="margin-top:22px">
      <div><div class="eyebrow">接下來要留意</div><ul class="list">${a.watch.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
        <div class="eyebrow" style="margin-top:18px">主要風險</div><ul class="list ev-neg">${a.risks.map((x) => `<li>${esc(x)}</li>`).join("") || "<li>沒有特別列出的風險</li>"}</ul></div>
      <div><div class="eyebrow">關鍵價位</div><div class="kv" style="grid-template-columns:1fr">${levels}
        <div><span>EMA21 / EMA50 / EMA200</span><span class="num">${num(lv.ema21)} / ${num(lv.ema50)} / ${num(lv.ema200)}</span></div>
        <div><span>52 週高 / 低</span><span class="num">${num(lv.high52)} / ${num(lv.low52)}</span></div>
        ${lv.fib ? `<div><span>斐波那契回撤（${esc(lv.fib.direction)}）</span><span class="num">38.2% ${num(lv.fib.levels["0.382"])} · 50% ${num(lv.fib.levels["0.5"])} · 61.8% ${num(lv.fib.levels["0.618"])}</span></div>` : ""}
        ${vp ? `<div><span>成交量控制點 / 價值區</span><span class="num">${num(vp.poc)} / ${num(vp.val)}–${num(vp.vah)}</span></div>` : ""}</div>
        ${vp ? volumeProfileSVG(vp, a.close) : ""}</div></div>
    ${opts.compact ? "" : `<div class="eyebrow" style="margin-top:24px">詳細分析</div><div class="sections">${secs}</div>`}
    <p class="note">以上由系統按固定規則綜合數據生成，屬描述性研究，不構成投資建議。</p></div>`;
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
  g += line(vp.poc, cssVar("--warn"), "POC") + line(close, cssVar("--fg"), "現價");
  return `<div class="eyebrow" style="margin-top:16px">成交量分佈（近 ${vp.window} 日；綠＝現價以下的籌碼，黃底＝真空區）</div><svg viewBox="0 0 ${W} ${H}" style="width:100%;height:${H}px">${g}</svg>`;
}

async function loadAnalyst(sym, el, compact = false) {
  el.innerHTML = '<div class="note-block"><div class="eyebrow">分析師報告</div><p class="note">正在綜合技術、籌碼、基本面、期權和催化劑…</p></div>';
  try {
    const r = await api(`/api/analyst/${encodeURIComponent(sym)}?session=${state.session}&live=true`);
    if (el.isConnected) el.innerHTML = analystPanel(r.note, { compact });
    return r;
  } catch (err) { if (el.isConnected) el.innerHTML = `<div class="note-block"><p class="note">無法生成分析師報告：${esc(err.message)}</p></div>`; }
  return null;
}

function candidatePanels(c, deep) {
  const comps = Object.entries(c.components).map(([k, v]) => `<div class="comp"><div>${tr(k)} <span class="muted">· ${state.snap.settings.weights[k]}</span></div><div class="bar"><span style="width:${v}%"></span></div><div class="n mono">${v.toFixed(0)}</div></div>`).join("");
  const pens = Object.entries(c.penalties).map(([k, v]) => `<span class="chip">${tr(k)} −${v}</span>`).join("") || '<span class="muted">沒有</span>';
  const h = c.history || {};
  const hrow = (st, label) => st && st.n ? `<tr><td>${label}</td><td class="n">${st.n}${st.small_sample ? " ⚠" : ""}</td><td class="n">${pct(st.win_rate, 0, false)}</td><td class="n ${cls(st.avg_r)}">${num(st.avg_r)}</td><td class="n">${pct(st.hit_target_rate, 0, false)}</td><td class="n">${pct(st.stop_rate, 0, false)}</td><td class="n">${num(st.avg_days, 1)}</td></tr>` : `<tr><td>${label}</td><td colspan="6" class="muted">沒有歷史</td></tr>`;
  const g = c.group;
  return `<div class="card"><h2>研究摘要 <small>${esc(tr(c.direction))} · ${esc(c.setup_name)}</small></h2><p>${esc(c.thesis)}</p>
    ${g ? `<p class="note">行業組：<b>${esc(g.group)}</b>，相對強弱排第 ${g.rank}／${g.of} 組${g.rank_change ? `（5 日${g.rank_change > 0 ? "上升" : "下跌"} ${Math.abs(g.rank_change)} 位）` : ""}。</p>` : ""}
    <div class="grid g2"><div><h3 class="up">支持證據</h3><ul class="list ev-pos">${c.supporting.map((x) => `<li>${esc(x)}</li>`).join("") || "<li class=muted>沒有</li>"}</ul></div>
    <div><h3 class="down">反對證據／風險</h3><ul class="list ev-neg">${c.contradicting.map((x) => `<li>${esc(x)}</li>`).join("") || "<li class=muted>沒有</li>"}</ul></div></div></div>
    <div class="card"><h2>評分拆解 <small>原始 ${num(c.raw_score, 1)} − 扣分 ${num(c.penalty_total, 0)} = ${num(c.score, 1)}</small></h2>${comps}<div style="margin-top:8px">扣分：${pens}</div></div>
    <div class="card"><h2>此形態的歷史回測 <small>全股票池，樣本內</small></h2><div class="tw"><table><thead><tr><th>樣本</th><th class="n">數量</th><th class="n">勝率</th><th class="n">平均 R</th><th class="n">達目標一</th><th class="n">止損</th><th class="n">日數</th></tr></thead>
    <tbody>${hrow(h.setup, "所有市況")}${hrow(h.setup_in_regime, `${esc(tr(h.regime))}市況`)}</tbody></table></div>
    <p class="note">翌日開市進場，用相同的止損／目標一／時間止損規則，已計滑點。存在倖存者偏差和樣本內偏差；⚠ 表示少於 30 筆。</p></div>`;
}

function planPanel(c) {
  const p = c.plan;
  const budget = c.direction === "long" ? state.snap.regime.long_risk_budget : state.snap.regime.short_risk_budget;
  return `<div class="card"><h2>風險計劃 <small>${p.horizon_days} 個交易日時間止損</small></h2><div class="kv">
    <div><span>${p.conditional_entry ? "觸發價（條件進場）" : "參考進場價"}</span><span class="num">${num(p.entry)}</span></div>
    <div><span>止損</span><span class="num down">${num(p.stop)}（${num(p.risk_atr, 1)} ATR）</span></div>
    <div><span>目標一</span><span class="num up">${num(p.target1)}</span></div><div><span>目標二</span><span class="num up">${num(p.target2)}</span></div>
    <div><span>風險報酬比（目標一）</span><span class="num">${num(p.reward_risk, 2)}</span></div><div><span>每股風險</span><span class="num">${pct(p.risk_pct, 2, false)}</span></div>
    <div><span>結構止損</span><span class="num ${p.wide_stop ? "warn" : ""}">${num(p.structural_stop_atr, 1)} ATR${p.wide_stop ? "（已設上限）" : ""}</span></div>
    <div><span>${p.horizon_days} 日 1σ 波幅（HV20）</span><span class="num">±${num(p.expected_move_hv)}</span></div></div>
    <p class="note">倉位大小由你決定。例子：以 10 萬美元帳戶承擔 0.5% 風險 = $500 ÷ ${num(Math.abs(p.entry - p.stop))} ≈ ${isNum(p.entry - p.stop) && p.entry !== p.stop ? Math.floor(500 / Math.abs(p.entry - p.stop)) : "n/a"} 股（再乘以市況倉位比例 ×${budget}）。</p>
    <div class="qa">加價格提醒：${(() => {
      const L = c.direction === "long", items = [[L ? "below" : "above", p.stop, "觸及止損"], [L ? "above" : "below", p.target1, "到達目標一"]];
      if (p.conditional_entry) items.unshift([L ? "above" : "below", p.entry, "觸發進場"]);
      return items.map(([k, lv, n]) => `<button data-qa="${n}" data-sym="${c.symbol}" data-kind="${k}" data-level="${lv}">${n} ${num(lv)}</button>`).join("");
    })()}</div>
    ${planBasis(c)}</div>`;
}

function planBasis(c) {
  const p = c.plan, long = c.direction === "long", b = state.btFull;
  let alt = "";
  const ss = b && (b.selected_side || {})[c.direction];
  if (ss && isNum(p.atr)) {
    const k = ss.stop.startsWith("atr") ? Number(ss.stop.slice(3)) : null, sg = long ? 1 : -1;
    const bl = (b.baseline || {})[`${c.direction}|${ss.stop}|${ss.target}|${ss.horizon}`];
    const tgt = ss.target === "trail" ? `不設固定目標，用移動止損（${long ? "持倉期最高價 − 3 ATR" : "持倉期最低價 + 3 ATR"}，約 ${num(3 * p.atr)}）`
      : ss.target.startsWith("r") ? `目標 ${ss.target.slice(1)}R ≈ ${k ? num(p.entry + sg * Number(ss.target.slice(1)) * k * p.atr) : "n/a"}` : esc(b.target_names[ss.target]);
    alt = `<p class="${ss.oos.avg_r > 0 ? "" : "down"}"><b>回測較佳方案（${long ? "做多" : "做空"}）</b>：${k ? `止損 ${num(p.entry - sg * k * p.atr)}（${k} ATR）` : esc(b.stop_names[ss.stop])}；${tgt}；${ss.horizon} 日時間止損。
      五年回測樣本外每筆 ${num(ss.oos.avg_r)}R（現行方案 ${num(b.live.oos.avg_r)}R）${bl ? `；同規則隨機進場 ${num(bl.baseline_oos.avg_r)}R` : ""}。${ss.oos.avg_r > 0 ? "" : "做空形態整體仍然虧損，只作迴避／對沖參考。"}
      <a href="#/research">詳情見研究頁</a></p>`;
  }
  return `<div class="plan-basis"><div class="eyebrow">為什麼這樣設定</div>
    ${p.stop_basis ? `<p><b>止損</b>：${esc(p.stop_basis)}</p>` : ""}${p.target_basis ? `<p><b>目標</b>：${esc(p.target_basis)}</p>` : ""}
    <p class="note">依據：${esc(p.method || "結構位 + ATR 噪音下限 + 歷史阻力／支持")}。</p>${alt}</div>`;
}

function fundamentalsPanel(f) {
  if (!f.available) return `<div class="card"><h2>基本面</h2><p class="note">不可用 — ${esc(f.reason)}</p></div>`;
  const m = f.metrics;
  const kv = [["季度截至 / 公布", `${m.latest_quarter_end} / ${m.latest_filed}`], ["營收按年", pct(m.revenue_yoy, 0)], ["EPS 按年", pct(m.eps_yoy, 0)],
    ["毛利率（近四季）", pct(m.gross_margin_ttm, 1, false)], ["經營利潤率（近四季）", pct(m.operating_margin_ttm, 1, false)],
    ["經營利潤率：本季 vs 去年同季", `${pct(m.operating_margin_q, 1, false)} vs ${pct(m.operating_margin_q_year_ago, 1, false)}`],
    ["自由現金流（近四季）", money(m.fcf_ttm)], ["自由現金流利潤率", pct(m.fcf_margin_ttm, 1, false)], ["現金及短期投資", money(m.cash_and_st_investments)],
    ["總負債", money(m.total_debt)], ["淨負債", money(m.net_debt)], ["負債／股東權益", num(m.debt_to_equity, 2)], ["流動比率", num(m.current_ratio, 2)],
    ["利息覆蓋率", isNum(m.interest_coverage) ? num(m.interest_coverage, 1) + "×" : "n/a"], ["已發行股數", money(m.shares_outstanding).replace("$", "")],
    ["股數按年變化", pct(m.shares_yoy, 1)], ["市值", money(m.market_cap)], ["市盈率（近四季）", num(m.pe_ttm, 1)], ["市銷率（近四季）", num(m.ps_ttm, 1)],
    ["自由現金流收益率", pct(m.fcf_yield, 1, false)], ["EV / EBIT", num(m.ev_to_ebit, 1)]];
  return `<div class="card"><h2>基本面 <small>${esc(f.source)}</small></h2><div class="kv">${kv.map(([k, v]) => `<div><span>${k}</span><span class="num">${v}</span></div>`).join("")}</div>
    <h3>重點提示</h3><ul class="list">${f.flags.map((x) => `<li class="${x.kind === "risk" ? (x.severity === "high" ? "down" : "warn") : "up"}">${esc(x.text)}</li>`).join("") || "<li class=muted>沒有</li>"}</ul>
    <h3>季度數據</h3>${table("fq", [{ key: "end", label: "季度截至" }, { key: "filed", label: "公布" }, { key: "revenue", label: "營收", n: 1, render: (q) => money(q.revenue) },
      { key: "gross_profit", label: "毛利", n: 1, render: (q) => money(q.gross_profit) }, { key: "operating_income", label: "經營利潤", n: 1, render: (q) => money(q.operating_income) },
      { key: "net_income", label: "淨利潤", n: 1, render: (q) => money(q.net_income) }, { key: "eps_diluted", label: "EPS", n: 1, render: (q) => num(q.eps_diluted) }], f.quarters, { sortKey: "end" })}
    <p class="note">只使用分析日或之前已公布的數據（避免偷看未來）。業績指引：${esc(f.guidance || "")}</p></div>`;
}

function optionsPanel(o) {
  if (!o.available) return `<div class="card"><h2>期權</h2><p class="note">不可用 — ${esc(o.reason)}</p></div>`;
  const maxOI = Math.max(1, ...o.top_call_oi_strikes.map((x) => x.open_interest), ...o.top_put_oi_strikes.map((x) => x.open_interest));
  const oiRows = (arr, color) => arr.map((x) => `<div class="comp" style="grid-template-columns:70px 1fr 70px"><div class="mono">${num(x.strike, x.strike % 1 ? 1 : 0)}</div><div class="bar"><span style="width:${(x.open_interest / maxOI) * 100}%;background:${color}"></span></div><div class="n mono">${(x.open_interest / 1000).toFixed(1)}k</div></div>`).join("");
  return `<div class="card"><h2>期權 <small>${esc(o.source)}</small></h2><div class="kv">
    <div><span>平價隱含波動率（${o.atm_iv_expiry}）</span><span class="num">${pct(o.atm_iv, 1, false)}</span></div><div><span>次月平價隱含波動率</span><span class="num">${pct(o.atm_iv_next, 1, false)}</span></div>
    <div><span>隱含 / 實際波動率（HV20）</span><span class="num ${o.iv_hv_ratio > 1.4 ? "warn" : ""}">${num(o.iv_hv_ratio, 2)}</span></div><div><span>IV Rank</span><span class="num">${isNum(o.iv_rank) ? num(o.iv_rank, 0) : "n/a"}</span></div>
    <div><span>預期波幅 → ${o.expected_move_expiry}</span><span class="num">±${num(o.expected_move)}（${pct(o.expected_move_pct, 1, false)}）</span></div><div><span>最大痛點（近月）</span><span class="num">${num(o.max_pain_front)}</span></div>
    <div><span>認購 / 認沽成交量</span><span class="num">${(o.call_volume / 1000).toFixed(1)}k / ${(o.put_volume / 1000).toFixed(1)}k</span></div><div><span>認沽認購比：成交 · 未平倉</span><span class="num">${num(o.pcr_volume)} · ${num(o.pcr_oi)}</span></div>
    <div><span>認購 / 認沽未平倉</span><span class="num">${(o.call_oi / 1000).toFixed(1)}k / ${(o.put_oi / 1000).toFixed(1)}k</span></div><div><span>Gamma 曝險估算（推算）</span><span class="num">${money(o.gex_estimate)}</span></div></div>
    ${o.iv_rank_note ? `<p class="note">${esc(o.iv_rank_note)}</p>` : ""}
    <h3>未平倉最多的行權價 — 認購（綠）／認沽（紅）</h3>${oiRows(o.top_call_oi_strikes, "var(--up)")}${oiRows(o.top_put_oi_strikes, "var(--down)")}
    ${o.unusual.length ? `<h3>成交量超過上日未平倉 2 倍（推算為開倉）</h3>${table("unu", [{ key: "side", label: "方向", render: (u) => (u.side === "call" ? "認購" : "認沽") }, { key: "expiry", label: "到期日" }, { key: "strike", label: "行權價", n: 1 }, { key: "volume", label: "成交量", n: 1, render: (u) => u.volume.toLocaleString() }, { key: "open_interest", label: "未平倉", n: 1, render: (u) => u.open_interest.toLocaleString() }, { key: "premium", label: "權利金", n: 1, render: (u) => money(u.premium) }], o.unusual, { sortKey: "premium" })}` : ""}
    <ul class="list">${o.interpretation.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
    <p class="note">${o.limitations.map(esc).join(" ")} ${esc(o.gex_note)}</p></div>`;
}

function positioningPanel(p) {
  const ins = p.insider_90d || {}, inst = p.institutional || {}, sv = p.short_volume || {}, ox = p.off_exchange || {};
  return `<div class="card"><h2>持倉與股東 <small>延遲披露數據</small></h2><div class="kv">
    <div><span>做空佔流通股</span><span class="num">${pct(p.short_percent_float, 1, false)}</span></div><div><span>回補天數</span><span class="num">${num(p.days_to_cover, 1)}</span></div>
    <div><span>做空數據日期</span><span class="num">${esc(p.short_interest_date || "n/a")}（${p.short_interest_age_days ?? "?"} 日前）</span></div><div><span>做空股數較上期</span><span class="num">${pct(p.short_interest_change, 0)}</span></div>
    <div><span>流通股</span><span class="num">${money(p.float_shares).replace("$", "")}</span></div><div><span>已發行股數</span><span class="num">${money(p.shares_outstanding).replace("$", "")}</span></div>
    <div><span>機構持股</span><span class="num">${pct(p.held_pct_institutions, 0, false)}</span></div><div><span>內部人士持股</span><span class="num">${pct(p.held_pct_insiders, 1, false)}</span></div>
    <div><span>內部人士買入 / 賣出（90 日）</span><span class="num">${ins.available ? `${ins.n_buys} / ${ins.n_sales}` : "n/a"}</span></div><div><span>內部人士淨買賣金額</span><span class="num ${cls(ins.net_value)}">${money(ins.net_value)}</span></div>
    <div><span>賣空成交佔比</span><span class="num">${sv.available ? pct(sv.latest.ratio, 0, false) + "（10 日平均 " + pct(sv.avg_ratio, 0, false) + "）" : "n/a"}</span></div>
    <div><span>場外交易佔比</span><span class="num">${ox.available && isNum(ox.off_exchange_share) ? pct(ox.off_exchange_share, 0, false) + "（" + ox.week_start + " 當週）" : "n/a"}</span></div></div>
    <ul class="list">${(p.notes || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
    ${inst.available ? `<h3>主要機構持有人（13F）</h3>${table("inst", [{ key: "holder", label: "持有人" }, { key: "pct_held", label: "持股 %", n: 1, render: (r) => pct(r.pct_held, 2, false) }, { key: "pct_change", label: "變化", n: 1, render: (r) => `<span class="${cls(r.pct_change)}">${pct(r.pct_change, 1)}</span>` }, { key: "date_reported", label: "截至" }], inst.top_holders.slice(0, 6), { sortKey: "pct_held" })}<p class="note">${esc(inst.note)}</p>` : ""}
    <p class="note">大宗交易：${esc((p.block_trades || {}).reason || "n/a")} ${esc(sv.note || "")} ${esc(ox.note || "")}</p></div>`;
}

function catalystsPanel(cat) {
  const n = cat.next_earnings;
  return `<div class="card"><h2>催化劑</h2>
    ${n ? `<p><b>下次財報：</b><span class="mono">${n.date}</span> ${esc(tr(n.timing) || "（時間未確認）")} · 反應交易日 ${n.reaction_session} · <b class="${n.sessions_until <= 7 ? "warn" : ""}">${n.sessions_until} 個交易日後</b> <span class="chip">${tr(n.status)}</span></p>` : '<p class="note">未有公布下次財報日期。</p>'}
    ${cat.earnings_history && cat.earnings_history.length ? `<p class="note">最近 ${cat.earnings_history.length} 次財報 EPS 超預期幅度：${cat.earnings_history.map((h) => `${h.date} ${isNum(h.surprise_pct) ? (h.surprise_pct > 0 ? "+" : "") + h.surprise_pct.toFixed(1) + "%" : "n/a"}`).join(" · ")}</p>` : ""}
    ${table("cev", [{ key: "event_date", label: "日期" }, { key: "event_type", label: "事件", wrap: 1, render: (e) => e.url ? `<a href="${esc(e.url)}" target="_blank" rel="noopener">${esc(e.description || e.event_type)}</a>` : esc(e.description || e.event_type) },
      { key: "sentiment", label: "通常影響", render: (e) => `<span class="${e.sentiment === "Positive" ? "up" : e.sentiment === "Negative" ? "down" : "muted"}">${tr(e.sentiment)}</span>` }, { key: "status", label: "狀態", render: (e) => tr(e.status) }, { key: "source", label: "來源" }], cat.events || [], { sortKey: "event_date" })}
    <p class="note">${(cat.limitations || []).map(esc).join(" ")}</p></div>`;
}

// ------------------------------------------------------------------ views: universe
function universe() {
  const s = state.snap, q = state.filters.q.toUpperCase();
  const rows = s.universe.filter((u) => !q || u.symbol.includes(q));
  const uf = s.universe_filter || {};
  app.innerHTML = `<div class="wall-head"><div><div class="eyebrow">高 Beta／高波動篩選 / <b>保留 ${s.universe.length} 檔</b></div>
    <h1 class="display">股票池 <em>universe</em></h1>
    <p>只保留 AI、半導體、高增長和高波動股票。每日自動篩選：120 日 Beta ≥ ${uf.rules ? uf.rules.min_beta : "1.3"}、
    60 日波動率 ≥ ${uf.rules ? pct(uf.rules.min_hv60, 0, false) : "35%"}、市值 ≥ ${uf.rules ? money(uf.rules.min_market_cap) : "$2B"}（槓桿 ETF 豁免）、
    日均成交額 ≥ ${uf.rules ? money(uf.rules.min_avg_dollar_volume) : "$100M"}。被排除的股票見「數據」頁。</p></div></div>
    <div class="card"><div class="filters"><label>搜尋 <input id="u-q" value="${esc(state.filters.q)}" style="width:90px"></label><span class="note">點擊任何一行查看圖表</span></div>
    ${table("uni", [
      { key: "symbol", label: "代號", render: (u) => `<b>${u.symbol}</b>${u.leveraged ? ' <span class="chip">槓桿 ETF</span>' : ""}` }, { key: "sector_etf", label: "板塊" },
      { key: "close", label: "收市", n: 1, render: (u) => num(u.close) },
      { key: "ret_1d", label: "1 日", n: 1, render: (u) => `<span class="${cls(u.ret_1d)}">${pct(u.ret_1d)}</span>` },
      { key: "ret_5d", label: "5 日", n: 1, render: (u) => `<span class="${cls(u.ret_5d)}">${pct(u.ret_5d)}</span>` },
      { key: "ret_20d", label: "20 日", n: 1, render: (u) => `<span class="${cls(u.ret_20d)}">${pct(u.ret_20d)}</span>` },
      { key: "beta", label: "Beta", n: 1, render: (u) => num(u.beta, 2) },
      { key: "hv60", label: "60 日波動", n: 1, render: (u) => pct(u.hv60, 0, false) },
      { key: "market_cap", label: "市值", n: 1, render: (u) => money(u.market_cap) },
      { key: "rel_20d", label: "相對 20 日", n: 1, render: (u) => `<span class="${cls(u.rel_20d)}">${pct(u.rel_20d)}</span>` },
      { key: "rs_pct", label: "RS 百分位", n: 1, render: (u) => num(u.rs_pct, 0) }, { key: "rsi14", label: "RSI14", n: 1, render: (u) => num(u.rsi14, 0) },
      { key: "volume_ratio", label: "量比", n: 1, render: (u) => num(u.volume_ratio, 2) }, { key: "extension_atr", label: "偏離 ATR", n: 1, render: (u) => num(u.extension_atr, 1) },
      { key: "trend", label: "趨勢", render: (u) => esc(tr(u.trend)) }, { key: "structure", label: "結構", render: (u) => esc(tr(u.structure)) },
      { key: "setups", label: "形態", render: (u) => u.setups.map((x) => `<span class="chip ${s.setup_directions[x]}">${esc(s.setup_names[x])}</span>`).join("") || '<span class="muted">—</span>', sort: (u) => u.setups.length },
      { key: "dollar_volume20", label: "日均成交額", n: 1, render: (u) => money(u.dollar_volume20) },
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
  app.innerHTML = `<div class="wall-head"><div><div class="eyebrow">財報 · SEC 文件 · 分析師 · 宏觀</div><h1 class="display">催化劑 <em>catalysts</em></h1></div></div>
  <div class="grid g2">
    <div class="card"><h2>即將公布財報 <small>入選股票</small></h2>${table("ue", [
      { key: "symbol", label: "代號", render: (e) => `<b>${e.symbol}</b>` }, { key: "date", label: "日期" }, { key: "timing", label: "時間", render: (e) => esc(tr(e.timing) || "未確認") },
      { key: "reaction_session", label: "反應交易日" }, { key: "sessions_until", label: "相距交易日", n: 1 }, { key: "eps_estimate", label: "預估 EPS", n: 1, render: (e) => num(e.eps_estimate) },
      { key: "status", label: "狀態", render: (e) => tr(e.status) }],
      s.upcoming_earnings || [], { sortKey: "sessions_until", sortDir: "asc", rowAttr: (e) => `data-sym="${e.symbol}"`, onClick: true })}</div>
    <div class="card"><h2>宏觀及市場結構日曆 <small>未來 30 日</small></h2>${table("mac", [
      { key: "event_date", label: "日期" }, { key: "event_type", label: "事件", wrap: 1 }, { key: "status", label: "狀態", render: (e) => tr(e.status) }, { key: "source", label: "來源", wrap: 1 }],
      s.macro_events || [], { sortKey: "event_date", sortDir: "asc" })}
      <p class="note">休市日：${(s.holidays || []).map((h) => `${h.date} ${esc(tr(h.name))}`).join(" · ") || "未來 45 日沒有"}。CPI、PCE 等日期可在 config/macro_events.yaml 加入。</p></div>
  </div>
  <div class="card" style="margin-top:14px"><h2>SEC 文件 <small>入選股票，最近 60 日</small></h2>${table("fil", [
    { key: "symbol", label: "代號", render: (f) => `<b>${f.symbol}</b>` }, { key: "event_date", label: "提交日" }, { key: "published_at", label: "接收時間（UTC）" }, { key: "form", label: "表格" },
    { key: "event_type", label: "類型", wrap: 1, render: (f) => f.url ? `<a href="${esc(f.url)}" target="_blank" rel="noopener">${esc(f.event_type)}</a>` : esc(f.event_type) },
    { key: "sentiment", label: "通常影響", render: (f) => sent(f.sentiment) }, { key: "dilution_risk", label: "稀釋", render: (f) => f.dilution_risk ? '<span class="down">是</span>' : "" }],
    filings, { sortKey: "event_date" })}${filings.length ? "" : '<p class="note">未載入 SEC 文件：請在 .env 設定 SEC_USER_AGENT（見頁頂提示）以啟用 SEC EDGAR。</p>'}</div>
  <div class="card" style="margin-top:14px"><h2>分析師評級 <small>最近 30 日</small></h2>${table("ana", [
    { key: "symbol", label: "代號", render: (a) => `<b>${a.symbol}</b>` }, { key: "date", label: "日期" }, { key: "description", label: "評級變動", wrap: 1 }, { key: "sentiment", label: "影響", render: (a) => sent(a.sentiment) }],
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
    <div class="card"><h2>跨資產 <small>美元 · 原油 · 黃金 · 比特幣（只作背景，不計入市場狀態）</small></h2>${table("mac", [
      { key: "name", label: "資產", render: (x) => `<b>${esc(x.name)}</b> <span class="muted mono">${esc(x.symbol)}</span>` },
      { key: "spark", label: "60 日", render: (x) => sparkLine(x.spark) },
      { key: "close", label: "最新", n: 1, render: (x) => (x.available ? num(x.close, x.close > 1000 ? 0 : 2) : '<span class="muted">沒有數據</span>') },
      { key: "ret_5d", label: "5 日", n: 1, render: (x) => `<span class="${cls(x.ret_5d)}">${pct(x.ret_5d)}</span>` },
      { key: "ret_20d", label: "20 日", n: 1, render: (x) => `<span class="${cls(x.ret_20d)}">${pct(x.ret_20d)}</span>` },
      { key: "z20", label: "20 日 z", n: 1, render: (x) => `<span class="${isNum(x.z20) && Math.abs(x.z20) >= 1.5 ? "warn" : ""}">${num(x.z20, 1)}</span>` },
      { key: "corr_qqq_60d", label: "與 QQQ 相關", n: 1, render: (x) => num(x.corr_qqq_60d, 2) }], m)}
      <ul class="list">${m.filter((x) => x.comment).map((x) => `<li><b>${esc(x.name)}</b>：${esc(x.comment)}</li>`).join("")}</ul></div>
    <div class="card"><h2>上漲／下跌家數 <small>${esc(ad.scope || "")}</small></h2>
      ${ad.available ? `<div class="grid g4 kpis"><div class="kpi"><span class="l">上漲</span><span class="v up">${ad.advancers}</span></div>
        <div class="kpi"><span class="l">下跌</span><span class="v down">${ad.decliners}</span></div>
        <div class="kpi"><span class="l">McClellan</span><span class="v ${cls(ad.mcclellan)}">${num(ad.mcclellan, 0)}</span></div>
        <div class="kpi"><span class="l">上漲成交量佔比</span><span class="v">${pct(ad.up_volume_share, 0, false)}</span></div></div>
        <div class="lg">A/D 線（累計上漲減下跌）與 McClellan 振盪指標 · 10 日淨上漲 ${ad.net_10d > 0 ? "+" : ""}${ad.net_10d}</div>
        <div id="ad-chart" class="chart" style="height:150px"></div><div id="mcc-chart" class="chart small"></div>
        <p class="note">${esc(ad.comment || "")} 全市場（NYSE）的上漲下跌家數沒有穩定的免費來源，這裏用本股票池計算，代表高 Beta 股的參與面。</p>` : '<p class="note">沒有足夠數據。</p>'}</div>
  </div>
  ${groups.length ? `<div class="card" style="margin-top:12px"><h2>行業組相對強弱排名 <small>${groups.length} 組 · 按成員 RS 中位數排名，箭咀為 5 日前名次變化</small></h2>${table("grp", [
    { key: "rank", label: "#", n: 1 },
    { key: "group", label: "行業組", render: (g) => `<b>${esc(g.group)}</b>` },
    { key: "rank_change", label: "5 日變化", n: 1, render: (g) => (g.rank_change ? `<span class="${cls(g.rank_change)}">${g.rank_change > 0 ? "▲" : "▼"}${Math.abs(g.rank_change)}</span>` : '<span class="muted">—</span>') },
    { key: "n", label: "成員", n: 1 },
    { key: "pct", label: "RS 百分位", n: 1, render: (g) => num(g.pct, 0) },
    { key: "ret_5d", label: "5 日（中位）", n: 1, render: (g) => `<span class="${cls(g.ret_5d)}">${pct(g.ret_5d)}</span>` },
    { key: "ret_20d", label: "20 日（中位）", n: 1, render: (g) => `<span class="${cls(g.ret_20d)}">${pct(g.ret_20d)}</span>` },
    { key: "breadth", label: "高於 EMA50", n: 1, render: (g) => pct(g.breadth, 0, false) },
    { key: "leaders", label: "最強成員", wrap: 1, render: (g) => g.leaders.map((x) => `<a href="#/ticker/${x}">${x}</a>`).join(" ") }],
    groups, { sortKey: "rank", sortDir: "asc" })}<p class="note">行業來自 Yahoo 公司資料（每 120 日更新）；只有一隻成員的行業不列出。RS 分數 = 0.5 × 相對 SPY 20 日 + 0.3 × 60 日 + 0.2 × 5 日。強勢行業組內的領頭股，較容易延續升勢。</p></div>` : ""}`;
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
  if (!e.available) return `<div class="card"><h2>財報反應</h2><p class="note">${esc(e.reason || "沒有數據。")}</p></div>`;
  const imp = e.implied;
  return `<div class="card"><h2>財報反應 <small>過去 ${e.n} 次 · 期權隱含 vs 歷史</small></h2>
    <div class="grid g4 kpis">
      <div class="kpi"><span class="l">期權隱含</span><span class="v">${imp ? "±" + pct(imp.move_pct, 1, false) : "—"}</span><span class="note">${imp ? `${imp.expiry} 跨式 ${num(imp.straddle)}` : "沒有覆蓋財報的到期日"}</span></div>
      <div class="kpi"><span class="l">歷史波幅中位</span><span class="v">±${pct(e.median_abs_move, 1, false)}</span><span class="note">平均 ±${pct(e.mean_abs_move, 1, false)}</span></div>
      <div class="kpi"><span class="l">隱含 ÷ 歷史</span><span class="v ${isNum(e.implied_vs_history) ? (e.implied_vs_history > 1.15 ? "warn" : e.implied_vs_history < 0.85 ? "up" : "") : "muted"}">${isNum(e.implied_vs_history) ? num(e.implied_vs_history, 2) + "×" : "—"}</span><span class="note">&gt; 1 期權較貴</span></div>
      <div class="kpi"><span class="l">上升比例</span><span class="v">${pct(e.up_share, 0, false)}</span><span class="note">最大 ${pct(e.max_up, 0)} ／ ${pct(e.max_down, 0)}</span></div></div>
    <p>${esc(e.note || "")}</p>${e.drift_note ? `<p>${esc(e.drift_note)}</p>` : ""}
    <p class="note">跳空後收市回吐的比例 ${pct(e.gap_fade_share, 0, false)}；財報後 10 日延續首日方向 ${pct(e.drift10_same_dir, 0, false)}（平均 ${pct(e.drift10_signed_mean, 1)}）。</p>
    ${table("earn", [{ key: "date", label: "公布" }, { key: "timing", label: "時間", render: (x) => ({ "before open": "開市前", "after close": "收市後", estimated: "推算" }[x.timing] || x.timing) },
      { key: "surprise_pct", label: "EPS 驚喜", n: 1, render: (x) => (isNum(x.surprise_pct) ? `<span class="${cls(x.surprise_pct)}">${num(x.surprise_pct, 1)}%</span>` : "—") },
      { key: "gap", label: "跳空", n: 1, render: (x) => `<span class="${cls(x.gap)}">${pct(x.gap)}</span>` },
      { key: "move", label: "當日", n: 1, render: (x) => `<span class="${cls(x.move)}">${pct(x.move)}</span>` },
      { key: "drift5", label: "其後 5 日", n: 1, render: (x) => `<span class="${cls(x.drift5)}">${pct(x.drift5)}</span>` },
      { key: "drift10", label: "其後 10 日", n: 1, render: (x) => `<span class="${cls(x.drift10)}">${pct(x.drift10)}</span>` }], e.events, { sortKey: "date" })}
    <p class="note">樣本只有 ${e.n} 次，屬描述性統計。「推算」= 數據商沒有提供公布時間，以跳空較大的一日作反應日。</p></div>`;
}

function ftdPanel(f) {
  if (!f) return "";
  if (!f.available) return `<div class="card"><h2>SEC 交收失敗（FTD）</h2><p class="note">${esc(f.reason)}</p></div>`;
  return `<div class="card"><h2>SEC 交收失敗（FTD） <small>${esc(f.period)}</small></h2>
    ${f.n ? `<div class="grid g4 kpis"><div class="kpi"><span class="l">最高</span><span class="v">${fmtVol(f.max)}</span><span class="note">${f.max_date}</span></div>
      <div class="kpi"><span class="l">最高 ÷ 日均量</span><span class="v ${f.max_vs_adv >= 0.1 ? "warn" : ""}">${pct(f.max_vs_adv, 1, false)}</span></div>
      <div class="kpi"><span class="l">最新</span><span class="v">${fmtVol(f.latest)}</span><span class="note">${f.latest_date}</span></div>
      <div class="kpi"><span class="l">紀錄日數</span><span class="v">${f.n}</span><span class="note">≥ 0.5% 日均量：${f.sessions_over_half_pct_adv ?? "—"}</span></div></div>
      <div class="ftd-bars">${(() => { const mx = Math.max(...f.series.map((x) => x.fails)); return f.series.map((x) => `<i style="height:${Math.max(2, x.fails / mx * 100)}%" title="${x.date}：${x.fails.toLocaleString()} 股"></i>`).join(""); })()}</div>` : ""}
    <p>${esc(f.note)}</p><p class="note">數據為 SEC 每半個月公布一次，通常滯後兩至四星期；數字是當日累計未交收的股數，不是新增。ETF 申購贖回時間差也會產生 FTD，單獨看意義有限。</p></div>`;
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
  if (!pf.labels) return '<p class="note">候選不足兩個，沒有相關矩陣。</p>';
  const idx = pf.labels.map((l, i) => [l, i]).filter(([l]) => !picked || picked.has(l.replace(" (空)", "")));
  const bg = (v) => `background:color-mix(in srgb, var(${v >= 0 ? "--down" : "--up"}) ${Math.round(Math.min(1, Math.abs(v)) * 70)}%, transparent)`;
  return `<div class="tw"><table class="heat corr"><thead><tr><th></th>${idx.map(([l]) => `<th class="n">${esc(l)}</th>`).join("")}</tr></thead><tbody>
    ${idx.map(([l, i]) => `<tr><th>${esc(l)}</th>${idx.map(([, j]) => `<td class="n" style="${i === j ? "" : bg(pf.correlation[i][j])}">${i === j ? "—" : num(pf.correlation[i][j], 2)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <p class="note">倉位相關 = 60 日日回報相關係數，做空倉位已反轉方向：紅色越深 = 越會一起賺一起蝕（風險疊加），綠色 = 互相對沖。</p>`;
}

async function portfolio() {
  const s = state.snap, pf = s.portfolio || {};
  const v = pfSettings();
  const draw = () => {
    const rows = pfSized(v);
    const picked = new Set(v.picked || rows.map((r) => r.symbol));
    const sel = rows.filter((r) => picked.has(r.symbol));
    const heat = sel.reduce((a, r) => a + r.dollar_risk, 0), heatPct = heat / v.account * 100;
    const lab = sel.map((r) => r.symbol + (r.direction === "long" ? "" : " (空)"));
    const ix = lab.map((l) => (pf.labels || []).indexOf(l)).filter((i) => i >= 0);
    let sum = 0; ix.forEach((i) => ix.forEach((j) => { sum += pf.correlation[i][j]; }));
    const eff = ix.length > 1 && sum > 0 ? ix.length * ix.length / sum : null;
    $("#pf-body").innerHTML = `<div class="grid g4 kpis">
        <div class="kpi"><span class="l">已選倉位</span><span class="v">${sel.length}</span><span class="note">做多 ${sel.filter((r) => r.direction === "long").length} ／ 做空 ${sel.filter((r) => r.direction === "short").length}</span></div>
        <div class="kpi"><span class="l">總風險（全部止損）</span><span class="v ${heatPct > v.maxHeat ? "down" : "up"}">${num(heatPct, 2)}%</span><span class="note">$${Math.round(heat).toLocaleString()} ／ 上限 ${num(v.maxHeat, 1)}%</span></div>
        <div class="kpi"><span class="l">獨立注碼（估算）</span><span class="v">${eff ? num(eff, 1) : "—"}</span><span class="note">n² ÷ Σ 相關係數</span></div>
        <div class="kpi"><span class="l">總持倉 ／ 淨持倉</span><span class="v">${pct(sel.reduce((a, r) => a + r.value, 0) / v.account, 0, false)}</span><span class="note">淨 ${pct(sel.reduce((a, r) => a + r.value * (r.direction === "long" ? 1 : -1), 0) / v.account, 0)}</span></div></div>
      <div class="heatbar"><i style="width:${Math.min(100, heatPct / Math.max(v.maxHeat, 0.01) * 100)}%" class="${heatPct > v.maxHeat ? "over" : ""}"></i></div>
      ${table("pf", [{ key: "pick", label: "選", render: (r) => `<input type="checkbox" data-pick="${r.symbol}" ${picked.has(r.symbol) ? "checked" : ""}>` },
        { key: "symbol", label: "代號", render: (r) => `<a href="#/ticker/${r.symbol}"><b>${r.symbol}</b></a> <span class="chip ${r.direction}">${r.direction === "long" ? "多" : "空"}</span>` },
        { key: "score", label: "分數", n: 1, render: (r) => num(r.score, 0) }, { key: "entry", label: "進場", n: 1, render: (r) => num(r.entry) },
        { key: "stop", label: "止損", n: 1, render: (r) => `<span class="down">${num(r.stop)}</span>` },
        { key: "per", label: "每股風險", n: 1, sort: (r) => Math.abs(r.entry - r.stop), render: (r) => num(Math.abs(r.entry - r.stop)) },
        { key: "shares", label: "股數", n: 1, render: (r) => `${r.shares.toLocaleString()}${r.capped ? ' <span class="warn" title="已按單一倉位上限收細">⚑</span>' : ""}` },
        { key: "value", label: "倉位金額", n: 1, render: (r) => "$" + Math.round(r.value).toLocaleString() },
        { key: "value_pct", label: "佔帳戶", n: 1, render: (r) => pct(r.value_pct, 1, false) },
        { key: "dollar_risk", label: "風險金額", n: 1, render: (r) => "$" + Math.round(r.dollar_risk).toLocaleString() }], rows, { sortKey: "score" })}
      <h3>倉位相關矩陣（已選）</h3>${corrHeat(pf, picked)}
      <ul class="list">${(pf.clusters || []).map((c) => `<li class="warn">高度相關（≥ 0.7），會一起觸發止損：${c.map(esc).join("、")}</li>`).join("")}</ul>`;
    app.querySelectorAll("[data-pick]").forEach((cb) => cb.addEventListener("change", () => {
      const cur = new Set(v.picked || rows.map((r) => r.symbol));
      cb.checked ? cur.add(cb.dataset.pick) : cur.delete(cb.dataset.pick);
      v.picked = [...cur]; pfSave(v); draw();
    }));
  };
  app.innerHTML = `<div class="wall-head"><div><div class="eyebrow">倉位 · 相關性 · 總風險 · 交易日誌 · 價格提醒</div><h1 class="display">組合 <em>portfolio</em></h1>
    <p>把今日的候選當作一個組合來看：每個倉位按止損距離計股數（固定風險比例），再看它們會不會一起輸。系統不會下單，所有交易紀錄由你自行輸入。</p></div></div>
    <section class="note-block"><div class="pf-inputs">
      <label>帳戶金額（US$）<input id="pf-acc" type="number" min="1000" step="1000" value="${v.account}"></label>
      <label>每筆風險（%）<input id="pf-risk" type="number" min="0.05" max="5" step="0.05" value="${v.risk}"></label>
      <label>單一倉位上限（%）<input id="pf-max" type="number" min="1" max="100" step="1" value="${v.maxPos}"></label>
      <label>總風險上限（%）<input id="pf-heat" type="number" min="0.5" max="20" step="0.5" value="${v.maxHeat}"></label>
      <span class="note">股數 = 帳戶 × 每筆風險 × 市況倉位比例（做多 ×${(pf.budgets || {}).long ?? 1}，做空 ×${(pf.budgets || {}).short ?? 1}）÷ |進場 − 止損|。設定只儲存在這個瀏覽器。</span></div>
      <div id="pf-body"></div></section>
    <section class="note-block"><div class="note-head"><div><div class="eyebrow">價格提醒</div><div class="hl">到價提醒</div></div></div>
      <form id="al-form" class="pf-inputs"><label>代號<input name="symbol" required placeholder="NVDA"></label>
        <label>條件<select name="kind"><option value="above">盤中升穿</option><option value="below">盤中跌穿</option><option value="close_above">收市高於</option><option value="close_below">收市低於</option></select></label>
        <label>價位<input name="level" type="number" step="0.01" required></label><label class="wide">備註<input name="note" placeholder="例如：突破後加注"></label>
        <button type="submit">加入提醒</button></form>
      <div id="al-list" class="note">載入中…</div>
      <p class="note">提醒在每次每日分析後及打開網頁時，用已儲存的日線檢查（盤中＝當日最高／最低價曾經碰到）。不是即時報價；在 Mac 上觸發時會彈出系統通知。</p></section>
    <section class="note-block"><div class="note-head"><div><div class="eyebrow">交易日誌 · 檢討</div><div class="hl">交易日誌</div></div></div>
      <form id="jn-form" class="pf-inputs"><label>代號<input name="symbol" required></label>
        <label>方向<select name="direction"><option value="long">做多</option><option value="short">做空</option></select></label>
        <label>形態<select name="setup"><option value="">—</option>${Object.entries(s.setup_names).map(([k, n]) => `<option value="${k}">${esc(n)}</option>`).join("")}</select></label>
        <label>進場日<input name="entry_date" type="date"></label><label>進場價<input name="entry_price" type="number" step="0.01" required></label>
        <label>初始止損<input name="stop" type="number" step="0.01" required></label><label>目標<input name="target" type="number" step="0.01"></label>
        <label>股數<input name="shares" type="number" step="1"></label><label>出場日<input name="exit_date" type="date"></label><label>出場價<input name="exit_price" type="number" step="0.01"></label>
        <label class="wide">備註<input name="notes" placeholder="為什麼進場／出場，有沒有跟計劃"></label><button type="submit">記錄交易</button></form>
      <div id="jn-list" class="note">載入中…</div></section>`;
  const bind = (id, key) => $(id).addEventListener("input", (e) => { const x = Number(e.target.value); if (isNum(x) && x > 0) { v[key] = x; pfSave(v); draw(); } });
  bind("#pf-acc", "account"); bind("#pf-risk", "risk"); bind("#pf-max", "maxPos"); bind("#pf-heat", "maxHeat");
  draw();
  const formData = (f) => Object.fromEntries([...new FormData(f).entries()].filter(([, x]) => x !== ""));
  const ro = !!window.__PREVIEW__;
  const loadAlerts = async () => {
    try {
      const a = await api("/api/alerts");
      $("#al-list").innerHTML = table("al", [{ key: "symbol", label: "代號", render: (x) => `<a href="#/ticker/${x.symbol}"><b>${x.symbol}</b></a>` },
        { key: "kind_name", label: "條件" }, { key: "level", label: "價位", n: 1, render: (x) => num(x.level) },
        { key: "last", label: "最新收市", n: 1, render: (x) => num(x.last) }, { key: "distance", label: "距離", n: 1, render: (x) => pct(x.distance) },
        { key: "active", label: "狀態", render: (x) => (x.active ? '<span class="muted">等待中</span>' : `<span class="warn">已觸發 ${x.triggered_session}（${num(x.triggered_price)}）</span>`) },
        { key: "note", label: "備註", wrap: 1 }, { key: "act", label: "", render: (x) => `${x.active ? "" : `<button data-rearm="${x.id}">重設</button>`} <button data-del-al="${x.id}">刪除</button>` }],
        a.alerts, { sortKey: "active" });
      app.querySelectorAll("[data-del-al]").forEach((b) => b.addEventListener("click", async () => { await api(`/api/alerts/${b.dataset.delAl}`, { method: "DELETE" }); loadAlerts(); }));
      app.querySelectorAll("[data-rearm]").forEach((b) => b.addEventListener("click", async () => { await api(`/api/alerts/${b.dataset.rearm}/rearm`, { method: "POST" }); loadAlerts(); }));
    } catch (err) { $("#al-list").textContent = err.message; }
  };
  const loadJournal = async () => {
    try {
      const j = await api("/api/journal"), st = j.stats || {};
      $("#jn-list").innerHTML = `<div class="grid g4 kpis"><div class="kpi"><span class="l">已平倉</span><span class="v">${st.n_closed || 0}</span><span class="note">持倉中 ${st.n_open || 0}</span></div>
        <div class="kpi"><span class="l">勝率</span><span class="v">${pct(st.win_rate, 0, false)}</span><span class="note">盈虧比 ${num(st.profit_factor)}</span></div>
        <div class="kpi"><span class="l">平均 R</span><span class="v ${cls(st.avg_r)}">${num(st.avg_r)}</span><span class="note">平均贏 ${num(st.avg_win_r)}R ／ 輸 ${num(st.avg_loss_r)}R</span></div>
        <div class="kpi"><span class="l">累計盈虧</span><span class="v ${cls(st.total_pnl)}">${isNum(st.total_pnl) ? "$" + Math.round(st.total_pnl).toLocaleString() : "—"}</span><span class="note">累計 ${num(st.total_r, 1)}R</span></div></div>
        ${(st.by_setup || []).length ? `<h3>按形態</h3>${table("jn-set", [{ key: "setup", label: "形態", render: (x) => esc(s.setup_names[x.setup] || x.setup) }, { key: "n", label: "筆數", n: 1 },
          { key: "win_rate", label: "勝率", n: 1, render: (x) => pct(x.win_rate, 0, false) }, { key: "avg_r", label: "平均 R", n: 1, render: (x) => `<span class="${cls(x.avg_r)}">${num(x.avg_r)}</span>` }], st.by_setup)}` : ""}
        ${table("jn", [{ key: "symbol", label: "代號", render: (t) => `<a href="#/ticker/${t.symbol}"><b>${t.symbol}</b></a> <span class="chip ${t.direction}">${t.direction === "long" ? "多" : "空"}</span>` },
          { key: "setup", label: "形態", render: (t) => esc(s.setup_names[t.setup] || t.setup || "") }, { key: "entry_date", label: "進場日" },
          { key: "entry_price", label: "進場", n: 1, render: (t) => num(t.entry_price) }, { key: "stop", label: "止損", n: 1, render: (t) => num(t.stop) },
          { key: "exit_price", label: "出場／現價", n: 1, render: (t) => (t.status === "closed" ? num(t.exit_price) : `<span class="muted">${num(t.mark)}</span>`) },
          { key: "r", label: "R", n: 1, sort: (t) => t.r ?? t.r_open, render: (t) => (t.status === "closed" ? `<span class="${cls(t.r)}">${num(t.r)}</span>` : `<span class="muted">${num(t.r_open)}（未平）</span>`) },
          { key: "pnl", label: "盈虧", n: 1, sort: (t) => t.pnl ?? t.pnl_open, render: (t) => { const x = t.status === "closed" ? t.pnl : t.pnl_open; return isNum(x) ? `<span class="${cls(x)}">$${Math.round(x).toLocaleString()}</span>` : "—"; } },
          { key: "flag", label: "", render: (t) => (t.flag ? `<span class="warn">${esc(t.flag)}</span>` : "") },
          { key: "notes", label: "備註", wrap: 1 },
          { key: "act", label: "", render: (t) => `${t.status !== "closed" ? `<button data-close="${t.id}">平倉</button>` : ""} <button data-del-jn="${t.id}">刪除</button>` }], j.trades, { sortKey: "entry_date" })}
        <p class="note">R = 方向 × (出場 − 進場) ÷ |進場 − 初始止損|。持倉中的交易按最新收市價估算。</p>`;
      app.querySelectorAll("[data-del-jn]").forEach((b) => b.addEventListener("click", async () => { if (confirm("刪除這筆紀錄？")) { await api(`/api/journal/${b.dataset.delJn}`, { method: "DELETE" }); loadJournal(); } }));
      app.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", async () => {
        const px = Number(prompt("出場價？")); if (!isNum(px) || px <= 0) return;
        await api(`/api/journal/${b.dataset.close}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ exit_price: px, exit_date: new Date().toISOString().slice(0, 10), status: "closed" }) });
        loadJournal();
      }));
    } catch (err) { $("#jn-list").textContent = err.message; }
  };
  const post = (url, f, after) => f.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (ro) { alert("預覽版不能儲存。"); return; }
    try { await api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(formData(f)) }); f.reset(); after(); }
    catch (err) { alert(err.message); }
  });
  post("/api/alerts", $("#al-form"), loadAlerts);
  post("/api/journal", $("#jn-form"), loadJournal);
  loadAlerts(); loadJournal();
}

async function quickAlert(symbol, kind, level, note) {
  if (window.__PREVIEW__) { alert("預覽版不能儲存提醒。"); return; }
  try {
    await api("/api/alerts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ symbol, kind, level, note }) });
    alert(`已加入提醒：${symbol} ${note}（${num(level)}）。可在「組合」頁管理。`);
  } catch (err) { alert(err.message); }
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-qa]");
  if (b) quickAlert(b.dataset.sym, b.dataset.kind, Number(b.dataset.level), b.dataset.qa);
});

// ------------------------------------------------------------------ SL / TP validation backtest (python backtest.py)
const BT_STOP_ORDER = ["structural", "atr1.0", "atr1.5", "atr2.0", "atr2.5", "atr3.0", "swing", "chip"];
const BT_TGT_ORDER = ["resistance", "atr2", "atr3", "atr4", "r2", "r3", "chip", "trail"];
const REGIMES = ["Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish"];
state.bt = { side: "long", hz: 10, period: "oos" };
const rv = (v, d = 2) => (isNum(v) ? `<span class="${cls(v)}">${v > 0 ? "+" : ""}${num(v, d)}R</span>` : '<span class="muted">—</span>');
const tval = (v) => (isNum(v) ? `<span class="${Math.abs(v) >= 2 ? (v > 0 ? "up" : "down") : "muted"}">${num(v, 1)}</span>` : "—");

function slTpExplainer(b) {
  const live = b && b.available ? b.live : null, d = (b && b.diagnostics) || {};
  const mae = d.winner_mae_atr || {}, mfe = d.mfe_r || {}, rr = d.rr_calibration || [];
  const edge = !!live && live.oos.avg_r > 0.05 && live.oos.t_stat > 2, tight = (d.stopped_then_target || 0) > 0.2;
  const below = rr.filter((r) => r.target_rate < r.breakeven).length;
  const verdictLine = (txt, ok) => `<li><b class="tag ${ok ? "up" : "down"}">${ok ? "合格" : "問題"}</b>${txt}</li>`;
  return `<section class="note-block bt-explain"><div class="note-head"><div><div class="eyebrow">止損 · 目標 · 時間止損</div>
      <div class="hl">SL／TP 是怎樣定的？用了什麼，沒有用什麼</div></div></div>
    <div class="grid g3 ruled">
      <div><h3>止損 SL</h3><p>按<b>價格結構</b>定：每個形態都有一個「前提失效」的價位，例如突破形態用突破位（前 20 日高位）和突破當日低位兩者較低者，
        回調形態用近 5 日低位和 EMA21 兩者較低者，再留 <b>0.25 ATR</b> 緩衝。</p>
        <p>然後用 <b>ATR（14 日平均真實波幅）</b>限制距離：少於 1 ATR 會放寬到 1 ATR（避免被日常雜訊掃走），多於 3 ATR 會收窄到 3 ATR 並標示「止損過寬」。</p></div>
      <div><h3>目標 TP</h3><p><b>目標一</b> = 上方最近、而且至少 0.75 ATR 以外的歷史高位（20 日、60 日、52 週高位，即前期阻力）；如果股價已在新高、上方沒有阻力，就用進場價 + 2 ATR。</p>
        <p><b>目標二</b> = 目標一 + 1 ATR 與 2 倍風險兩者較遠者。做空形態全部反過來（用歷史低位作支持）。</p></div>
      <div><h3>時間止損</h3><p>持倉 <b>7 個交易日</b>後，未到止損或目標就在收市出場：短線形態的動力通常在一至兩星期內消退。</p>
        <p class="note">籌碼（成交量分佈）<b>沒有</b>用在現行 SL／TP，但回測把「籌碼密集區／價值區下沿止損」和「上方籌碼區／真空區頂部目標」當作替代方案測試（見下表）。
        期權<b>沒有</b>用：沒有免費的歷史引伸波幅數據可以回測，期權預期波幅只在個股頁作參考。</p></div>
    </div>
    ${live ? `<h3>回測怎樣說這套 SL／TP？</h3><ul class="list verdict">
      ${verdictLine(`整體 ${live.all.n.toLocaleString()} 筆：勝率 ${pct(live.all.win_rate, 0, false)}，平均 ${rv(live.all.avg_r)}；樣本外 ${rv(live.oos.avg_r)}（t ${tval(live.oos.t_stat)}）——${edge ? "<b>有統計上顯著的優勢</b>" : "<b>沒有可靠優勢</b>"}。`, edge)}
      ${verdictLine(`止損${tight ? "可能太緊" : "不算太緊"}：被止損後仍然到達目標的有 ${pct(d.stopped_then_target, 0, false)}；最終達標的交易途中最大浮虧中位數 ${num(mae.p50)} ATR、90% 不超過 ${num(mae.p90)} ATR，而平均止損距離是 ${num(d.avg_risk_atr)} ATR。`, !tight)}
      ${verdictLine(`目標${below === rr.length && rr.length ? "太遠：所有計劃 R:R 區間的實際達標率都低於打和所需" : `：${rr.length - below}／${rr.length} 個計劃 R:R 區間的達標率高於打和所需`}；交易最遠走到的盈利中位數是 ${num(mfe.p50)}R，${pct(mfe.ge2, 0, false)} 曾經到 2R。`, below < rr.length)}
      ${verdictLine(`做多 ${rv(live.long.avg_r)}，做空 ${rv(live.short.avg_r)}：做空形態在這個高 Beta 股票池整體虧損。`, live.short.avg_r > 0)}
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
    return `<td class="n ${live ? "live" : ""}" style="${bg(c.avg_r)}" title="${c.n || 0} 筆 · 勝率 ${pct(c.win_rate, 0, false)} · t ${num(c.t_stat, 1)}">${isNum(c.avg_r) ? (c.avg_r > 0 ? "+" : "") + num(c.avg_r) : "—"}</td>`;
  }).join("")}</tr>`).join("");
  const seg = (k, opts) => `<div class="seg" data-bt="${k}">${opts.map(([val, lab]) => `<button data-v="${val}" class="${String(v[k]) === String(val) ? "active" : ""}">${lab}</button>`).join("")}</div>`;
  return `<div class="heat-ctl">${seg("side", [["long", "做多"], ["short", "做空"]])} ${seg("hz", [[5, "5 日"], [7, "7 日"], [10, "10 日"]])} ${seg("period", [["is", "樣本內"], ["oos", "樣本外"]])}</div>
    <div class="tw"><table class="heat"><thead><tr><th>止損 ↓ ／ 目標 →</th>${head}</tr></thead><tbody>${rows}</tbody></table></div>
    <p class="note">每格 = 平均每筆 R（1R = 止損距離）。滑鼠停在格上看筆數、勝率和 t 值。外框標示的是現行方案（結構止損 + 阻力目標 + 7 日）。</p>`;
}

function btBestTable(b, side) {
  const base = b.baseline || {};
  const rows = (b.best_is[side] || []).map((g, i) => {
    const bl = base[`${side}|${g.stop}|${g.target}|${g.horizon}`];
    return { rank: i + 1, ...g, bl_oos: bl ? bl.baseline_oos.avg_r : null, edge: bl ? bl.edge_oos : null };
  });
  const lv = b.grid.find((g) => g.side === side && g.stop === "structural" && g.target === "resistance" && g.horizon === 7);
  if (lv) rows.push({ rank: "現行", ...lv, bl_oos: null, edge: null });
  return table("bt-best-" + side, [
    { key: "rank", label: "#", render: (r) => (r.rank === "現行" ? '<b class="warn">現行</b>' : r.rank) },
    { key: "stop", label: "止損", render: (r) => esc(b.stop_names[r.stop]) }, { key: "target", label: "目標", render: (r) => esc(b.target_names[r.target]) },
    { key: "horizon", label: "持倉", n: 1, render: (r) => r.horizon + " 日" },
    { key: "is_r", label: "樣本內 R", n: 1, sort: (r) => r.is.avg_r, render: (r) => rv(r.is.avg_r) },
    { key: "oos_r", label: "樣本外 R", n: 1, sort: (r) => r.oos.avg_r, render: (r) => rv(r.oos.avg_r) },
    { key: "oos_w", label: "樣本外勝率", n: 1, sort: (r) => r.oos.win_rate, render: (r) => pct(r.oos.win_rate, 0, false) },
    { key: "oos_pf", label: "盈虧比", n: 1, sort: (r) => r.oos.profit_factor, render: (r) => num(r.oos.profit_factor) },
    { key: "oos_t", label: "t 值", n: 1, sort: (r) => r.oos.t_stat, render: (r) => tval(r.oos.t_stat) },
    { key: "oos_n", label: "筆數", n: 1, sort: (r) => r.oos.n, render: (r) => (r.oos.n || 0).toLocaleString() },
    { key: "bl_oos", label: "隨機進場（樣本外）", n: 1, render: (r) => rv(r.bl_oos) },
    { key: "edge", label: "形態額外優勢", n: 1, render: (r) => rv(r.edge) }], rows);
}

function quarterBars(q, title) {
  const ks = Object.keys(q || {}).sort(), vals = ks.map((k) => q[k]);
  if (!ks.length) return "";
  const W = 640, H = 130, m = Math.max(0.1, ...vals.filter(isNum).map(Math.abs)), bw = W / ks.length;
  const y0 = H / 2, sc = (H / 2 - 14) / m;
  const bars = ks.map((k, i) => {
    const v = isNum(q[k]) ? q[k] : 0, h = Math.abs(v) * sc;
    return `<rect x="${i * bw + 2}" y="${v > 0 ? y0 - h : y0}" width="${Math.max(1, bw - 4)}" height="${Math.max(0.5, h)}" fill="var(${v > 0 ? "--up" : "--down"})"><title>${k}：${num(v)}R</title></rect>`
      + (i % 4 === 0 ? `<text x="${i * bw + 2}" y="${H - 1}" class="ax">${k.slice(0, 4)}</text>` : "");
  }).join("");
  return `<div class="qbars"><div class="eyebrow">${esc(title)}</div><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="${esc(title)}">
    <line x1="0" x2="${W}" y1="${y0}" y2="${y0}" stroke="var(--line3)"/>${bars}</svg></div>`;
}

function icTable(f, id) {
  if (!f) return "";
  const rows = f.factors.slice(0, 14);
  const bar = (v) => `<span class="icbar ${v > 0 ? "pos" : "neg"}"><i style="width:${Math.min(100, Math.abs(v) / 0.12 * 100)}%"></i></span>`;
  return table(id, [{ key: "name", label: "因子" },
    { key: "ic_is", label: "樣本內 IC", n: 1, render: (r) => `${bar(r.ic_is)} <span class="${cls(r.ic_is)}">${num(r.ic_is, 3)}</span>` },
    { key: "ic_oos", label: "樣本外 IC", n: 1, render: (r) => `${bar(r.ic_oos)} <span class="${cls(r.ic_oos)}">${num(r.ic_oos, 3)}</span>` },
    { key: "stable", label: "方向一致", render: (r) => (r.stable ? '<span class="up">是</span>' : '<span class="muted">否</span>') }], rows);
}

function calibNote(f) {
  const q = ((f || {}).calibrated || {}).oos_quintiles;
  if (!q || q.length < 5) return "";
  const r = q.map((x) => x.avg_r), mono = r.every((v, i) => i === 0 || v >= r[i - 1] - 0.01), spread = r[4] - r[0];
  return `用樣本內 IC 加權重新組合的分數，在樣本外由低至高分五組：${r.map((v) => num(v)).join(" ／ ")} R。`
    + (mono && spread > 0.05 ? "大致單調向上，代表這些因子有用，可以考慮調整評分權重。" : "並不單調，代表重新組合後的分數在樣本外站不住，所以暫時不改評分權重。");
}

function backtestSection(b) {
  if (!b || !b.available) {
    return `<section class="note-block"><div class="hl">SL／TP 回測驗證</div><p class="note">${esc((b && b.reason) || "沒有回測結果。")}</p>
      <button id="bt-run" class="wide">重新回測</button> <span id="bt-status" class="note"></span></section>`;
  }
  const live = b.live, L = (b.selected_side || {}).long, S = (b.selected_side || {}).short, sc = b.score_calibration || {}, d = b.diagnostics || {};
  const blL = L && (b.baseline || {})[`long|${L.stop}|${L.target}|${L.horizon}`];
  const fx = b.factors || {};
  return `<section class="note-block bt"><div class="note-head"><div><div class="eyebrow">${b.first} → ${b.last} · ${b.n_symbols} 檔 · ${b.n_signals.toLocaleString()} 個信號 · 樣本內至 ${b.cutoff}，之後為樣本外</div>
      <div class="hl">SL／TP 回測驗證：現行止損目標準不準？</div></div>
      <div class="meta">現行方案樣本外<b class="${cls(live.oos.avg_r)}">${num(live.oos.avg_r)}R</b>t ${num(live.oos.t_stat, 1)}</div></div>
    <ul class="list verdict">${(b.verdict || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
    <div class="grid g4 kpis">
      <div class="kpi"><span class="l">現行方案勝率</span><span class="v">${pct(live.all.win_rate, 0, false)}</span><span class="note">平均 ${rv(live.all.avg_r)}</span></div>
      <div class="kpi"><span class="l">做多最佳（樣本外）</span><span class="v ${cls(L && L.oos.avg_r)}">${L ? num(L.oos.avg_r) + "R" : "—"}</span><span class="note">${L ? esc(b.stop_names[L.stop] + " + " + b.target_names[L.target] + " + " + L.horizon + " 日") : ""}</span></div>
      <div class="kpi"><span class="l">同規則隨機進場</span><span class="v ${cls(blL && blL.baseline_oos.avg_r)}">${blL ? num(blL.baseline_oos.avg_r) + "R" : "—"}</span><span class="note">形態額外優勢 ${blL ? rv(blL.edge_oos) : "—"}</span></div>
      <div class="kpi"><span class="l">評分與結果相關</span><span class="v ${Math.abs(sc.spearman || 0) < 0.03 ? "muted" : cls(sc.spearman)}">${num(sc.spearman, 2)}</span><span class="note">等級相關係數（0 = 無預測力）</span></div>
    </div>
    <h3>止損 × 目標 熱圖</h3><div id="bt-heat">${btHeat(b)}</div>
    <h3>做多：樣本內最佳 8 組，再看樣本外</h3>${btBestTable(b, "long")}
    <h3>做空：樣本內最佳 8 組，再看樣本外</h3>${btBestTable(b, "short")}
    ${S && S.oos.avg_r <= 0 ? `<p class="note down">做空最佳組合樣本外仍然 ${num(S.oos.avg_r)}R，只有 ${pct(S.positive_quarters, 0, false)} 季度為正：在這個高 Beta 股票池，做空形態目前只適合作迴避／對沖參考。</p>` : ""}
    <p class="note">選法：只用樣本內（較早 60% 的日子）排名，樣本外（之後 40%）的數字從來沒有參與挑選，所以樣本外才是真正的驗證。「隨機進場」= 同一批股票每 5 個交易日進場一次、不看任何形態，用同一套出場規則；
      如果隨機進場也賺，代表利潤主要來自市場趨勢和出場規則，而不是形態。</p>
    <div class="grid g2">
      <div><h3>計劃風險報酬比 vs 實際達標率</h3>${table("bt-rr", [{ key: "bucket", label: "計劃 R:R" }, { key: "n", label: "筆數", n: 1 },
        { key: "target_rate", label: "實際達標", n: 1, render: (r) => pct(r.target_rate, 1, false) },
        { key: "breakeven", label: "打和所需", n: 1, render: (r) => pct(r.breakeven, 0, false) },
        { key: "ok", label: "", render: (r) => (r.target_rate >= r.breakeven ? '<span class="up">✓</span>' : '<span class="down">✗</span>') },
        { key: "avg_r", label: "平均 R", n: 1, render: (r) => rv(r.avg_r) }], d.rr_calibration || [])}
        <p class="note">計劃 R:R 越高，目標越遠，達標率就跌得越快；所以「R:R 高」並不代表交易更好。</p></div>
      <div><h3>途中浮盈浮虧</h3><div class="kv">
        <div><span>被止損後仍然到達目標</span><span class="num">${pct(d.stopped_then_target, 0, false)}</span></div>
        <div><span>達標交易的最大浮虧（中位／75%／90%）</span><span class="num">${num((d.winner_mae_atr || {}).p50)} ／ ${num((d.winner_mae_atr || {}).p75)} ／ ${num((d.winner_mae_atr || {}).p90)} ATR</span></div>
        <div><span>平均止損距離</span><span class="num">${num(d.avg_risk_atr)} ATR</span></div><div><span>平均目標距離</span><span class="num">${num(d.avg_reward_atr)} ATR</span></div>
        <div><span>最大浮盈中位數（MFE）</span><span class="num">${num((d.mfe_r || {}).p50)}R</span></div>
        <div><span>曾經到 1R ／ 2R ／ 3R</span><span class="num">${pct((d.mfe_r || {}).ge1, 0, false)} ／ ${pct((d.mfe_r || {}).ge2, 0, false)} ／ ${pct((d.mfe_r || {}).ge3, 0, false)}</span></div>
        <div><span>時間止損出場時賺錢的比例</span><span class="num">${pct(d.time_exits_positive, 0, false)}</span></div></div></div>
    </div>
    <div class="grid g2">
      <div><h3>現行評分分組</h3>${table("bt-sc", [{ key: "bucket", label: "分數分組" }, { key: "score_range", label: "分數" }, { key: "n", label: "筆數", n: 1 },
        { key: "win_rate", label: "勝率", n: 1, render: (r) => pct(r.win_rate, 0, false) }, { key: "avg_r", label: "平均 R", n: 1, render: (r) => rv(r.avg_r) }], sc.quintiles || [])}
        <h3>因子預測力（做多，與每筆 R 的等級相關）</h3>${icTable(fx.long, "bt-ic-l")}
        <p class="note">IC 是因子與交易結果的等級相關係數：|IC| 0.02–0.05 屬弱、0.05 以上在短線交易已算有用。「方向一致」= 樣本內外正負號相同。
          ${calibNote(fx.long)}</p></div>
      <div><h3>因子預測力（做空）</h3>${icTable(fx.short, "bt-ic-s")}
        <h3>按形態（現行方案）</h3>${table("bt-set", [{ key: "name", label: "形態" }, { key: "n", label: "筆數", n: 1 },
          { key: "win_rate", label: "勝率", n: 1, render: (r) => pct(r.win_rate, 0, false) }, { key: "avg_r", label: "平均 R", n: 1, render: (r) => rv(r.avg_r) },
          { key: "t_stat", label: "t 值", n: 1, render: (r) => tval(r.t_stat) },
          { key: "best", label: "該形態樣本內最佳 → 樣本外", wrap: 1, render: (r) => { const x = ((b.best_by_setup[r.key] || {}).best || [])[0]; return x ? `${esc(b.stop_names[x.stop])} + ${esc(b.target_names[x.target])} + ${x.horizon} 日：${rv(x.is.avg_r)} → ${rv(x.oos.avg_r)}` : "—"; } }],
          Object.entries(b.by_setup).map(([k, v]) => ({ key: k, ...v })), { sortKey: "avg_r" })}</div>
    </div>
    <h3>按市況（現行方案，平均 R）</h3>${table("bt-reg", [{ key: "side", label: "方向" }, ...REGIMES.map((r) => ({ key: r, label: tr(r), n: 1, sort: (x) => (x[r] || {}).avg_r,
      render: (x) => (x[r] && x[r].n ? `${rv(x[r].avg_r)} <span class="muted">(${x[r].n})</span>` : '<span class="muted">—</span>') }))],
      [["long", "做多"], ["short", "做空"]].map(([k, lab]) => ({ side: lab, ...Object.fromEntries(REGIMES.map((r) => [r, (b.by_regime_side || {})[`${r}|${k}`]])) })))}
    <div class="grid g2">${quarterBars(Object.fromEntries(Object.entries(b.by_quarter || {}).map(([k, v]) => [k, v.avg_r])), "每季平均 R：現行方案")}
      ${L ? quarterBars(L.quarters, `每季平均 R：做多最佳（${b.stop_names[L.stop]} + ${b.target_names[L.target]} + ${L.horizon} 日）`) : ""}</div>
    <p class="note">${esc(b.survivorship_note || "")} 其他限制：只用日線（同一根 K 線同時碰到止損和目標時當作止損）、每邊滑點 ${b.slippage_bps} 基點、未計佣金和借券成本、同日多個相關股票的信號會重疊。
      生成於 ${esc(b.generated_at || "")}，用時 ${b.elapsed_seconds} 秒。 <button id="bt-run">重新回測</button> <span id="bt-status"></span></p>
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
    if (window.__PREVIEW__) { run.disabled = true; run.title = "預覽版不能運行"; }
    run.addEventListener("click", async () => {
      try { await api("/api/backtest/run", { method: "POST" }); } catch (err) { $("#bt-status").textContent = err.message; return; }
      const poll = async () => {
        const st = await api("/api/backtest/status");
        $("#bt-status").textContent = st.running ? `運行中… ${(st.log || []).slice(-1)[0] || ""}` : st.status === "success" ? "完成，重新載入…" : `失敗：${(st.log || []).slice(-1)[0] || ""}`;
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
  const rows = Object.entries(bt.by_setup || {}).map(([k, v]) => ({ key: k, name: s.setup_names[k] || tr(k), ...v })).filter((r) => r.n);
  const regimes = ["Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish"];
  const reg = Object.entries(bt.by_setup_regime || {}).map(([k, m]) => ({ name: s.setup_names[k] || k, ...Object.fromEntries(regimes.map((r) => [r, m[r]])) }));
  app.innerHTML = `<div class="wall-head"><div><div class="eyebrow">${bt.first_signal || ""} → ${bt.last_signal || ""} / <b>${bt.n_trades} 筆模擬交易</b></div><h1 class="display">研究 <em>research</em></h1>
    <p>止損和目標是怎樣定的、在五年歷史上準不準、有沒有更好的設定；下面再列出每日報告附帶的形態回測和計算方法。</p></div></div>
    ${slTpExplainer(full)}
    ${backtestSection(full)}
    <div class="card" style="margin-top:14px"><h2>形態歷史回測 <small>每日報告附帶 · ${bt.horizon_days} 日時間止損 · 滑點 ${bt.slippage_bps} 基點</small></h2>
    ${table("bt", [{ key: "name", label: "形態" }, { key: "n", label: "樣本", n: 1 }, { key: "win_rate", label: "勝率", n: 1, render: (r) => pct(r.win_rate, 0, false) },
      { key: "avg_r", label: "平均 R", n: 1, render: (r) => `<span class="${cls(r.avg_r)}">${num(r.avg_r)}</span>` }, { key: "median_r", label: "中位 R", n: 1, render: (r) => num(r.median_r) },
      { key: "profit_factor", label: "盈虧比", n: 1, render: (r) => num(r.profit_factor) }, { key: "hit_target_rate", label: "達目標一", n: 1, render: (r) => pct(r.hit_target_rate, 0, false) },
      { key: "stop_rate", label: "止損", n: 1, render: (r) => pct(r.stop_rate, 0, false) }, { key: "avg_days", label: "平均日數", n: 1, render: (r) => num(r.avg_days, 1) },
      { key: "avg_mfe", label: "平均最大浮盈", n: 1, render: (r) => pct(r.avg_mfe) }, { key: "avg_mae", label: "平均最大浮虧", n: 1, render: (r) => pct(r.avg_mae) },
      { key: "fwd_5d_mean", label: "5 日後（按方向）", n: 1, render: (r) => pct(r.fwd_5d_mean, 2) }, { key: "fwd_10d_mean", label: "10 日後（按方向）", n: 1, render: (r) => pct(r.fwd_10d_mean, 2) }], rows, { sortKey: "avg_r" })}
    <p class="note">基準（所有股票日，做多）：5 日平均回報 ${pct((bt.baseline || {}).fwd_5d_mean, 2)}，上升機率 ${pct((bt.baseline || {}).fwd_5d_positive, 0, false)}（樣本 ${(bt.baseline || {}).n}）。
    「按方向」為按交易方向計算的收市對收市回報。偏差：倖存者偏差（使用今日股票池）、樣本內、相關股票的信號重疊、未計佣金和借券成本、只用日線。
    這些數字只用作檢查規則是否合理，不代表未來回報。</p></div>
    <div class="card" style="margin-top:14px"><h2>按信號當日市況的平均 R</h2>${table("btr", [{ key: "name", label: "形態" }, ...regimes.map((r) => ({ key: r, label: tr(r), n: 1, sort: (x) => (x[r] || {}).avg_r, render: (x) => x[r] && x[r].n ? `<span class="${cls(x[r].avg_r)}">${num(x[r].avg_r)}</span> <span class="muted">(${x[r].n})</span>` : '<span class="muted">—</span>' }))], reg)}</div>
    <div class="card" style="margin-top:14px"><h2>計算方法</h2><div id="meth" class="note">載入中…</div></div>`;
  try {
    const m = await api("/api/methodology");
    $("#meth").innerHTML = [["市場狀態", m.regime], ["交易形態", m.setups], ["風險計劃", m.plan], ["評分與篩選", m.scoring], ["歷史回測", m.backtest]]
      .map(([t, d]) => `<h3>${t}</h3><pre class="doc">${esc((d || "").trim())}</pre>`).join("")
      + `<h3>權重 · 扣分 · 級別（config/terminal.yaml）</h3><pre class="doc">${esc(JSON.stringify({ 權重: m.weights, 扣分: m.penalties, 級別: m.tiers }, null, 2))}</pre>`;
  } catch (err) { $("#meth").textContent = err.message; }
  if (full && full.available) wireBacktest(full); else wireBacktest(null);
}

// ------------------------------------------------------------------ views: data quality
async function quality() {
  const s = state.snap, q = s.quality, u = q.update || {}, uf = s.universe_filter || {};
  app.innerHTML = `<div class="wall-head"><div><div class="eyebrow">數據檢查 · 篩選 · 運行紀錄</div><h1 class="display">數據 <em>data</em></h1></div></div>
    <div class="grid g4">
    <div class="card kpi"><span class="l">數據正常</span><span class="v up">${q.counts.ok}</span></div>
    <div class="card kpi"><span class="l">警告</span><span class="v warn">${q.counts.warning}</span></div>
    <div class="card kpi"><span class="l">數據問題排除</span><span class="v down">${q.counts.excluded}</span></div>
    <div class="card kpi"><span class="l">篩選後保留</span><span class="v">${(uf.kept || []).length || s.universe.length}</span></div></div>
    <div class="card" style="margin-top:14px"><h2>被高 Beta／高波動篩選排除 <small>${(uf.excluded || []).length} 檔</small></h2>
    ${table("uf", [{ key: "symbol", label: "代號", render: (x) => `<b>${x.symbol}</b>` }, { key: "beta", label: "Beta", n: 1, render: (x) => num(x.beta, 2) },
      { key: "hv60", label: "60 日波動", n: 1, render: (x) => pct(x.hv60, 0, false) }, { key: "market_cap", label: "市值", n: 1, render: (x) => money(x.market_cap) },
      { key: "dollar_volume20", label: "日均成交額", n: 1, render: (x) => money(x.dollar_volume20) }, { key: "reason", label: "原因", wrap: 1 }], uf.excluded || [], { sortKey: "beta" })}</div>
    <div class="grid g2" style="margin-top:14px">
    <div class="card"><h2>代號檢查 <small>過期 · 缺日 · 開高低收 · 成交量 · 報價停滯 · 異常跳動</small></h2>${table("dq", [
      { key: "symbol", label: "代號", render: (r) => `<b>${r.symbol}</b>` }, { key: "status", label: "狀態", render: (r) => `<span class="${r.status === "ok" ? "up" : r.status === "warning" ? "warn" : "down"}">${tr(r.status)}</span>` },
      { key: "last_date", label: "最新 K 線" }, { key: "sessions", label: "交易日數", n: 1 }, { key: "missing_sessions", label: "缺少（1 年）", n: 1 }, { key: "issues", label: "問題", wrap: 1, render: (r) => esc(r.issues.join(" ")), sort: (r) => r.issues.length }],
      q.symbols, { sortKey: "issues" })}</div>
    <div class="card"><h2>更新摘要</h2><div class="kv"><div><span>增量更新</span><span class="num">${(u.incremental || []).length}</span></div><div><span>完整下載</span><span class="num">${(u.full || []).length}</span></div>
      <div><span>重新調整（公司行動）</span><span class="num">${(u.readjusted || []).join(", ") || 0}</span></div><div><span>失敗</span><span class="num">${(u.failed || []).join(", ") || 0}</span></div>
      <div><span>寫入行數</span><span class="num">${u.rows_written ?? "n/a"}</span></div><div><span>丟棄的未完成 K 線</span><span class="num">${u.discarded_future_bars ?? 0}</span></div></div>
      <h3>運行紀錄</h3><div id="runs" class="note">載入中…</div>
      <h3>每日報告</h3><ul class="list" id="reports"></ul></div></div>`;
  try {
    const r = await api("/api/runs");
    $("#runs").innerHTML = table("runs", [{ key: "run_id", label: "運行", n: 1 }, { key: "session_date", label: "交易日" }, { key: "status", label: "狀態", render: (x) => `<span class="${x.status.startsWith("success") ? "up" : x.status === "failed" ? "down" : "warn"}">${tr(x.status)}</span>` },
      { key: "regime_label", label: "市場狀態", render: (x) => tr(x.regime_label) }, { key: "n_candidates", label: "候選", n: 1 }, { key: "finished_at", label: "完成時間" }], r.runs, { sortKey: "run_id" });
    const { sessions } = await api("/api/sessions");
    $("#reports").innerHTML = sessions.map((x) => `<li><a href="/reports/${x}/report.html" target="_blank">${x} 報告</a></li>`).join("");
  } catch (err) { $("#runs").textContent = err.message; }
}

boot();

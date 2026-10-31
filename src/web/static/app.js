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
  const views = { overview, scanner, universe, catalysts, research, quality, ticker };
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
  </div>`;
  const el = $("#regime-chart");
  const ch = makeChart(el, 220);
  const line = ch.addBaselineSeries({ baseValue: { type: "price", price: 0 }, topLineColor: cssVar("--up"), bottomLineColor: cssVar("--down"),
    topFillColor1: "rgba(38,166,154,0.25)", topFillColor2: "rgba(38,166,154,0.02)", bottomFillColor1: "rgba(239,83,80,0.02)", bottomFillColor2: "rgba(239,83,80,0.25)", lineWidth: 2 });
  line.setData(r.history.filter((h) => isNum(h.composite)).map((h) => ({ time: h.date, value: h.composite })));
  for (const [p, t] of [[45, "強勢看多"], [15, "看多"], [-15, "看空"], [-45, "強勢看空"]])
    line.createPriceLine({ price: p, color: cssVar("--dim"), lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: t });
  ch.timeScale().fitContent();
  drawRRG($("#rrg"), s.sectors);
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
      <div class="grid" style="align-content:start">${planPanel(c)}${deep.catalysts ? catalystsPanel(deep.catalysts) : ""}</div></div>
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
      ${deep.catalysts ? catalystsPanel(deep.catalysts) : ""}
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
  return `<div class="card"><h2>研究摘要 <small>${esc(tr(c.direction))} · ${esc(c.setup_name)}</small></h2><p>${esc(c.thesis)}</p>
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
    <p class="note">倉位大小由你決定。例子：以 10 萬美元帳戶承擔 0.5% 風險 = $500 ÷ ${num(Math.abs(p.entry - p.stop))} ≈ ${isNum(p.entry - p.stop) && p.entry !== p.stop ? Math.floor(500 / Math.abs(p.entry - p.stop)) : "n/a"} 股（再乘以市況倉位比例 ×${budget}）。</p></div>`;
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

// ------------------------------------------------------------------ views: research
async function research() {
  const s = state.snap, bt = s.backtest;
  const rows = Object.entries(bt.by_setup || {}).map(([k, v]) => ({ key: k, name: s.setup_names[k] || tr(k), ...v })).filter((r) => r.n);
  const regimes = ["Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish"];
  const reg = Object.entries(bt.by_setup_regime || {}).map(([k, m]) => ({ name: s.setup_names[k] || k, ...Object.fromEntries(regimes.map((r) => [r, m[r]])) }));
  app.innerHTML = `<div class="wall-head"><div><div class="eyebrow">${bt.first_signal || ""} → ${bt.last_signal || ""} / <b>${bt.n_trades} 筆模擬交易</b></div><h1 class="display">研究 <em>research</em></h1>
    <p>每個形態的交易計劃在歷史數據上的模擬結果：${bt.horizon_days} 個交易日時間止損，每邊滑點 ${bt.slippage_bps} 基點。</p></div></div>
    <div class="card"><h2>形態歷史回測</h2>
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

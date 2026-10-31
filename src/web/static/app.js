/* Swing Research Terminal - single-page dashboard (vanilla JS, no build step). Research only. */
"use strict";

const state = { snap: null, session: null, sort: {}, filters: { tier: "", setup: "", hideEarnings: false, q: "" },
  charts: [], scannerSide: "long" };
const $ = (sel, el = document) => el.querySelector(sel);
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
  try {
    const { sessions } = await api("/api/sessions");
    const run = await api("/api/run/status");
    if (run.running) watchRun(true);
    const sel = $("#session");
    sel.innerHTML = sessions.map((s) => `<option>${s}</option>`).join("");
    if (!sessions.length) {
      app.innerHTML = run.running
        ? `<div class="card empty"><h2>Downloading market data…</h2><p>The first run downloads ~2 years of history for the
            universe and analyses it (2–4 minutes). This page opens the report automatically when it is ready.</p></div>`
        : `<div class="card empty"><h2>No data yet</h2><p>Run <code>python daily_run.py</code> or press <b>Run</b> above.
            The first run downloads ~2 years of history for the universe (a few minutes).</p></div>`;
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
  b.textContent = `${reg.label} ${isNum(reg.composite) ? (reg.composite > 0 ? "+" : "") + reg.composite.toFixed(0) : ""}`;
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
  state.charts.forEach((c) => { c._gone = true; c._observer.disconnect(); try { c.remove(); } catch (_) { /* already gone */ } });
  state.charts = [];
  const [, view = "overview", arg] = (location.hash || "#/overview").split("/");
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === view));
  const views = { overview, scanner, universe, catalysts, research, quality, ticker };
  (views[view] || overview)(arg);
  window.scrollTo(0, 0);
}

// ------------------------------------------------------------------ chart helpers
function makeChart(el, height, opts = {}) {
  const chart = LightweightCharts.createChart(el, {
    height, layout: { background: { type: "solid", color: cssVar("--panel") }, textColor: cssVar("--muted"), fontSize: 11 },
    grid: { vertLines: { color: cssVar("--line") }, horzLines: { color: cssVar("--line") } },
    rightPriceScale: { borderColor: cssVar("--line") }, timeScale: { borderColor: cssVar("--line"), visible: opts.timeVisible !== false },
    crosshair: { mode: 0 }, handleScroll: true, handleScale: true, ...opts.extra,
  });
  const observer = new ResizeObserver(() => { if (!chart._gone) chart.applyOptions({ width: el.clientWidth }); });
  observer.observe(el);
  chart._observer = observer;
  state.charts.push(chart);
  return chart;
}
function syncCharts(charts) {
  charts.forEach((c) => c.timeScale().subscribeVisibleLogicalRangeChange((range) => {
    if (!range) return;
    charts.forEach((o) => { if (o !== c) o.timeScale().setVisibleLogicalRange(range); });
  }));
}

// ------------------------------------------------------------------ views: overview
function overview() {
  const s = state.snap, r = s.regime;
  const needle = isNum(r.composite) ? ((r.composite + 100) / 200) * 100 : 50;
  const comps = r.components.map((c) => {
    const w = isNum(c.score) ? Math.abs(c.score) / 2 * 50 : 0;
    const left = isNum(c.score) && c.score < 0 ? 50 - w : 50;
    const color = isNum(c.score) ? (c.score >= 0 ? "var(--up)" : "var(--down)") : "var(--dim)";
    return `<div class="comp"><div>${esc(c.name)} <span class="muted">· ${c.weight}</span></div>
      <div class="zbar"><div class="mid"></div><div class="fill" style="left:${left}%;width:${w}%;background:${color}"></div></div>
      <div class="n mono ${cls(c.score)}">${isNum(c.score) ? (c.score > 0 ? "+" : "") + c.score.toFixed(2) : "n/a"}</div>
      <div class="ev">${c.evidence.map(esc).join(" ")}</div></div>`;
  }).join("");
  const b = r.breadth || {}, v = r.volatility || {}, rt = r.rates || {};
  const tops = (side) => s.candidates[side].slice(0, 6).map((c) => `<tr class="click" data-sym="${c.symbol}"><td><b>${c.symbol}</b></td><td>${esc(c.setup_name)}</td><td class="n">${c.score.toFixed(0)}</td><td>${tier(c.tier)}</td><td class="n">${num(c.plan.reward_risk, 1)}R</td></tr>`).join("") || `<tr><td class="empty">None</td></tr>`;
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
      <div class="card"><h2>Top short research candidates</h2><div class="tw"><table><tbody>${tops("short")}</tbody></table></div></div>
      <div class="card"><h2>Next events</h2><ul class="list">${(s.macro_events || []).slice(0, 6).map((e) => `<li><span class="mono">${e.event_date}</span> ${esc(e.event_type)} <span class="chip">${e.status}</span></li>`).join("") || "<li>None in the next 30 days</li>"}
        ${(s.upcoming_earnings || []).filter((e) => e.sessions_until <= 10).slice(0, 8).map((e) => `<li><span class="mono">${e.date}</span> <a href="#/ticker/${e.symbol}">${e.symbol}</a> earnings ${esc(e.timing || "(time n/a)")} <span class="chip">${e.status}</span></li>`).join("")}</ul></div>
    </div>
  </div>
  <div class="grid g2" style="margin-top:12px">
    <div class="card"><h2>Sector rotation <small>RS vs SPY: trend (x) vs momentum (y), 10-session tails</small></h2><svg id="rrg" class="rrg"></svg></div>
    <div class="card"><h2>Sector & industry ETFs</h2>${table("sec", [
      { key: "rank_20d", label: "#", n: 1 },
      { key: "symbol", label: "ETF", render: (x) => `<b>${x.symbol}</b> <span class="muted">${esc(x.name)}</span>` },
      { key: "ret_5d", label: "5D", n: 1, render: (x) => `<span class="${cls(x.ret_5d)}">${pct(x.ret_5d)}</span>` },
      { key: "ret_20d", label: "20D", n: 1, render: (x) => `<span class="${cls(x.ret_20d)}">${pct(x.ret_20d)}</span>` },
      { key: "rel_20d", label: "Rel 20D", n: 1, render: (x) => `<span class="${cls(x.rel_20d)}">${pct(x.rel_20d)}</span>` },
      { key: "rel_60d", label: "Rel 60D", n: 1, render: (x) => `<span class="${cls(x.rel_60d)}">${pct(x.rel_60d)}</span>` },
      { key: "quadrant", label: "Quadrant", render: (x) => `<span class="${{ Leading: "up", Lagging: "down", Improving: "", Weakening: "warn" }[x.quadrant] || "muted"}">${x.quadrant}</span>` },
    ], s.sectors, { sortKey: "rank_20d", sortDir: "asc" })}</div>
  </div>`;
  const el = $("#regime-chart");
  const ch = makeChart(el, 220);
  const line = ch.addBaselineSeries({ baseValue: { type: "price", price: 0 }, topLineColor: cssVar("--up"), bottomLineColor: cssVar("--down"),
    topFillColor1: "rgba(38,166,154,0.25)", topFillColor2: "rgba(38,166,154,0.02)", bottomFillColor1: "rgba(239,83,80,0.02)", bottomFillColor2: "rgba(239,83,80,0.25)", lineWidth: 2 });
  line.setData(r.history.filter((h) => isNum(h.composite)).map((h) => ({ time: h.date, value: h.composite })));
  for (const [p, t] of [[45, "Strong Bullish"], [15, "Bullish"], [-15, "Bearish"], [-45, "Strong Bearish"]])
    line.createPriceLine({ price: p, color: cssVar("--dim"), lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: t });
  ch.timeScale().fitContent();
  drawRRG($("#rrg"), s.sectors);
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
  out += quad.map(([t, x, y, a, c]) => `<text x="${x}" y="${y}" text-anchor="${a}" fill="${cssVar(c)}" font-size="11" opacity="0.8">${t}</text>`).join("");
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
  const s = state.snap, f = state.filters, side = state.scannerSide;
  const all = s.candidates[side];
  const setups = [...new Set(all.map((c) => c.setup))];
  const rows = all.filter((c) => (!f.tier || c.tier === f.tier) && (!f.setup || c.setup === f.setup)
    && (!f.hideEarnings || !c.penalties.earnings_in_window) && (!f.q || c.symbol.includes(f.q.toUpperCase())));
  const cols = [
    { key: "rank", label: "#", n: 1 },
    { key: "symbol", label: "Symbol", render: (c) => `<b>${c.symbol}</b> <span class="muted">${esc(c.sector_etf || "")}</span>` },
    { key: "setup_name", label: "Setup", render: (c) => `${esc(c.setup_name)}${c.other_setups.length ? ` <span class="chip">+${c.other_setups.length}</span>` : ""}` },
    { key: "score", label: "Score", n: 1, render: (c) => `<div class="scorecell"><div class="bar"><span style="width:${c.score}%"></span></div><span class="mono">${c.score.toFixed(0)}</span></div>` },
    { key: "tier", label: "Tier", render: (c) => tier(c.tier), sort: (c) => c.score },
    { key: "entry", label: "Entry", n: 1, render: (c) => num(c.plan.entry) + (c.plan.conditional_entry ? ' <span class="chip">cond.</span>' : ""), sort: (c) => c.plan.entry },
    { key: "stop", label: "Stop", n: 1, render: (c) => num(c.plan.stop), sort: (c) => c.plan.stop },
    { key: "t1", label: "T1", n: 1, render: (c) => num(c.plan.target1), sort: (c) => c.plan.target1 },
    { key: "rr", label: "R/R", n: 1, render: (c) => `<span class="${c.plan.reward_risk >= 2 ? "up" : c.plan.reward_risk < 1.5 ? "down" : ""}">${num(c.plan.reward_risk, 1)}</span>`, sort: (c) => c.plan.reward_risk },
    { key: "risk", label: "Risk %", n: 1, render: (c) => pct(c.plan.risk_pct, 1, false), sort: (c) => c.plan.risk_pct },
    { key: "rs", label: "RS pct", n: 1, render: (c) => num(c.technical.rs_percentile, 0), sort: (c) => c.technical.rs_percentile },
    { key: "earn", label: "Earnings", render: (c) => { const e = earningsOf(c); return e ? `<span class="${e.sessions_until <= 7 ? "warn" : ""}">${e.date} (${e.sessions_until})</span>` : '<span class="muted">—</span>'; }, sort: (c) => (earningsOf(c) || {}).sessions_until ?? 999 },
    { key: "ev", label: "Evidence +/−", n: 1, render: (c) => `<span class="up">${c.supporting.length}</span> / <span class="down">${c.contradicting.length}</span>`, sort: (c) => c.supporting.length - c.contradicting.length },
    { key: "pen", label: "Penalties", render: (c) => Object.keys(c.penalties).map((k) => `<span class="chip">${k.replace(/_/g, " ")}</span>`).join("") || '<span class="muted">—</span>', sort: (c) => c.penalty_total },
    { key: "hist", label: "Setup history", render: (c) => { const h = (c.history || {}).setup; return h && h.n ? `<span class="mono">${(h.win_rate * 100).toFixed(0)}% · ${h.avg_r >= 0 ? "+" : ""}${h.avg_r.toFixed(2)}R</span>` : "n/a"; }, sort: (c) => ((c.history || {}).setup || {}).avg_r },
  ];
  app.innerHTML = `<div class="card"><h2>Swing candidates <small>ranked research priority, not trade signals · regime ${esc(s.regime.label)}</small></h2>
    <div class="tabs"><button data-side="long" class="${side === "long" ? "active" : ""}">Long (${s.candidates.long.length})</button><button data-side="short" class="${side === "short" ? "active" : ""}">Short (${s.candidates.short.length})</button></div>
    <div class="filters">
      <label>Tier <select id="f-tier"><option value="">All</option>${["A", "B", "C", "Watch"].map((t) => `<option ${f.tier === t ? "selected" : ""}>${t}</option>`).join("")}</select></label>
      <label>Setup <select id="f-setup"><option value="">All</option>${setups.map((x) => `<option value="${x}" ${f.setup === x ? "selected" : ""}>${esc(s.setup_names[x])}</option>`).join("")}</select></label>
      <label><input type="checkbox" id="f-earn" ${f.hideEarnings ? "checked" : ""}> Hide earnings inside the holding window</label>
      <label>Search <input id="f-q" value="${esc(f.q)}" style="width:80px"></label>
      <span class="note">${rows.length} shown · click a row for the full research view</span>
    </div>
    ${table("scan-" + side, cols, rows, { sortKey: "rank", sortDir: "asc", onClick: true, rowAttr: (c) => `data-sym="${c.symbol}"` })}
    <p class="note">Entry = reference (today's close; "cond." = only on a break of the trigger). Stops / targets follow fixed rules (Research → plan).
    Score = weighted sub-scores − penalties. History = universe-wide simulation of the same plan (in-sample).</p></div>`;
  app.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => { state.scannerSide = b.dataset.side; state.filters.setup = ""; route(); }));
  $("#f-tier").onchange = (e) => { f.tier = e.target.value; route(); };
  $("#f-setup").onchange = (e) => { f.setup = e.target.value; route(); };
  $("#f-earn").onchange = (e) => { f.hideEarnings = e.target.checked; route(); };
  $("#f-q").oninput = (e) => { f.q = e.target.value; clearTimeout(state.qt); state.qt = setTimeout(() => { route(); const i = $("#f-q"); i.focus(); i.setSelectionRange(i.value.length, i.value.length); }, 250); };
}

// ------------------------------------------------------------------ views: ticker
async function ticker(sym) {
  sym = (sym || "").toUpperCase();
  const s = state.snap;
  const cands = [...s.candidates.long, ...s.candidates.short].filter((c) => c.symbol === sym);
  const deep = s.deep_dive[sym] || {};
  const uni = s.universe.find((u) => u.symbol === sym);
  const c = cands[0];
  app.innerHTML = `<div class="grid g-side">
    <div class="grid" style="align-content:start">
      <div class="card"><h2><span><b style="font-size:16px;color:var(--fg)">${esc(sym)}</b> <span class="muted">${esc((deep.profile || {}).longName || "")}</span></span><small id="t-asof"></small></h2>
        <div class="legend"><span><i style="background:#f2b84b"></i>EMA9</span><span><i style="background:#4ea1ff"></i>EMA21</span><span><i style="background:#a78bfa"></i>EMA50</span><span><i style="background:#8592a0"></i>EMA200</span><span><i style="background:#5b6672"></i>EMA250</span><span><i style="background:rgba(133,146,160,0.6)"></i>BOLL 20/1.8</span>${c ? '<span><i style="background:#26a69a"></i>T1/T2</span><span><i style="background:#ef5350"></i>Stop</span>' : ""}<span>▲▼ setup starts</span></div>
        <div id="c-price" class="chart"></div><div class="legend"><span>Volume · MAVOL20</span></div><div id="c-vol" class="chart small"></div>
        <div class="legend"><span><i style="background:#f2b84b"></i>RSI6</span><span><i style="background:#4ea1ff"></i>RSI14</span></div><div id="c-rsi" class="chart small"></div>
        <div class="legend"><span><i style="background:#4ea1ff"></i>DIF</span><span><i style="background:#f2b84b"></i>DEA</span><span>Histogram (MACD 12/26/9)</span></div><div id="c-macd" class="chart small"></div>
      </div>
      ${c ? candidatePanels(c, deep) : `<div class="card"><h2>Not a candidate this session</h2><p class="note">${uni ? `Active setups: ${uni.setups.length ? uni.setups.map((x) => esc(s.setup_names[x])).join(", ") : "none"}. Trend ${esc(uni.trend)}, structure ${esc(uni.structure)}, RS percentile ${num(uni.rs_pct, 0)}.` : "Not in the screened universe - chart only if prices are stored."}</p></div>`}
      ${deep.fundamentals ? fundamentalsPanel(deep.fundamentals) : ""}
    </div>
    <div class="grid" style="align-content:start">
      ${c ? planPanel(c) : ""}
      <div class="card"><h2>Technical snapshot</h2><div id="t-snap" class="kv"></div><h3>Support / resistance (60 sessions)</h3><div id="t-levels" class="note">…</div></div>
      ${deep.options ? optionsPanel(deep.options) : ""}
      ${deep.positioning ? positioningPanel(deep.positioning) : ""}
      ${deep.catalysts ? catalystsPanel(deep.catalysts) : ""}
      <div class="card"><h2>Candidate history</h2><div id="t-hist" class="note">…</div></div>
    </div></div>`;
  try {
    const d = await api(`/api/chart/${encodeURIComponent(sym)}?days=200&session=${state.session}`);
    renderTickerCharts(d, c);
    $("#t-asof").textContent = `as of ${d.as_of}`;
    const L = d.last;
    $("#t-snap").innerHTML = [["Close", num(L.close)], ["ATR(14)", `${num(L.atr)} (${num(L.atr_pct, 1)}%)`], ["RSI6 / RSI14", `${num(L.rsi6, 0)} / ${num(L.rsi14, 0)}`],
      ["Trend", esc(L.trend)], ["Swing structure", esc(L.swing_structure)], ["Structure state", esc(L.structure)], ["Swing condition", esc(L.swing_condition)],
      ["Volume / MAVOL20", num(L.volume_ratio, 2) + "×"], ["Extension (ATR from EMA21)", num(L.extension_atr, 2)], ["HV20", pct(L.hv20, 0, false)],
      ["Rel. 20D / 60D vs SPY", `${pct(L.rel_20d)} / ${pct(L.rel_60d)}`], ["From 52w high", pct(L.from_52w_high)]]
      .map(([k, v]) => `<div><span>${k}</span><span class="num">${v}</span></div>`).join("");
    $("#t-levels").innerHTML = ["resistance", "support"].map((side) => `<div><b class="${side === "support" ? "up" : "down"}">${side}</b>: ${d.levels[side].map((l) => `<span class="mono">${num(l.price)}</span> <span class="muted">(${esc(l.label)}, ${num(l.distance_atr, 1)} ATR)</span>`).join(" · ") || "none in range"}</div>`).join("");
  } catch (err) {
    $("#c-price").innerHTML = `<div class="empty">Chart unavailable: ${esc(err.message)}</div>`;
  }
  try {
    const h = await api(`/api/candidate-history/${encodeURIComponent(sym)}`);
    $("#t-hist").innerHTML = h.history.length ? table("chist", [
      { key: "session_date", label: "Session" }, { key: "direction", label: "Side" }, { key: "setup", label: "Setup", render: (x) => esc(s.setup_names[x.setup] || x.setup) },
      { key: "score", label: "Score", n: 1, render: (x) => num(x.score, 0) }, { key: "tier", label: "Tier", render: (x) => tier(x.tier) }], h.history, { sortKey: "session_date" }) : "Never listed as a candidate.";
  } catch (_) { $("#t-hist").textContent = "n/a"; }
}

function renderTickerCharts(d, c) {
  const price = makeChart($("#c-price"), 420);
  const candles = price.addCandlestickSeries({ upColor: cssVar("--up"), downColor: cssVar("--down"), borderVisible: false, wickUpColor: cssVar("--up"), wickDownColor: cssVar("--down") });
  candles.setData(d.candles);
  const line = (data, color, width = 1.5, style = 0) => { const sr = price.addLineSeries({ color, lineWidth: width, lineStyle: style, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false }); sr.setData(data); return sr; };
  line(d.bb_upper, "rgba(133,146,160,0.6)", 1, 2); line(d.bb_lower, "rgba(133,146,160,0.6)", 1, 2);
  line(d.ema250, "#5b6672", 1); line(d.ema200, "#8592a0", 1.2); line(d.ema50, "#a78bfa"); line(d.ema21, "#4ea1ff"); line(d.ema9, "#f2b84b");
  const recent = new Set(d.markers.slice(-3).map((m) => m.time + m.setup));  // label only the latest few
  candles.setMarkers(d.markers.map((m) => ({ time: m.time, position: m.direction === "long" ? "belowBar" : "aboveBar", color: m.direction === "long" ? cssVar("--up") : cssVar("--down"), shape: m.direction === "long" ? "arrowUp" : "arrowDown", text: recent.has(m.time + m.setup) ? m.setup.replace("_", " ") : "" })));
  if (c) {
    const p = c.plan, pl = (price_, color, title, style = 0) => candles.createPriceLine({ price: price_, color, lineWidth: 1, lineStyle: style, axisLabelVisible: true, title });
    pl(p.entry, cssVar("--accent"), p.conditional_entry ? "trigger" : "ref", 2); pl(p.stop, cssVar("--down"), "stop"); pl(p.target1, cssVar("--up"), "T1"); pl(p.target2, cssVar("--up"), "T2", 2);
  }
  const vol = makeChart($("#c-vol"), 110, { timeVisible: false });
  vol.addHistogramSeries({ priceFormat: { type: "volume" }, priceLineVisible: false, lastValueVisible: false }).setData(d.volume);
  vol.addLineSeries({ color: "#f2b84b", lineWidth: 1, priceLineVisible: false, lastValueVisible: false }).setData(d.mavol20);
  const rsi = makeChart($("#c-rsi"), 110, { timeVisible: false });
  const r6 = rsi.addLineSeries({ color: "#f2b84b", lineWidth: 1.2, priceLineVisible: false }); r6.setData(d.rsi6);
  rsi.addLineSeries({ color: "#4ea1ff", lineWidth: 1.5, priceLineVisible: false }).setData(d.rsi14);
  [70, 30].forEach((v) => r6.createPriceLine({ price: v, color: cssVar("--dim"), lineWidth: 1, lineStyle: 2, axisLabelVisible: false }));
  const macd = makeChart($("#c-macd"), 110);
  macd.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false }).setData(d.macd_hist.map((x) => ({ ...x, color: x.value >= 0 ? "rgba(38,166,154,0.6)" : "rgba(239,83,80,0.6)" })));
  macd.addLineSeries({ color: "#4ea1ff", lineWidth: 1.2, priceLineVisible: false }).setData(d.macd);
  macd.addLineSeries({ color: "#f2b84b", lineWidth: 1.2, priceLineVisible: false }).setData(d.macd_signal);
  const all = [price, vol, rsi, macd];
  all.forEach((x) => x.timeScale().fitContent());
  syncCharts(all);
}

function candidatePanels(c, deep) {
  const comps = Object.entries(c.components).map(([k, v]) => `<div class="comp"><div>${k.replace(/_/g, " ")} <span class="muted">· ${state.snap.settings.weights[k]}</span></div><div class="bar"><span style="width:${v}%"></span></div><div class="n mono">${v.toFixed(0)}</div></div>`).join("");
  const pens = Object.entries(c.penalties).map(([k, v]) => `<span class="chip">${k.replace(/_/g, " ")} −${v}</span>`).join("") || '<span class="muted">none</span>';
  const h = c.history || {};
  const hrow = (st, label) => st && st.n ? `<tr><td>${label}</td><td class="n">${st.n}${st.small_sample ? " ⚠" : ""}</td><td class="n">${pct(st.win_rate, 0, false)}</td><td class="n ${cls(st.avg_r)}">${num(st.avg_r)}</td><td class="n">${pct(st.hit_target_rate, 0, false)}</td><td class="n">${pct(st.stop_rate, 0, false)}</td><td class="n">${num(st.avg_days, 1)}</td></tr>` : `<tr><td>${label}</td><td colspan="6" class="muted">no history</td></tr>`;
  return `<div class="card"><h2>Research thesis <small>${esc(c.direction)} · ${esc(c.setup_name)}</small></h2><p>${esc(c.thesis)}</p>
    <div class="grid g2"><div><h3 class="up">Supporting evidence</h3><ul class="list ev-pos">${c.supporting.map((x) => `<li>${esc(x)}</li>`).join("") || "<li class=muted>none</li>"}</ul></div>
    <div><h3 class="down">Contradicting evidence / risks</h3><ul class="list ev-neg">${c.contradicting.map((x) => `<li>${esc(x)}</li>`).join("") || "<li class=muted>none</li>"}</ul></div></div></div>
    <div class="card"><h2>Score breakdown <small>raw ${num(c.raw_score, 1)} − penalties ${num(c.penalty_total, 0)} = ${num(c.score, 1)}</small></h2>${comps}<div style="margin-top:8px">Penalties: ${pens}</div></div>
    <div class="card"><h2>Historical simulation of this setup <small>universe-wide, in-sample</small></h2><div class="tw"><table><thead><tr><th>Sample</th><th class="n">N</th><th class="n">Win</th><th class="n">Avg R</th><th class="n">T1 hit</th><th class="n">Stopped</th><th class="n">Days</th></tr></thead>
    <tbody>${hrow(h.setup, "All regimes")}${hrow(h.setup_in_regime, `In ${esc(h.regime)} regimes`)}</tbody></table></div>
    <p class="note">Next-open entry, same stop / T1 / time-stop rules, slippage included. Survivorship and in-sample bias apply; ⚠ = fewer than 30 trades.</p></div>`;
}

function planPanel(c) {
  const p = c.plan;
  return `<div class="card"><h2>Risk plan <small>${p.horizon_days}-session time stop</small></h2><div class="kv">
    <div><span>${p.conditional_entry ? "Trigger (conditional)" : "Reference entry"}</span><span class="num">${num(p.entry)}</span></div>
    <div><span>Stop</span><span class="num down">${num(p.stop)} (${num(p.risk_atr, 1)} ATR)</span></div>
    <div><span>Target 1</span><span class="num up">${num(p.target1)}</span></div><div><span>Target 2</span><span class="num up">${num(p.target2)}</span></div>
    <div><span>Reward / risk (T1)</span><span class="num">${num(p.reward_risk, 2)}</span></div><div><span>Risk per share</span><span class="num">${pct(p.risk_pct, 2, false)}</span></div>
    <div><span>Structural stop</span><span class="num ${p.wide_stop ? "warn" : ""}">${num(p.structural_stop_atr, 1)} ATR${p.wide_stop ? " (capped)" : ""}</span></div>
    <div><span>1σ ${p.horizon_days}-day move (HV20)</span><span class="num">±${num(p.expected_move_hv)}</span></div></div>
    <p class="note">Position size is your decision. Example: risking 0.5% of a $100k account = $500 / ${num(Math.abs(p.entry - p.stop))} ≈ ${isNum(p.entry - p.stop) && p.entry !== p.stop ? Math.floor(500 / Math.abs(p.entry - p.stop)) : "n/a"} shares (× regime budget ${c.direction === "long" ? state.snap.regime.long_risk_budget : state.snap.regime.short_risk_budget}).</p></div>`;
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
  app.innerHTML = `<div class="card"><h2>Screened universe <small>${s.universe.length} stocks · click for chart</small></h2>
    <div class="filters"><label>Search <input id="u-q" value="${esc(state.filters.q)}" style="width:90px"></label></div>
    ${table("uni", [
      { key: "symbol", label: "Symbol", render: (u) => `<b>${u.symbol}</b>` }, { key: "sector_etf", label: "ETF" },
      { key: "close", label: "Close", n: 1, render: (u) => num(u.close) },
      { key: "ret_1d", label: "1D", n: 1, render: (u) => `<span class="${cls(u.ret_1d)}">${pct(u.ret_1d)}</span>` },
      { key: "ret_5d", label: "5D", n: 1, render: (u) => `<span class="${cls(u.ret_5d)}">${pct(u.ret_5d)}</span>` },
      { key: "ret_20d", label: "20D", n: 1, render: (u) => `<span class="${cls(u.ret_20d)}">${pct(u.ret_20d)}</span>` },
      { key: "rel_20d", label: "Rel 20D", n: 1, render: (u) => `<span class="${cls(u.rel_20d)}">${pct(u.rel_20d)}</span>` },
      { key: "rs_pct", label: "RS pct", n: 1, render: (u) => num(u.rs_pct, 0) }, { key: "rsi14", label: "RSI14", n: 1, render: (u) => num(u.rsi14, 0) },
      { key: "volume_ratio", label: "Vol/MAVOL", n: 1, render: (u) => num(u.volume_ratio, 2) }, { key: "extension_atr", label: "Ext. ATR", n: 1, render: (u) => num(u.extension_atr, 1) },
      { key: "atr_pct", label: "ATR %", n: 1, render: (u) => num(u.atr_pct, 1) }, { key: "from_52w_high", label: "From 52wH", n: 1, render: (u) => pct(u.from_52w_high) },
      { key: "trend", label: "Trend" }, { key: "structure", label: "Structure" },
      { key: "setups", label: "Active setups", render: (u) => u.setups.map((x) => `<span class="chip ${s.setup_directions[x]}">${esc(s.setup_names[x])}</span>`).join("") || '<span class="muted">—</span>', sort: (u) => u.setups.length },
      { key: "dollar_volume20", label: "$ Vol 20D", n: 1, render: (u) => money(u.dollar_volume20) },
    ], rows, { sortKey: "rs_pct", rowAttr: (u) => `data-sym="${u.symbol}"`, onClick: true })}</div>`;
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
  const sent = (v) => `<span class="${v === "Positive" ? "up" : v === "Negative" ? "down" : "muted"}">${v}</span>`;
  app.innerHTML = `<div class="grid g2">
    <div class="card"><h2>Upcoming earnings <small>shortlisted names</small></h2>${table("ue", [
      { key: "symbol", label: "Symbol", render: (e) => `<b>${e.symbol}</b>` }, { key: "date", label: "Date" }, { key: "timing", label: "Timing", render: (e) => esc(e.timing || "not confirmed") },
      { key: "reaction_session", label: "Reaction session" }, { key: "sessions_until", label: "Sessions", n: 1 }, { key: "eps_estimate", label: "EPS est.", n: 1, render: (e) => num(e.eps_estimate) }, { key: "status", label: "Status" }],
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

// ------------------------------------------------------------------ views: research
async function research() {
  const s = state.snap, bt = s.backtest;
  const rows = Object.entries(bt.by_setup || {}).map(([k, v]) => ({ key: k, name: s.setup_names[k] || k.replace("ALL_", "All ").toLowerCase(), ...v })).filter((r) => r.n);
  const regimes = ["Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish"];
  const reg = Object.entries(bt.by_setup_regime || {}).map(([k, m]) => ({ name: s.setup_names[k] || k, ...Object.fromEntries(regimes.map((r) => [r, m[r]])) }));
  app.innerHTML = `<div class="card"><h2>Historical setup simulation <small>${bt.first_signal || ""} → ${bt.last_signal || ""} · ${bt.n_trades} trades · ${bt.horizon_days}-session time stop · ${bt.slippage_bps} bps slippage/side</small></h2>
    ${table("bt", [{ key: "name", label: "Setup" }, { key: "n", label: "N", n: 1 }, { key: "win_rate", label: "Win", n: 1, render: (r) => pct(r.win_rate, 0, false) },
      { key: "avg_r", label: "Avg R", n: 1, render: (r) => `<span class="${cls(r.avg_r)}">${num(r.avg_r)}</span>` }, { key: "median_r", label: "Median R", n: 1, render: (r) => num(r.median_r) },
      { key: "profit_factor", label: "PF", n: 1, render: (r) => num(r.profit_factor) }, { key: "hit_target_rate", label: "T1 hit", n: 1, render: (r) => pct(r.hit_target_rate, 0, false) },
      { key: "stop_rate", label: "Stopped", n: 1, render: (r) => pct(r.stop_rate, 0, false) }, { key: "avg_days", label: "Days", n: 1, render: (r) => num(r.avg_days, 1) },
      { key: "avg_mfe", label: "Avg MFE", n: 1, render: (r) => pct(r.avg_mfe) }, { key: "avg_mae", label: "Avg MAE", n: 1, render: (r) => pct(r.avg_mae) },
      { key: "fwd_5d_mean", label: "Fwd 5D (dir.)", n: 1, render: (r) => pct(r.fwd_5d_mean, 2) }, { key: "fwd_10d_mean", label: "Fwd 10D (dir.)", n: 1, render: (r) => pct(r.fwd_10d_mean, 2) }], rows, { sortKey: "avg_r" })}
    <p class="note">Baseline (every stock-day, long): mean 5D ${pct((bt.baseline || {}).fwd_5d_mean, 2)}, positive ${pct((bt.baseline || {}).fwd_5d_positive, 0, false)} (N=${(bt.baseline || {}).n}).
    "Fwd" columns are close-to-close returns in the trade direction. Biases: survivorship (today's universe), in-sample rules, overlapping correlated signals, no commissions / borrow fees, daily bars only.
    Treat these as a sanity check of the rules — not as expected future performance.</p></div>
    <div class="card" style="margin-top:12px"><h2>Average R by market regime at signal time</h2>${table("btr", [{ key: "name", label: "Setup" }, ...regimes.map((r) => ({ key: r, label: r, n: 1, sort: (x) => (x[r] || {}).avg_r, render: (x) => x[r] && x[r].n ? `<span class="${cls(x[r].avg_r)}">${num(x[r].avg_r)}</span> <span class="muted">(${x[r].n})</span>` : '<span class="muted">—</span>' }))], reg)}</div>
    <div class="card" style="margin-top:12px"><h2>Methodology</h2><div id="meth" class="note">Loading…</div></div>`;
  try {
    const m = await api("/api/methodology");
    $("#meth").innerHTML = [["Market regime", m.regime], ["Setups", m.setups], ["Risk plan", m.plan], ["Scoring", m.scoring], ["Historical simulation", m.backtest]]
      .map(([t, d]) => `<h3>${t}</h3><pre class="doc">${esc((d || "").trim())}</pre>`).join("")
      + `<h3>Weights · penalties · tiers (config/terminal.yaml)</h3><pre class="doc">${esc(JSON.stringify({ weights: m.weights, penalties: m.penalties, tiers: m.tiers }, null, 2))}</pre>`;
  } catch (err) { $("#meth").textContent = err.message; }
}

// ------------------------------------------------------------------ views: data quality
async function quality() {
  const s = state.snap, q = s.quality, u = q.update || {};
  app.innerHTML = `<div class="grid g4">
    <div class="card kpi"><span class="l">Symbols OK</span><span class="v up">${q.counts.ok}</span></div>
    <div class="card kpi"><span class="l">Warnings</span><span class="v warn">${q.counts.warning}</span></div>
    <div class="card kpi"><span class="l">Excluded</span><span class="v down">${q.counts.excluded}</span></div>
    <div class="card kpi"><span class="l">Coverage</span><span class="v">${pct(q.coverage, 0, false)}</span></div></div>
    <div class="grid g2" style="margin-top:12px">
    <div class="card"><h2>Symbol checks <small>stale bar · missing sessions · OHLC · volume · stale quotes · jumps</small></h2>${table("dq", [
      { key: "symbol", label: "Symbol", render: (r) => `<b>${r.symbol}</b>` }, { key: "status", label: "Status", render: (r) => `<span class="${r.status === "ok" ? "up" : r.status === "warning" ? "warn" : "down"}">${r.status}</span>` },
      { key: "last_date", label: "Last bar" }, { key: "sessions", label: "Sessions", n: 1 }, { key: "missing_sessions", label: "Missing (1y)", n: 1 }, { key: "issues", label: "Issues", wrap: 1, render: (r) => esc(r.issues.join(" ")), sort: (r) => r.issues.length }],
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

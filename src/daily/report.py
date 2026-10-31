"""Daily research report: Markdown (single source) and a self-contained, printable HTML version."""

from __future__ import annotations

import html
import re
from typing import Any

SIDE_NAMES = {"long": "Long", "short": "Short"}


def _pct(v: Any, d: int = 1, sign: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%"


def _num(v: Any, d: int = 2) -> str:
    return "n/a" if v is None else f"{v:,.{d}f}"


def _money(v: Any) -> str:
    if v is None:
        return "n/a"
    a = abs(v)
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"${v / div:,.1f}{suffix}"
    return f"${v:,.0f}"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_None._\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    return "\n".join(lines) + "\n"


def _earn(c: dict[str, Any], deep: dict[str, Any]) -> str:
    n = (deep.get(c["symbol"], {}).get("catalysts") or {}).get("next_earnings")
    if not n:
        return "n/a" if not c.get("deep_dive") else "none listed"
    return f"{n['date']} ({n['sessions_until']}s)"


def build_markdown(d: dict[str, Any]) -> str:
    s, reg, deep = d["session"], d["regime"], d.get("deep_dive", {})
    out: list[str] = []
    out.append(f"# Daily Swing Research Report — {s['date']} ({s['weekday']})\n")
    out.append(f"> {d['disclaimer']}\n")
    out.append(f"Generated {d['generated_at']} · next session {s['next_session']}"
               + (" · **early close session**" if s.get("early_close") else "") + "\n")

    # executive summary
    out.append("## 1. Executive summary\n")
    out.append(f"**Market regime: {reg['label']}** (composite {_num(reg['composite'], 0)} on a -100…+100 scale"
               + (f", {round(reg['composite_5d_change']) + 0:+d} vs 5 sessions ago" if reg.get("composite_5d_change") is not None else "")
               + ").\n")
    out.append(f"{reg['guidance']} Suggested risk budget vs normal: longs ×{reg['long_risk_budget']}, "
               f"shorts ×{reg['short_risk_budget']}.\n")
    for side in ("long", "short"):
        top = d["candidates"][side][:5]
        if top:
            out.append(f"**Top {SIDE_NAMES[side].lower()} research candidates:** "
                       + ", ".join(f"{c['symbol']} ({c['setup_name']}, {c['score']:.0f}, tier {c['tier']})" for c in top)
                       + "\n")
    lead = [x for x in d["sectors"] if x["quadrant"] == "Leading"][:4]
    lag = [x for x in d["sectors"] if x["quadrant"] == "Lagging"][:4]
    out.append("**Sector rotation:** leading " + (", ".join(f"{x['symbol']} ({x['name']})" for x in lead) or "none")
               + "; lagging " + (", ".join(f"{x['symbol']} ({x['name']})" for x in lag) or "none") + ".\n")
    if d.get("macro_events"):
        nxt = d["macro_events"][:4]
        out.append("**Upcoming macro events:** " + "; ".join(f"{e['event_date']} {e['event_type']} ({e['status']})"
                                                             for e in nxt) + ".\n")
    if d.get("warnings"):
        out.append("**Run warnings:**\n" + "\n".join(f"- {w}" for w in d["warnings"]) + "\n")

    # regime
    out.append("## 2. Market regime\n")
    out.append(_table(["Component", "Weight", "Score (−2…+2)", "Contribution", "Evidence"],
                      [[c["name"], c["weight"], _num(c["score"], 2), _num(c["contribution"], 1),
                        " ".join(c["evidence"])] for c in reg["components"]]))
    out.append("\nClassification: ≥ +45 Strong Bullish · ≥ +15 Bullish · > −15 Neutral · > −45 Bearish · "
               "else Strong Bearish (Strong labels require SPY on the matching side of its EMA50).\n")
    out.append("### Indices, volatility and credit\n")
    out.append(_table(["Symbol", "Close", "1D", "5D", "20D", "YTD", "vs EMA21", "vs EMA50", "vs EMA200",
                       "From 52w high", "EMA stack"],
                      [[i["symbol"], _num(i["close"]), _pct(i["ret_1d"]), _pct(i["ret_5d"]), _pct(i["ret_20d"]),
                        _pct(i["ret_ytd"]), _pct(i["vs_ema21"]), _pct(i["vs_ema50"]), _pct(i["vs_ema200"]),
                        _pct(i["from_52w_high"]), i["ema_stack"]] for i in d["indices"]]))
    out.append("### Sector rotation (vs SPY)\n")
    out.append(_table(["Rank", "ETF", "Sector", "5D", "20D", "Rel 20D", "Rel 60D", "RS trend", "RS mom.", "Quadrant",
                       ">EMA50"],
                      [[x["rank_20d"], x["symbol"], x["name"], _pct(x["ret_5d"]), _pct(x["ret_20d"]),
                        _pct(x["rel_20d"]), _pct(x["rel_60d"]), _pct(x["rs_trend"]), _pct(x["rs_momentum"]),
                        x["quadrant"], "yes" if x["above_ema50"] else "no"] for x in d["sectors"]]))

    # candidates
    for n, side in ((3, "long"), (4, "short")):
        cands = d["candidates"][side]
        out.append(f"## {n}. {SIDE_NAMES[side]} candidates\n")
        out.append(_table(["#", "Symbol", "Setup", "Score", "Tier", "Ref. entry", "Stop", "T1", "T2", "R/R",
                           "Risk %", "Earnings", "Setup history"],
                          [[c["rank"], c["symbol"], c["setup_name"], f"{c['score']:.0f}", c["tier"],
                            _num(c["plan"]["entry"]) + (" (cond.)" if c["plan"].get("conditional_entry") else ""),
                            _num(c["plan"]["stop"]), _num(c["plan"]["target1"]), _num(c["plan"]["target2"]),
                            _num(c["plan"]["reward_risk"], 1), _pct(c["plan"]["risk_pct"], 1, False),
                            _earn(c, deep), c["history"]["note"]] for c in cands]))
        for c in cands[:8]:
            out.append(candidate_detail(c, deep.get(c["symbol"], {})))

    # catalysts
    out.append("## 5. Catalyst calendar\n")
    out.append("### Upcoming earnings (shortlisted names)\n")
    out.append(_table(["Symbol", "Date", "Timing", "Reaction session", "Sessions away", "EPS est.", "Status"],
                      [[e["symbol"], e["date"], e.get("timing") or "n/a", e["reaction_session"], e["sessions_until"],
                        _num(e.get("eps_estimate")), e["status"]] for e in d.get("upcoming_earnings", [])]))
    out.append("### Macro and market-structure events (next 30 days)\n")
    out.append(_table(["Date", "Event", "Status", "Source"],
                      [[e["event_date"], e["event_type"], e["status"], e["source"]] for e in d.get("macro_events", [])]))
    if d.get("holidays"):
        out.append("Market holidays ahead: " + ", ".join(f"{h['date']} {h['name']}" for h in d["holidays"]) + "\n")
    notable = []
    for sym, dd in deep.items():
        for f in (dd.get("catalysts") or {}).get("filings", []):
            if f["sentiment"] == "Negative" or f.get("dilution_risk") or f.get("item") in ("1.01", "2.01", "8.01"):
                notable.append([sym, f["event_date"], f["form"], f["event_type"], f["sentiment"],
                                f"[link]({f['url']})" if f.get("url") else ""])
    out.append("### Notable recent SEC filings (shortlisted names, 60 days)\n")
    out.append(_table(["Symbol", "Filed", "Form", "Type", "Typical implication", "Filing"], notable[:40]))

    # history
    bt = d["backtest"]
    out.append("## 6. Historical setup statistics (universe-wide simulation)\n")
    out.append(f"Each setup's plan simulated on {bt.get('first_signal', 'n/a')} … {bt.get('last_signal', 'n/a')}: "
               f"next-open entry, stop / T1 / {bt['horizon_days']}-session time stop, {bt['slippage_bps']:.0f} bps "
               "slippage per side. Biases: survivorship (today's universe), in-sample, overlapping correlated "
               "signals, no commissions / borrow costs, daily bars only.\n")
    rows = []
    for key, st in bt.get("by_setup", {}).items():
        if not st.get("n"):
            continue
        name = d["setup_names"].get(key, key)
        rows.append([name, st["n"], _pct(st["win_rate"], 0, False), _num(st["avg_r"]), _num(st["median_r"]),
                     _num(st.get("profit_factor")), _pct(st["hit_target_rate"], 0, False),
                     _pct(st["stop_rate"], 0, False), _num(st["avg_days"], 1), _pct(st.get("fwd_5d_mean"), 2)])
    out.append(_table(["Setup", "N", "Win", "Avg R", "Median R", "PF", "T1 hit", "Stopped", "Avg days",
                       "Fwd 5D (dir.)"], rows))
    base = bt.get("baseline") or {}
    if base:
        out.append(f"Baseline (all stock-days, long): mean 5D return {_pct(base.get('fwd_5d_mean'), 2)}, "
                   f"positive {_pct(base.get('fwd_5d_positive'), 0, False)} of the time (N={base.get('n')}).\n")

    # quality
    q = d["quality"]
    out.append("## 7. Data quality and limitations\n")
    out.append(f"Symbols: {q['counts']} · coverage {q['coverage'] * 100:.0f}% · usable stocks {q['n_usable_stocks']}. "
               f"Update: {len(q['update'].get('incremental', []))} incremental, {len(q['update'].get('full', []))} full, "
               f"{len(q['update'].get('readjusted', []))} re-adjusted, {len(q['update'].get('failed', []))} failed.\n")
    issues = [[r["symbol"], r["status"], " ".join(r["issues"])] for r in q["symbols"] if r["status"] != "ok"]
    out.append(_table(["Symbol", "Status", "Issues"], issues))
    out.append("- Prices: Yahoo Finance daily bars (unofficial API), split/dividend-adjusted.\n"
               "- Fundamentals: SEC XBRL, point-in-time (filed on/before the session).\n"
               "- Options: Yahoo end-of-day chains; OI is prior-day; flow direction is not observable.\n"
               "- Positioning: short interest, 13F and FINRA data are delayed disclosures (dates shown).\n"
               "- Breadth is measured on the screened universe, not the whole exchange.\n")
    out.append("## 8. Methodology\n")
    out.append("Regime, setups, plans and scores follow fixed, documented rules (see README section "
               "'Research terminal'). Composite = Σ weight × sub-score / 100 − penalties; weights: "
               + ", ".join(f"{k} {v}" for k, v in d["settings"]["weights"].items()) + ".\n")
    return "\n".join(out)


def candidate_detail(c: dict[str, Any], deep: dict[str, Any]) -> str:
    p = c["plan"]
    lines = [f"### {c['rank']}. {c['symbol']} — {SIDE_NAMES[c['direction']]} · {c['setup_name']} · "
             f"score {c['score']:.0f} (tier {c['tier']})\n"]
    prof = deep.get("profile") or {}
    if prof.get("longName"):
        lines.append(f"{prof['longName']} · {prof.get('sector') or ''} / {prof.get('industry') or ''} · "
                     f"market cap {_money(prof.get('marketCap'))} · sector ETF {c['sector_etf']} "
                     f"({c.get('sector_quadrant') or 'n/a'})\n")
    lines.append(c["thesis"] + "\n")
    comp = ", ".join(f"{k} {v:.0f}" for k, v in c["components"].items())
    pen = ", ".join(f"{k} −{v:g}" for k, v in c["penalties"].items()) or "none"
    lines.append(f"- **Sub-scores (0-100):** {comp}\n- **Penalties:** {pen} (raw {c['raw_score']:.1f})")
    if p.get("expected_move_hv"):
        lines.append(f"- **Typical {p['horizon_days']}-session move (HV20, 1σ):** ±{_num(p['expected_move_hv'])} "
                     f"(±{p['expected_move_hv'] / p['entry'] * 100:.1f}%)")
    h = c.get("history") or {}
    lines.append(f"- **Setup history:** {h.get('note')} · in a {h.get('regime')} regime: {h.get('note_regime')}")
    if c["supporting"]:
        lines.append("- **Supporting evidence:** " + " · ".join(c["supporting"]))
    if c["contradicting"]:
        lines.append("- **Contradicting evidence / risks:** " + " · ".join(c["contradicting"]))
    f = deep.get("fundamentals") or {}
    if f.get("available"):
        m = f["metrics"]
        est = "Yahoo" in (f.get("source") or "")
        lines.append(f"- **Fundamentals ({'Yahoo fallback' if est else 'SEC XBRL'}, quarter ended "
                     f"{m.get('latest_quarter_end')}, {'est. published' if est else 'filed'} {m.get('latest_filed')}):** "
                     f"revenue YoY {_pct(m.get('revenue_yoy'), 0)}, EPS YoY {_pct(m.get('eps_yoy'), 0)}, "
                     f"gross margin {_pct(m.get('gross_margin_ttm'), 0, False)}, op. margin "
                     f"{_pct(m.get('operating_margin_ttm'), 0, False)}, FCF TTM {_money(m.get('fcf_ttm'))}, "
                     f"net debt {_money(m.get('net_debt'))}, shares YoY {_pct(m.get('shares_yoy'), 1)}, "
                     f"P/E {_num(m.get('pe_ttm'), 1)}, P/S {_num(m.get('ps_ttm'), 1)}")
    elif f:
        lines.append(f"- **Fundamentals:** not available — {f.get('reason')}")
    o = deep.get("options") or {}
    if o.get("available"):
        lines.append(f"- **Options ({o.get('atm_iv_expiry')}):** ATM IV {_pct(o.get('atm_iv'), 0, False)}, "
                     f"IV/HV20 {_num(o.get('iv_hv_ratio'))}, P/C vol {_num(o.get('pcr_volume'))}, P/C OI "
                     f"{_num(o.get('pcr_oi'))}, expected move to {o.get('expected_move_expiry')} "
                     f"±{_pct(o.get('expected_move_pct'), 1, False)}, IV rank "
                     f"{_num(o.get('iv_rank'), 0) if o.get('iv_rank') is not None else 'insufficient history'}"
                     + ("; " + " ".join(o.get("interpretation", [])) if o.get("interpretation") else ""))
    elif o:
        lines.append(f"- **Options:** not available — {o.get('reason')}")
    pos = deep.get("positioning") or {}
    if pos.get("notes"):
        lines.append("- **Positioning:** " + " ".join(pos["notes"]))
    cat = deep.get("catalysts") or {}
    ne = cat.get("next_earnings")
    if ne:
        lines.append(f"- **Next earnings:** {ne['date']} {ne.get('timing') or '(time n/a)'} — {ne['sessions_until']} "
                     f"session(s) away ({ne['status']})")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ minimal Markdown -> HTML
def _inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t


def markdown_to_html(md: str) -> str:
    lines, out, i = md.splitlines(), [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= {"-"}:
            head = [c.strip() for c in line.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            out.append("<div class=tw><table><thead><tr>" + "".join(f"<th>{_inline(h)}</th>" for h in head)
                       + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r)
                                                          + "</tr>" for r in rows) + "</tbody></table></div>")
            continue
        m = re.match(r"^(#{1,4}) (.*)", line)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif line.startswith("> "):
            out.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
        i += 1
    return "\n".join(out)


CSS = """
:root{--bg:#fff;--fg:#1b1f24;--muted:#5b6470;--line:#d8dde3;--head:#f3f5f7;--accent:#0b5cad}
@media (prefers-color-scheme:dark){:root{--bg:#0f1318;--fg:#e3e7ec;--muted:#98a2ad;--line:#2a313a;--head:#171c22;--accent:#6fb0f0}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
max-width:1200px;margin:0 auto;padding:16px}
h1{font-size:22px}h2{font-size:18px;border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:28px}
h3{font-size:15px;margin-top:20px}a{color:var(--accent)}blockquote{margin:8px 0;padding:8px 12px;border-left:3px solid #c98a00;
background:var(--head);color:var(--muted)}.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12.5px;
font-variant-numeric:tabular-nums}th,td{border:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
th{background:var(--head)}li{margin:2px 0}@media print{body{max-width:none}a{color:inherit}}
"""


def build_html(d: dict[str, Any]) -> str:
    body = markdown_to_html(build_markdown(d))
    title = f"Swing Research {d['session']['date']}"
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>{title}</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")

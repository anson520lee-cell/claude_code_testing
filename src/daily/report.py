"""Daily research report: Markdown (single source) and a self-contained, printable HTML version."""

from __future__ import annotations

import html
import re
from typing import Any

from src.i18n.zh import tr

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

    out.append("## 1. Summary\n")
    change = reg.get("composite_5d_change")
    out.append(f"**Market regime: {tr(reg['label'])}** (composite {_num(reg['composite'], 0)} on a −100 to +100 scale"
               + (f", {round(change) + 0:+d} vs 5 sessions ago" if change is not None else "") + ").\n")
    out.append(f"{reg['guidance']} Suggested size vs normal: long ×{reg['long_risk_budget']}, short ×{reg['short_risk_budget']}.\n")
    for side in ("long", "short"):
        top = d["candidates"][side][:5]
        if top:
            out.append(f"**Top {SIDE_NAMES[side].lower()} candidates:** "
                       + ", ".join(f"{c['symbol']} ({c['setup_name']}, {c['score']:.0f}, tier {tr(c['tier'])})" for c in top) + "\n")
    lead = [x for x in d["sectors"] if x["quadrant"] == "Leading"][:4]
    lag = [x for x in d["sectors"] if x["quadrant"] == "Lagging"][:4]
    out.append("**Sector rotation:** leading " + (", ".join(f"{x['symbol']} ({x['name']})" for x in lead) or "none")
               + "; lagging " + (", ".join(f"{x['symbol']} ({x['name']})" for x in lag) or "none") + ".\n")
    if d.get("macro_events"):
        out.append("**Upcoming macro events:** " + "; ".join(f"{e['event_date']} {e['event_type']} ({tr(e['status'])})"
                                                     for e in d["macro_events"][:4]) + ".\n")
    if d.get("warnings"):
        out.append("**Run warnings:**\n" + "\n".join(f"- {w}" for w in d["warnings"]) + "\n")

    out.append("## 2. Market regime\n")
    out.append(_table(["Component", "Weight", "Score (−2 to +2)", "Contribution", "Evidence"],
                      [[tr(c["name"]), c["weight"], _num(c["score"], 2), _num(c["contribution"], 1), " ".join(c["evidence"])]
                       for c in reg["components"]]))
    out.append("\nLabels: ≥ +45 Strong Bullish · ≥ +15 Bullish · > −15 Neutral · > −45 Bearish · else Strong Bearish (Strong labels need SPY on the matching side of EMA50).\n")
    out.append("### Indices, volatility and credit\n")
    out.append(_table(["Symbol", "Close", "1d", "5d", "20d", "YTD", "vs EMA21", "vs EMA50", "vs EMA200",
                       "From 52w high", "EMA stack"],
                      [[i["symbol"], _num(i["close"]), _pct(i["ret_1d"]), _pct(i["ret_5d"]), _pct(i["ret_20d"]),
                        _pct(i["ret_ytd"]), _pct(i["vs_ema21"]), _pct(i["vs_ema50"]), _pct(i["vs_ema200"]),
                        _pct(i["from_52w_high"]), tr(i["ema_stack"])] for i in d["indices"]]))
    out.append("### Sector rotation (vs SPY)\n")
    out.append(_table(["Rank", "ETF", "Sector", "5d", "20d", "Rel 20d", "Rel 60d", "RS trend", "RS momentum", "Quadrant", "Above EMA50"],
                      [[x["rank_20d"], x["symbol"], tr(x["name"]), _pct(x["ret_5d"]), _pct(x["ret_20d"]),
                        _pct(x["rel_20d"]), _pct(x["rel_60d"]), _pct(x["rs_trend"]), _pct(x["rs_momentum"]),
                        x["quadrant"], "yes" if x["above_ema50"] else "no"] for x in d["sectors"]]))
    if d.get("macro"):
        out.append("### Cross-asset (context only)\n")
        out.append(_table(["Asset", "Last", "5d", "20d", "20d z", "Corr. with QQQ (60d)", "Read"],
                          [[m["name"], _num(m.get("close")), _pct(m.get("ret_5d")), _pct(m.get("ret_20d")), _num(m.get("z20"), 1),
                            _num(m.get("corr_qqq_60d")), m.get("comment", "")] for m in d["macro"] if m.get("available")]))
    ad = d.get("advance_decline") or {}
    if ad.get("available"):
        out.append(f"### Advance / decline ({ad['scope']})\n")
        out.append(f"Advancers {ad['advancers']}, decliners {ad['decliners']}, unchanged {ad['unchanged']}; McClellan {ad['mcclellan']:+.0f}; "
                   f"10-day net advances {ad['net_10d']:+d}. {ad.get('comment', '')}\n")
    if d.get("industry_groups"):
        g = d["industry_groups"]
        out.append(f"### Industry-group relative strength (top 10 of {len(g)})\n")
        out.append(_table(["Rank", "Group", "5d change", "Members", "5d", "20d", "Leaders"],
                          [[x["rank"], x["group"], f"{x['rank_change']:+d}", x["n"], _pct(x["ret_5d"]), _pct(x["ret_20d"]),
                            ", ".join(x["leaders"])] for x in g[:10]]))
    if d.get("alerts_recent"):
        out.append("### Price alerts triggered in the last 7 days\n")
        out.append(_table(["Symbol", "Condition", "Level", "Triggered", "Price", "Note"],
                          [[a["symbol"], a["kind"], _num(a["level"]), a["triggered_session"], _num(a["triggered_price"]), a.get("note") or ""]
                           for a in d["alerts_recent"]]))

    for n, side in ((3, "long"), (4, "short")):
        cands = d["candidates"][side]
        out.append(f"## {n}. {SIDE_NAMES[side]} candidates\n")
        out.append(_table(["#", "Symbol", "Setup", "Score", "Tier", "Ref. entry", "Stop", "T1", "T2", "R/R",
                           "Risk %", "Earnings", "Setup history"],
                          [[c["rank"], c["symbol"], c["setup_name"], f"{c['score']:.0f}", tr(c["tier"]),
                            _num(c["plan"]["entry"]) + (" (cond.)" if c["plan"].get("conditional_entry") else ""),
                            _num(c["plan"]["stop"]), _num(c["plan"]["target1"]), _num(c["plan"]["target2"]),
                            _num(c["plan"]["reward_risk"], 1), _pct(c["plan"]["risk_pct"], 1, False),
                            _earn(c, deep), c["history"]["note"]] for c in cands]))
        for c in cands[:8]:
            out.append(candidate_detail(c, deep.get(c["symbol"], {})))

    out.append("## 5. Catalyst calendar\n")
    out.append("### Upcoming earnings (listed names)\n")
    out.append(_table(["Symbol", "Date", "Timing", "Reaction session", "Sessions away", "EPS estimate", "Status"],
                      [[e["symbol"], e["date"], tr(e.get("timing")) or "unconfirmed", e["reaction_session"], e["sessions_until"],
                        _num(e.get("eps_estimate")), tr(e["status"])] for e in d.get("upcoming_earnings", [])]))
    out.append("### Macro and market-structure events (next 30 days)\n")
    out.append(_table(["Date", "Event", "Status", "Source"],
                      [[e["event_date"], e["event_type"], tr(e["status"]), e["source"]] for e in d.get("macro_events", [])]))
    if d.get("holidays"):
        out.append("Market holidays ahead: " + ", ".join(f"{h['date']} {h['name']}" for h in d["holidays"]) + "\n")
    notable = []
    for sym, dd in deep.items():
        for f in (dd.get("catalysts") or {}).get("filings", []):
            if f["sentiment"] == "Negative" or f.get("dilution_risk") or f.get("item") in ("1.01", "2.01", "8.01"):
                notable.append([sym, f["event_date"], f["form"], f["event_type"], tr(f["sentiment"]),
                                f"[link]({f['url']})" if f.get("url") else ""])
    out.append("### Notable recent SEC filings (shortlisted names, 60 days)\n")
    out.append(_table(["Symbol", "Filed", "Form", "Type", "Typical implication", "Filing"], notable[:40]))

    bt = d["backtest"]
    out.append("## 6. Setup history (whole universe)\n")
    out.append(f"Each setup's plan simulated from {bt.get('first_signal', 'n/a')} to {bt.get('last_signal', 'n/a')}: "
               f"entry at the next open, exit at the stop / T1 / {bt['horizon_days']}-day time stop, {bt['slippage_bps']:.0f} bps slippage per side. "
               "Biases: survivorship (today's universe), in-sample, overlapping correlated signals, no commissions or borrow costs, daily bars only.\n")
    rows = []
    for key, st in bt.get("by_setup", {}).items():
        if not st.get("n"):
            continue
        rows.append([d["setup_names"].get(key, tr(key)), st["n"], _pct(st["win_rate"], 0, False), _num(st["avg_r"]),
                     _num(st["median_r"]), _num(st.get("profit_factor")), _pct(st["hit_target_rate"], 0, False),
                     _pct(st["stop_rate"], 0, False), _num(st["avg_days"], 1), _pct(st.get("fwd_5d_mean"), 2)])
    out.append(_table(["Setup", "N", "Win rate", "Avg R", "Median R", "Profit factor", "Hit T1", "Stopped", "Avg days", "5d fwd (signed)"], rows))
    base = bt.get("baseline") or {}
    if base:
        out.append(f"Baseline (all stock-days, long): mean 5D return {_pct(base.get('fwd_5d_mean'), 2)}, "
                   f"positive {_pct(base.get('fwd_5d_positive'), 0, False)} of the time (N={base.get('n')}).\n")

    q = d["quality"]
    out.append("## 7. Data quality and limitations\n")
    out.append(f"Symbols: ok {q['counts']['ok']}, warning {q['counts']['warning']}, excluded {q['counts']['excluded']} · "
               f"coverage {q['coverage'] * 100:.0f}% · usable stocks {q['n_usable_stocks']}. "
               f"Update: incremental {len(q['update'].get('incremental', []))}, full {len(q['update'].get('full', []))}, "
               f"re-adjusted {len(q['update'].get('readjusted', []))}, failed {len(q['update'].get('failed', []))}.\n")
    issues = [[r["symbol"], tr(r["status"]), " ".join(r["issues"])] for r in q["symbols"] if r["status"] != "ok"]
    out.append(_table(["Symbol", "Status", "Issues"], issues))
    uf = d.get("universe_filter") or {}
    if uf.get("excluded"):
        out.append("### Excluded by the high-beta / high-volatility filter\n")
        out.append(_table(["Symbol", "Beta", "60d volatility", "Market cap", "Dollar volume", "Reason"],
                          [[x["symbol"], _num(x.get("beta"), 2), _pct(x.get("hv60"), 0, False), _money(x.get("market_cap")),
                            _money(x.get("dollar_volume20")), x["reason"]] for x in uf["excluded"]]))
    out.append("- Prices: Yahoo Finance daily bars (unofficial API), adjusted for splits and dividends.\n"
               "- Fundamentals: SEC XBRL, only data published on or before the analysis date.\n"
               "- Options: Yahoo end-of-day chains; open interest is the prior session's and flow direction is unknown.\n"
               "- Positioning: short interest, 13F and FINRA data are delayed (dates shown).\n"
               "- Breadth is measured on the universe, not the whole market.\n")
    out.append("## 8. Method\n")
    out.append("Regime, setups, plans and scores follow fixed, documented rules (see the dashboard's Research page). "
               "Score = Σ weight × sub-score / 100 − penalties; weights: "
               + ", ".join(f"{tr(k)} {v}" for k, v in d["settings"]["weights"].items()) + ".\n")
    return "\n".join(out)


def candidate_detail(c: dict[str, Any], deep: dict[str, Any]) -> str:
    p = c["plan"]
    lines = [f"### {c['rank']}. {c['symbol']} — {SIDE_NAMES[c['direction']]} · {c['setup_name']} · "
             f"score {c['score']:.0f} (tier {c['tier']})\n"]
    prof = deep.get("profile") or {}
    if prof.get("longName"):
        lines.append(f"{prof['longName']} · {prof.get('sector') or ''} / {prof.get('industry') or ''} · "
                     f"Market cap {_money(prof.get('marketCap'))} · sector ETF {c['sector_etf']} ({tr(c.get('sector_quadrant')) or 'n/a'})\n")
    a = c.get("analyst")
    if a:
        lines.append(f"> **Analyst call: {a['bias']}** ({a['bias_score']:+.0f}/100, {a['confidence']} confidence) — {a['headline']}\n")
        lines.append(a["summary"] + "\n")
        lines.append(_table(["Scenario", "Trigger", "Target", "Invalidation"], [[x["name"], x["trigger"], x["target"], x["invalidation"]] for x in a["scenarios"]]))
        lines.append("**Scorecard:** " + ", ".join(f"{x['name']} {x['score']:+.1f}" for x in a["scorecard"]) + "\n")
        lines.append("**Watch:** " + "; ".join(a["watch"]) + "\n")
    lines.append(c["thesis"] + "\n")
    comp = ", ".join(f"{tr(k)} {v:.0f}" for k, v in c["components"].items())
    pen = ", ".join(f"{k} −{v:g}" for k, v in c["penalties"].items()) or "none"
    lines.append(f"- **Sub-scores (0-100):** {comp}\n- **Penalties:** {pen} (raw {c['raw_score']:.1f})")
    if p.get("expected_move_hv"):
        lines.append(f"- **Typical {p['horizon_days']}-session move (HV20, 1σ):** ±{_num(p['expected_move_hv'])} "
                     f" (±{p['expected_move_hv'] / p['entry'] * 100:.1f}%)")
    h = c.get("history") or {}
    lines.append(f"- **Setup history:** {h.get('note')} · in a {h.get('regime')} regime: {h.get('note_regime')}")
    if p.get("stop_basis"):
        lines.append(f"- **Why this stop:** {p['stop_basis']}")
        lines.append(f"- **Why these targets:** {p.get('target_basis', '')}")
    if c.get("group"):
        g = c["group"]
        lines.append(f"- **Industry group:** {g['group']}, RS rank {g['rank']} of {g['of']}")
    es = (deep or {}).get("earnings_study") or {}
    if es.get("available"):
        lines.append(f"- **Earnings reactions:** {es.get('note', '')} {es.get('drift_note', '')}")
    ftd = (deep or {}).get("ftd") or {}
    if ftd.get("available") and ftd.get("n"):
        lines.append(f"- **Fails to deliver:** {ftd['note']}")
    if c["supporting"]:
        lines.append("- **Supporting evidence:** " + " · ".join(c["supporting"]))
    if c["contradicting"]:
        lines.append("- **Contradicting evidence / risks:** " + " · ".join(c["contradicting"]))
    f = deep.get("fundamentals") or {}
    if f.get("available"):
        m = f["metrics"]
        est = "Yahoo" in (f.get("source") or "")
        lines.append(f"- **Fundamentals ({'Yahoo fallback' if est else 'SEC XBRL'}, quarter to {m.get('latest_quarter_end')}, "
                     f"{'est. published' if est else 'filed'} {m.get('latest_filed')}):** "
                     f"revenue YoY {_pct(m.get('revenue_yoy'), 0)}, EPS YoY {_pct(m.get('eps_yoy'), 0)}, "
                     f"gross margin {_pct(m.get('gross_margin_ttm'), 0, False)}, operating margin {_pct(m.get('operating_margin_ttm'), 0, False)}, "
                     f"FCF {_money(m.get('fcf_ttm'))}, net debt {_money(m.get('net_debt'))}, shares YoY {_pct(m.get('shares_yoy'), 1)}, "
                     f"P/E {_num(m.get('pe_ttm'), 1)}, P/S {_num(m.get('ps_ttm'), 1)}")
    elif f:
        lines.append(f"- **Fundamentals:** not available — {f.get('reason')}")
    o = deep.get("options") or {}
    if o.get("available"):
        lines.append(f"- **Options ({o.get('atm_iv_expiry')}):** ATM IV {_pct(o.get('atm_iv'), 0, False)}, "
                     f"IV/HV20 {_num(o.get('iv_hv_ratio'))}, put/call volume {_num(o.get('pcr_volume'))}, put/call OI {_num(o.get('pcr_oi'))}, "
                     f"expected move to {o.get('expected_move_expiry')} ±{_pct(o.get('expected_move_pct'), 1, False)}, IV rank "
                     f"{_num(o.get('iv_rank'), 0) if o.get('iv_rank') is not None else 'not enough history'}"
                     + ("; " + " ".join(o.get("interpretation", [])) if o.get("interpretation") else ""))
    elif o:
        lines.append(f"- **Options:** not available — {o.get('reason')}")
    pos = deep.get("positioning") or {}
    if pos.get("notes"):
        lines.append("- **Positioning:** " + " ".join(pos["notes"]))
    ne = (deep.get("catalysts") or {}).get("next_earnings")
    if ne:
        lines.append(f"- **Next earnings:** {ne['date']} {tr(ne.get('timing')) or '(time unconfirmed)'} — in {ne['sessions_until']} sessions ({tr(ne['status'])})")
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
    return (f"<!doctype html><html lang=zh-Hant><head><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>{title}</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")

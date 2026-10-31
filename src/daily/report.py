"""Daily research report: Markdown (single source) and a self-contained, printable HTML version."""

from __future__ import annotations

import html
import re
from typing import Any

from src.i18n.zh import tr

SIDE_NAMES = {"long": "做多", "short": "做空"}


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
        return "n/a" if not c.get("deep_dive") else "未有公布"
    return f"{n['date']}（{n['sessions_until']} 日）"


def build_markdown(d: dict[str, Any]) -> str:
    s, reg, deep = d["session"], d["regime"], d.get("deep_dive", {})
    out: list[str] = []
    out.append(f"# 每日波段研究報告 — {s['date']}（{tr(s['weekday'])}）\n")
    out.append(f"> {d['disclaimer']}\n")
    out.append(f"生成時間 {d['generated_at']} · 下一個交易日 {s['next_session']}"
               + (" · **提早收市日**" if s.get("early_close") else "") + "\n")

    out.append("## 1. 摘要\n")
    change = reg.get("composite_5d_change")
    out.append(f"**市場狀態：{tr(reg['label'])}**（綜合分數 {_num(reg['composite'], 0)}，範圍 −100 至 +100"
               + (f"，較 5 個交易日前 {round(change) + 0:+d}" if change is not None else "") + "）。\n")
    out.append(f"{reg['guidance']} 建議倉位比例（相對正常）：做多 ×{reg['long_risk_budget']}，做空 ×{reg['short_risk_budget']}。\n")
    for side in ("long", "short"):
        top = d["candidates"][side][:5]
        if top:
            out.append(f"**{SIDE_NAMES[side]}研究候選前列：** "
                       + "、".join(f"{c['symbol']}（{c['setup_name']}，{c['score']:.0f} 分，{tr(c['tier'])} 級）" for c in top) + "\n")
    lead = [x for x in d["sectors"] if x["quadrant"] == "Leading"][:4]
    lag = [x for x in d["sectors"] if x["quadrant"] == "Lagging"][:4]
    out.append("**板塊輪動：** 領先 " + ("、".join(f"{x['symbol']}（{tr(x['name'])}）" for x in lead) or "無")
               + "；落後 " + ("、".join(f"{x['symbol']}（{tr(x['name'])}）" for x in lag) or "無") + "。\n")
    if d.get("macro_events"):
        out.append("**即將發生的宏觀事件：** " + "；".join(f"{e['event_date']} {e['event_type']}（{tr(e['status'])}）"
                                                     for e in d["macro_events"][:4]) + "。\n")
    if d.get("warnings"):
        out.append("**運行提示：**\n" + "\n".join(f"- {w}" for w in d["warnings"]) + "\n")

    out.append("## 2. 市場狀態\n")
    out.append(_table(["項目", "權重", "分數（−2 至 +2）", "貢獻", "依據"],
                      [[tr(c["name"]), c["weight"], _num(c["score"], 2), _num(c["contribution"], 1), " ".join(c["evidence"])]
                       for c in reg["components"]]))
    out.append("\n分級：≥ +45 強勢看多 · ≥ +15 看多 · > −15 中性 · > −45 看空 · 其餘為強勢看空（「強勢」級別要求 SPY 位於 EMA50 相應一方）。\n")
    out.append("### 指數、波動率與信貸\n")
    out.append(_table(["代號", "收市", "1 日", "5 日", "20 日", "年初至今", "相對 EMA21", "相對 EMA50", "相對 EMA200",
                       "距 52 週高位", "均線排列"],
                      [[i["symbol"], _num(i["close"]), _pct(i["ret_1d"]), _pct(i["ret_5d"]), _pct(i["ret_20d"]),
                        _pct(i["ret_ytd"]), _pct(i["vs_ema21"]), _pct(i["vs_ema50"]), _pct(i["vs_ema200"]),
                        _pct(i["from_52w_high"]), tr(i["ema_stack"])] for i in d["indices"]]))
    out.append("### 板塊輪動（相對 SPY）\n")
    out.append(_table(["排名", "ETF", "板塊", "5 日", "20 日", "相對 20 日", "相對 60 日", "RS 趨勢", "RS 動量", "象限", "高於 EMA50"],
                      [[x["rank_20d"], x["symbol"], tr(x["name"]), _pct(x["ret_5d"]), _pct(x["ret_20d"]),
                        _pct(x["rel_20d"]), _pct(x["rel_60d"]), _pct(x["rs_trend"]), _pct(x["rs_momentum"]),
                        tr(x["quadrant"]), "是" if x["above_ema50"] else "否"] for x in d["sectors"]]))

    for n, side in ((3, "long"), (4, "short")):
        cands = d["candidates"][side]
        out.append(f"## {n}. {SIDE_NAMES[side]}候選\n")
        out.append(_table(["#", "代號", "形態", "分數", "級別", "參考進場", "止損", "目標一", "目標二", "風險報酬", "風險 %",
                           "財報", "形態歷史"],
                          [[c["rank"], c["symbol"], c["setup_name"], f"{c['score']:.0f}", tr(c["tier"]),
                            _num(c["plan"]["entry"]) + ("（條件）" if c["plan"].get("conditional_entry") else ""),
                            _num(c["plan"]["stop"]), _num(c["plan"]["target1"]), _num(c["plan"]["target2"]),
                            _num(c["plan"]["reward_risk"], 1), _pct(c["plan"]["risk_pct"], 1, False),
                            _earn(c, deep), c["history"]["note"]] for c in cands]))
        for c in cands[:8]:
            out.append(candidate_detail(c, deep.get(c["symbol"], {})))

    out.append("## 5. 催化劑日曆\n")
    out.append("### 即將公布財報（入選股票）\n")
    out.append(_table(["代號", "日期", "時間", "反應交易日", "相距交易日", "預估 EPS", "狀態"],
                      [[e["symbol"], e["date"], tr(e.get("timing")) or "未確認", e["reaction_session"], e["sessions_until"],
                        _num(e.get("eps_estimate")), tr(e["status"])] for e in d.get("upcoming_earnings", [])]))
    out.append("### 宏觀及市場結構事件（未來 30 日）\n")
    out.append(_table(["日期", "事件", "狀態", "來源"],
                      [[e["event_date"], e["event_type"], tr(e["status"]), e["source"]] for e in d.get("macro_events", [])]))
    if d.get("holidays"):
        out.append("即將休市：" + "、".join(f"{h['date']} {tr(h['name'])}" for h in d["holidays"]) + "\n")
    notable = []
    for sym, dd in deep.items():
        for f in (dd.get("catalysts") or {}).get("filings", []):
            if f["sentiment"] == "Negative" or f.get("dilution_risk") or f.get("item") in ("1.01", "2.01", "8.01"):
                notable.append([sym, f["event_date"], f["form"], f["event_type"], tr(f["sentiment"]),
                                f"[連結]({f['url']})" if f.get("url") else ""])
    out.append("### 值得留意的近期 SEC 文件（入選股票，60 日內）\n")
    out.append(_table(["代號", "提交日", "表格", "類型", "通常影響", "文件"], notable[:40]))

    bt = d["backtest"]
    out.append("## 6. 形態歷史回測（全股票池）\n")
    out.append(f"在 {bt.get('first_signal', 'n/a')} 至 {bt.get('last_signal', 'n/a')} 期間模擬每個形態的計劃："
               f"翌日開市進場，按止損／目標一／{bt['horizon_days']} 日時間止損出場，每邊滑點 {bt['slippage_bps']:.0f} 基點。"
               "偏差：倖存者偏差（使用現時股票池）、樣本內、信號重疊且彼此相關、未計佣金及借券成本、只用日線。\n")
    rows = []
    for key, st in bt.get("by_setup", {}).items():
        if not st.get("n"):
            continue
        rows.append([d["setup_names"].get(key, tr(key)), st["n"], _pct(st["win_rate"], 0, False), _num(st["avg_r"]),
                     _num(st["median_r"]), _num(st.get("profit_factor")), _pct(st["hit_target_rate"], 0, False),
                     _pct(st["stop_rate"], 0, False), _num(st["avg_days"], 1), _pct(st.get("fwd_5d_mean"), 2)])
    out.append(_table(["形態", "樣本", "勝率", "平均 R", "中位 R", "盈虧比", "達目標一", "止損", "平均日數", "5 日後（按方向）"], rows))
    base = bt.get("baseline") or {}
    if base:
        out.append(f"基準（所有股票日，做多）：5 日平均回報 {_pct(base.get('fwd_5d_mean'), 2)}，"
                   f"上升機率 {_pct(base.get('fwd_5d_positive'), 0, False)}（樣本 {base.get('n')}）。\n")

    q = d["quality"]
    out.append("## 7. 數據品質與限制\n")
    out.append(f"代號：正常 {q['counts']['ok']}、警告 {q['counts']['warning']}、排除 {q['counts']['excluded']} · "
               f"覆蓋率 {q['coverage'] * 100:.0f}% · 可用股票 {q['n_usable_stocks']}。"
               f"更新：增量 {len(q['update'].get('incremental', []))}、完整 {len(q['update'].get('full', []))}、"
               f"重新調整 {len(q['update'].get('readjusted', []))}、失敗 {len(q['update'].get('failed', []))}。\n")
    issues = [[r["symbol"], tr(r["status"]), " ".join(r["issues"])] for r in q["symbols"] if r["status"] != "ok"]
    out.append(_table(["代號", "狀態", "問題"], issues))
    uf = d.get("universe_filter") or {}
    if uf.get("excluded"):
        out.append("### 被高 Beta／高波動篩選排除的股票\n")
        out.append(_table(["代號", "Beta", "60 日波動率", "市值", "日均成交額", "原因"],
                          [[x["symbol"], _num(x.get("beta"), 2), _pct(x.get("hv60"), 0, False), _money(x.get("market_cap")),
                            _money(x.get("dollar_volume20")), x["reason"]] for x in uf["excluded"]]))
    out.append("- 價格：Yahoo Finance 日線（非官方接口），已按拆股和派息調整。\n"
               "- 基本面：SEC XBRL，只用分析日或之前已公布的數據。\n"
               "- 期權：Yahoo 收市期權鏈；未平倉合約為上一交易日數字，無法得知資金流方向。\n"
               "- 持倉：做空數據、13F 和 FINRA 都是延遲披露（已標示日期）。\n"
               "- 市場寬度以股票池計算，並非全市場。\n")
    out.append("## 8. 方法\n")
    out.append("市場狀態、形態、交易計劃和評分都按固定並有文件記錄的規則計算（見網頁「研究」頁）。"
               "綜合分數 = Σ 權重 × 分項分數 / 100 − 扣分；權重："
               + "、".join(f"{tr(k)} {v}" for k, v in d["settings"]["weights"].items()) + "。\n")
    return "\n".join(out)


def candidate_detail(c: dict[str, Any], deep: dict[str, Any]) -> str:
    p = c["plan"]
    lines = [f"### {c['rank']}. {c['symbol']} — {SIDE_NAMES[c['direction']]} · {c['setup_name']} · "
             f"{c['score']:.0f} 分（{tr(c['tier'])} 級）\n"]
    prof = deep.get("profile") or {}
    if prof.get("longName"):
        lines.append(f"{prof['longName']} · {prof.get('sector') or ''} / {prof.get('industry') or ''} · "
                     f"市值 {_money(prof.get('marketCap'))} · 板塊 ETF {c['sector_etf']}（{tr(c.get('sector_quadrant')) or 'n/a'}）\n")
    a = c.get("analyst")
    if a:
        lines.append(f"> **分析師判斷：{a['bias']}**（{a['bias_score']:+.0f}/100，信心{a['confidence']}）— {a['headline']}\n")
        lines.append(a["summary"] + "\n")
        lines.append(_table(["情景", "條件", "目標", "失效"], [[x["name"], x["trigger"], x["target"], x["invalidation"]] for x in a["scenarios"]]))
        lines.append("**評分卡：** " + "、".join(f"{x['name']} {x['score']:+.1f}" for x in a["scorecard"]) + "\n")
        lines.append("**留意：** " + "；".join(a["watch"]) + "\n")
    lines.append(c["thesis"] + "\n")
    comp = "、".join(f"{tr(k)} {v:.0f}" for k, v in c["components"].items())
    pen = "、".join(f"{tr(k)} −{v:g}" for k, v in c["penalties"].items()) or "無"
    lines.append(f"- **分項分數（0-100）：** {comp}\n- **扣分：** {pen}（原始分數 {c['raw_score']:.1f}）")
    if p.get("expected_move_hv"):
        lines.append(f"- **{p['horizon_days']} 日典型波幅（HV20，1 個標準差）：** ±{_num(p['expected_move_hv'])} "
                     f"（±{p['expected_move_hv'] / p['entry'] * 100:.1f}%）")
    h = c.get("history") or {}
    lines.append(f"- **形態歷史：** {h.get('note')} · 在{tr(h.get('regime'))}市況下：{h.get('note_regime')}")
    if c["supporting"]:
        lines.append("- **支持證據：** " + " · ".join(c["supporting"]))
    if c["contradicting"]:
        lines.append("- **反對證據／風險：** " + " · ".join(c["contradicting"]))
    f = deep.get("fundamentals") or {}
    if f.get("available"):
        m = f["metrics"]
        est = "Yahoo" in (f.get("source") or "")
        lines.append(f"- **基本面（{'Yahoo 後備' if est else 'SEC XBRL'}，季度截至 {m.get('latest_quarter_end')}，"
                     f"{'估計公布' if est else '提交'} {m.get('latest_filed')}）：** "
                     f"營收按年 {_pct(m.get('revenue_yoy'), 0)}，EPS 按年 {_pct(m.get('eps_yoy'), 0)}，"
                     f"毛利率 {_pct(m.get('gross_margin_ttm'), 0, False)}，經營利潤率 {_pct(m.get('operating_margin_ttm'), 0, False)}，"
                     f"自由現金流 {_money(m.get('fcf_ttm'))}，淨負債 {_money(m.get('net_debt'))}，股數按年 {_pct(m.get('shares_yoy'), 1)}，"
                     f"市盈率 {_num(m.get('pe_ttm'), 1)}，市銷率 {_num(m.get('ps_ttm'), 1)}")
    elif f:
        lines.append(f"- **基本面：** 不可用 — {f.get('reason')}")
    o = deep.get("options") or {}
    if o.get("available"):
        lines.append(f"- **期權（{o.get('atm_iv_expiry')}）：** 平價隱含波動率 {_pct(o.get('atm_iv'), 0, False)}，"
                     f"IV/HV20 {_num(o.get('iv_hv_ratio'))}，認沽認購成交比 {_num(o.get('pcr_volume'))}，未平倉比 {_num(o.get('pcr_oi'))}，"
                     f"到 {o.get('expected_move_expiry')} 預期波幅 ±{_pct(o.get('expected_move_pct'), 1, False)}，IV Rank "
                     f"{_num(o.get('iv_rank'), 0) if o.get('iv_rank') is not None else '歷史不足'}"
                     + ("；" + " ".join(o.get("interpretation", [])) if o.get("interpretation") else ""))
    elif o:
        lines.append(f"- **期權：** 不可用 — {o.get('reason')}")
    pos = deep.get("positioning") or {}
    if pos.get("notes"):
        lines.append("- **持倉：** " + " ".join(pos["notes"]))
    ne = (deep.get("catalysts") or {}).get("next_earnings")
    if ne:
        lines.append(f"- **下次財報：** {ne['date']} {tr(ne.get('timing')) or '（時間未確認）'} — {ne['sessions_until']} 個交易日後（{tr(ne['status'])}）")
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
    title = f"波段研究報告 {d['session']['date']}"
    return (f"<!doctype html><html lang=zh-Hant><head><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>{title}</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")

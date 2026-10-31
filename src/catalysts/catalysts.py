"""Catalyst intelligence: earnings, SEC filings, analyst actions and macro events.

Every catalyst record has:
    company, event_type, event_date, published_at, source, sentiment
    (Positive / Negative / Uncertain), status (Confirmed / Estimated / Speculative),
    description and a link where one exists.

Classification rules are explicit (``FORM_RULES``, ``ITEM_RULES``). Sentiment describes the
TYPICAL implication of an event type for existing shareholders - e.g. a shelf registration
(S-3) is "Negative" because it enables future share sales - not a judgement of the specific
document, which the tool does not read. "Uncertain" is used whenever the direction depends on
content (e.g. 8-K Item 1.01 material agreements, Item 5.02 officer changes).

FDA decisions, government contracts, product launches and M&A are only visible when the
company files them (8-K Items 1.01 / 2.01 / 7.01 / 8.01); there is no free structured FDA /
contract calendar, so they appear as "8-K: other events - read filing" rather than being guessed.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.i18n.zh import tr
from src.market import calendar

logger = logging.getLogger(__name__)

FORM_RULES: dict[str, tuple[str, str, bool]] = {  # form -> (event type, sentiment, dilution risk)
    "10-Q": ("季度報告（10-Q）", "Uncertain", False),
    "10-K": ("年度報告（10-K）", "Uncertain", False),
    "10-Q/A": ("修訂季度報告", "Negative", False),
    "10-K/A": ("修訂年度報告", "Negative", False),
    "S-3": ("儲架註冊（S-3）：可在日後發行股票／債券", "Negative", True),
    "S-3ASR": ("自動儲架註冊（S-3ASR）", "Negative", True),
    "S-1": ("註冊聲明（S-1）：發行股票", "Negative", True),
    "F-3": ("儲架註冊（F-3）", "Negative", True),
    "S-8": ("員工股權計劃註冊（S-8）", "Uncertain", False),
    "424B5": ("招股補充文件（424B5）：發行進行中（可能是按市價發行 ATM）", "Negative", True),
    "424B3": ("招股文件（424B3）：轉售／發行", "Negative", True),
    "424B7": ("招股文件（424B7）：股東配售／二次發行", "Negative", True),
    "424B4": ("最終招股文件（424B4）：已定價發行", "Negative", True),
    "424B2": ("招股文件（424B2）：票據／結構性產品", "Uncertain", False),
    "SC 13D": ("持股超過 5% 並有意影響公司（13D，激進投資者）", "Uncertain", False),
    "SC 13D/A": ("13D 修訂", "Uncertain", False),
    "SC 13G": ("被動持股超過 5%（13G）", "Uncertain", False),
    "SC TO-T": ("第三方收購要約", "Positive", False),
    "SC 14D9": ("公司對收購要約的建議", "Uncertain", False),
    "DEFM14A": ("合併委託書", "Uncertain", False),
    "PREM14A": ("初步合併委託書", "Uncertain", False),
    "144": ("內部人士擬出售股份通知（Form 144）", "Negative", False),
    "25-NSE": ("除牌通知", "Negative", False),
    "NT 10-Q": ("延遲提交通知（10-Q）", "Negative", False),
    "NT 10-K": ("延遲提交通知（10-K）", "Negative", False),
}
ITEM_RULES: dict[str, tuple[str, str, bool]] = {  # 8-K item -> (event type, sentiment, dilution risk)
    "1.01": ("簽訂重大協議（合約／交易／融資）", "Uncertain", False),
    "1.02": ("終止重大協議", "Negative", False),
    "1.03": ("破產或接管", "Negative", False),
    "1.05": ("重大網絡安全事件", "Negative", False),
    "2.01": ("完成收購或出售資產", "Uncertain", False),
    "2.02": ("業績公布（財報新聞稿）", "Uncertain", False),
    "2.03": ("新增債務", "Uncertain", False),
    "2.04": ("觸發債務加速償還", "Negative", False),
    "2.05": ("重組／退出成本", "Negative", False),
    "2.06": ("重大減值", "Negative", False),
    "3.01": ("除牌／不符上市規則通知", "Negative", False),
    "3.02": ("未註冊股份發行（私人配售／可換股／認股證）", "Negative", True),
    "3.03": ("股東權利重大修改", "Uncertain", False),
    "4.01": ("更換核數師", "Negative", False),
    "4.02": ("過往財務報表不可依賴（需重述）", "Negative", False),
    "5.01": ("控制權變更", "Uncertain", False),
    "5.02": ("董事／高管離任或委任", "Uncertain", False),
    "5.03": ("修改公司章程／細則", "Uncertain", False),
    "5.07": ("股東投票結果", "Uncertain", False),
    "7.01": ("公平披露（簡報／業務更新）", "Uncertain", False),
    "8.01": ("其他事項，請閱讀文件（可能涉及 FDA、合約、產品消息）", "Uncertain", False),
    "9.01": ("財務報表及附件", "Uncertain", False),
}
IGNORED_FORMS = {"4", "4/A", "3", "5", "SD", "11-K", "IRANNOTICE", "CORRESP", "UPLOAD", "FWP", "S-8 POS",
                 "DEFA14A", "ARS", "8-A12B", "CERT"}

POSITIVE_GRADES = ("buy", "outperform", "overweight", "strong buy", "positive", "accumulate", "add")
NEGATIVE_GRADES = ("sell", "underperform", "underweight", "reduce", "negative", "strong sell")


def _age(as_of: date, day: str | None) -> int | None:
    try:
        return (as_of - date.fromisoformat(str(day)[:10])).days
    except ValueError:
        return None


def classify_filings(filings: list[dict[str, Any]], as_of: date, ticker: str, days: int = 60) -> list[dict[str, Any]]:
    out = []
    for f in filings:
        form, filed = f.get("form") or "", f.get("filingDate")
        age = _age(as_of, filed)
        if age is None or age < 0 or age > days or form in IGNORED_FORMS:
            continue
        accepted = f.get("acceptanceDateTime")
        base = {"company": ticker, "form": form, "event_date": filed, "published_at": accepted,
                "source": "SEC EDGAR", "status": "Confirmed", "url": f.get("url"), "age_days": age}
        if form in ("8-K", "8-K/A"):
            items = [i.strip() for i in str(f.get("items") or "").split(",") if i.strip()]
            relevant = [i for i in items if i not in ("9.01",)] or items
            for item in relevant:
                etype, sentiment, dilution = ITEM_RULES.get(item, (f"8-K 第 {item} 項", "Uncertain", False))
                out.append({**base, "event_type": f"8-K {item}: {etype}", "sentiment": sentiment,
                            "dilution_risk": dilution, "item": item})
            if not relevant:
                out.append({**base, "event_type": "8-K（未列出項目）", "sentiment": "Uncertain",
                            "dilution_risk": False})
        elif form in FORM_RULES:
            etype, sentiment, dilution = FORM_RULES[form]
            out.append({**base, "event_type": etype, "sentiment": sentiment, "dilution_risk": dilution})
        elif form.startswith("424B"):
            out.append({**base, "event_type": f"招股文件（{form}）", "sentiment": "Uncertain", "dilution_risk": False})
    return out


def earnings_info(dates: pd.DataFrame | None, cal: dict[str, Any] | None, session: date) -> dict[str, Any]:
    """Next earnings date (with timing) and the last 4 reported surprises."""
    out: dict[str, Any] = {"next": None, "history": []}
    if dates is not None and not dates.empty:
        frame = dates.copy()
        idx = pd.DatetimeIndex(frame.index)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        frame.index = idx
        frame = frame.sort_index()
        future = frame[frame.index.normalize() > pd.Timestamp(session)]
        if not future.empty:
            ts = future.index[0]
            out["next"] = _earnings_event(ts, session, "Yahoo Finance 財報日曆",
                                          estimate=_num(future.iloc[0].get("EPS Estimate")))
        past = frame[frame.index.normalize() <= pd.Timestamp(session)].tail(4)
        for ts, r in past.iloc[::-1].iterrows():
            out["history"].append({"date": ts.strftime("%Y-%m-%d"), "eps_estimate": _num(r.get("EPS Estimate")),
                                   "eps_reported": _num(r.get("Reported EPS")), "surprise_pct": _num(r.get("Surprise(%)"))})
    if out["next"] is None and cal:
        dates_list = cal.get("Earnings Date") or []
        future = [d for d in dates_list if isinstance(d, (date, datetime)) and pd.Timestamp(d).date() > session]
        if future:
            out["next"] = _earnings_event(pd.Timestamp(future[0]), session, "Yahoo Finance 日曆（只有日期）",
                                          estimate=_num(cal.get("Earnings Average")))
    surprises = [h["surprise_pct"] for h in out["history"] if h["surprise_pct"] is not None]
    out["beats_last4"] = sum(s > 0 for s in surprises) if surprises else None
    out["avg_surprise_pct"] = float(np.mean(surprises)) if surprises else None
    return out


def _earnings_event(ts: pd.Timestamp, session: date, source: str, estimate: float | None) -> dict[str, Any]:
    """Timing from the timestamp. Releases happen before the open or after the close; Yahoo uses
    placeholder intraday times (e.g. 15:00) and midnight for UNCONFIRMED dates, so those are treated
    as "time not confirmed" and the earliest possible reaction session (the date itself) is used -
    the conservative choice for event risk."""
    minutes = ts.hour * 60 + ts.minute
    if minutes == 0 or 9 * 60 + 30 <= minutes < 16 * 60:
        timing = None
    elif minutes < 9 * 60 + 30:
        timing = "before open"
    else:
        timing = "after close"
    day = ts.date()
    reaction = calendar.next_trading_day(day) if timing == "after close" else day
    if not calendar.is_trading_day(reaction):
        reaction = calendar.next_trading_day(reaction)
    confirmed = timing is not None
    return {"date": day.isoformat(), "timing": timing, "reaction_session": reaction.isoformat(),
            "sessions_until": calendar.trading_days_between(session, reaction), "eps_estimate": estimate,
            "source": source, "status": "Confirmed" if confirmed else "Estimated",
            "note": None if confirmed else "數據商未確認公布時間，日期仍可能改變。"}


def analyst_actions(table: pd.DataFrame | None, as_of: date, days: int = 30) -> list[dict[str, Any]]:
    if table is None or table.empty:
        return []
    t = table.reset_index()
    t["GradeDate"] = pd.to_datetime(t["GradeDate"], errors="coerce")
    t = t[(t["GradeDate"] <= pd.Timestamp(as_of) + pd.Timedelta(days=1))
          & (t["GradeDate"] >= pd.Timestamp(as_of) - pd.Timedelta(days=days))]
    out = []
    for _, r in t.sort_values("GradeDate", ascending=False).head(10).iterrows():
        action = str(r.get("Action") or "").lower()
        grade = str(r.get("ToGrade") or "")
        g = grade.lower()
        if action == "up":
            sentiment = "Positive"
        elif action == "down":
            sentiment = "Negative"
        elif any(p in g for p in NEGATIVE_GRADES):
            sentiment = "Negative"
        elif any(p in g for p in POSITIVE_GRADES):
            sentiment = "Positive" if action in ("init", "reit", "main") and r.get("priceTargetAction") != "Lowers" \
                else "Uncertain"
        else:
            sentiment = "Uncertain"
        pt, prior = _num(r.get("currentPriceTarget")), _num(r.get("priorPriceTarget"))
        verb = {"up": "上調評級", "down": "下調評級", "init": "首次覆蓋", "main": "維持評級", "reit": "重申評級"}.get(action, "評級變動")
        desc = f"{r.get('Firm')}：{verb} {r.get('FromGrade') or ''} → {grade}".replace("  ", " ")
        if pt:
            desc += f"，目標價 {prior:.0f} → {pt:.0f}" if prior else f"，目標價 {pt:.0f}"
        out.append({"date": r["GradeDate"].strftime("%Y-%m-%d"), "firm": r.get("Firm"), "action": action,
                    "to_grade": grade, "price_target": pt, "prior_price_target": prior, "description": desc,
                    "sentiment": sentiment, "status": "Confirmed", "source": "Yahoo Finance（分析師評級）",
                    "age_days": _age(as_of, r["GradeDate"].strftime("%Y-%m-%d"))})
    return out


def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# ----------------------------------------------------------------------- macro calendar
def macro_events(session: date, configured: list[dict[str, Any]] | None, days: int = 30) -> list[dict[str, Any]]:
    """Configured macro events (FOMC, CPI ... from config/macro_events.yaml) + rule-based ones:
    monthly options expiration (3rd Friday; quarterly 'triple witching' in Mar/Jun/Sep/Dec) and the
    US employment report (usually the first Friday - marked Estimated)."""
    end = session + timedelta(days=days)
    out = []
    for e in configured or []:
        d = date.fromisoformat(str(e["date"]))
        if session < d <= end:
            out.append({"company": "MARKET", "event_type": e["event"], "event_date": d.isoformat(),
                        "published_at": None, "source": e.get("source", "config/macro_events.yaml"),
                        "sentiment": "Uncertain", "status": e.get("status", "Confirmed"),
                        "description": e.get("note", ""), "importance": e.get("importance", "high")})
    month = date(session.year, session.month, 1)
    for _ in range(3):
        third_friday = month + timedelta(days=(4 - month.weekday()) % 7 + 14)
        expiry = third_friday if calendar.is_trading_day(third_friday) else calendar.previous_trading_day(third_friday)
        if session < expiry <= end:
            quarterly = expiry.month in (3, 6, 9, 12)
            out.append({"company": "MARKET", "event_type": "季度期權到期日（三巫日）"
                        if quarterly else "每月期權到期日", "event_date": expiry.isoformat(),
                        "published_at": None, "source": "交易所規則（每月第三個星期五）", "sentiment": "Uncertain",
                        "status": "Confirmed", "description": "大量未平倉合約的行權價附近可能出現價格黏附和成交量急增。",
                        "importance": "medium"})
        first_friday = month + timedelta(days=(4 - month.weekday()) % 7)
        if session < first_friday <= end and not any(e["event_type"].startswith("美國非農") for e in out
                                                       if e["event_date"][:7] == first_friday.isoformat()[:7]):
            out.append({"company": "MARKET", "event_type": "美國非農就業報告",
                        "event_date": first_friday.isoformat(), "published_at": None,
                        "source": "經驗規則（每月第一個星期五），請到 bls.gov 核實", "sentiment": "Uncertain",
                        "status": "Estimated", "description": "美東 08:30 公布；部分月份日期會調整。",
                        "importance": "high"})
        month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
    return sorted(out, key=lambda e: e["event_date"])


def build_catalysts(ticker: str, session: date, earnings_dates: pd.DataFrame | None, cal: dict | None,
                    filings: list[dict[str, Any]], upgrades: pd.DataFrame | None) -> dict[str, Any]:
    earnings = earnings_info(earnings_dates, cal, session)
    filed = classify_filings(filings, session, ticker)
    analyst = analyst_actions(upgrades, session)
    events = []
    if earnings["next"]:
        n = earnings["next"]
        events.append({"company": ticker, "event_type": "財報公布", "event_date": n["date"],
                       "published_at": None, "source": n["source"], "sentiment": "Uncertain", "status": n["status"],
                       "description": f"{tr(n.get('timing')) or '時間未確認'}；反應交易日 {n['reaction_session']}"
                                      + (f"；預估 EPS {n['eps_estimate']:.2f}" if n.get("eps_estimate") else ""),
                       "upcoming": True})
    for f in filed:
        events.append({**f, "description": f["event_type"], "upcoming": False})
    for a in analyst:
        events.append({"company": ticker, "event_type": "分析師評級／目標價變動", "event_date": a["date"],
                       "published_at": a["date"], "source": a["source"], "sentiment": a["sentiment"],
                       "status": "Confirmed", "description": a["description"], "upcoming": False})
    latest_results = next((f for f in filed if f.get("item") == "2.02"), None)
    return {"next_earnings": earnings["next"], "earnings_history": earnings["history"],
            "beats_last4": earnings["beats_last4"], "avg_surprise_pct": earnings["avg_surprise_pct"],
            "filings": filed, "analyst": analyst, "events": events, "latest_results_filing": latest_results,
            "limitations": ["FDA 決定、合約和產品發布只有在公司向 SEC 提交 8-K 時才會出現，沒有免費的結構化日曆。",
                            "「通常影響」只按文件類型判斷，系統沒有閱讀文件內容，請打開連結查看。"]}

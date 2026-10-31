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

from src.market import calendar

logger = logging.getLogger(__name__)

FORM_RULES: dict[str, tuple[str, str, bool]] = {  # form -> (event type, sentiment, dilution risk)
    "10-Q": ("Quarterly report (10-Q)", "Uncertain", False),
    "10-K": ("Annual report (10-K)", "Uncertain", False),
    "10-Q/A": ("Amended quarterly report", "Negative", False),
    "10-K/A": ("Amended annual report", "Negative", False),
    "S-3": ("Shelf registration (S-3) - enables future share / debt sales", "Negative", True),
    "S-3ASR": ("Automatic shelf registration (S-3ASR)", "Negative", True),
    "S-1": ("Registration statement (S-1) - share offering", "Negative", True),
    "F-3": ("Shelf registration (F-3)", "Negative", True),
    "S-8": ("Employee stock plan registration (S-8)", "Uncertain", False),
    "424B5": ("Prospectus supplement (424B5) - offering in progress (may be an ATM program)", "Negative", True),
    "424B3": ("Prospectus (424B3) - resale / offering", "Negative", True),
    "424B7": ("Prospectus (424B7) - secondary / selling-holder offering", "Negative", True),
    "424B4": ("Final prospectus (424B4) - priced offering", "Negative", True),
    "424B2": ("Prospectus (424B2) - notes / structured offering", "Uncertain", False),
    "SC 13D": ("Activist / >5% holder with control intent (13D)", "Uncertain", False),
    "SC 13D/A": ("13D amendment", "Uncertain", False),
    "SC 13G": ("Passive >5% holder (13G)", "Uncertain", False),
    "SC TO-T": ("Third-party tender offer", "Positive", False),
    "SC 14D9": ("Recommendation on a tender offer", "Uncertain", False),
    "DEFM14A": ("Merger proxy", "Uncertain", False),
    "PREM14A": ("Preliminary merger proxy", "Uncertain", False),
    "144": ("Notice of proposed insider sale (Form 144)", "Negative", False),
    "25-NSE": ("Delisting notification", "Negative", False),
    "NT 10-Q": ("Late filing notice (10-Q)", "Negative", False),
    "NT 10-K": ("Late filing notice (10-K)", "Negative", False),
}
ITEM_RULES: dict[str, tuple[str, str, bool]] = {  # 8-K item -> (event type, sentiment, dilution risk)
    "1.01": ("Material definitive agreement (contract / deal / financing)", "Uncertain", False),
    "1.02": ("Termination of a material agreement", "Negative", False),
    "1.03": ("Bankruptcy or receivership", "Negative", False),
    "1.05": ("Material cybersecurity incident", "Negative", False),
    "2.01": ("Completed acquisition or disposition", "Uncertain", False),
    "2.02": ("Results of operations (earnings release)", "Uncertain", False),
    "2.03": ("New debt obligation", "Uncertain", False),
    "2.04": ("Debt acceleration trigger", "Negative", False),
    "2.05": ("Restructuring / exit costs", "Negative", False),
    "2.06": ("Material impairment", "Negative", False),
    "3.01": ("Delisting / listing-standard notice", "Negative", False),
    "3.02": ("Unregistered sale of equity (private placement / convertible / warrants)", "Negative", True),
    "3.03": ("Material modification of shareholder rights", "Uncertain", False),
    "4.01": ("Change of auditor", "Negative", False),
    "4.02": ("Non-reliance on prior financial statements (restatement)", "Negative", False),
    "5.01": ("Change in control", "Uncertain", False),
    "5.02": ("Director / officer departure or appointment", "Uncertain", False),
    "5.03": ("Charter / bylaw amendment", "Uncertain", False),
    "5.07": ("Shareholder vote results", "Uncertain", False),
    "7.01": ("Regulation FD disclosure (presentation / update)", "Uncertain", False),
    "8.01": ("Other events - read filing (may include FDA, contracts, product news)", "Uncertain", False),
    "9.01": ("Financial statements and exhibits", "Uncertain", False),
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
                etype, sentiment, dilution = ITEM_RULES.get(item, (f"8-K item {item}", "Uncertain", False))
                out.append({**base, "event_type": f"8-K {item}: {etype}", "sentiment": sentiment,
                            "dilution_risk": dilution, "item": item})
            if not relevant:
                out.append({**base, "event_type": "8-K (no items listed)", "sentiment": "Uncertain",
                            "dilution_risk": False})
        elif form in FORM_RULES:
            etype, sentiment, dilution = FORM_RULES[form]
            out.append({**base, "event_type": etype, "sentiment": sentiment, "dilution_risk": dilution})
        elif form.startswith("424B"):
            out.append({**base, "event_type": f"Prospectus ({form})", "sentiment": "Uncertain", "dilution_risk": False})
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
            out["next"] = _earnings_event(ts, session, "Yahoo Finance earnings calendar",
                                          estimate=_num(future.iloc[0].get("EPS Estimate")))
        past = frame[frame.index.normalize() <= pd.Timestamp(session)].tail(4)
        for ts, r in past.iloc[::-1].iterrows():
            out["history"].append({"date": ts.strftime("%Y-%m-%d"), "eps_estimate": _num(r.get("EPS Estimate")),
                                   "eps_reported": _num(r.get("Reported EPS")), "surprise_pct": _num(r.get("Surprise(%)"))})
    if out["next"] is None and cal:
        dates_list = cal.get("Earnings Date") or []
        future = [d for d in dates_list if isinstance(d, (date, datetime)) and pd.Timestamp(d).date() > session]
        if future:
            out["next"] = _earnings_event(pd.Timestamp(future[0]), session, "Yahoo Finance calendar (date only)",
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
            "note": None if confirmed else "Release time not confirmed by the provider; the date may still change."}


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
        desc = f"{r.get('Firm')}: {action or 'action'} {r.get('FromGrade') or ''} -> {grade}".replace("  ", " ")
        if pt:
            desc += f", PT {prior:.0f} -> {pt:.0f}" if prior else f", PT {pt:.0f}"
        out.append({"date": r["GradeDate"].strftime("%Y-%m-%d"), "firm": r.get("Firm"), "action": action,
                    "to_grade": grade, "price_target": pt, "prior_price_target": prior, "description": desc,
                    "sentiment": sentiment, "status": "Confirmed", "source": "Yahoo Finance (analyst actions)",
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
            out.append({"company": "MARKET", "event_type": "Quarterly options expiration (triple witching)"
                        if quarterly else "Monthly options expiration", "event_date": expiry.isoformat(),
                        "published_at": None, "source": "Exchange rule (3rd Friday)", "sentiment": "Uncertain",
                        "status": "Confirmed", "description": "Pin risk / volume spike around large open-interest strikes.",
                        "importance": "medium"})
        first_friday = month + timedelta(days=(4 - month.weekday()) % 7)
        if session < first_friday <= end and not any(e["event_type"].startswith("Employment") for e in out
                                                       if e["event_date"][:7] == first_friday.isoformat()[:7]):
            out.append({"company": "MARKET", "event_type": "Employment report (nonfarm payrolls)",
                        "event_date": first_friday.isoformat(), "published_at": None,
                        "source": "Rule of thumb (first Friday) - verify at bls.gov", "sentiment": "Uncertain",
                        "status": "Estimated", "description": "08:30 ET; date shifts in some months.",
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
        events.append({"company": ticker, "event_type": "Earnings release", "event_date": n["date"],
                       "published_at": None, "source": n["source"], "sentiment": "Uncertain", "status": n["status"],
                       "description": f"{n.get('timing') or 'time n/a'}; reaction session {n['reaction_session']}"
                                      + (f"; EPS est. {n['eps_estimate']:.2f}" if n.get("eps_estimate") else ""),
                       "upcoming": True})
    for f in filed:
        events.append({**f, "description": f["event_type"], "upcoming": False})
    for a in analyst:
        events.append({"company": ticker, "event_type": "Analyst rating / target change", "event_date": a["date"],
                       "published_at": a["date"], "source": a["source"], "sentiment": a["sentiment"],
                       "status": "Confirmed", "description": a["description"], "upcoming": False})
    latest_results = next((f for f in filed if f.get("item") == "2.02"), None)
    return {"next_earnings": earnings["next"], "earnings_history": earnings["history"],
            "beats_last4": earnings["beats_last4"], "avg_surprise_pct": earnings["avg_surprise_pct"],
            "filings": filed, "analyst": analyst, "events": events, "latest_results_filing": latest_results,
            "limitations": ["FDA decisions, contracts and product launches are only captured when filed with the "
                            "SEC (8-K); no free structured calendar exists for them.",
                            "Sentiment labels describe the typical implication of the filing TYPE; the documents "
                            "themselves are not read - open the link."]}

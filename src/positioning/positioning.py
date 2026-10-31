"""Market positioning and ownership - every figure carries its as-of date and reporting delay.

Sources and their delays:
* Short interest, days to cover, float, shares outstanding, % held by institutions / insiders:
  Yahoo Finance key statistics. Short interest is the FINRA bi-monthly settlement-date figure,
  published ~8 business days after settlement (so it is 1-4 weeks old when you see it).
* Institutional holders: 13F filings, due 45 days after quarter end - positions can be up to
  ~4.5 months old and show long equity positions only (no shorts, no options hedges).
* Insider transactions: SEC Form 4 (due within 2 business days of the trade), via Yahoo.
* Daily short-sale VOLUME: FINRA Reg SHO consolidated file (CNMS), next morning. This counts
  short-sale-flagged trades, most of which are market makers providing liquidity - it is NOT
  short interest and is ~40-50% for most stocks. Only its change vs its own average is informative.
* Off-exchange volume: FINRA OTC transparency weekly data (ATS + non-ATS), published 2-4 weeks
  after the week ends. Off-exchange share = off-exchange shares / consolidated volume that week.
* Block trades: no free, reliable public source - shown as not available.
"""

from __future__ import annotations

import io
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

from src.market import calendar

logger = logging.getLogger(__name__)

REGSHO_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{day:%Y%m%d}.txt"
OTC_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"


def _ts_to_date(value: Any) -> str | None:
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def from_info(info: dict[str, Any]) -> dict[str, Any]:
    short_date = _ts_to_date(info.get("dateShortInterest"))
    shares_short, prior = _num(info.get("sharesShort")), _num(info.get("sharesShortPriorMonth"))
    return {
        "shares_short": shares_short,
        "short_interest_date": short_date,
        "short_interest_prior": prior,
        "short_interest_change": (shares_short / prior - 1) if shares_short and prior else None,
        "short_percent_float": _num(info.get("shortPercentOfFloat")),
        "short_percent_outstanding": _num(info.get("sharesPercentSharesOut")),
        "days_to_cover": _num(info.get("shortRatio")),
        "float_shares": _num(info.get("floatShares")),
        "shares_outstanding": _num(info.get("sharesOutstanding")),
        "held_pct_institutions": _num(info.get("heldPercentInstitutions")),
        "held_pct_insiders": _num(info.get("heldPercentInsiders")),
    }


def insider_summary(table: pd.DataFrame | None, as_of: date, days: int = 90) -> dict[str, Any]:
    """Open-market purchases vs sales in the last ``days`` days (Form 4 via Yahoo)."""
    if table is None or table.empty:
        return {"available": False}
    t = table.copy()
    t["date"] = pd.to_datetime(t.get("Start Date"), errors="coerce")
    t = t[(t["date"] >= pd.Timestamp(as_of) - pd.Timedelta(days=days)) & (t["date"] <= pd.Timestamp(as_of))]
    text = t.get("Text", pd.Series("", index=t.index)).fillna("").str.lower()
    tx = t.get("Transaction", pd.Series("", index=t.index)).fillna("").str.lower()
    buys = t[text.str.contains("purchase") | tx.str.contains("purchase")]
    sells = t[text.str.contains("sale") | tx.str.contains("sale")]
    value = lambda f: float(pd.to_numeric(f.get("Value"), errors="coerce").fillna(0).sum())  # noqa: E731
    recent = [{"date": r["date"].strftime("%Y-%m-%d"), "insider": r.get("Insider"), "position": r.get("Position"),
               "text": r.get("Text"), "shares": _num(r.get("Shares")), "value": _num(r.get("Value"))}
              for _, r in t.sort_values("date", ascending=False).head(8).iterrows()]
    return {"available": True, "window_days": days, "n_buys": int(len(buys)), "n_sales": int(len(sells)),
            "buy_value": value(buys), "sale_value": value(sells), "net_value": value(buys) - value(sells),
            "recent": recent,
            "note": "Sales are often pre-planned (10b5-1) or tax-related; open-market purchases are the rarer signal."}


def institutional_summary(table: pd.DataFrame | None) -> dict[str, Any]:
    if table is None or table.empty:
        return {"available": False}
    rows = []
    for _, r in table.head(10).iterrows():
        reported = r.get("Date Reported")
        rows.append({"holder": r.get("Holder"), "pct_held": _num(r.get("pctHeld")), "shares": _num(r.get("Shares")),
                     "value": _num(r.get("Value")), "pct_change": _num(r.get("pctChange")),
                     "date_reported": pd.Timestamp(reported).strftime("%Y-%m-%d") if pd.notna(reported) else None})
    changes = [r["pct_change"] for r in rows if r["pct_change"] is not None]
    return {"available": True, "top_holders": rows,
            "net_change_top10_avg": float(np.mean(changes)) if changes else None,
            "note": "13F data: quarter-end long positions filed up to 45 days later; not real-time positioning."}


class FinraData:
    """Caches FINRA daily short-sale volume files (all symbols) and weekly OTC summaries."""

    def __init__(self, cache_dir: Path, offline: bool = False, getter: Callable[..., Any] | None = None,
                 poster: Callable[..., Any] | None = None) -> None:
        self.dir = Path(cache_dir) / "finra"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.get = getter or requests.get
        self.post = poster or requests.post
        self._days: dict[date, pd.DataFrame | None] = {}

    def short_volume_day(self, day: date) -> pd.DataFrame | None:
        if day in self._days:
            return self._days[day]
        path = self.dir / f"regsho_{day:%Y%m%d}.txt"
        text = path.read_text() if path.exists() else None
        if text is None and not self.offline:
            try:
                response = self.get(REGSHO_URL.format(day=day), timeout=20)
                if response.status_code == 200 and "|" in response.text[:200]:
                    text = response.text
                    path.write_text(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FINRA short volume %s unavailable: %s", day, exc)
        frame = None
        if text:
            frame = pd.read_csv(io.StringIO(text), sep="|")
            frame = frame[pd.to_numeric(frame["TotalVolume"], errors="coerce").notna()].set_index("Symbol")
        self._days[day] = frame
        return frame

    def short_volume(self, symbol: str, session: date, days: int = 10) -> dict[str, Any]:
        rows, day = [], session
        for _ in range(days):
            frame = self.short_volume_day(day)
            if frame is not None and symbol in frame.index:
                r = frame.loc[symbol]
                total = float(r["TotalVolume"])
                rows.append({"date": day.isoformat(), "short_volume": float(r["ShortVolume"]), "total_volume": total,
                             "ratio": float(r["ShortVolume"]) / total if total else None})
            day = calendar.previous_trading_day(day)
        if not rows:
            return {"available": False}
        ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
        return {"available": True, "latest": rows[0], "avg_ratio": float(np.mean(ratios)) if ratios else None,
                "days": rows, "source": "FINRA Reg SHO daily short sale volume (consolidated NMS)",
                "note": "Short-sale VOLUME (mostly market-maker liquidity), not short interest."}

    def off_exchange(self, symbol: str, session: date, consolidated: pd.Series | None) -> dict[str, Any]:
        """Most recent published week of ATS + non-ATS OTC volume (2-4 week delay)."""
        if self.offline:
            return {"available": False, "reason": "offline"}
        monday = session - timedelta(days=session.weekday())
        for weeks_back in range(2, 7):
            week = monday - timedelta(weeks=weeks_back)
            totals: dict[str, float] = {}
            try:
                for code in ("ATS_W_SMBL", "OTC_W_SMBL"):
                    body = {"limit": 5, "compareFilters": [
                        {"compareType": "equal", "fieldName": "issueSymbolIdentifier", "fieldValue": symbol},
                        {"compareType": "equal", "fieldName": "weekStartDate", "fieldValue": week.isoformat()},
                        {"compareType": "equal", "fieldName": "tierIdentifier", "fieldValue": "T1"},
                        {"compareType": "equal", "fieldName": "summaryTypeCode", "fieldValue": code}]}
                    response = self.post(OTC_URL, json=body, headers={"Accept": "application/json"}, timeout=20)
                    if response.status_code != 200 or not response.text.strip():
                        continue
                    data = response.json()
                    totals[code] = float(sum(float(d.get("totalWeeklyShareQuantity") or 0) for d in data))
            except Exception as exc:  # noqa: BLE001
                logger.warning("FINRA OTC data for %s unavailable: %s", symbol, exc)
                return {"available": False, "reason": "FINRA OTC transparency API unavailable."}
            if totals:
                off = sum(totals.values())
                week_end = week + timedelta(days=4)
                total = None
                if consolidated is not None:
                    window = consolidated[(consolidated.index >= pd.Timestamp(week))
                                          & (consolidated.index <= pd.Timestamp(week_end))]
                    total = float(window.sum()) if not window.empty else None
                return {"available": True, "week_start": week.isoformat(), "ats_shares": totals.get("ATS_W_SMBL"),
                        "otc_non_ats_shares": totals.get("OTC_W_SMBL"), "off_exchange_shares": off,
                        "consolidated_shares": total, "off_exchange_share": off / total if total else None,
                        "source": "FINRA OTC Transparency (weekly, Tier 1)",
                        "note": f"Week of {week.isoformat()} - published with a 2-4 week delay."}
        return {"available": False, "reason": "No published FINRA OTC week found in the last 6 weeks."}


def analyze_positioning(info: dict[str, Any] | None, insider: pd.DataFrame | None, institutions: pd.DataFrame | None,
                        as_of: date, finra: FinraData | None = None, symbol: str = "",
                        consolidated_volume: pd.Series | None = None) -> dict[str, Any]:
    out = from_info(info or {})
    out["insider_90d"] = insider_summary(insider, as_of)
    out["institutional"] = institutional_summary(institutions)
    if finra is not None:
        out["short_volume"] = finra.short_volume(symbol, as_of)
        out["off_exchange"] = finra.off_exchange(symbol, as_of, consolidated_volume)
    out["block_trades"] = {"available": False, "reason": "No free, reliable public block-trade feed."}
    lag = None
    if out.get("short_interest_date"):
        lag = (as_of - date.fromisoformat(out["short_interest_date"])).days
    out["short_interest_age_days"] = lag
    out["notes"] = interpret(out)
    return out


def interpret(p: dict[str, Any]) -> list[str]:
    notes = []
    si, dtc = p.get("short_percent_float"), p.get("days_to_cover")
    if si is not None:
        level = "very high" if si >= 0.20 else "high" if si >= 0.10 else "moderate" if si >= 0.05 else "low"
        notes.append(f"Short interest {si * 100:.1f}% of float ({level}), as of {p.get('short_interest_date')} "
                     f"({p.get('short_interest_age_days')} days old).")
    if dtc is not None and dtc >= 5:
        notes.append(f"Days to cover {dtc:.1f}: crowded short - squeeze risk on positive news, but also persistent "
                     "bearish conviction.")
    ch = p.get("short_interest_change")
    if ch is not None and abs(ch) >= 0.15:
        notes.append(f"Short interest {'up' if ch > 0 else 'down'} {abs(ch) * 100:.0f}% vs the prior report.")
    ins = p.get("insider_90d") or {}
    if ins.get("available"):
        if ins["n_buys"]:
            notes.append(f"{ins['n_buys']} insider purchase(s) in 90 days (${ins['buy_value'] / 1e6:.1f}M).")
        if ins["n_sales"]:
            notes.append(f"{ins['n_sales']} insider sale(s) in 90 days (${ins['sale_value'] / 1e6:.1f}M) - often "
                         "planned / compensation-related.")
    sv = p.get("short_volume") or {}
    if sv.get("available") and sv.get("avg_ratio") and sv["latest"].get("ratio"):
        diff = sv["latest"]["ratio"] - sv["avg_ratio"]
        if abs(diff) >= 0.10:
            notes.append(f"Short-sale volume share {sv['latest']['ratio'] * 100:.0f}% vs {sv['avg_ratio'] * 100:.0f}% "
                         "10-day average (FINRA; mostly market making - weak signal).")
    return notes

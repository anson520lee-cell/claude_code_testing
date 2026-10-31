"""Point-in-time quarterly fundamentals from SEC XBRL company facts.

Look-ahead protection: every value keeps its fiscal period END and the date the filing
was FILED. For an analysis as of date D only facts filed on or before D are used, and for
each period the ORIGINAL (earliest-filed) value is used - later restatements, which are
repeated in subsequent 10-Ks, are ignored.

Quarterly values: XBRL reports income / cash-flow items for the quarter (3 months) and
year-to-date (6, 9, 12 months). Missing quarters are derived as YTD(S, E) - YTD(S, E')
where both share the same start S and E' is the previous quarter end (this is how Q4 is
obtained from the 10-K: FY - 9M YTD). Cash-flow statements are YTD-only, so this matters.

Only the us-gaap taxonomy is read. Foreign private issuers (20-F, IFRS) are reported as
"not available" instead of being guessed. Management guidance is not in XBRL (it is in
earnings press releases, 8-K Item 2.02) - the report links that filing instead.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

DURATION_CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                "RevenuesNetOfInterestExpense"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
}
INSTANT_CONCEPTS = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "short_term_investments": ["ShortTermInvestments", "MarketableSecuritiesCurrent",
                               "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "long_term_debt": ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "long_term_debt_noncurrent": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligationsNoncurrent"],
    "long_term_debt_current": ["LongTermDebtCurrent"],
    "short_term_debt": ["ShortTermBorrowings", "CommercialPaper"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
}
PER_SHARE = {"eps_diluted"}
# Weighted share counts cannot be derived by YTD differences. EPS can only approximately
# (share count changes within the year); derived EPS quarters are flagged "derived".
NON_ADDITIVE = {"diluted_shares"}


def _records(facts: dict, concepts: list[str], as_of: date, unit_hint: str = "USD") -> list[dict]:
    """All facts of the listed concepts filed on or before ``as_of`` (10-Q / 10-K family).

    Companies switch tags over time (e.g. SalesRevenueNet -> RevenueFromContractWith...), so
    every concept is read; ``priority`` (position in the list) breaks ties for the same period.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    out = []
    for priority, concept in enumerate(concepts):
        node = gaap.get(concept)
        if not node:
            continue
        units = node.get("units", {})
        key = next((u for u in units if u == unit_hint), None) or next(iter(units), None)
        if key is None:
            continue
        out.extend({**r, "concept": concept, "priority": priority} for r in units[key]
                   if r.get("filed") and date.fromisoformat(r["filed"]) <= as_of
                   and r.get("form") in ("10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT"))
    return out


def _original(rows: list[dict], keyfunc) -> dict:
    """Per period key: the preferred concept, and within it the earliest-filed record
    (point-in-time original value, not a later restatement)."""
    best: dict = {}
    for r in rows:
        k = keyfunc(r)
        rank = (r.get("priority", 0), r["filed"])
        if k not in best or rank < (best[k].get("priority", 0), best[k]["filed"]):
            best[k] = r
    return best


def quarterly_series(facts: dict, metric: str, as_of: date) -> pd.DataFrame:
    """Columns: end, value, filed, derived (bool). One row per fiscal quarter, oldest first."""
    unit = "USD/shares" if metric in PER_SHARE else ("shares" if metric == "diluted_shares" else "USD")
    rows = [r for r in _records(facts, DURATION_CONCEPTS[metric], as_of, unit) if r.get("start")]
    if not rows:
        return pd.DataFrame(columns=["end", "value", "filed", "derived"])
    periods = _original(rows, lambda r: (r["start"], r["end"]))
    by_end: dict[str, dict] = {}
    for (start, end), r in periods.items():
        days = (date.fromisoformat(end) - date.fromisoformat(start)).days
        if 75 <= days <= 105:
            by_end[end] = {"end": end, "value": float(r["val"]), "filed": r["filed"], "derived": False}
    if metric not in NON_ADDITIVE:
        ytd = {k: v for k, v in periods.items()
               if (date.fromisoformat(k[1]) - date.fromisoformat(k[0])).days > 105}
        for (start, end), r in sorted(ytd.items(), key=lambda kv: kv[0][1]):
            if end in by_end:
                continue
            # previous YTD from the same start ending ~1 quarter earlier
            prior = [(k, v) for k, v in periods.items() if k[0] == start and k[1] < end
                     and 75 <= (date.fromisoformat(end) - date.fromisoformat(k[1])).days <= 105]
            if prior:
                (_, _), pv = max(prior, key=lambda kv: kv[0][1])
                by_end[end] = {"end": end, "value": float(r["val"]) - float(pv["val"]),
                               "filed": max(r["filed"], pv["filed"]), "derived": True}
    frame = pd.DataFrame(sorted(by_end.values(), key=lambda x: x["end"]))
    return frame


def instant_value(facts: dict, metric: str, as_of: date) -> dict | None:
    rows = _records(facts, INSTANT_CONCEPTS[metric], as_of, "USD")
    rows = [r for r in rows if not r.get("start")]
    if not rows:
        return None
    periods = _original(rows, lambda r: r["end"])
    end = max(periods)
    r = periods[end]
    return {"end": end, "value": float(r["val"]), "filed": r["filed"], "concept": r["concept"]}


def shares_outstanding(facts: dict, as_of: date) -> pd.DataFrame:
    node = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {})
    rows = [r for r in node.get("units", {}).get("shares", [])
            if r.get("filed") and date.fromisoformat(r["filed"]) <= as_of]
    if not rows:
        return pd.DataFrame(columns=["end", "value", "filed"])
    periods = _original(rows, lambda r: r["end"])
    return pd.DataFrame([{"end": k, "value": float(v["val"]), "filed": v["filed"]}
                         for k, v in sorted(periods.items())])


def _year_ago(series: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    if series.empty:
        return None, None
    latest = series.iloc[-1]
    target = pd.Timestamp(latest["end"]) - pd.Timedelta(days=364)
    ends = pd.to_datetime(series["end"])
    close = (ends - target).abs() <= pd.Timedelta(days=20)
    return latest, (series[close].iloc[-1] if close.any() else None)


def _ttm(series: pd.DataFrame) -> float | None:
    if len(series) < 4:
        return None
    last4 = series.tail(4)
    ends = pd.to_datetime(last4["end"])
    span = (ends.iloc[-1] - ends.iloc[0]).days
    return float(last4["value"].sum()) if 250 <= span <= 300 else None


def analyze_fundamentals(facts: dict | None, as_of: date, price: float | None = None) -> dict[str, Any]:
    if not facts:
        return {"available": False, "reason": "No SEC XBRL company facts (not an SEC filer, or not found)."}
    if "us-gaap" not in facts.get("facts", {}):
        return {"available": False, "reason": "Company reports under IFRS / as a foreign private issuer (20-F); "
                                              "US-GAAP quarterly data not available."}
    q = {m: quarterly_series(facts, m, as_of) for m in DURATION_CONCEPTS}
    if q["revenue"].empty and q["net_income"].empty:
        return {"available": False, "reason": "No usable quarterly US-GAAP facts filed as of this date."}
    out: dict[str, Any] = {"available": True, "as_of": as_of.isoformat(), "source": "SEC EDGAR XBRL company facts",
                           "quarters": [], "metrics": {}, "flags": []}
    m = out["metrics"]
    latest_rev, ago_rev = _year_ago(q["revenue"])
    if latest_rev is not None:
        m["latest_quarter_end"] = latest_rev["end"]
        m["latest_filed"] = latest_rev["filed"]
        m["revenue_q"] = latest_rev["value"]
        m["revenue_yoy"] = (latest_rev["value"] / ago_rev["value"] - 1) if ago_rev is not None and ago_rev["value"] > 0 else None
    for name in ("revenue", "gross_profit", "operating_income", "net_income", "ocf", "capex", "interest_expense", "sbc"):
        m[f"{name}_ttm"] = _ttm(q[name])
    if m.get("gross_profit_ttm") is None and m.get("revenue_ttm") and _ttm(q["cost_of_revenue"]) is not None:
        m["gross_profit_ttm"] = m["revenue_ttm"] - _ttm(q["cost_of_revenue"])
    rev_ttm = m.get("revenue_ttm")
    for name, label in (("gross_profit", "gross_margin"), ("operating_income", "operating_margin"),
                        ("net_income", "net_margin")):
        m[f"{label}_ttm"] = (m[f"{name}_ttm"] / rev_ttm) if rev_ttm and m.get(f"{name}_ttm") is not None else None
    # quarter margin vs year-ago quarter
    def quarter_margin(metric: str) -> tuple[float | None, float | None]:
        s = q[metric].set_index("end")["value"] if not q[metric].empty else pd.Series(dtype=float)
        if latest_rev is None or latest_rev["end"] not in s.index:
            return None, None
        now = s[latest_rev["end"]] / latest_rev["value"] if latest_rev["value"] else None
        before = (s[ago_rev["end"]] / ago_rev["value"]) if ago_rev is not None and ago_rev["end"] in s.index \
            and ago_rev["value"] else None
        return now, before
    m["operating_margin_q"], m["operating_margin_q_year_ago"] = quarter_margin("operating_income")
    m["gross_margin_q"], m["gross_margin_q_year_ago"] = quarter_margin("gross_profit")
    eps_latest, eps_ago = _year_ago(q["eps_diluted"])
    if eps_latest is not None:
        m["eps_q"] = eps_latest["value"]
        m["eps_q_end"] = eps_latest["end"]
        m["eps_yoy"] = ((eps_latest["value"] - eps_ago["value"]) / abs(eps_ago["value"])
                        if eps_ago is not None and eps_ago["value"] not in (0, None) else None)
    m["eps_ttm"] = _ttm(q["eps_diluted"])
    if m.get("ocf_ttm") is not None:
        m["fcf_ttm"] = m["ocf_ttm"] - (m.get("capex_ttm") or 0.0)
        m["fcf_margin_ttm"] = m["fcf_ttm"] / rev_ttm if rev_ttm else None
    inst = {k: instant_value(facts, k, as_of) for k in INSTANT_CONCEPTS}
    cash = sum(v["value"] for k, v in inst.items() if k in ("cash", "short_term_investments") and v)
    debt = inst["long_term_debt"]["value"] if inst["long_term_debt"] else \
        sum(inst[k]["value"] for k in ("long_term_debt_noncurrent", "long_term_debt_current") if inst[k])
    debt += inst["short_term_debt"]["value"] if inst["short_term_debt"] else 0.0
    m["cash_and_st_investments"] = cash if inst["cash"] else None
    m["total_debt"] = debt
    m["net_debt"] = debt - cash if inst["cash"] else None
    m["balance_sheet_date"] = inst["cash"]["end"] if inst["cash"] else (inst["assets"]["end"] if inst["assets"] else None)
    eq = inst["equity"]["value"] if inst["equity"] else None
    m["equity"] = eq
    m["debt_to_equity"] = debt / eq if eq and eq > 0 else None
    ca, cl = inst["current_assets"], inst["current_liabilities"]
    m["current_ratio"] = ca["value"] / cl["value"] if ca and cl and cl["value"] else None
    if m.get("operating_income_ttm") and m.get("interest_expense_ttm"):
        m["interest_coverage"] = m["operating_income_ttm"] / m["interest_expense_ttm"]
    shares = shares_outstanding(facts, as_of)
    if not shares.empty:
        m["shares_outstanding"] = float(shares["value"].iloc[-1])
        m["shares_date"] = shares["end"].iloc[-1]
        latest_s, ago_s = _year_ago(shares)
        if ago_s is not None and ago_s["value"]:
            m["shares_yoy"] = latest_s["value"] / ago_s["value"] - 1
    dil_latest, dil_ago = _year_ago(q["diluted_shares"])
    if m.get("shares_yoy") is None and dil_latest is not None and dil_ago is not None and dil_ago["value"]:
        m["shares_yoy"] = dil_latest["value"] / dil_ago["value"] - 1
    if price and m.get("shares_outstanding"):
        mcap = price * m["shares_outstanding"]
        m["market_cap"] = mcap
        m["pe_ttm"] = price / m["eps_ttm"] if m.get("eps_ttm") and m["eps_ttm"] > 0 else None
        m["ps_ttm"] = mcap / rev_ttm if rev_ttm else None
        m["fcf_yield"] = m["fcf_ttm"] / mcap if m.get("fcf_ttm") is not None else None
        if m.get("net_debt") is not None and m.get("operating_income_ttm"):
            m["ev_to_ebit"] = (mcap + m["net_debt"]) / m["operating_income_ttm"] if m["operating_income_ttm"] > 0 else None
    # last 8 quarters table
    table = q["revenue"].tail(8)
    for _, r in table.iterrows():
        row = {"end": r["end"], "filed": r["filed"], "revenue": r["value"]}
        for name in ("gross_profit", "operating_income", "net_income", "eps_diluted"):
            s = q[name].set_index("end")["value"] if not q[name].empty else pd.Series(dtype=float)
            row[name] = float(s[r["end"]]) if r["end"] in s.index else None
        out["quarters"].append(row)
    out["flags"] = fundamental_flags(m)
    out["guidance"] = ("Not machine-readable in XBRL. Guidance is published in earnings press releases "
                       "(8-K Item 2.02) - see the linked filing in Catalysts.")
    return out


def fundamental_flags(m: dict[str, Any]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []

    def add(kind: str, severity: str, text: str) -> None:
        flags.append({"kind": kind, "severity": severity, "text": text})

    g = m.get("revenue_yoy")
    if g is not None:
        if g >= 0.20:
            add("strength", "info", f"Revenue +{g * 100:.0f}% YoY (quarter ended {m.get('latest_quarter_end')}).")
        elif g <= -0.10:
            add("risk", "medium", f"Revenue {g * 100:.0f}% YoY - shrinking business.")
    e = m.get("eps_yoy")
    if e is not None and m.get("eps_q") is not None:
        if e >= 0.25 and m["eps_q"] > 0:
            add("strength", "info", f"Diluted EPS +{e * 100:.0f}% YoY.")
        elif e <= -0.25:
            add("risk", "medium", f"Diluted EPS {e * 100:.0f}% YoY.")
    om, om_ago = m.get("operating_margin_q"), m.get("operating_margin_q_year_ago")
    if om is not None and om_ago is not None and om - om_ago <= -0.05:
        add("risk", "medium", f"Operating margin fell {(om_ago - om) * 100:.1f}pp YoY ({om * 100:.1f}% vs "
                              f"{om_ago * 100:.1f}%).")
    elif om is not None and om_ago is not None and om - om_ago >= 0.05:
        add("strength", "info", f"Operating margin up {(om - om_ago) * 100:.1f}pp YoY.")
    fcf, cash = m.get("fcf_ttm"), m.get("cash_and_st_investments")
    if fcf is not None and fcf < 0:
        if cash is not None and cash > 0:
            runway = cash / (-fcf / 4)
            m["cash_runway_quarters"] = runway
            if runway < 4:
                add("risk", "high", f"Negative free cash flow; cash covers only ~{runway:.1f} quarters of burn - "
                                    "financing (dilution / debt) likely.")
            else:
                add("risk", "medium", f"Negative TTM free cash flow (runway ~{runway:.0f} quarters).")
        else:
            add("risk", "medium", "Negative TTM free cash flow.")
    elif fcf is not None and m.get("fcf_margin_ttm") is not None and m["fcf_margin_ttm"] >= 0.15:
        add("strength", "info", f"Strong cash generation: FCF margin {m['fcf_margin_ttm'] * 100:.0f}% (TTM).")
    ic = m.get("interest_coverage")
    if ic is not None and ic < 1.5:
        add("risk", "high", f"Interest coverage {ic:.1f}x (operating income barely covers interest).")
    nd, oi = m.get("net_debt"), m.get("operating_income_ttm")
    if nd is not None and oi is not None and oi > 0 and nd / oi > 4:
        add("risk", "medium", f"Net debt is {nd / oi:.1f}x TTM operating income - elevated leverage.")
    if m.get("equity") is not None and m["equity"] < 0:
        add("risk", "high", "Negative stockholders' equity.")
    cr = m.get("current_ratio")
    if cr is not None and cr < 1.0:
        add("risk", "medium", f"Current ratio {cr:.2f} (< 1).")
    sy = m.get("shares_yoy")
    if sy is not None:
        if sy >= 0.10:
            add("risk", "high", f"Share count +{sy * 100:.0f}% YoY - significant dilution.")
        elif sy >= 0.03:
            add("risk", "medium", f"Share count +{sy * 100:.1f}% YoY - ongoing dilution.")
        elif sy <= -0.02:
            add("strength", "info", f"Share count {sy * 100:.1f}% YoY (buybacks).")
    ni = m.get("net_income_ttm")
    if ni is not None and ni < 0:
        add("risk", "medium", "Net loss over the last four quarters.")
    return flags


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# --------------------------------------------------------------------------- Yahoo fallback
# Yahoo quarterly statement rows -> the us-gaap concept analyze_fundamentals() reads.
YAHOO_DURATION = {
    "TotalRevenue": "Revenues", "CostOfRevenue": "CostOfRevenue", "GrossProfit": "GrossProfit",
    "OperatingIncome": "OperatingIncomeLoss", "NetIncome": "NetIncomeLoss", "DilutedEPS": "EarningsPerShareDiluted",
    "InterestExpense": "InterestExpense", "DilutedAverageShares": "WeightedAverageNumberOfDilutedSharesOutstanding",
    "OperatingCashFlow": "NetCashProvidedByUsedInOperatingActivities",
    "CapitalExpenditure": "PaymentsToAcquirePropertyPlantAndEquipment", "StockBasedCompensation": "ShareBasedCompensation",
}
YAHOO_INSTANT = {
    "CashAndCashEquivalents": "CashAndCashEquivalentsAtCarryingValue",
    "OtherShortTermInvestments": "ShortTermInvestments", "LongTermDebt": "LongTermDebtNoncurrent",
    "CurrentDebt": "LongTermDebtCurrent", "StockholdersEquity": "StockholdersEquity", "TotalAssets": "Assets",
    "CurrentAssets": "AssetsCurrent", "CurrentLiabilities": "LiabilitiesCurrent",
}
ESTIMATED_FILING_LAG_DAYS = 45  # 10-Q deadline is 40-45 days after quarter end; 10-K 60-90 (conservative for Q1-Q3)


def facts_from_yahoo(statements: dict[str, pd.DataFrame], shares: float | None = None,
                     shares_date: date | None = None) -> dict | None:
    """Build an XBRL-like facts dict from Yahoo quarterly statements (pretty=False row names).

    Yahoo gives period ends but no filing dates, so each value's filing date is ESTIMATED as
    period end + 45 days (+75 days for the fiscal-year quarter is not knowable here, so 45 is
    used for all - the analysis states that publication dates are estimated).
    """
    gaap: dict[str, Any] = {}
    found = False
    for name, frame in statements.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        mapping = YAHOO_INSTANT if name == "balance" else YAHOO_DURATION
        for row, concept in mapping.items():
            if row not in frame.index:
                continue
            unit = "USD/shares" if concept.startswith("EarningsPerShare") else \
                "shares" if concept.startswith("WeightedAverage") else "USD"
            for column, value in frame.loc[row].items():
                if pd.isna(value):
                    continue
                end = pd.Timestamp(column).date()
                val = -float(value) if row == "CapitalExpenditure" else float(value)
                record = {"end": end.isoformat(), "val": val, "form": "10-Q",
                          "filed": (end + pd.Timedelta(days=ESTIMATED_FILING_LAG_DAYS)).isoformat()}
                if name != "balance":
                    record["start"] = (end - pd.Timedelta(days=90)).isoformat()
                gaap.setdefault(concept, {"units": {}})["units"].setdefault(unit, []).append(record)
                found = True
    if not found:
        return None
    facts: dict[str, Any] = {"facts": {"us-gaap": gaap}}
    if shares and shares_date:
        facts["facts"]["dei"] = {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": shares_date.isoformat(), "val": shares, "filed": shares_date.isoformat()}]}}}
    return facts

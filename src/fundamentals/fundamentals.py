"""Retrieve and normalise fundamental data.

Two kinds of data are collected:

1. **Snapshot metrics**: the provider's current values (market cap, P/E, EV/Sales ...).
   These are *reported* by Yahoo Finance, not calculated here. Stored with
   period_type = 'snapshot' and period_end = the date they were retrieved, so
   repeated runs build up a history of valuation over time.

2. **Annual statement metrics**: revenue, profit, cash flow and balance-sheet
   items per fiscal year (Yahoo provides about 4 years). A few ratios are
   *calculated* from these (growth and margins); their source says "Calculated".

Nothing is estimated or filled in: a value the provider does not supply is
simply absent and shown as "Not available" in the report.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

YAHOO_SOURCE = "Yahoo Finance (via yfinance)"

# (metric name, Yahoo "info" key, unit type, label shown in the report)
# unit type: "money_trading" = trading currency, "money_financial" = reporting
# currency, "ratio", "per_share", "shares".
SNAPSHOT_METRICS: list[tuple[str, str, str, str]] = [
    ("market_cap", "marketCap", "money_trading", "Market capitalization"),
    ("enterprise_value", "enterpriseValue", "money_trading", "Enterprise value"),
    ("revenue_ttm", "totalRevenue", "money_financial", "Revenue (trailing 12 months)"),
    ("revenue_growth_yoy_quarterly", "revenueGrowth", "ratio",
     "Revenue growth (latest quarter vs. same quarter a year earlier)"),
    ("gross_margin_ttm", "grossMargins", "ratio", "Gross margin (trailing 12 months)"),
    ("operating_margin", "operatingMargins", "ratio", "Operating margin (as reported by Yahoo)"),
    ("net_income_ttm", "netIncomeToCommon", "money_financial", "Net income to common (trailing 12 months)"),
    ("eps_ttm", "trailingEps", "per_share", "EPS (trailing 12 months)"),
    ("free_cash_flow_ttm", "freeCashflow", "money_financial",
     "Free cash flow (Yahoo 'levered' FCF, trailing 12 months)"),
    ("ebitda_ttm", "ebitda", "money_financial", "EBITDA (trailing 12 months)"),
    ("total_cash", "totalCash", "money_financial", "Cash and short-term investments (latest quarter)"),
    ("total_debt", "totalDebt", "money_financial", "Total debt (latest quarter)"),
    ("shares_outstanding", "sharesOutstanding", "shares", "Shares outstanding"),
    ("pe_trailing", "trailingPE", "ratio", "P/E (trailing)"),
    ("pe_forward", "forwardPE", "ratio", "P/E (forward, based on analyst estimates)"),
    ("ev_to_sales", "enterpriseToRevenue", "ratio", "EV / Sales"),
    ("ev_to_ebitda", "enterpriseToEbitda", "ratio", "EV / EBITDA"),
]

# (metric name, statement, candidate Yahoo row names in order of preference, unit type, label)
STATEMENT_METRICS: list[tuple[str, str, list[str], str, str]] = [
    ("revenue", "income", ["TotalRevenue"], "money_financial", "Revenue"),
    ("gross_profit", "income", ["GrossProfit"], "money_financial", "Gross profit"),
    ("operating_income", "income", ["OperatingIncome", "TotalOperatingIncomeAsReported"],
     "money_financial", "Operating income"),
    ("net_income", "income", ["NetIncomeCommonStockholders", "NetIncome"], "money_financial", "Net income"),
    ("eps_diluted", "income", ["DilutedEPS"], "per_share", "Diluted EPS"),
    ("ebitda", "income", ["EBITDA"], "money_financial", "EBITDA"),
    ("operating_cash_flow", "cashflow", ["OperatingCashFlow"], "money_financial", "Operating cash flow"),
    ("capital_expenditure", "cashflow", ["CapitalExpenditure"], "money_financial", "Capital expenditure"),
    ("free_cash_flow", "cashflow", ["FreeCashFlow"], "money_financial", "Free cash flow"),
    ("cash_and_equivalents", "balance", ["CashAndCashEquivalents"], "money_financial", "Cash and cash equivalents"),
    ("total_debt", "balance", ["TotalDebt"], "money_financial", "Total debt"),
    ("shares_outstanding", "balance", ["OrdinarySharesNumber", "ShareIssued"], "shares", "Shares outstanding"),
]

# Ratios calculated by this program from the annual statements.
CALCULATED_METRICS: dict[str, tuple[str, str]] = {
    "revenue_growth": ("ratio", "Revenue growth vs. previous fiscal year (calculated)"),
    "gross_margin": ("ratio", "Gross margin = gross profit / revenue (calculated)"),
    "operating_margin": ("ratio", "Operating margin = operating income / revenue (calculated)"),
    "net_margin": ("ratio", "Net margin = net income / revenue (calculated)"),
}

PROFILE_KEYS = {
    "name": ("longName", "shortName"),
    "exchange": ("fullExchangeName", "exchange"),
    "sector": ("sector",),
    "industry": ("industry",),
    "currency": ("currency",),
    "financial_currency": ("financialCurrency",),
    "country": ("country",),
    "quote_type": ("quoteType",),
}


def to_finite_float(value: Any) -> float | None:
    """Convert a provider value to float; return None for missing, text or infinite values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def extract_profile(info: dict[str, Any]) -> dict[str, Any]:
    """Pick the company-profile fields (name, sector, ...) from Yahoo's info dictionary."""
    profile: dict[str, Any] = {}
    for field, keys in PROFILE_KEYS.items():
        profile[field] = next((info[k] for k in keys if info.get(k) not in (None, "")), None)
    return profile


def _unit(unit_type: str, profile: dict[str, Any]) -> str:
    trading = profile.get("currency") or "currency unknown"
    financial = profile.get("financial_currency") or profile.get("currency") or "currency unknown"
    return {
        "money_trading": trading,
        "money_financial": financial,
        "per_share": f"{financial}/share",
        "ratio": "ratio",
        "shares": "shares",
    }[unit_type]


def snapshot_records(ticker: str, info: dict[str, Any], retrieved: date) -> list[dict[str, Any]]:
    """Normalise Yahoo's snapshot metrics. Missing values produce no record."""
    profile = extract_profile(info)
    records = []
    for metric, key, unit_type, label in SNAPSHOT_METRICS:
        value = to_finite_float(info.get(key))
        if value is None:
            continue
        records.append({
            "ticker": ticker, "period_type": "snapshot", "period_end": retrieved.isoformat(),
            "metric": metric, "label": label, "value": value, "unit": _unit(unit_type, profile),
            "source": f"{YAHOO_SOURCE}, info['{key}']", "retrieved_at": retrieved.isoformat(),
        })
    return records


def _statement_value(table: pd.DataFrame, rows: list[str], column: Any) -> tuple[float | None, str | None]:
    for row in rows:
        if row in table.index:
            value = to_finite_float(table.at[row, column])
            if value is not None:
                return value, row
    return None, None


def annual_records(ticker: str, statements: dict[str, pd.DataFrame], info: dict[str, Any],
                   retrieved: date) -> list[dict[str, Any]]:
    """Normalise annual statement values and calculate growth and margins."""
    profile = extract_profile(info)
    records: list[dict[str, Any]] = []
    values: dict[str, dict[str, float]] = {}  # period_end -> metric -> value

    for metric, statement, rows, unit_type, label in STATEMENT_METRICS:
        table = statements.get(statement)
        if table is None or table.empty:
            continue
        for column in table.columns:
            period_end = pd.Timestamp(column).strftime("%Y-%m-%d")
            value, used_row = _statement_value(table, rows, column)
            if value is None:
                continue
            values.setdefault(period_end, {})[metric] = value
            records.append({
                "ticker": ticker, "period_type": "annual", "period_end": period_end,
                "metric": metric, "label": label, "value": value, "unit": _unit(unit_type, profile),
                "source": f"{YAHOO_SOURCE}, annual {statement} statement ({used_row})",
                "retrieved_at": retrieved.isoformat(),
            })

    periods = sorted(values)
    for i, period_end in enumerate(periods):
        v = values[period_end]
        revenue = v.get("revenue")
        calculated: dict[str, float] = {}
        if revenue and revenue > 0:
            for metric, numerator in (("gross_margin", "gross_profit"),
                                      ("operating_margin", "operating_income"),
                                      ("net_margin", "net_income")):
                if numerator in v:
                    calculated[metric] = v[numerator] / revenue
            if i > 0:
                previous = values[periods[i - 1]].get("revenue")
                days_apart = (pd.Timestamp(period_end) - pd.Timestamp(periods[i - 1])).days
                # Only compare consecutive fiscal years (about 365 days apart).
                if previous and previous > 0 and 300 <= days_apart <= 430:
                    calculated["revenue_growth"] = revenue / previous - 1
        for metric, value in calculated.items():
            unit_type, label = CALCULATED_METRICS[metric]
            records.append({
                "ticker": ticker, "period_type": "annual", "period_end": period_end,
                "metric": metric, "label": label, "value": value, "unit": unit_type,
                "source": "Calculated from Yahoo Finance annual statements",
                "retrieved_at": retrieved.isoformat(),
            })
    return records


def collect_fundamentals(provider: Any, ticker: str, retrieved: date | None = None) -> dict[str, Any]:
    """Download and normalise all fundamental data for a ticker.

    Never raises for missing data or network failures: problems are returned
    in the "warnings" list so the rest of the analysis can continue.

    Returns:
        {"profile": {...}, "records": [...], "warnings": [...]}
    """
    retrieved = retrieved or date.today()
    warnings: list[str] = []
    info: dict[str, Any] = {}
    try:
        info = provider.get_company_info(ticker) or {}
    except Exception as exc:  # noqa: BLE001 - any failure here must not stop the analysis
        warnings.append(f"Company information / snapshot fundamentals unavailable: {exc}")
        logger.warning("Company info for %s unavailable: %s", ticker, exc)

    statements: dict[str, pd.DataFrame] = {}
    try:
        statements = provider.get_financial_statements(ticker) or {}
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Annual financial statements unavailable: {exc}")
        logger.warning("Financial statements for %s unavailable: %s", ticker, exc)

    records = snapshot_records(ticker, info, retrieved) + annual_records(ticker, statements, info, retrieved)
    if info and not any(r["period_type"] == "snapshot" for r in records):
        warnings.append("The data provider returned no snapshot fundamentals for this ticker.")
    if not any(r["period_type"] == "annual" for r in records):
        warnings.append("No annual financial-statement data was available for this ticker.")
    return {"profile": extract_profile(info), "records": records, "warnings": warnings}


def fundamentals_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Records as a tidy table (the layout written to fundamentals.csv)."""
    columns = ["ticker", "period_type", "period_end", "metric", "label", "value", "unit", "source", "retrieved_at"]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records).reindex(columns=columns)


def annual_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Annual metrics pivoted to one row per metric and one column per fiscal year."""
    annual = [r for r in records if r["period_type"] == "annual"]
    if not annual:
        return pd.DataFrame()
    df = pd.DataFrame(annual)
    order = [m[0] for m in STATEMENT_METRICS] + list(CALCULATED_METRICS)
    table = df.pivot_table(index="metric", columns="period_end", values="value", aggfunc="first")
    table = table.reindex([m for m in order if m in table.index])
    return table[sorted(table.columns)]


def metric_labels() -> dict[str, str]:
    """Map every metric name to its human-readable label (used for data loaded from the database)."""
    labels = {m[0]: m[3] for m in SNAPSHOT_METRICS}
    labels.update({m[0]: m[4] for m in STATEMENT_METRICS})
    labels.update({k: v[1] for k, v in CALCULATED_METRICS.items()})
    return labels

"""Write the report files: summary.md, events.csv, fundamentals.csv and analysis.json.

The Markdown report labels every section so facts, calculations and unknowns
are never mixed up:

* [Reported]   - taken directly from the data provider
* [Calculated] - computed by this program (formulas in README.md)
* Unknown      - not available from the data used; never guessed
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.fundamentals.fundamentals import CALCULATED_METRICS, SNAPSHOT_METRICS, STATEMENT_METRICS, annual_table

logger = logging.getLogger(__name__)

PERCENT_METRICS = {
    "revenue_growth_yoy_quarterly", "gross_margin_ttm", "operating_margin",
    "revenue_growth", "gross_margin", "net_margin",
}
MULTIPLE_METRICS = {"pe_trailing", "pe_forward", "ev_to_sales", "ev_to_ebitda"}
NOT_AVAILABLE = "Not available"


# ------------------------------------------------------------------ formatting
def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def fmt_pct(value: Any, decimals: int = 2, signed: bool = True) -> str:
    """0.0123 -> '+1.23%'. Missing -> 'n/a'."""
    if _is_missing(value):
        return "n/a"
    return f"{value * 100:+.{decimals}f}%" if signed else f"{value * 100:.{decimals}f}%"


def fmt_num(value: Any, decimals: int = 2) -> str:
    if _is_missing(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def fmt_big(value: Any) -> str:
    """1_234_000_000 -> '1.23B'."""
    if _is_missing(value):
        return "n/a"
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= divisor:
            return f"{value / divisor:,.2f}{suffix}"
    return f"{value:,.2f}"


def fmt_date(value: Any) -> str:
    if _is_missing(value):
        return "n/a"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def fmt_metric(metric: str, value: Any, unit: str | None) -> str:
    """Format a fundamental metric according to its type."""
    if _is_missing(value):
        return NOT_AVAILABLE
    if metric in PERCENT_METRICS:
        return fmt_pct(value, 1, signed=metric in {"revenue_growth", "revenue_growth_yoy_quarterly"})
    if metric in MULTIPLE_METRICS:
        return f"{value:,.1f}x"
    if unit == "shares":
        return f"{fmt_big(value)} shares"
    if unit and unit.endswith("/share"):
        return f"{value:,.2f} {unit}"
    return f"{fmt_big(value)} {unit}" if unit else fmt_big(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Build a Markdown table. '|' characters in cells are escaped."""
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------ JSON output
def to_jsonable(value: Any) -> Any:
    """Recursively convert pandas/numpy objects to JSON-safe Python values (NaN -> null)."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_dict())
    if isinstance(value, pd.Timestamp):
        # Plain dates as 'YYYY-MM-DD'; timestamps with a time keep it.
        return value.strftime("%Y-%m-%d") if value == value.normalize() else value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if value is pd.NaT or value is pd.NA:
        return None
    return value


# ------------------------------------------------------------------ file writers
# Stored in the database for provenance, but too long to be useful in the CSV/JSON
# (the settings used are already listed once in analysis.json).
EXPORT_EXCLUDED_COLUMNS = ["detection_settings"]


def export_events(events: pd.DataFrame) -> pd.DataFrame:
    return events.drop(columns=[c for c in EXPORT_EXCLUDED_COLUMNS if c in events.columns])


def write_events_csv(events: pd.DataFrame, path: Path) -> Path:
    out = export_events(events)
    if not out.empty:
        out["event_date"] = pd.to_datetime(out["event_date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    return path


def write_fundamentals_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def write_analysis_json(results: dict[str, Any], path: Path) -> Path:
    payload = {
        "ticker": results["ticker"],
        "generated_at_utc": results["generated_at"],
        "run_id": results.get("run_id"),
        "disclaimer": "Descriptive research output. Contains no investment recommendations.",
        "data_sources": results["data_sources"],
        "settings": results["settings"],
        "profile": results.get("profile"),
        "statistics": results["stats"],
        "benchmark": results.get("benchmark_summary"),
        "events": export_events(results["events"]),
        "event_outcomes": results["event_outcomes"],
        "fundamentals": results["fundamentals"]["records"],
        "warnings": results["warnings"],
        "database": results.get("database"),
        "files": results.get("files"),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)
    return path


# ------------------------------------------------------------------ Markdown
def _section_company(r: dict[str, Any]) -> str:
    profile = r.get("profile") or {}
    stats = r["stats"]
    sources = r["data_sources"]
    unknown = "Unknown"
    rows = [
        ["Ticker", r["ticker"]],
        ["Company name", profile.get("name") or unknown],
        ["Exchange", profile.get("exchange") or unknown],
        ["Sector", profile.get("sector") or unknown],
        ["Industry", profile.get("industry") or unknown],
        ["Trading currency", profile.get("currency") or unknown],
        ["Price data source", sources["prices"]],
        ["Price history analysed", f"{fmt_date(stats['start_date'])} to {fmt_date(stats['end_date'])} "
                                   f"({stats['trading_days']:,} trading days)"],
        ["Price series used for returns", sources["price_column_description"]],
    ]
    return "## 1. Company / Ticker\n\n*[Reported] by the data provider.*\n\n" + md_table(["Field", "Value"], rows)


def _section_market_overview(r: dict[str, Any]) -> str:
    s = r["stats"]
    ma = s["moving_averages"]
    rows = [
        ["Last close", f"{fmt_num(s['last_close'])} on {fmt_date(s['end_date'])}"],
        ["52-week high (closing)", f"{fmt_num(s['high_52w_close'])} on {fmt_date(s['high_52w_date'])}"],
        ["52-week low (closing)", f"{fmt_num(s['low_52w_close'])} on {fmt_date(s['low_52w_date'])}"],
        ["Last close vs 52-week high", fmt_pct(s["close_vs_52w_high"])],
    ]
    for n in (20, 50):
        entry = ma.get(f"ma_{n}")
        if entry:
            rows.append([f"{n}-day moving average", f"{fmt_num(entry['value'])} "
                                                    f"(last close is {fmt_pct(entry['close_vs_ma'])} vs MA)"])
        else:
            rows.append([f"{n}-day moving average", "n/a (not enough history)"])
    rows.append(["Current drawdown from highest price in period", fmt_pct(s["drawdown"]["current_drawdown"])])
    vol = s["volume"]
    rows.append(["Last day's volume", f"{fmt_big(vol['last_volume'])} shares "
                                      f"({fmt_num(vol['last_volume_ratio'])}x the previous-window average)"])
    return ("## 2. Market Overview\n\n*[Calculated] from reported daily prices.*\n\n"
            + md_table(["Measure", "Value"], rows))


def _section_performance(r: dict[str, Any]) -> str:
    s = r["stats"]
    rows = [[label, fmt_pct(value)] for label, value in s["performance"].items()]
    text = "## 3. Recent Performance\n\n*[Calculated] total return based on the price series used for returns.*\n\n"
    text += md_table(["Window", "Return"], rows)
    if s.get("annualised_return") is not None:
        text += f"\n\nAnnualised return over the full period: **{fmt_pct(s['annualised_return'])}** per year."
    return text


def _section_volatility(r: dict[str, Any], files: dict[str, str]) -> str:
    s = r["stats"]
    v, d, dd = s["volatility"], s["distribution"], s["drawdown"]
    windows = r["settings"]["statistics"]
    rows = [
        [f"Rolling volatility, last {windows['volatility_window']} days (annualised)",
         fmt_pct(v["current_rolling_annualised"], signed=False)],
        ["Volatility, last 12 months (annualised)", fmt_pct(v["last_year_annualised"], signed=False)],
        ["Volatility, full period (annualised)", fmt_pct(v["full_period_annualised"], signed=False)],
        ["Daily return standard deviation", fmt_pct(v["daily_std"], 3, signed=False)],
        ["Mean / median daily return", f"{fmt_pct(d['mean'], 3)} / {fmt_pct(d['median'], 3)}"],
        ["Skewness / excess kurtosis", f"{fmt_num(d['skewness'])} / {fmt_num(d['excess_kurtosis'])}"],
        ["Best day", f"{fmt_pct(d['max'])} on {fmt_date(d['max_date'])}"],
        ["Worst day", f"{fmt_pct(d['min'])} on {fmt_date(d['min_date'])}"],
        ["Share of up days", fmt_pct(d["share_positive_days"], 1, signed=False)],
        ["Maximum drawdown", f"{fmt_pct(dd['max_drawdown'])} (peak {fmt_date(dd['peak_date'])} -> "
                             f"trough {fmt_date(dd['trough_date'])})"],
        ["Recovered to previous peak?", fmt_date(dd["recovery_date"]) if dd["recovery_date"] is not None
         else "Not recovered within the data"],
    ]
    text = "## 4. Volatility\n\n*[Calculated]*\n\n" + md_table(["Measure", "Value"], rows)
    tail = s.get("tail_frequency")
    if tail:
        ratio = tail["ratio_to_normal"]
        text += (
            f"\n\n**Fat tails check:** {tail['days_beyond_threshold']} of {tail['days_scored']:,} scored days "
            f"({fmt_pct(tail['observed_share'], 2, signed=False)}) had |Return Z-score| >= {tail['threshold']:g}. "
            f"A normal distribution would give {fmt_pct(tail['normal_expected_share'], 2, signed=False)}"
            + (f", so such days occurred **{ratio:.1f}x** as often as a normal model predicts." if ratio else ".")
        )
    if "returns_distribution" in files:
        text += f"\n\n![Return distribution]({files['returns_distribution']})"
    return text


def _event_rules_text(settings: dict[str, Any]) -> str:
    e, st = settings["events"], settings["statistics"]
    rules = []
    if e.get("return_zscore_threshold") is not None:
        rules.append(f"|Return Z-score| >= **{e['return_zscore_threshold']:g}** "
                     f"(baseline: previous {st['zscore_window']} trading days)")
    if e.get("abs_return_threshold") is not None:
        rules.append(f"|Daily return| >= **{e['abs_return_threshold'] * 100:g}%**")
    if e.get("volume_ratio_threshold") is not None:
        rules.append(f"Volume >= **{e['volume_ratio_threshold']:g}x** the average of the previous "
                     f"{st['volume_window']} trading days")
    if e.get("volume_zscore_threshold") is not None:
        rules.append(f"Volume Z-score >= **{e['volume_zscore_threshold']:g}** "
                     f"(baseline: previous {st['volume_window']} trading days)")
    return "A day is flagged when **any** of these rules is true:\n\n" + "\n".join(f"- {rule}" for rule in rules)


def _section_events(r: dict[str, Any], files: dict[str, str]) -> str:
    events: pd.DataFrame = r["events_frame"]
    settings = r["settings"]
    limit = settings["report"]["max_events_in_summary"]
    text = ("## 5. Abnormal Events\n\n*[Calculated] from prices and volume. The **cause** of each event is "
            "**Unknown** unless evidence is listed; the program never guesses a cause.*\n\n")
    text += _event_rules_text(settings) + "\n\n"

    if events.empty:
        text += "**No abnormal events were detected in this period with the current thresholds.**\n"
    else:
        direction = events["direction"].value_counts()
        price = events["is_price_event"].fillna(False).astype(bool)
        volume = events["is_volume_event"].fillna(False).astype(bool)
        unknown = int((events["event_category"] == "Unknown").sum())
        counts = [
            ["Total events", len(events)],
            ["Up / down / flat days", f"{direction.get('up', 0)} / {direction.get('down', 0)} / "
                                      f"{direction.get('flat', 0)}"],
            ["Price move with abnormal volume", int((price & volume).sum())],
            ["Price move only", int((price & ~volume).sum())],
            ["Volume only", int((volume & ~price).sum())],
            ["Linked to an earnings-release date (date match, unverified)", len(events) - unknown],
            ["Cause unknown (no evidence found)", unknown],
        ]
        text += md_table(["Count", "Value"], counts) + "\n\n"
        recent = events.sort_values("event_date", ascending=False).head(limit)
        shown = f"all {len(events)}" if len(events) <= limit else f"the {limit} most recent of {len(events)}"
        text += f"Showing {shown} events (every event is in `events.csv`). Forward returns are `n/a` when " \
                "not enough trading days have passed yet.\n\n"
        rows = []
        for _, e in recent.iterrows():
            rows.append([
                fmt_date(e["event_date"]), fmt_num(e["close"]), fmt_pct(e["daily_return"]),
                fmt_num(e["return_zscore"]), fmt_num(e["volume_ratio"]),
                e["trigger_rules"].replace(",", ", "),
                fmt_pct(e["fwd_return_1d"]), fmt_pct(e["fwd_return_5d"]), fmt_pct(e["fwd_return_20d"]),
                e["event_category"], e["verification_status"],
            ])
        text += md_table(
            ["Date", "Close", "Return", "Z-score", "Vol. ratio", "Rules fired", "Fwd 1D", "Fwd 5D", "Fwd 20D",
             "Category", "Verification"],
            rows,
        )
        evidence = recent[recent["event_description"].notna()]
        if not evidence.empty:
            text += "\n\n**Evidence attached to the events above** (source: " \
                    f"{evidence['source'].iloc[0]}):\n\n"
            text += "\n".join(f"- **{fmt_date(e['event_date'])}**: {e['event_description']}"
                              for _, e in evidence.iterrows())
        text += (
            "\n\n**Unknown:** for events in the category `Unknown` no evidence was found in the data used. "
            "Possible causes (news, filings, analyst actions, sector or market moves) must be researched "
            "manually; record findings in the `events` table (see README)."
        )
    for key, caption in (("price_chart", "Price chart"), ("event_chart", "Event chart"),
                         ("volume_chart", "Volume chart")):
        if key in files:
            text += f"\n\n![{caption}]({files[key]})"
    return text


def _section_fundamentals(r: dict[str, Any]) -> str:
    fundamentals = r["fundamentals"]
    records = fundamentals["records"]
    text = ("## 6. Fundamental Snapshot\n\n*[Reported] by Yahoo Finance unless marked (calculated). "
            f"Values the provider does not supply are shown as \"{NOT_AVAILABLE}\" - nothing is estimated.*\n\n")
    snapshot = {rec["metric"]: rec for rec in records if rec["period_type"] == "snapshot"}
    rows = []
    for metric, key, _unit_type, label in SNAPSHOT_METRICS:
        rec = snapshot.get(metric)
        rows.append([label, fmt_metric(metric, rec["value"], rec["unit"]) if rec else NOT_AVAILABLE,
                     f"Yahoo `{key}`" if rec else "-"])
    retrieved = next(iter(snapshot.values()))["retrieved_at"] if snapshot else None
    text += md_table(["Metric", "Value", "Source field"], rows)
    if retrieved:
        text += f"\n\nSnapshot retrieved on {retrieved}."

    table = annual_table(records)
    if not table.empty:
        labels = {m[0]: m[4] for m in STATEMENT_METRICS}
        labels.update({k: v[1] for k, v in CALCULATED_METRICS.items()})
        units = {rec["metric"]: rec["unit"] for rec in records if rec["period_type"] == "annual"}
        headers = ["Metric"] + [f"FY ending {c}" for c in table.columns]
        rows = []
        for metric, values in table.iterrows():
            rows.append([labels.get(metric, metric)] +
                        [fmt_metric(metric, values[c], units.get(metric)) for c in table.columns])
        text += "\n\n### Annual financial statements\n\n*[Reported] statement values; rows marked " \
                "(calculated) were computed from them.*\n\n" + md_table(headers, rows)
    else:
        text += "\n\n### Annual financial statements\n\nNot available from the data provider for this run."
    if fundamentals["warnings"]:
        text += "\n\n" + "\n".join(f"- Note: {w}" for w in fundamentals["warnings"])
    return text


def _section_benchmark(r: dict[str, Any], files: dict[str, str]) -> str:
    summary = r.get("benchmark_summary")
    text = "## 7. Benchmark Comparison\n\n"
    if not summary or not summary.get("available"):
        reason = (summary or {}).get("reason", "Benchmark comparison was disabled or the benchmark data "
                                               "could not be downloaded.")
        return text + f"Not available: {reason}"
    ticker, bench = r["ticker"], summary["benchmark"]
    text += (f"*[Calculated]* Benchmark: **{bench}**. Abnormal return = {ticker} return - {bench} return "
             f"(simple difference, not risk-adjusted). Overlapping history: "
             f"{fmt_date(summary['overlap_start'])} to {fmt_date(summary['overlap_end'])}.\n\n")
    rows = [[c["window"], fmt_pct(c["stock_return"]), fmt_pct(c["benchmark_return"]), fmt_pct(c["difference"])]
            for c in summary["comparison"]]
    text += md_table(["Window", ticker, bench, "Difference"], rows)
    text += "\n\n" + md_table(["Measure", "Value"], [
        [f"Beta vs {bench} (daily returns, full period)", fmt_num(summary["beta"])],
        [f"Correlation with {bench} (daily returns)", fmt_num(summary["correlation"])],
        ["Std. of daily abnormal returns (annualised)",
         fmt_pct(summary["abnormal_return_annualised_std"], signed=False)],
        ["Cumulative relative performance (full period)", fmt_pct(summary["final_relative_performance"])],
    ])
    if "benchmark_chart" in files:
        text += f"\n\n![Benchmark comparison]({files['benchmark_chart']})"
    return text


def _section_outcomes(r: dict[str, Any]) -> str:
    outcomes: pd.DataFrame = r["event_outcomes_frame"]
    text = ("## 8. Historical Event Outcomes\n\n*[Calculated]* What happened **after** the events detected "
            f"in this run for {r['ticker']}. Forward return = price h trading days later / event-day close - 1.\n\n")
    if outcomes.empty or (outcomes["group"] != "All trading days (baseline)").sum() == 0:
        return text + "No events were detected, so there are no event outcomes to summarise."
    rows = [[o["group"], o["n_events"], fmt_pct(o["mean_1d"]), fmt_pct(o["mean_5d"]), fmt_pct(o["mean_20d"]),
             fmt_pct(o["median_20d"]), fmt_pct(o["pct_positive_5d"], 0, signed=False),
             fmt_pct(o["pct_positive_20d"], 0, signed=False), o["n_20d"]]
            for _, o in outcomes.iterrows()]
    text += md_table(["Group", "Events", "Avg 1D", "Avg 5D", "Avg 20D", "Median 20D", "% positive 5D",
                      "% positive 20D", "N with 20D data"], rows)
    if outcomes["mean_abnormal_20d"].notna().any():
        rows = [[o["group"], fmt_pct(o["mean_abnormal_1d"]), fmt_pct(o["mean_abnormal_5d"]),
                 fmt_pct(o["mean_abnormal_20d"])] for _, o in outcomes.iterrows()]
        text += "\n\n**Average forward abnormal return (stock minus benchmark over the same days):**\n\n"
        text += md_table(["Group", "1D", "5D", "20D"], rows)
    text += (
        "\n\n**How to read this:** for *up-move* groups, \"% positive\" is how often the move continued; for "
        "*down-move* groups it is how often the move reversed. Compare every group with the "
        "\"All trading days (baseline)\" row.\n\n"
        "**Caveats:** these are descriptive statistics for one stock. Small groups (fewer than ~20 events) "
        "can be dominated by one or two outcomes; forward windows of events close together overlap, so "
        "events are not independent; no statistical significance test is applied. Past outcomes do not "
        "predict future ones. Use `python event_stats.py` to pool events across all analysed tickers."
    )
    return text


def _section_limitations(r: dict[str, Any]) -> str:
    items = [
        "Market data comes from Yahoo Finance through the unofficial `yfinance` library. It can contain "
        "errors, gaps or later revisions and is not an official exchange record.",
        f"Returns use the {r['data_sources']['price_column_description'].lower()}. The price chart and moving "
        "averages use the unadjusted close as displayed by the provider.",
        "Event detection is purely statistical. An event means the move or volume was unusual relative to "
        "the stock's own recent history - not that anything fundamental happened.",
        "Event causes are only linked to earnings-release DATES from Yahoo's earnings calendar. News, SEC "
        "filings, analyst actions, index changes and sector/market moves are not checked automatically.",
        "Z-scores assume the recent past is a fair baseline; after a quiet period, ordinary moves can score "
        "high, and during volatile periods large moves can score low.",
        "Fundamental data reflects what the provider reports at retrieval time; periods and definitions "
        "(e.g. Yahoo's 'levered' free cash flow) may differ from company filings.",
        "The abnormal return is a simple difference from the benchmark; it does not adjust for beta.",
    ]
    items += r["warnings"]
    return "## 9. Data Limitations\n\n" + "\n".join(f"- {item}" for item in items)


def build_summary_markdown(r: dict[str, Any]) -> str:
    """Assemble the full Markdown report."""
    files = r.get("files", {})
    name = (r.get("profile") or {}).get("name")
    title = f"# {r['ticker']}{f' ({name})' if name else ''} - Event-Driven Research Report"
    header = (
        f"{title}\n\n"
        f"Generated {r['generated_at']} UTC · data through {fmt_date(r['stats']['end_date'])}"
        + (f" · analysis run #{r['run_id']}" if r.get("run_id") else "") + "\n\n"
        "> **Descriptive research only.** This report describes what happened in the data. It contains no "
        "investment recommendations, price targets, predictions or trading signals.\n\n"
        "**Labels used:** **[Reported]** = taken directly from the data provider · **[Calculated]** = computed "
        "by this program (formulas in README.md) · **Unknown** / **Not available** = not available from the "
        "data used; never guessed."
    )
    sections = [
        header,
        _section_company(r),
        _section_market_overview(r),
        _section_performance(r),
        _section_volatility(r, files),
        _section_events(r, files),
        _section_fundamentals(r),
        _section_benchmark(r, files),
        _section_outcomes(r),
        _section_limitations(r),
    ]
    listing = "\n".join(f"- `{name}`" for name in sorted(set(files.values()) | {
        "summary.md", "events.csv", "fundamentals.csv", "analysis.json", "run.log"}))
    sections.append("## Files in this folder\n\n" + listing)
    return "\n\n".join(sections) + "\n"


def write_summary_markdown(results: dict[str, Any], path: Path) -> Path:
    path.write_text(build_summary_markdown(results), encoding="utf-8")
    return path

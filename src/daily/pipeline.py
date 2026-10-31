"""The daily research run (``python daily_run.py``) - read this file first.

    1. Identify the latest COMPLETED US session (exchange calendar + settle time)
    2. Update daily bars for indices, volatility, credit, sector ETFs and the stock universe
    3. Validate data quality; refuse to score stale / broken symbols
    4. Market regime (with evidence), sector rotation, index table, Treasury yields
    5. Features + setup detection for every stock; historical simulation of every setup
    6. Stage-1 technical score for every (stock, setup); liquidity filter
    7. Stage-2 deep dive for the shortlist: SEC fundamentals (point-in-time), options,
       positioning, catalysts -> penalties and supporting / contradicting evidence
    8. Final ranking per direction, tiers, historical setup statistics attached
    9. Save to the database and write reports/daily/<session>/ (JSON, Markdown, HTML)

Research and decision support only: nothing here connects to a broker or places orders.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from src.catalysts.catalysts import build_catalysts, macro_events
from src.daily import store
from src.daily.settings import all_symbols, path
from src.data import market_data, quality
from src.derivatives.options import analyze_chains
from src.fundamentals.sec_fundamentals import analyze_fundamentals, facts_from_yahoo
from src.market import calendar, regime
from src.positioning.positioning import analyze_positioning
from src.scanner.backtest import run_backtest
from src.scanner.features import build_features
from src.scanner.plan import plan_row
from src.scanner.scoring import apply_enrichment, build_candidate, cross_sectional_ranks, stats_note
from src.scanner.setups import SETUP_DIRECTION, SETUP_NAMES, active_setups, detect_setups

logger = logging.getLogger(__name__)
MACRO_PATH = Path(__file__).resolve().parents[2] / "config" / "macro_events.yaml"
DISCLAIMER = ("Research and decision-support output only - not investment advice, not a trade signal. "
              "No orders are placed. All statistics are historical and in-sample; markets change.")


class DailyRunError(Exception):
    pass


def _load_macro() -> list[dict[str, Any]]:
    if not MACRO_PATH.exists():
        return []
    with MACRO_PATH.open(encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("events", []) or []


def run_daily(settings: dict[str, Any], now: datetime | None = None, session: date | None = None,
              offline: bool = False, downloader: Any = None, enrichment: Any = None, fred_getter: Any = None,
              skip_deep: bool = False, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda msg: logger.info(msg))
    t0 = time.time()
    warnings: list[str] = []
    target = session or calendar.latest_completed_session(now, settings["data"].get("settle_minutes", 30))
    say(f"Target session: {target} ({target:%A})")
    conn = store.connect(path(settings, "database"), path(settings, "backups_dir"))
    run_id = store.start_run(conn, target)
    try:
        result = _run(conn, settings, target, offline, downloader, enrichment, fred_getter, skip_deep, say, warnings)
        result["run_id"] = run_id
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        out_dir = write_outputs(result, settings)
        status = "success_with_warnings" if warnings else "success"
        store.finish_run(conn, run_id, status, regime_label=result["regime"]["label"],
                         regime_score=result["regime"]["composite"], n_universe=len(settings["universe"]),
                         n_usable=result["quality"]["n_usable_stocks"],
                         n_candidates=len(result["candidates"]["long"]) + len(result["candidates"]["short"]),
                         report_path=str(out_dir), warnings=warnings)
        result["status"] = status
        result["output_dir"] = str(out_dir)
        say(f"Done in {result['elapsed_seconds']}s -> {out_dir}")
        return result
    except Exception as exc:
        store.finish_run(conn, run_id, "failed", error=f"{type(exc).__name__}: {exc}", warnings=warnings)
        raise
    finally:
        conn.close()


def _run(conn, settings, target, offline, downloader, enrichment, fred_getter, skip_deep, say, warnings):
    universe: dict[str, str] = settings["universe"]
    symbols = all_symbols(settings)
    sc = settings["scanner"]

    # 2. data update
    say(f"Updating prices for {len(symbols)} symbols{' (offline: skipped)' if offline else ''}")
    update = market_data.update_prices(conn, symbols, target, settings, downloader=downloader, offline=offline,
                                       progress=say)
    if update["failed"]:
        warnings.append(f"No fresh data for {len(update['failed'])} symbol(s): {', '.join(update['failed'][:15])}")
    if update["readjusted"]:
        warnings.append(f"Corporate action detected (history re-downloaded): {', '.join(update['readjusted'])}")
    prices = market_data.load_universe_prices(conn, symbols, target)

    # provider check: has SPY published the target session?
    if not quality.provider_has_session(prices, "SPY", target):
        spy = prices.get("SPY")
        if spy is None or spy.empty:
            raise DailyRunError("No SPY data - cannot analyse the market (check the internet connection).")
        latest = spy.index[-1].date()
        warnings.append(f"The data provider has not published {target} yet; the analysis uses the latest available "
                        f"session {latest}.")
        target = latest
        prices = {s: f[f.index <= pd.Timestamp(target)] for s, f in prices.items()}

    # 3. quality
    q = quality.check_universe(prices, symbols, target, int(settings["data"].get("min_history_days", 260)))
    store.save_quality(conn, target, q)
    status_by_symbol = {r["symbol"]: r["status"] for r in q["symbols"]}
    usable_stocks = [s for s in universe if status_by_symbol.get(s) != "excluded"]
    excluded = [r for r in q["symbols"] if r["status"] == "excluded"]
    if excluded:
        warnings.append(f"{len(excluded)} symbol(s) excluded by data-quality checks: "
                        + ", ".join(r["symbol"] for r in excluded[:15]))
    say(f"Data quality: {q['counts']}")

    # 4. regime
    yields = market_data.fetch_fred(settings["market"].get("yields_fred", []), target, getter=fred_getter) \
        if not offline else pd.DataFrame()
    if not offline and yields.empty:
        warnings.append("Treasury yields (FRED) unavailable - the rates component is excluded from the regime.")
    history = regime.compute_regime_history(prices, usable_stocks, settings, yields)
    reg = regime.summarize(history)
    store.save_regime(conn, history, list(regime.COMPONENT_WEIGHTS))
    index = prices["SPY"].index
    etfs = {**settings["market"]["sector_etfs"], **settings["market"].get("industry_etfs", {})}
    sectors = regime.sector_rotation(prices, etfs, index)
    quadrant = {s["symbol"]: s["quadrant"] for s in sectors}
    indices = regime.index_table(prices, [*settings["market"]["indices"], *settings["market"]["volatility"],
                                          *settings["market"]["credit_rates"]], index)
    say(f"Market regime: {reg['label']} ({reg['composite']:+.0f})")

    # 5. features, setups, history
    spy_close = prices["SPY"]["adj_close"]
    features: dict[str, pd.DataFrame] = {}
    setups: dict[str, pd.DataFrame] = {}
    for symbol in usable_stocks:
        etf = universe[symbol]
        sector_close = prices[etf]["adj_close"] if etf in prices else None
        try:
            f = build_features(prices[symbol], spy_close, sector_close)
            features[symbol], setups[symbol] = f, detect_setups(f)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{symbol}: feature calculation failed ({type(exc).__name__}: {exc})")
    say(f"Features for {len(features)} stocks; simulating setup history")
    bt = run_backtest(features, setups, sc, history["label"])
    trades_frame = bt.pop("trades_frame", None)

    # 6. stage 1
    latest = pd.DataFrame({s: f.iloc[-1] for s, f in features.items()}).T
    ranks = cross_sectional_ranks(_numeric(latest))
    liq = settings["liquidity"]
    candidates: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for symbol, f in features.items():
        row = f.iloc[-1]
        price, dv = float(row["close"]), float(row["dollar_volume20"]) if pd.notna(row["dollar_volume20"]) else 0.0
        names = active_setups(setups[symbol].iloc[-1])
        if not names:
            continue
        if price < liq["min_price"] or dv < liq["min_avg_dollar_volume"]:
            filtered.append({"symbol": symbol, "reason": f"illiquid (price {price:.2f}, $vol {dv / 1e6:.1f}M)"})
            continue
        best: dict[str, dict] = {}
        for name in names:
            plan = plan_row(f, name, sc, int(sc["plan_horizon_days"]))
            context = {"regime": reg["label"], "sector_quadrant": quadrant.get(universe[symbol]),
                       "sector_etf": universe[symbol], "data_quality": status_by_symbol.get(symbol, "ok")}
            cand = build_candidate(symbol, row, name, float(setups[symbol].iloc[-1][f"quality_{name}"]), plan,
                                   ranks.loc[symbol], context, settings)
            cand["other_setups"] = [n for n in names if n != name]
            d = cand["direction"]
            if d not in best or cand["score"] > best[d]["score"]:
                best[d] = cand
        candidates.extend(best.values())
    say(f"Stage 1: {len(candidates)} (stock, direction) candidates with an active setup")

    # 7. stage 2
    per_side = max(1, int(sc["shortlist_size"]) // 2)
    shortlist = []
    for d in ("long", "short"):
        side = sorted((c for c in candidates if c["direction"] == d), key=lambda c: -c["score"])
        shortlist.extend(side[:per_side])
    deep: dict[str, dict[str, Any]] = {}
    if not skip_deep and shortlist:
        if enrichment is None:
            from src.daily.enrichment import YahooSecEnrichment
            enrichment = YahooSecEnrichment(settings["data"]["sec_user_agent"], path(settings, "cache_dir"), target,
                                            offline=offline)
        symbols_deep = list(dict.fromkeys(c["symbol"] for c in shortlist))
        say(f"Stage 2: deep dive on {len(symbols_deep)} symbols")
        finra = getattr(enrichment, "finra", None)
        with ThreadPoolExecutor(max_workers=6) as pool:
            raws = dict(zip(symbols_deep, pool.map(enrichment.fetch, symbols_deep)))
        for symbol in symbols_deep:
            deep[symbol] = enrich(symbol, raws[symbol], features[symbol], target, settings, conn, finra)
        for c in shortlist:
            apply_enrichment(c, deep[c["symbol"]], settings)
            c["deep_dive"] = True
        sec_states = {d.get("sec_status") for d in deep.values()}
        if "not_configured" in sec_states:
            warnings.append("SEC EDGAR not used: set SEC_USER_AGENT=\"Your Name your@email\" (SEC fair-access policy). "
                            "Fundamentals fell back to Yahoo statements; SEC filings / dilution checks are missing.")
        elif "blocked" in sec_states:
            warnings.append("SEC EDGAR refused requests (HTTP 403) - check SEC_USER_AGENT. Fundamentals fell back to "
                            "Yahoo statements; SEC filings / dilution checks are missing.")
    elif skip_deep:
        warnings.append("Deep dive skipped (--skip-deep): no fundamentals / options / positioning / catalysts.")

    # 8. final ranking
    final = {"long": [], "short": []}
    keep = int(sc["final_list_size"])
    for d in ("long", "short"):
        side = sorted((c for c in candidates if c["direction"] == d),
                      key=lambda c: (-int(c.get("deep_dive", False)), -c["score"]))
        side = sorted(side[: max(keep, per_side)], key=lambda c: -c["score"])[:keep]
        for rank, c in enumerate(side, 1):
            c["rank"] = rank
            st = bt["by_setup"].get(c["setup"])
            st_reg = (bt["by_setup_regime"].get(c["setup"]) or {}).get(reg["label"])
            c["history"] = {"setup": st, "setup_in_regime": st_reg, "regime": reg["label"],
                            "note": stats_note(st), "note_regime": stats_note(st_reg)}
            c["thesis"] = thesis(c)
        final[d] = side
    store.save_candidates(conn, target, final["long"] + final["short"])

    universe_table = []
    for symbol, f in features.items():
        row = f.iloc[-1]
        universe_table.append({
            "symbol": symbol, "sector_etf": universe[symbol], "close": row["close"], "ret_1d": row["return_1d"],
            "ret_5d": row["return_5d"], "ret_20d": row["return_20d"], "rsi14": row["rsi14"],
            "rel_20d": row["rel_20d"], "rs_pct": ranks.loc[symbol, "rs_pct"], "volume_ratio": row["volume_ratio"],
            "extension_atr": row["extension_atr"], "trend": row["trend"], "structure": row["swing_structure"],
            "setups": active_setups(setups[symbol].iloc[-1]), "atr_pct": row["atr_pct"],
            "from_52w_high": row["from_52w_high"], "dollar_volume20": row["dollar_volume20"],
            "data_quality": status_by_symbol.get(symbol)})

    events = macro_events(target, _load_macro())
    upcoming_earnings = sorted(
        [{"symbol": s, **d["catalysts"]["next_earnings"]} for s, d in deep.items()
         if d.get("catalysts", {}).get("next_earnings")], key=lambda e: e["date"])
    return {
        "session": calendar.session_info(target),
        "generated_at": store.utc_now(),
        "disclaimer": DISCLAIMER,
        "regime": reg,
        "sectors": sectors,
        "indices": indices,
        "candidates": final,
        "deep_dive": deep,
        "universe": universe_table,
        "filtered_illiquid": filtered,
        "backtest": bt,
        "setup_names": SETUP_NAMES,
        "setup_directions": SETUP_DIRECTION,
        "macro_events": events,
        "upcoming_earnings": upcoming_earnings,
        "holidays": calendar.upcoming_holidays(target),
        "quality": {"counts": q["counts"], "coverage": q["coverage"], "symbols": q["symbols"],
                    "n_usable_stocks": len(usable_stocks), "update": update},
        "warnings": warnings,
        "settings": {k: settings[k] for k in ("weights", "penalties", "tiers", "scanner", "liquidity")},
        "trade_count": int(len(trades_frame)) if trades_frame is not None else 0,
    }


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("rel_5d", "rel_20d", "rel_60d", "rel_sector_20d"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def enrich(symbol: str, raw: dict[str, Any], f: pd.DataFrame, session: date, settings: dict[str, Any], conn,
           finra: Any) -> dict[str, Any]:
    row = f.iloc[-1]
    price = float(row["close"])
    out: dict[str, Any] = {}
    try:
        facts, source = raw.get("sec_facts"), "SEC EDGAR XBRL company facts"
        if facts is None and raw.get("yahoo_statements"):
            info = raw.get("info") or {}
            facts = facts_from_yahoo(raw["yahoo_statements"], info.get("sharesOutstanding"), session)
            source = ("Yahoo Finance quarterly statements (fallback: SEC unavailable). Publication dates ESTIMATED "
                      "as quarter end + 45 days.")
        out["fundamentals"] = analyze_fundamentals(facts, session, price)
        if out["fundamentals"].get("available"):
            out["fundamentals"]["source"] = source
        elif raw.get("sec_status") != "ok":
            out["fundamentals"]["reason"] = (f"SEC EDGAR {raw.get('sec_status', 'unavailable').replace('_', ' ')} and "
                                             "no usable Yahoo quarterly statements.")
    except Exception as exc:  # noqa: BLE001
        out["fundamentals"] = {"available": False, "reason": f"Fundamental analysis failed: {exc}"}
    out["sec_status"] = raw.get("sec_status", "ok")
    try:
        out["catalysts"] = build_catalysts(symbol, session, raw.get("earnings_dates"), raw.get("calendar"),
                                           raw.get("filings") or [], raw.get("upgrades"))
        store.save_catalysts(conn, symbol, out["catalysts"]["events"])
    except Exception as exc:  # noqa: BLE001
        out["catalysts"] = {"events": [], "filings": [], "analyst": [], "next_earnings": None,
                            "error": f"Catalyst analysis failed: {exc}"}
    chains = raw.get("option_chains")
    if chains:
        hv20 = float(row["hv20"]) if pd.notna(row["hv20"]) else None
        try:
            out["options"] = analyze_chains(chains, price, hv20, int(settings["scanner"]["plan_horizon_days"]),
                                            store.iv_history(conn, symbol, session), session)
            store.save_options(conn, symbol, session, out["options"])
        except Exception as exc:  # noqa: BLE001
            out["options"] = {"available": False, "reason": f"Options analysis failed: {exc}"}
    else:
        out["options"] = {"available": False, "reason": "No options data (none listed, or download failed / offline)."}
    try:
        volume = f["volume"]
        out["positioning"] = analyze_positioning(raw.get("info"), raw.get("insider"), raw.get("institutions"),
                                                 session, finra, symbol, volume)
    except Exception as exc:  # noqa: BLE001
        out["positioning"] = {"error": f"Positioning analysis failed: {exc}"}
    info = raw.get("info") or {}
    out["profile"] = {k: info.get(k) for k in ("longName", "sector", "industry", "marketCap", "exchange",
                                               "beta", "fullTimeEmployees", "website")}
    return out


def thesis(c: dict[str, Any]) -> str:
    """One-paragraph, evidence-based summary of the candidate (no advice language)."""
    p, t = c["plan"], c["technical"]
    side = "Long" if c["direction"] == "long" else "Short"
    parts = [f"{side} setup: {c['setup_name']} (quality {c['setup_quality']:.0f}/100)."]
    if p.get("conditional_entry"):
        parts.append(f"Conditional entry on a move through {p['entry']:.2f}.")
    parts.append(f"Plan: reference {p['entry']:.2f}, stop {p['stop']:.2f} ({p['risk_atr']:.1f} ATR, "
                 f"{p['risk_pct'] * 100:.1f}%), T1 {p['target1']:.2f} ({p['reward_risk']:.1f}R), "
                 f"T2 {p['target2']:.2f}, time stop {p['horizon_days']} sessions.")
    if t.get("rs_percentile") is not None:
        parts.append(f"RS percentile {t['rs_percentile']:.0f}; trend '{t.get('trend')}', structure "
                     f"{t.get('swing_structure')}.")
    h = (c.get("history") or {}).get("setup")
    if h and h.get("n"):
        parts.append(f"History of this setup (universe, {h['n']} trades): win {h['win_rate'] * 100:.0f}%, "
                     f"avg {h['avg_r']:+.2f}R.")
    parts.append(f"{len(c['supporting'])} supporting vs {len(c['contradicting'])} contradicting observations.")
    return " ".join(parts)


# ----------------------------------------------------------------------------- outputs
def write_outputs(result: dict[str, Any], settings: dict[str, Any]) -> Path:
    from src.daily.report import build_html, build_markdown

    out_dir = path(settings, "reports_dir") / result["session"]["date"]
    out_dir.mkdir(parents=True, exist_ok=True)
    data = store.jsonable(result)
    (out_dir / "data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    (out_dir / "report.md").write_text(build_markdown(data), encoding="utf-8")
    (out_dir / "report.html").write_text(build_html(data), encoding="utf-8")
    latest = path(settings, "reports_dir") / "latest.json"
    latest.write_text(json.dumps({"session": result["session"]["date"], "path": str(out_dir)}), encoding="utf-8")
    return out_dir

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
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from src.analyst.analyst import analyze
from src.catalysts.catalysts import build_catalysts, macro_events
from src.daily import store
from src.daily.settings import all_symbols, path
from src.data import market_data, quality
from src.derivatives.options import analyze_chains
from src.fundamentals.sec_fundamentals import analyze_fundamentals, facts_from_yahoo
from src.market import calendar, regime
from src.positioning.positioning import analyze_positioning
from src.catalysts.earnings_study import earnings_study
from src.data.profiles import industries, market_caps
from src.market import macro as macro_mod
from src.portfolio import journal
from src.portfolio.risk import portfolio_report
from src.positioning.ftd import analyze_ftd, load_ftd
from src.scanner.backtest import run_backtest
from src.scanner.universe_filter import apply_universe_filter
from src.scanner.features import build_features
from src.scanner.plan import plan_row
from src.scanner.scoring import apply_enrichment, backtest_setup_bonus, build_candidate, cross_sectional_ranks, stats_note
from src.scanner.setups import SETUP_DIRECTION, SETUP_NAMES, active_setups, detect_setups

logger = logging.getLogger(__name__)
MACRO_PATH = Path(__file__).resolve().parents[2] / "config" / "macro_events.yaml"
DISCLAIMER = ("Research and decision-support output only - not investment advice, not a trade signal. "
              "No orders are placed. All statistics are historical and in-sample; markets change.")


class DailyRunError(Exception):
    pass


class RunInProgressError(DailyRunError):
    """Another daily run (cron / launchd job, dashboard button, Dock launcher) holds the lock."""


@contextmanager
def run_lock(lock_dir: Path):
    """Exclusive, non-blocking file lock so two runs never write the database at the same time.
    The OS releases it automatically if the process dies. (No-op where fcntl is unavailable.)"""
    try:
        import fcntl
    except ImportError:  # Windows
        yield
        return
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / ".daily_run.lock", "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RunInProgressError("Another daily run is already in progress.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


LAB_DEFAULT = Path(__file__).resolve().parents[2] / "config" / "strategy_lab.default.json"
DISCOVERY_DEFAULT = Path(__file__).resolve().parents[2] / "config" / "discovery.default.json"
INSIDER_DEFAULT = Path(__file__).resolve().parents[2] / "config" / "insider.default.json"


def load_discovery(backtest_dir: Path) -> dict[str, Any] | None:
    """The newest discovery.json (styles not used by the terminal, incl. the index study), else the shipped copy."""
    for f in [*sorted(backtest_dir.glob("*/discovery.json"))[-1:], DISCOVERY_DEFAULT]:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("index"):
                return d
        except (OSError, ValueError):
            continue
    return None


def load_insider(backtest_dir: Path) -> dict[str, Any] | None:
    """The newest insider.json (SEC Form 4 insider-buying study), else the shipped copy."""
    for f in [*sorted(backtest_dir.glob("*/insider.json"))[-1:], INSIDER_DEFAULT]:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("styles"):
                return d
        except (OSError, ValueError):
            continue
    return None


def load_strategy_lab(backtest_dir: Path) -> dict[str, Any] | None:
    """The newest strategy_lab.json (chosen exits per setup, adopted indicator model). Until this machine has run
    the lab itself (monthly), the copy shipped with the code (config/strategy_lab.default.json) is used."""
    try:
        files = sorted(backtest_dir.glob("*/strategy_lab.json"))
        if files:
            return json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    try:
        return json.loads(LAB_DEFAULT.read_text(encoding="utf-8")) if LAB_DEFAULT.exists() else None
    except (OSError, ValueError):
        return None


def _load_macro() -> list[dict[str, Any]]:
    if not MACRO_PATH.exists():
        return []
    with MACRO_PATH.open(encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("events", []) or []


def run_daily(settings: dict[str, Any], now: datetime | None = None, session: date | None = None,
              offline: bool = False, downloader: Any = None, enrichment: Any = None, fred_getter: Any = None,
              skip_deep: bool = False, progress: Callable[[str], None] | None = None,
              market_cap_fetch: Callable[[str], float | None] | None = None) -> dict[str, Any]:
    with run_lock(path(settings, "database").parent):
        return _run_daily(settings, now, session, offline, downloader, enrichment, fred_getter, skip_deep, progress,
                          market_cap_fetch)


def _run_daily(settings, now, session, offline, downloader, enrichment, fred_getter, skip_deep, progress,
               market_cap_fetch=None):
    say = progress or (lambda msg: logger.info(msg))
    t0 = time.time()
    warnings: list[str] = []
    target = session or calendar.latest_completed_session(now, settings["data"].get("settle_minutes", 30))
    say(f"Target session: {target} ({target:%A})")
    conn = store.connect(path(settings, "database"), path(settings, "backups_dir"))
    run_id = store.start_run(conn, target)
    try:
        result = _run(conn, settings, target, offline, downloader, enrichment, fred_getter, skip_deep, say, warnings,
                      market_cap_fetch)
        result["run_id"] = run_id
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        out_dir = write_outputs(result, settings)
        status = "success_with_warnings" if warnings else "success"
        # record the session that was ACTUALLY analysed: when the provider had not published the requested
        # session yet (or the network was down) the run fell back to an older one, and --if-new must retry
        store.finish_run(conn, run_id, status, session_date=result["session"]["date"], regime_label=result["regime"]["label"],
                         regime_score=result["regime"]["composite"], n_universe=len(settings["universe"]),
                         n_usable=result["quality"]["n_usable_stocks"],
                         n_candidates=len(result["candidates"]["long"]) + len(result["candidates"]["short"]),
                         report_path=str(out_dir), warnings=warnings)
        result["status"] = status
        result["output_dir"] = str(out_dir)
        try:
            from src.daily import housekeeping
            hk = housekeeping.run(settings, conn)
            if any(hk.values()):
                say("Housekeeping: " + ", ".join(f"{k} {v}" for k, v in hk.items() if v))
        except Exception as exc:  # noqa: BLE001 - never fail a finished run over clean-up
            logger.warning("housekeeping failed: %s", exc)
        say(f"Done in {result['elapsed_seconds']}s -> {out_dir}")
        return result
    except Exception as exc:
        store.finish_run(conn, run_id, "failed", error=f"{type(exc).__name__}: {exc}", warnings=warnings)
        raise
    finally:
        conn.close()


def _run(conn, settings, target, offline, downloader, enrichment, fred_getter, skip_deep, say, warnings,
         market_cap_fetch=None):
    universe: dict[str, str] = settings["universe"]
    symbols = all_symbols(settings)
    sc = settings["scanner"]

    # 2. data update
    # symbols with an active price alert or an open journal trade keep updating even after they leave the
    # universe (monthly rebuild), so alerts still trigger and open trades stay marked to market
    try:
        tracked = {r[0] for r in conn.execute("SELECT symbol FROM price_alerts WHERE active = 1 UNION "
                                              "SELECT symbol FROM trade_journal WHERE status != 'closed'")}
    except Exception:  # noqa: BLE001
        tracked = set()
    extra = sorted(tracked - set(symbols))
    say(f"Updating prices for {len(symbols) + len(extra)} symbols{' (offline: skipped)' if offline else ''}")
    update = market_data.update_prices(conn, symbols + extra, target, settings, downloader=downloader, offline=offline,
                                       progress=say)
    failed = update["failed"]
    if not offline and len(failed) >= max(20, 0.8 * (len(symbols) + len(extra))):
        warnings.append("Almost every price download failed: no internet connection, or Yahoo changed its site and "
                        "yfinance needs an update (the hourly job upgrades it automatically).")
    elif failed:
        # names with no bar for 3+ weeks were delisted / acquired / renamed: they are excluded by the quality
        # check; the monthly rebuild drops them from the generated universe (core list: edit terminal.yaml)
        cutoff = (target - timedelta(days=21)).isoformat()
        dead = [s for s in failed
                if (last := conn.execute("SELECT MAX(date) FROM price_data WHERE ticker = ?", (s,)).fetchone()[0])
                and str(last)[:10] < cutoff]
        rest = [s for s in failed if s not in dead]
        if rest:
            warnings.append(f"No fresh data for {len(rest)} symbol(s): {', '.join(rest[:15])}")
        if dead:
            warnings.append(f"No data for 3+ weeks (probably delisted, acquired or renamed - skipped): "
                            f"{', '.join(dead[:15])}")
    if update["readjusted"]:
        warnings.append(f"Corporate action detected (history re-downloaded): {', '.join(update['readjusted'])}")
    prices = market_data.load_universe_prices(conn, symbols, target)

    # provider check: has SPY published the target session?
    if not quality.provider_has_session(prices, "SPY", target):
        spy = prices.get("SPY")
        if spy is None or spy.empty:
            raise DailyRunError("No SPY data - cannot analyse the market (check the internet connection).")
        latest = spy.index[-1].date()
        warnings.append(f"The provider has not published {target} yet; using the latest available session {latest}.")
        target = latest
        prices = {s: f[f.index <= pd.Timestamp(target)] for s, f in prices.items()}

    # 3. quality (cross-asset macro series trade on other calendars - context only, not quality-gated)
    macro_syms = set(settings["market"].get("macro", {}))
    q = quality.check_universe(prices, [s for s in symbols if s not in macro_syms], target,
                               int(settings["data"].get("min_history_days", 260)))
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
    macro_table = macro_mod.macro_table(prices, settings["market"].get("macro", {}), index)
    try:
        from src.backtest.discovery import index_timing_today
        index_timing = index_timing_today(prices, load_discovery(path(settings, "reports_dir").parent / "backtest"))
    except Exception as exc:  # noqa: BLE001 - context only
        logger.warning("index timing failed: %s", exc)
        index_timing = []
    breadth_ad = macro_mod.advance_decline(prices, usable_stocks, index)
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
            warnings.append(f"{symbol}: indicator calculation failed ({type(exc).__name__}: {exc})")
    # 5b. focus filter: high beta / high volatility / market cap / dollar volume
    uf_cfg = settings.get("universe_filter") or {}
    caps = (market_caps(list(features), path(settings, "cache_dir"), target,
                        int(uf_cfg.get("market_cap_refresh_days", 7)), offline=offline, fetch=market_cap_fetch)
            if uf_cfg.get("enabled", True) else {})
    universe_filter = apply_universe_filter(features, settings, caps)
    features = {s: features[s] for s in universe_filter["kept"]}
    setups = {s: setups[s] for s in universe_filter["kept"]}
    if not features:
        raise DailyRunError("No stock passed the high-beta / high-volatility filter - check universe_filter in config/terminal.yaml.")
    say(f"Focus filter: kept {len(features)}, excluded {len(universe_filter['excluded'])}")
    try:
        from src.market.universe_builder import load_auto
        seed = (load_auto() or {}).get("details") or {}
        inds = industries(list(features), path(settings, "cache_dir"), target, seed=seed, offline=offline)
    except Exception as exc:  # noqa: BLE001
        inds = {}
        warnings.append(f"Industry lookup failed ({type(exc).__name__}); industry groups fall back to sector ETFs.")
    lev = set(settings.get("leveraged_etfs") or [])
    inds = {s: ("Leveraged / inverse ETFs" if s in lev else v) for s, v in {**inds, **{x: None for x in lev if x in features}}.items()}
    groups = macro_mod.industry_groups(features, inds, {s: universe[s] for s in features})
    group_of = macro_mod.group_lookup(groups)
    say(f"Indicators computed for {len(features)} stocks; simulating setup history")
    bt = run_backtest(features, setups, sc, history["label"])
    trades_frame = bt.pop("trades_frame", None)

    # 5c. evidence-based setup weighting from the latest SL/TP backtest
    try:
        bt_dir = path(settings, "reports_dir").parent / "backtest"
        bt_session = json.loads((bt_dir / "latest.json").read_text())["session"]
        bt_full = json.loads((bt_dir / bt_session / "backtest.json").read_text())
    except (OSError, ValueError, KeyError):
        bt_full = None
    lab = load_strategy_lab(path(settings, "reports_dir").parent / "backtest")
    exit_rules = {s: r for s, r in ((lab or {}).get("exits") or {}).get("chosen", {}).items() if s != "all"}
    if lab and bt_full is not None:
        bt_full = {**bt_full, "lab": lab["exits"]}
    elif lab:
        bt_full = {"by_setup": {"_": {}}, "session": lab.get("session"), "lab": lab["exits"]}
    bonus, bonus_source = backtest_setup_bonus(bt_full, sc)
    settings = {**settings, "scanner": {**sc, "setup_bonus": bonus, "exit_rules": exit_rules}}
    sc = settings["scanner"]
    if exit_rules:
        say("Exits from the strategy lab: " + "; ".join(f"{s} {r['stop']}/{r['target']}/{r['horizon']}d"
                                                        for s, r in exit_rules.items()))
    say(f"Setup weighting ({bonus_source}): " + ", ".join(f"{k} {v:+g}" for k, v in sorted(bonus.items(), key=lambda x: -x[1])))

    # 6. stage 1
    latest = pd.DataFrame({s: f.iloc[-1] for s, f in features.items()}).T
    ranks = cross_sectional_ranks(_numeric(latest))
    liq = settings["liquidity"]
    candidates: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for symbol, f in features.items():
        row = f.iloc[-1]
        price, dv = float(row["close"]), float(row["dollar_volume20"]) if pd.notna(row["dollar_volume20"]) else 0.0
        allowed = set(sc.get("directions") or ["long", "short"])
        enabled = set(sc.get("enabled_setups") or SETUP_DIRECTION)
        names = [n for n in active_setups(setups[symbol].iloc[-1]) if SETUP_DIRECTION[n] in allowed and n in enabled]
        if not names:
            continue
        if price < liq["min_price"] or dv < liq["min_avg_dollar_volume"]:
            filtered.append({"symbol": symbol, "reason": f"illiquid (price {price:.2f}, $vol {dv / 1e6:.1f}M)"})
            continue
        best: dict[str, dict] = {}
        for name in names:
            plan = plan_row(f, name, sc, int((sc.get("setup_horizon") or {}).get(name, sc["plan_horizon_days"])))
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
    sides = [d for d in ("long", "short") if d in set(sc.get("directions") or ["long", "short"])]
    per_side = max(1, int(sc["shortlist_size"]) // len(sides))
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
        ftd = None
        sec = getattr(enrichment, "sec", None)
        if sec is not None and getattr(enrichment, "sec_enabled", False):
            try:
                ftd = load_ftd(sec, target)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"SEC fails-to-deliver data failed to load: {type(exc).__name__}")
        for symbol in symbols_deep:
            deep[symbol] = enrich(symbol, raws[symbol], features[symbol], target, settings, conn, finra, ftd)
        for c in shortlist:
            apply_enrichment(c, deep[c["symbol"]], settings)
            c["deep_dive"] = True
        sec_states = {d.get("sec_status") for d in deep.values()}
        if "not_configured" in sec_states:
            warnings.append("SEC EDGAR not used: set SEC_USER_AGENT=\"Your Name your@email\" (SEC fair-access policy). "
                            "Yahoo statements; SEC filings / dilution checks are missing.")
        elif "blocked" in sec_states:
            warnings.append("SEC EDGAR refused requests (HTTP 403) - check SEC_USER_AGENT. Fundamentals fell back to "
                            "Yahoo statements; SEC filings / dilution checks are missing.")
    elif skip_deep:
        warnings.append("Deep dive skipped (--skip-deep): no fundamentals / options / positioning / catalysts.")

    # 8. final ranking (+ analyst note for every listed candidate)
    sector_by_etf = {x["symbol"]: x for x in sectors}
    final = {"long": [], "short": []}
    keep = int(sc["final_list_size"])
    for d in ("long", "short"):
        side = sorted((c for c in candidates if c["direction"] == d),
                      key=lambda c: (-int(c.get("deep_dive", False)), -c["score"]))
        side = sorted(side[: max(keep, per_side)], key=lambda c: -c["score"])[:keep]
        for rank, c in enumerate(side, 1):
            c["rank"] = rank
            c["group"] = group_of.get(c["symbol"])
            st = bt["by_setup"].get(c["setup"])
            st_reg = (bt["by_setup_regime"].get(c["setup"]) or {}).get(reg["label"])
            c["history"] = {"setup": st, "setup_in_regime": st_reg, "regime": reg["label"],
                            "note": stats_note(st), "note_regime": stats_note(st_reg)}
            rule = (c["plan"].get("exit_rule") or {})
            oos = rule.get("oos") or {}
            if oos.get("n"):
                # the daily 2-year simulation uses the structural plan; the strategy lab tested THIS exit
                c["history"]["lab"] = oos
                c["history"]["note"] = (f"5-year test of this exit, out of sample: N={oos['n']}, win {oos['win_rate'] * 100:.0f}%, "
                                        f"avg {oos['avg_r']:+.2f}R")
            c["thesis"] = thesis(c)
            try:
                c["analyst"] = analyze(c["symbol"], features[c["symbol"]], c, deep.get(c["symbol"]), reg,
                                       sector_by_etf.get(universe[c["symbol"]]), spy_close,
                                       float(ranks.loc[c["symbol"], "rs_pct"]), [SETUP_NAMES[x] for x in [c["setup"], *c["other_setups"]]])
            except Exception as exc:  # noqa: BLE001 - the note is optional
                warnings.append(f"{c['symbol']}: analyst note failed ({type(exc).__name__}: {exc})")
        final[d] = side
    store.save_candidates(conn, target, final["long"] + final["short"])
    pcfg = settings.get("portfolio") or {}
    try:
        portfolio = portfolio_report(features, final, pcfg, {"long": float(reg.get("long_risk_budget") or 1),
                                                             "short": float(reg.get("short_risk_budget") or 1)},
                                     int(pcfg.get("candidates_per_side", 10)))
    except Exception as exc:  # noqa: BLE001
        portfolio = {"error": f"{type(exc).__name__}: {exc}"}
    news_market, news_digest = refresh_news(conn, settings, list(features), final, deep, offline, say)
    fired = journal.check_alerts(conn)
    alerts_recent = journal.recent_triggers(conn, target - timedelta(days=7))
    for a in fired:
        warnings.append(f"Price alert: {a['symbol']} {journal.ALERT_KINDS.get(a['kind'], a['kind']).lower()} {a['level']:g}"
                        f" ({a['triggered_session']}, {a['triggered_price']:.2f})" + (f" - {a['note']}" if a.get("note") else ""))

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
            "beta": row["beta120"], "hv60": row["hv60"], "market_cap": caps.get(symbol),
            "industry": inds.get(symbol), "group_rank": (group_of.get(symbol) or {}).get("rank"),
            "leveraged": symbol in set(settings.get("leveraged_etfs") or []),
            "data_quality": status_by_symbol.get(symbol)})

    configured = _load_macro()
    events = macro_events(target, configured)
    fomc = sorted(str(e.get("date")) for e in configured if "FOMC" in str(e.get("event", "")))
    if fomc and fomc[-1] < (target + timedelta(days=45)).isoformat():
        warnings.append(f"config/macro_events.yaml lists FOMC dates only up to {fomc[-1]}: add the next year's meetings "
                        "(federalreserve.gov/monetarypolicy/fomccalendars.htm) so rate decisions keep showing as events.")
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
        "macro": macro_table,
        "index_timing": index_timing,
        "advance_decline": breadth_ad,
        "industry_groups": groups,
        "portfolio": portfolio,
        "alerts_fired": fired,
        "market_news": news_market,
        "news_digest": news_digest,
        "alerts_recent": alerts_recent,
        "candidates": final,
        "deep_dive": deep,
        "universe": universe_table,
        "filtered_illiquid": filtered,
        "universe_filter": universe_filter,
        "backtest": bt,
        "setup_names": SETUP_NAMES,
        "setup_bonus": {"bonus": bonus, "source": bonus_source},
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
           finra: Any, ftd: pd.DataFrame | None = None) -> dict[str, Any]:
    row = f.iloc[-1]
    price = float(row["close"])
    out: dict[str, Any] = {}
    try:
        facts, source = raw.get("sec_facts"), "SEC EDGAR XBRL company facts"
        if facts is None and raw.get("yahoo_statements"):
            info = raw.get("info") or {}
            facts = facts_from_yahoo(raw["yahoo_statements"], info.get("sharesOutstanding"), session)
            source = "Yahoo Finance quarterly statements (fallback when SEC is unavailable). Publication dates estimated as quarter end + 45 days."
        out["fundamentals"] = analyze_fundamentals(facts, session, price)
        if out["fundamentals"].get("available"):
            out["fundamentals"]["source"] = source
        elif raw.get("sec_status") != "ok":
            out["fundamentals"]["reason"] = "SEC EDGAR is not configured or unavailable, and Yahoo has no usable quarterly statements."
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
    try:
        out["earnings_study"] = earnings_study(raw.get("earnings_dates"), f, session,
                                               (out.get("catalysts") or {}).get("next_earnings"), chains)
    except Exception as exc:  # noqa: BLE001
        out["earnings_study"] = {"available": False, "reason": f"Earnings study failed: {exc}"}
    out["ftd"] = analyze_ftd(ftd, symbol, f["volume"]) if ftd is not None else {
        "available": False, "reason": "No SEC fails-to-deliver data: set SEC_USER_AGENT in .env."}
    info = raw.get("info") or {}
    out["profile"] = {k: info.get(k) for k in ("longName", "sector", "industry", "marketCap", "exchange",
                                               "beta", "fullTimeEmployees", "website", "sharesOutstanding", "floatShares",
                                               "trailingPE", "forwardPE", "priceToBook", "dividendYield", "trailingAnnualDividendYield")}
    return out


def refresh_news(conn, settings: dict[str, Any], symbols: list[str], final: dict[str, list], deep: dict[str, Any],
                 offline: bool, say: Callable[[str], None]) -> tuple[list[dict], dict[str, Any]]:
    """Fetch news for every universe stock (+ Google News for listed candidates) and the market; store it,
    attach a digest to each candidate. Never fails the run."""
    from src.catalysts import news as news_mod
    cfg = settings.get("news") or {}
    if not cfg.get("enabled", True):
        return [], {}
    try:
        client = news_mod.NewsClient(path(settings, "cache_dir"), cache_minutes=int(cfg.get("cache_minutes", 50)),
                                     max_age_days=int(cfg.get("max_age_days", 21)), offline=offline)
        listed = [c for side in final.values() for c in side]
        names = {s: ((deep.get(s) or {}).get("profile") or {}).get("longName") for s in symbols}
        say(f"News: fetching headlines for {len(symbols)} stocks and the market")
        per = client.many(symbols, names, google_for={c["symbol"] for c in listed})
        for sym, items in per.items():
            news_mod.save(conn, sym, items)
        market = client.market_news()
        news_mod.save(conn, "_MARKET", market)
        marker = path(settings, "cache_dir") / "news" / "last_universe_refresh"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        for c in listed:
            items = per.get(c["symbol"], [])
            c["news"] = {**news_mod.summarize(items), "items": items[:12]}
            for flag, text in (("offering", "a share offering / dilution"), ("legal", "a lawsuit or investigation"),
                               ("guidance", "a guidance change")):
                hits = [x for x in items[:12] if flag in x["tags"] and x["sentiment"] == "negative"]
                if hits and flag in c["news"]["flags"]:
                    c["contradicting"].append(f"News in the last {c['news']['days']} days mentions {text}: \"{hits[0]['title']}\" ({hits[0].get('publisher') or ''}).")
        total = sum(len(v) for v in per.values())
        digest = {"stocks": len(per), "headlines": total, "with_news": sum(1 for v in per.values() if v),
                  "fetched_at": store.utc_now(), "sources": ["Yahoo Finance", "Nasdaq", "Google News"]}
        say(f"News: {total} headlines for {digest['with_news']} stocks, {len(market)} market headlines")
        return market[:60], digest
    except Exception as exc:  # noqa: BLE001
        logger.warning("news refresh failed: %s", exc)
        return [], {"error": f"{type(exc).__name__}: {exc}"}


def thesis(c: dict[str, Any]) -> str:
    """One-paragraph, evidence-based summary of the candidate (no advice language)."""
    p, t = c["plan"], c["technical"]
    side = "Long" if c["direction"] == "long" else "Short"
    parts = [f"{side} setup: {c['setup_name']} (quality {c['setup_quality']:.0f}/100)."]
    if p.get("conditional_entry"):
        parts.append(f"Conditional entry on a move through {p['entry']:.2f}.")
    parts.append(f"Plan: reference {p['entry']:.2f}, stop {p['stop']:.2f} ({p['risk_atr']:.1f} ATR, "
                 f"{p['risk_pct'] * 100:.1f}%), " + (f"T1 {p['target1']:.2f} ({p['reward_risk']:.1f}R), "
                 if p.get("target1") is not None else f"trailing stop {p.get('trail_atr')} ATR (no fixed target), ")
                 + (f"T2 {p['target2']:.2f}, " if p.get("target2") is not None else "")
                 + f"time stop {p['horizon_days']} sessions.")
    if t.get("rs_percentile") is not None:
        parts.append(f"RS percentile {t['rs_percentile']:.0f}; trend '{t.get('trend')}', structure "
                     f"{t.get('swing_structure')}.")
    h = (c.get("history") or {}).get("setup")
    if h and h.get("n"):
        parts.append(f"Setup history (universe, {h['n']} trades): win rate {h['win_rate'] * 100:.0f}%, average {h['avg_r']:+.2f}R.")
    parts.append(f"{len(c['supporting'])} supporting and {len(c['contradicting'])} contradicting points.")
    return "".join(parts)


# ----------------------------------------------------------------------------- outputs
def write_outputs(result: dict[str, Any], settings: dict[str, Any]) -> Path:
    from src.daily.report import build_html, build_markdown

    out_dir = path(settings, "reports_dir") / result["session"]["date"]
    out_dir.mkdir(parents=True, exist_ok=True)
    data = store.jsonable(result)
    (out_dir / "data.json.gz").unlink(missing_ok=True)       # a re-run of an old, compressed session
    store.write_atomic(out_dir / "data.json", json.dumps(data, indent=1))
    store.write_atomic(out_dir / "report.md", build_markdown(data))
    store.write_atomic(out_dir / "report.html", build_html(data))
    store.write_atomic(path(settings, "reports_dir") / "latest.json",
                       json.dumps({"session": result["session"]["date"], "path": str(out_dir)}))
    return out_dir

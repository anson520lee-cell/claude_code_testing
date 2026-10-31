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
from datetime import date, datetime
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
from src.i18n.zh import tr
from src.market import calendar, regime
from src.positioning.positioning import analyze_positioning
from src.data.profiles import market_caps
from src.scanner.backtest import run_backtest
from src.scanner.universe_filter import apply_universe_filter
from src.scanner.features import build_features
from src.scanner.plan import plan_row
from src.scanner.scoring import apply_enrichment, build_candidate, cross_sectional_ranks, stats_note
from src.scanner.setups import SETUP_DIRECTION, SETUP_NAMES, active_setups, detect_setups

logger = logging.getLogger(__name__)
MACRO_PATH = Path(__file__).resolve().parents[2] / "config" / "macro_events.yaml"
DISCLAIMER = ("只供研究和決策參考，不構成投資建議，也不是交易信號。系統不會下單。"
              "所有統計都是歷史及樣本內結果，市場會變。")


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
    say(f"目標交易日：{target}")
    conn = store.connect(path(settings, "database"), path(settings, "backups_dir"))
    run_id = store.start_run(conn, target)
    try:
        result = _run(conn, settings, target, offline, downloader, enrichment, fred_getter, skip_deep, say, warnings,
                      market_cap_fetch)
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
        say(f"完成，用時 {result['elapsed_seconds']} 秒 → {out_dir}")
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
    say(f"更新 {len(symbols)} 個代號的價格{'（離線：略過）' if offline else ''}")
    update = market_data.update_prices(conn, symbols, target, settings, downloader=downloader, offline=offline,
                                       progress=say)
    if update["failed"]:
        warnings.append(f"{len(update['failed'])} 個代號沒有取得新數據：{', '.join(update['failed'][:15])}")
    if update["readjusted"]:
        warnings.append(f"偵測到公司行動（拆股／派息），已重新下載歷史：{', '.join(update['readjusted'])}")
    prices = market_data.load_universe_prices(conn, symbols, target)

    # provider check: has SPY published the target session?
    if not quality.provider_has_session(prices, "SPY", target):
        spy = prices.get("SPY")
        if spy is None or spy.empty:
            raise DailyRunError("沒有 SPY 數據，無法分析市場（請檢查網絡連線）。")
        latest = spy.index[-1].date()
        warnings.append(f"數據商尚未發布 {target} 的數據，本次分析使用最新可用的交易日 {latest}。")
        target = latest
        prices = {s: f[f.index <= pd.Timestamp(target)] for s, f in prices.items()}

    # 3. quality
    q = quality.check_universe(prices, symbols, target, int(settings["data"].get("min_history_days", 260)))
    store.save_quality(conn, target, q)
    status_by_symbol = {r["symbol"]: r["status"] for r in q["symbols"]}
    usable_stocks = [s for s in universe if status_by_symbol.get(s) != "excluded"]
    excluded = [r for r in q["symbols"] if r["status"] == "excluded"]
    if excluded:
        warnings.append(f"{len(excluded)} 個代號因數據品質問題被排除："
                        + ", ".join(r["symbol"] for r in excluded[:15]))
    say(f"數據品質：正常 {q['counts']['ok']}，警告 {q['counts']['warning']}，排除 {q['counts']['excluded']}")

    # 4. regime
    yields = market_data.fetch_fred(settings["market"].get("yields_fred", []), target, getter=fred_getter) \
        if not offline else pd.DataFrame()
    if not offline and yields.empty:
        warnings.append("無法取得國債收益率（FRED），市場狀態暫不計入利率一項。")
    history = regime.compute_regime_history(prices, usable_stocks, settings, yields)
    reg = regime.summarize(history)
    store.save_regime(conn, history, list(regime.COMPONENT_WEIGHTS))
    index = prices["SPY"].index
    etfs = {**settings["market"]["sector_etfs"], **settings["market"].get("industry_etfs", {})}
    sectors = regime.sector_rotation(prices, etfs, index)
    quadrant = {s["symbol"]: s["quadrant"] for s in sectors}
    indices = regime.index_table(prices, [*settings["market"]["indices"], *settings["market"]["volatility"],
                                          *settings["market"]["credit_rates"]], index)
    say(f"市場狀態：{tr(reg['label'])}（{reg['composite']:+.0f}）")

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
            warnings.append(f"{symbol}：指標計算失敗（{type(exc).__name__}: {exc}）")
    # 5b. focus filter: high beta / high volatility / market cap / dollar volume
    uf_cfg = settings.get("universe_filter") or {}
    caps = (market_caps(list(features), path(settings, "cache_dir"), target,
                        int(uf_cfg.get("market_cap_refresh_days", 7)), offline=offline, fetch=market_cap_fetch)
            if uf_cfg.get("enabled", True) else {})
    universe_filter = apply_universe_filter(features, settings, caps)
    features = {s: features[s] for s in universe_filter["kept"]}
    setups = {s: setups[s] for s in universe_filter["kept"]}
    if not features:
        raise DailyRunError("沒有股票通過高 Beta／高波動篩選，請檢查 config/terminal.yaml 的 universe_filter 設定。")
    say(f"高 Beta／高波動篩選：保留 {len(features)} 檔，排除 {len(universe_filter['excluded'])} 檔")
    say(f"已計算 {len(features)} 檔股票的指標，正在回測形態歷史")
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
            filtered.append({"symbol": symbol, "reason": f"流動性不足（股價 {price:.2f}，日均成交額 {dv / 1e6:.1f}M）"})
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
    say(f"第一階段：{len(candidates)} 個出現形態的候選")

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
        say(f"第二階段：深入分析 {len(symbols_deep)} 檔股票")
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
            warnings.append("未使用 SEC EDGAR：請在 .env 設定 SEC_USER_AGENT=\"你的名字 你的電郵\"（SEC 使用規定）。"
                            "基本面暫用 Yahoo 數據，SEC 文件和股權稀釋檢查缺失。")
        elif "blocked" in sec_states:
            warnings.append("SEC EDGAR 拒絕了請求（HTTP 403），請檢查 SEC_USER_AGENT。"
                            "基本面暫用 Yahoo 數據，SEC 文件和股權稀釋檢查缺失。")
    elif skip_deep:
        warnings.append("已略過深入分析（--skip-deep）：沒有基本面、期權、持倉和催化劑數據。")

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
            st = bt["by_setup"].get(c["setup"])
            st_reg = (bt["by_setup_regime"].get(c["setup"]) or {}).get(reg["label"])
            c["history"] = {"setup": st, "setup_in_regime": st_reg, "regime": reg["label"],
                            "note": stats_note(st), "note_regime": stats_note(st_reg)}
            c["thesis"] = thesis(c)
            try:
                c["analyst"] = analyze(c["symbol"], features[c["symbol"]], c, deep.get(c["symbol"]), reg,
                                       sector_by_etf.get(universe[c["symbol"]]), spy_close,
                                       float(ranks.loc[c["symbol"], "rs_pct"]), [SETUP_NAMES[x] for x in [c["setup"], *c["other_setups"]]])
            except Exception as exc:  # noqa: BLE001 - the note is optional
                warnings.append(f"{c['symbol']}：分析師報告生成失敗（{type(exc).__name__}: {exc}）")
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
            "beta": row["beta120"], "hv60": row["hv60"], "market_cap": caps.get(symbol),
            "leveraged": symbol in set(settings.get("leveraged_etfs") or []),
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
        "universe_filter": universe_filter,
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
        facts, source = raw.get("sec_facts"), "SEC EDGAR XBRL 財務數據"
        if facts is None and raw.get("yahoo_statements"):
            info = raw.get("info") or {}
            facts = facts_from_yahoo(raw["yahoo_statements"], info.get("sharesOutstanding"), session)
            source = "Yahoo Finance 季度報表（SEC 不可用時的後備）。公布日期以季末 + 45 日估算。"
        out["fundamentals"] = analyze_fundamentals(facts, session, price)
        if out["fundamentals"].get("available"):
            out["fundamentals"]["source"] = source
        elif raw.get("sec_status") != "ok":
            out["fundamentals"]["reason"] = "SEC EDGAR 未啟用或不可用，Yahoo 也沒有可用的季度報表。"
    except Exception as exc:  # noqa: BLE001
        out["fundamentals"] = {"available": False, "reason": f"基本面分析失敗：{exc}"}
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
            out["options"] = {"available": False, "reason": f"期權分析失敗：{exc}"}
    else:
        out["options"] = {"available": False, "reason": "沒有期權數據（沒有上市期權、下載失敗或離線模式）。"}
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
    side = "做多" if c["direction"] == "long" else "做空"
    parts = [f"{side}形態：{c['setup_name']}（質素 {c['setup_quality']:.0f}/100）。"]
    if p.get("conditional_entry"):
        parts.append(f"條件進場：價格升穿 {p['entry']:.2f} 才進場。")
    parts.append(f"計劃：參考價 {p['entry']:.2f}，止損 {p['stop']:.2f}（{p['risk_atr']:.1f} ATR，{p['risk_pct'] * 100:.1f}%），"
                 f"目標一 {p['target1']:.2f}（{p['reward_risk']:.1f}R），目標二 {p['target2']:.2f}，"
                 f"{p['horizon_days']} 個交易日時間止損。")
    if t.get("rs_percentile") is not None:
        parts.append(f"相對強弱第 {t['rs_percentile']:.0f} 百分位；趨勢「{tr(t.get('trend'))}」，"
                     f"結構「{tr(t.get('swing_structure'))}」。")
    h = (c.get("history") or {}).get("setup")
    if h and h.get("n"):
        parts.append(f"此形態的歷史表現（股票池，{h['n']} 筆）：勝率 {h['win_rate'] * 100:.0f}%，平均 {h['avg_r']:+.2f}R。")
    parts.append(f"支持證據 {len(c['supporting'])} 項，反對證據 {len(c['contradicting'])} 項。")
    return "".join(parts)


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

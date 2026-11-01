"""Web dashboard API (FastAPI). Read-only research views + an optional "run now" trigger.

Endpoints (JSON unless noted):
    GET  /                          dashboard (single-page app)
    GET  /api/sessions              sessions with a stored daily report
    GET  /api/snapshot?session=     the full daily snapshot (data.json)
    GET  /api/chart/{symbol}        OHLCV + indicators + setup markers from the database
    GET  /api/regime/history        stored regime composite / label per date
    GET  /api/runs                  run log (terminal_runs)
    GET  /api/candidate-history/{symbol}   past appearances in the candidate lists
    GET  /api/options-history/{symbol}     stored ATM IV snapshots
    GET  /api/methodology           the documented rules (module docstrings)
    GET/POST /api/journal, PUT/DELETE /api/journal/{id}     your own trade journal (manual records)
    GET/POST /api/alerts, DELETE /api/alerts/{id}, POST /api/alerts/{id}/rearm   price alerts
    GET  /api/backtest              latest SL / TP validation backtest (python backtest.py)
    GET  /api/backtest/status, POST /api/backtest/run   re-run that backtest in the background
    GET  /api/run/status, POST /api/run    run the daily pipeline in the background
    GET  /reports/{session}/report.html    the printable report

No endpoint places orders or talks to a broker - this is a research terminal.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.daily import store
from src.daily.settings import load_terminal_settings, path
from src.database import db
from src.market import calendar
from src.scanner import setups as setups_module
from src.scanner.features import build_features
from src.scanner.setups import LONG_SETUPS, SHORT_SETUPS, detect_setups
from src.technical.indicators import adjusted_ohlcv, ema
from src.technical.structure import support_resistance

STATIC = Path(__file__).with_name("static")


def create_app(settings: dict[str, Any] | None = None) -> FastAPI:
    settings = settings or load_terminal_settings()
    reports_dir = path(settings, "reports_dir")
    db_path = path(settings, "database")
    app = FastAPI(title="Swing Research Terminal", version="1.0",
                  description="Quantitative swing-trading RESEARCH terminal. No order execution.")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    run_state: dict[str, Any] = {"running": False, "started_at": None, "finished_at": None, "status": None,
                                 "message": None, "log": []}
    lock = threading.Lock()

    def connect() -> sqlite3.Connection:
        if not db_path.exists():
            raise HTTPException(404, "No database yet - run `python daily_run.py` first.")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def sessions() -> list[str]:
        if not reports_dir.exists():
            return []
        return sorted((p.name for p in reports_dir.iterdir() if (p / "data.json").exists()), reverse=True)

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/sessions")
    def api_sessions() -> dict[str, Any]:
        return {"sessions": sessions()}

    @app.get("/api/snapshot")
    def api_snapshot(session: str | None = None) -> FileResponse:
        available = sessions()
        if not available:
            raise HTTPException(404, "No daily report yet - run `python daily_run.py` (or press Run in the header).")
        chosen = session or available[0]
        if chosen not in available:
            raise HTTPException(404, f"No report for session {chosen}")
        return FileResponse(reports_dir / chosen / "data.json", media_type="application/json")

    @app.get("/reports/{session}/report.html", response_class=HTMLResponse)
    def report_html(session: str) -> FileResponse:
        if session not in sessions():
            raise HTTPException(404, "Unknown session")
        return FileResponse(reports_dir / session / "report.html")

    @app.get("/api/chart/{symbol}")
    def api_chart(symbol: str, days: int = Query(180, ge=30, le=600), session: str | None = None) -> dict[str, Any]:
        symbol = symbol.upper()
        conn = connect()
        try:
            prices = db.load_prices(conn, symbol)
            spy = db.load_prices(conn, "SPY")
            etf = settings["universe"].get(symbol)
            sector = db.load_prices(conn, etf) if etf else None
        finally:
            conn.close()
        if prices.empty:
            raise HTTPException(404, f"No stored prices for {symbol}")
        if session:
            cut = pd.Timestamp(session)
            prices, spy = prices[prices.index <= cut], spy[spy.index <= cut]
            sector = sector[sector.index <= cut] if sector is not None else None
        if len(prices) < 60:
            raise HTTPException(422, f"Too little history for {symbol} ({len(prices)} sessions)")
        f = build_features(prices, spy["adj_close"] if not spy.empty else None,
                           sector["adj_close"] if sector is not None and not sector.empty else None)
        st = detect_setups(f)
        view = f.tail(days)
        sv = st.tail(days)

        def series(column: str) -> list[dict[str, Any]]:
            return [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
                    for d, v in view[column].items() if pd.notna(v) and np.isfinite(v)]

        candles = [{"time": d.strftime("%Y-%m-%d"), "open": round(float(r.open), 4), "high": round(float(r.high), 4),
                    "low": round(float(r.low), 4), "close": round(float(r.close), 4)} for d, r in view.iterrows()]
        volume = [{"time": d.strftime("%Y-%m-%d"), "value": float(r.volume),
                   "color": "rgba(38,166,154,0.55)" if r.close >= r.open else "rgba(239,83,80,0.55)"}
                  for d, r in view.iterrows() if pd.notna(r.volume)]
        markers = []
        for name in (*LONG_SETUPS, *SHORT_SETUPS):  # first day of each setup episode
            mask = sv[f"setup_{name}"].astype(bool)
            for d in sv.index[mask & ~mask.shift(1, fill_value=False)]:
                markers.append({"time": d.strftime("%Y-%m-%d"), "setup": name,
                                "direction": setups_module.SETUP_DIRECTION[name]})
        markers.sort(key=lambda m: m["time"])
        levels = support_resistance(f, 60, 3)
        last = f.iloc[-1]
        prev_close = float(f["close"].iloc[-2]) if len(f) > 1 else None
        hi, lo, cl, op, vo = (float(last[k]) for k in ("high", "low", "close", "open", "volume"))
        year = f.tail(252)
        quote = {"open": op, "high": hi, "low": lo, "close": cl, "prev_close": prev_close,
                 "change": cl - prev_close if prev_close else None, "change_pct": cl / prev_close - 1 if prev_close else None,
                 "volume": vo, "dollar_volume": vo * (hi + lo + cl) / 3, "avg_price": (hi + lo + cl) / 3,
                 "amplitude": (hi - lo) / prev_close if prev_close else None, "volume_ratio": last.get("volume_ratio"),
                 "high52": float(year["high"].max()), "low52": float(year["low"].min()),
                 "hist_high": float(f["high"].max()), "hist_low": float(f["low"].min()),
                 "hist_since": f.index[0].strftime("%Y-%m-%d"), "beta": last.get("beta120"), "hv20": last.get("hv20"),
                 "hv60": last.get("hv60"), "atr": last.get("atr")}
        from src.analyst.techniques import volume_profile
        from src.backtest.crash_study import crash_history
        try:
            crashes = crash_history(f)
        except Exception:  # noqa: BLE001 - optional
            crashes = None
        try:
            vp = volume_profile(f, 120)
            vp = {k: vp.get(k) for k in ("poc", "vah", "val", "window", "hvn", "vacuum")} if vp.get("poc") else None
        except Exception:  # noqa: BLE001 - optional overlay
            vp = None
        return store.jsonable({
            "symbol": symbol, "sector_etf": etf, "candles": candles, "volume": volume,
            "ema9": series("ema9"), "ema21": series("ema21"), "ema50": series("ema50"), "ema200": series("ema200"),
            "ema250": series("ema250"), "bb_upper": series("bb_upper"), "bb_lower": series("bb_lower"),
            "bb_mid": series("bb_mid"), "vwap20": series("vwap20"), "mavol20": series("mavol20"),
            "rsi6": series("rsi6"), "rsi14": series("rsi14"), "macd": series("macd"),
            "macd_signal": series("macd_signal"), "macd_hist": series("macd_hist"), "obv": series("obv"),
            "rel_20d": series("rel_20d"), "markers": markers,
            "levels": {side: [{"price": lv.price, "label": lv.label(), "distance_atr": lv.distance_atr}
                              for lv in levels[side]] for side in ("support", "resistance")},
            "last": {k: last.get(k) for k in ("close", "atr", "atr_pct", "rsi6", "rsi14", "trend", "swing_structure",
                                              "volume_ratio", "extension_atr", "hv20", "high252", "low252",
                                              "from_52w_high", "rel_20d", "rel_60d", "swing_condition", "structure")},
            "as_of": f.index[-1].strftime("%Y-%m-%d"), "quote": quote, "volume_profile": vp, "crash_history": crashes,
        })

    @app.get("/api/minicharts")
    def api_minicharts(symbols: str, days: int = Query(70, ge=20, le=250), session: str | None = None) -> dict[str, Any]:
        """Compact OHLC + EMA9/21/50 for many symbols at once (the candidate card wall)."""
        conn = connect()
        out: dict[str, Any] = {}
        try:
            for symbol in [s.strip().upper() for s in symbols.split(",") if s.strip()][:80]:
                prices = db.load_prices(conn, symbol)
                if session:
                    prices = prices[prices.index <= pd.Timestamp(session)]
                if len(prices) < 30:
                    continue
                column = "adj_close" if prices["adj_close"].notna().all() and (prices["adj_close"] > 0).all() else "close"
                o = adjusted_ohlcv(prices, column)
                lines = {span: ema(o["close"], span) for span in (9, 21, 50)}
                view = o.tail(days)
                r = lambda s: [round(float(v), 4) if pd.notna(v) else None for v in s]  # noqa: E731
                out[symbol] = {"t": [d.strftime("%Y-%m-%d") for d in view.index], "o": r(view["open"]),
                               "h": r(view["high"]), "l": r(view["low"]), "c": r(view["close"]),
                               **{f"ema{k}": r(v.tail(days)) for k, v in lines.items()}}
        finally:
            conn.close()
        return out

    @app.get("/api/i18n")
    def api_i18n() -> dict[str, str]:
        from src.i18n.zh import LABELS
        return LABELS

    analyst_cache: dict[tuple, Any] = {}

    @app.get("/api/analyst/{symbol}")
    def api_analyst(symbol: str, session: str | None = None, live: bool = False) -> dict[str, Any]:
        """Analyst note for any stored symbol. Uses the session snapshot (regime, sectors, deep dive) and,
        with live=true, fetches fundamentals / options / positioning / catalysts for names not in the deep dive."""
        from src.analyst.analyst import analyze
        from src.daily.pipeline import enrich

        symbol = symbol.upper()
        available = sessions()
        chosen = session if session in available else (available[0] if available else None)
        key = (symbol, chosen, live)
        if key in analyst_cache:
            return analyst_cache[key]
        snap = json.loads((reports_dir / chosen / "data.json").read_text()) if chosen else {}
        conn = connect()
        try:
            prices, spy = db.load_prices(conn, symbol), db.load_prices(conn, "SPY")
            etf = settings["universe"].get(symbol)
            sector_p = db.load_prices(conn, etf) if etf else None
        finally:
            conn.close()
        if prices.empty or len(prices) < 120:
            raise HTTPException(404, f"Not enough stored prices for {symbol} (needs at least 120 sessions)")
        if chosen:
            cut = pd.Timestamp(chosen)
            prices, spy = prices[prices.index <= cut], spy[spy.index <= cut]
            sector_p = sector_p[sector_p.index <= cut] if sector_p is not None else None
        spy_close = spy["adj_close"] if not spy.empty else None
        f = build_features(prices, spy_close, sector_p["adj_close"] if sector_p is not None and not sector_p.empty else None)
        cand = next((c for side in ("long", "short") for c in (snap.get("candidates") or {}).get(side, []) if c["symbol"] == symbol), None)
        deep = (snap.get("deep_dive") or {}).get(symbol)
        if deep is None and live:
            from src.daily.enrichment import YahooSecEnrichment
            sess_date = pd.Timestamp(chosen).date() if chosen else f.index[-1].date()
            fetcher = YahooSecEnrichment(settings["data"].get("sec_user_agent"), path(settings, "cache_dir"), sess_date)
            conn2 = store.connect(db_path, path(settings, "backups_dir"))
            try:
                deep = enrich(symbol, fetcher.fetch(symbol), f, sess_date, settings, conn2, fetcher.finra)
            finally:
                conn2.close()
        sector_row = next((x for x in snap.get("sectors", []) if x["symbol"] == etf), None)
        uni = next((u for u in snap.get("universe", []) if u["symbol"] == symbol), None)
        note = analyze(symbol, f, cand, deep, snap.get("regime"), sector_row, spy_close, (uni or {}).get("rs_pct"))
        out = store.jsonable({"note": note, "deep": deep, "candidate": cand})
        analyst_cache[key] = out
        return out

    @app.get("/api/regime/history")
    def api_regime_history(limit: int = Query(260, ge=10, le=2000)) -> dict[str, Any]:
        conn = connect()
        try:
            rows = conn.execute("SELECT date, composite, label, components FROM regime_history ORDER BY date DESC "
                                "LIMIT ?", (limit,)).fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            conn.close()
        return {"history": [{"date": r["date"], "composite": r["composite"], "label": r["label"],
                             "components": json.loads(r["components"] or "{}")} for r in reversed(rows)]}

    @app.get("/api/runs")
    def api_runs(limit: int = 50) -> dict[str, Any]:
        conn = connect()
        try:
            rows = conn.execute("SELECT * FROM terminal_runs ORDER BY run_id DESC LIMIT ?", (limit,)).fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            conn.close()
        return {"runs": [dict(r) for r in rows]}

    @app.get("/api/candidate-history/{symbol}")
    def api_candidate_history(symbol: str) -> dict[str, Any]:
        conn = connect()
        try:
            rows = conn.execute("SELECT session_date, direction, setup, score, tier, rank, entry, stop, target1, "
                                "reward_risk FROM scan_candidates WHERE symbol = ? ORDER BY session_date DESC LIMIT 60",
                                (symbol.upper(),)).fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            conn.close()
        return {"symbol": symbol.upper(), "history": [dict(r) for r in rows]}

    @app.get("/api/options-history/{symbol}")
    def api_options_history(symbol: str) -> dict[str, Any]:
        conn = connect()
        try:
            rows = conn.execute("SELECT * FROM options_snapshots WHERE symbol = ? ORDER BY date", (symbol.upper(),)).fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            conn.close()
        return {"symbol": symbol.upper(), "history": [dict(r) for r in rows]}

    @app.get("/api/methodology")
    def api_methodology() -> dict[str, Any]:
        from src.i18n.zh import methodology
        return {**methodology(),
                "weights": settings["weights"], "penalties": settings["penalties"], "tiers": settings["tiers"]}

    def tconn() -> sqlite3.Connection:
        return store.connect(db_path, path(settings, "backups_dir"))

    def guarded(fn, *args):
        conn = tconn()
        try:
            return fn(conn, *args)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            conn.close()

    @app.get("/api/journal")
    def api_journal() -> dict[str, Any]:
        from src.portfolio import journal
        return store.jsonable(guarded(journal.journal_review))

    @app.post("/api/journal")
    def api_journal_add(data: dict[str, Any] = Body(...)) -> dict[str, Any]:
        from src.portfolio import journal
        return {"id": guarded(journal.add_trade, data)}

    @app.put("/api/journal/{trade_id}")
    def api_journal_update(trade_id: int, data: dict[str, Any] = Body(...)) -> dict[str, Any]:
        from src.portfolio import journal
        guarded(journal.update_trade, trade_id, data)
        return {"ok": True}

    @app.delete("/api/journal/{trade_id}")
    def api_journal_delete(trade_id: int) -> dict[str, Any]:
        from src.portfolio import journal
        guarded(journal.delete_trade, trade_id)
        return {"ok": True}

    @app.get("/api/alerts")
    def api_alerts() -> dict[str, Any]:
        from src.portfolio import journal

        def work(conn):
            fired = journal.check_alerts(conn)
            return {"alerts": journal.list_alerts(conn), "fired": fired, "kinds": journal.ALERT_KINDS}
        return store.jsonable(guarded(work))

    @app.post("/api/alerts")
    def api_alerts_add(data: dict[str, Any] = Body(...)) -> dict[str, Any]:
        from src.portfolio import journal
        try:
            level = float(data.get("level"))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "level must be a number") from exc
        return {"id": guarded(journal.add_alert, str(data.get("symbol") or ""), str(data.get("kind") or ""), level, data.get("note"))}

    @app.delete("/api/alerts/{alert_id}")
    def api_alerts_delete(alert_id: int) -> dict[str, Any]:
        from src.portfolio import journal
        guarded(journal.delete_alert, alert_id)
        return {"ok": True}

    @app.post("/api/alerts/{alert_id}/rearm")
    def api_alerts_rearm(alert_id: int) -> dict[str, Any]:
        from src.portfolio import journal
        guarded(journal.rearm_alert, alert_id)
        return {"ok": True}

    backtest_dir = reports_dir.parent / "backtest"
    bt_state: dict[str, Any] = {"running": False, "started_at": None, "finished_at": None, "status": None, "log": []}

    @app.get("/api/backtest")
    def api_backtest() -> dict[str, Any]:
        latest = backtest_dir / "latest.json"
        if not latest.exists():
            return {"available": False, "reason": "No backtest yet: press Re-run backtest or run python backtest.py in a terminal."}
        session = json.loads(latest.read_text())["session"]
        f = backtest_dir / session / "backtest.json"
        if not f.exists():
            return {"available": False, "reason": f"Backtest file for {session} not found."}
        res = json.loads(f.read_text(encoding="utf-8"))
        keep = ("n", "win_rate", "avg_r", "t_stat", "profit_factor", "target_rate", "stop_rate", "avg_days")
        res["grid"] = [{**{k: g[k] for k in ("stop", "target", "horizon", "side")},
                        "is": {k: g["is"].get(k) for k in keep}, "oos": {k: g["oos"].get(k) for k in keep}} for g in res.get("grid", [])]
        res.pop("method", None)
        return res

    @app.get("/api/crash-study")
    def api_crash_study() -> dict[str, Any]:
        latest = backtest_dir / "latest.json"
        if not latest.exists():
            return {"available": False}
        f = backtest_dir / json.loads(latest.read_text())["session"] / "crash_study.json"
        if not f.exists():
            return {"available": False, "reason": "No crash study yet: run python crash_study.py."}
        res = json.loads(f.read_text(encoding="utf-8"))
        res["available"] = True
        return res

    @app.get("/api/backtest/status")
    def api_backtest_status() -> dict[str, Any]:
        return {k: (v[-30:] if k == "log" else v) for k, v in bt_state.items()}

    @app.post("/api/backtest/run")
    def api_backtest_run() -> dict[str, Any]:
        import subprocess
        import sys
        with lock:
            if bt_state["running"]:
                raise HTTPException(409, "A backtest is already running.")
            bt_state.update(running=True, started_at=datetime.now(timezone.utc).isoformat(), finished_at=None, status="running", log=[])

        def work() -> None:
            root = Path(__file__).resolve().parents[2]
            try:
                for script in ("backtest.py", "crash_study.py"):
                    proc = subprocess.Popen([sys.executable, script], cwd=root, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in proc.stdout or []:
                        bt_state["log"].append(line.rstrip())
                    if proc.wait() != 0:
                        bt_state["status"] = "failed"
                        return
                bt_state["status"] = "success"
            except Exception as exc:  # noqa: BLE001
                bt_state.update(status="failed")
                bt_state["log"].append(f"{type(exc).__name__}: {exc}")
            finally:
                bt_state.update(running=False, finished_at=datetime.now(timezone.utc).isoformat())

        threading.Thread(target=work, daemon=True).start()
        return {"started": True}

    @app.get("/api/run/status")
    def api_run_status() -> dict[str, Any]:
        return run_state

    def already_current() -> str | None:
        """Reason to skip an ``if_new`` run: the latest completed session is already reported,
        or a run finished in the last 30 minutes (the provider may not have published yet)."""
        target = calendar.latest_completed_session(settle_minutes=settings["data"].get("settle_minutes", 30))
        if target.isoformat() in sessions():
            return f"Session {target} is already up to date."
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            try:
                row = conn.execute("SELECT MAX(finished_at) FROM terminal_runs WHERE status LIKE 'success%'").fetchone()
            except sqlite3.OperationalError:
                row = None
            finally:
                conn.close()
            if row and row[0]:
                finished = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
                if (datetime.now(timezone.utc) - finished).total_seconds() < 1800:
                    return "A run finished less than 30 minutes ago."
        return None

    @app.post("/api/run")
    def api_run(skip_deep: bool = False, offline: bool = False, if_new: bool = False) -> dict[str, Any]:
        if if_new:
            reason = already_current()
            if reason:
                return {"started": False, "reason": reason}
        with lock:
            if run_state["running"]:
                raise HTTPException(409, "A run is already in progress.")
            run_state.update(running=True, started_at=datetime.now(timezone.utc).isoformat(), finished_at=None,
                             status="running", message=None, log=[])

        def work() -> None:
            from src.daily.pipeline import run_daily
            try:
                result = run_daily(settings, offline=offline, skip_deep=skip_deep,
                                   progress=lambda m: run_state["log"].append(m))
                run_state.update(status=result["status"], message=f"Session {result['session']['date']} done")
            except Exception as exc:  # noqa: BLE001
                run_state.update(status="failed", message=f"{type(exc).__name__}: {exc}")
                run_state["log"].append(traceback.format_exc())
            finally:
                run_state.update(running=False, finished_at=datetime.now(timezone.utc).isoformat())

        threading.Thread(target=work, daemon=True).start()
        return {"started": True}

    return app

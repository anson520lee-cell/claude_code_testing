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
from fastapi import FastAPI, HTTPException, Query
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
            "as_of": f.index[-1].strftime("%Y-%m-%d"),
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
        from src.i18n.zh import METHODOLOGY
        return {**METHODOLOGY,
                "weights": settings["weights"], "penalties": settings["penalties"], "tiers": settings["tiers"]}

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

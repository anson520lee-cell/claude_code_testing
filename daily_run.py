"""Daily research run for the swing trading terminal.

    python daily_run.py                    # latest completed US session
    python daily_run.py --offline          # re-analyse stored data, no internet
    python daily_run.py --skip-deep        # technical scan only (fast)
    python daily_run.py --if-new           # do nothing if this session was already processed (for cron)

Schedule after the close, e.g. crontab (server clock in UTC, 21:45 UTC = 16:45/17:45 New York):
    45 21 * * 1-5  cd /path/to/repo && .venv/bin/python daily_run.py --if-new >> data/daily_run.log 2>&1

Research / decision support only - this program never places orders or connects to a broker.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import traceback
from datetime import date

from src.config import ConfigError
from src.daily.pipeline import DailyRunError, run_daily
from src.daily.settings import load_terminal_settings, path
from src.market import calendar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily quantitative swing research run (no trading).")
    parser.add_argument("--config", help="terminal settings file (default config/terminal.yaml)")
    parser.add_argument("--session", help="analyse this session (YYYY-MM-DD) instead of the latest completed one")
    parser.add_argument("--offline", action="store_true", help="use stored data only (no downloads)")
    parser.add_argument("--skip-deep", action="store_true", help="skip fundamentals / options / positioning / catalysts")
    parser.add_argument("--skip-backtest", action="store_true", help="do not refresh the monthly SL / TP validation backtest")
    parser.add_argument("--if-new", action="store_true", help="exit if the session was already processed successfully")
    parser.add_argument("--rebuild-universe", action="store_true",
                        help="rebuild config/universe_auto.yaml now (otherwise automatically every 30 days)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = load_terminal_settings(args.config)
    except ConfigError as exc:
        print(f"Settings error: {exc}", file=sys.stderr)
        return 2
    session = date.fromisoformat(args.session) if args.session else None
    if session and not calendar.is_trading_day(session):
        print(f"{session} is not a US trading day.", file=sys.stderr)
        return 2
    if args.if_new:
        target = session or calendar.latest_completed_session(settle_minutes=settings["data"].get("settle_minutes", 30))
        db_file = path(settings, "database")
        if db_file.exists():
            with sqlite3.connect(db_file) as conn:
                try:
                    done = conn.execute("SELECT 1 FROM terminal_runs WHERE session_date = ? AND status LIKE 'success%'",
                                        (target.isoformat(),)).fetchone()
                except sqlite3.OperationalError:
                    done = None
            if done:
                print(f"Session {target} already processed - nothing to do.")
                return 0
    builder = settings.get("universe_builder") or {}
    if not args.offline and builder.get("enabled", True):
        from src.daily.settings import merge_auto_universe
        from src.market.universe_builder import build_universe, is_stale, load_auto
        if args.rebuild_universe or is_stale(load_auto(), date.today(), int(builder.get("refresh_days", 30))):
            try:
                settings["universe"] = dict(settings["core_universe"])
                build_universe(settings, progress=print)
                merge_auto_universe(settings)
            except Exception as exc:  # noqa: BLE001 - keep the previous universe
                print(f"Universe rebuild failed, keeping the current list: {type(exc).__name__}: {exc}", file=sys.stderr)
                settings = load_terminal_settings(args.config)
    try:
        result = run_daily(settings, session=session, offline=args.offline, skip_deep=args.skip_deep, progress=print)
    except DailyRunError as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    reg = result["regime"]
    print(f"\nSession {result['session']['date']}: regime {reg['label']} ({reg['composite']:+.0f})")
    for side in ("long", "short"):
        top = result["candidates"][side][:5]
        print(f"Top {side}: " + (", ".join(f"{c['symbol']}({c['score']:.0f}{c['tier']})" for c in top) or "none"))
    for w in result["warnings"]:
        print(f"warning: {w}")
    notify_alerts(result.get("alerts_fired") or [])
    print(f"Report: {result['output_dir']}/report.html  ·  Dashboard: python serve.py")
    if not (args.offline or args.skip_backtest) and backtest_stale(settings):
        print("\nSL/TP backtest is older than 30 days (or missing); re-running it (about 5 minutes)...")
        try:
            import backtest
            backtest.main([])
        except Exception as exc:  # noqa: BLE001 - the daily report is already written
            print(f"Backtest failed (the daily report is unaffected): {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


def notify_alerts(fired: list[dict]) -> None:
    """macOS notification for each price alert triggered by this run (no-op elsewhere)."""
    if not fired or sys.platform != "darwin":
        return
    import subprocess
    from src.portfolio.journal import ALERT_KINDS
    for a in fired[:5]:
        text = f"{a['symbol']} {ALERT_KINDS.get(a['kind'], a['kind']).lower()} {a['level']:g} ({a['triggered_price']:.2f})"
        text = text.replace('"', "").replace("\\", "")
        script = f'display notification "{text}" with title "Swing Terminal price alert"'
        try:
            subprocess.run(["osascript", "-e", script], check=False, timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass


def backtest_stale(settings: dict, max_age_days: int = 30) -> bool:
    import json
    latest = path(settings, "reports_dir").parent / "backtest" / "latest.json"
    try:
        session = date.fromisoformat(json.loads(latest.read_text())["session"])
    except (OSError, ValueError, KeyError):
        return True
    return (date.today() - session).days > max_age_days


if __name__ == "__main__":
    sys.exit(main())

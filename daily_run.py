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
    parser.add_argument("--news-only", action="store_true", help="only refresh news headlines (universe + market)")
    parser.add_argument("--if-new", action="store_true", help="exit if the session was already processed successfully")
    parser.add_argument("--restore-user-data", metavar="FILE",
                        help="re-import journal trades and price alerts from data/backups/user_data_*.json")
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
    if args.news_only:
        news_only(settings)
        return 0
    if args.restore_user_data:
        from src.daily import housekeeping, store
        conn = store.connect(path(settings, "database"), path(settings, "backups_dir"))
        try:
            print(f"Restored {housekeeping.restore_user_data(conn, args.restore_user_data)} journal / alert rows.")
        finally:
            conn.close()
        return 0
    if args.if_new:
        target = session or calendar.latest_completed_session(settle_minutes=settings["data"].get("settle_minutes", 30))
        db_file = path(settings, "database")
        if db_file.exists():
            done, fallbacks = session_status(db_file, target)
            if done or fallbacks >= 6:
                print(f"Session {target} already processed." if done else
                      f"The data provider still has no bars for {target} after {fallbacks} attempts "
                      "(market closure or provider gap) - not retrying until the next session.")
                if (settings.get("news") or {}).get("hourly_refresh", True):
                    try:
                        news_only(settings)
                    except Exception as exc:  # noqa: BLE001 - a news hiccup must not fail the hourly job
                        print(f"News refresh failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 0
    builder = settings.get("universe_builder") or {}
    if not args.offline and builder.get("enabled", True):
        from src.daily.settings import merge_auto_universe
        from src.market.universe_builder import build_universe, is_stale, load_auto
        failed = path(settings, "database").parent / ".universe_rebuild_failed"
        cooling = failed.exists() and (date.today() - date.fromtimestamp(failed.stat().st_mtime)).days < 3
        if args.rebuild_universe or (is_stale(load_auto(), date.today(), int(builder.get("refresh_days", 30)))
                                     and not cooling):
            try:
                settings["universe"] = dict(settings["core_universe"])
                build_universe(settings, progress=print)
                merge_auto_universe(settings)
                failed.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001 - keep the previous universe, retry in 3 days
                print(f"Universe rebuild failed, keeping the current list: {type(exc).__name__}: {exc}", file=sys.stderr)
                failed.parent.mkdir(parents=True, exist_ok=True)
                failed.write_text(f"{date.today()} {type(exc).__name__}: {exc}")
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
    if not args.offline and provider_broken(result):
        maybe_upgrade_yfinance(settings)
    print(f"Report: {result['output_dir']}/report.html  ·  Dashboard: python serve.py")
    if not (args.offline or args.skip_backtest) and backtest_stale(settings):
        print("\nSL/TP backtest is older than 30 days (or missing); re-running it with the crash study and the "
              "strategy lab (about 20-45 minutes)...")
        failed_marker = path(settings, "reports_dir").parent / "backtest" / "last_failure"
        try:
            import backtest
            import crash_study
            import strategy_lab
            if backtest.main([]) != 0:
                raise RuntimeError("backtest.py exited with an error")
            crash_study.main([])
            print("Strategy lab: exits per setup and indicator weights (about 15-40 minutes)...")
            strategy_lab.main([])
            import discover
            discover.main([])
            failed_marker.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - the daily report is already written
            failed_marker.parent.mkdir(parents=True, exist_ok=True)
            failed_marker.write_text(f"{date.today()} {type(exc).__name__}: {exc}")
            print(f"Backtest failed (the daily report is unaffected; retried in 3 days): {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    return 0


def provider_broken(result: dict) -> bool:
    """Almost every symbol came back empty (>= 80%, at least 20): Yahoo changed something, or no internet."""
    upd = (result.get("quality") or {}).get("update") or {}
    failed = len(upd.get("failed") or [])
    ok = len(upd.get("incremental") or []) + len(upd.get("full") or [])
    return failed >= 20 and failed >= 4 * ok


def maybe_upgrade_yfinance(settings: dict, cooldown_hours: float = 24) -> bool:
    """Yahoo changes its pages every few months and old yfinance versions then return nothing. When that
    happens, upgrade yfinance inside the app's own virtual environment (at most once a day) - the next
    hourly run, and the dashboard after its automatic restart, use the fixed version."""
    import subprocess
    import time as _time
    data_dir = path(settings, "database").parent
    marker = data_dir / ".yfinance_upgrade_attempt"
    if marker.exists() and _time.time() - marker.stat().st_mtime < cooldown_hours * 3600:
        return False
    data_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(date.today().isoformat())
    print("Almost every price download failed - trying to upgrade yfinance...")
    try:
        proc = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", "yfinance"],
                              capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"yfinance upgrade failed: {exc}", file=sys.stderr)
        return False
    if proc.returncode != 0:
        print("yfinance upgrade failed: " + (proc.stderr or proc.stdout or "").strip()[-400:], file=sys.stderr)
        return False
    (data_dir / ".deps_updated").touch()            # start_terminal.sh restarts the dashboard server
    print("yfinance upgraded; the next run uses the new version.")
    return True


def session_status(db_file, target: date) -> tuple[bool, int]:
    """(already analysed, number of runs that asked for ``target`` but only found older data)."""
    conn = sqlite3.connect(db_file, timeout=30)
    try:
        done = conn.execute("SELECT 1 FROM terminal_runs WHERE session_date = ? AND status LIKE 'success%'",
                            (target.isoformat(),)).fetchone() is not None
        fallbacks = conn.execute("SELECT COUNT(*) FROM terminal_runs WHERE requested_session = ? AND session_date != ? "
                                 "AND status LIKE 'success%'", (target.isoformat(), target.isoformat())).fetchone()[0]
    except sqlite3.OperationalError:
        done, fallbacks = False, 0
    finally:
        conn.close()
    return done, int(fallbacks)


def news_only(settings: dict, full: bool = False) -> None:
    """Refresh headlines: every call does the latest candidates (+ Google News) and the market; the whole
    universe is refreshed at most every ``universe_every_hours`` (default 6) unless ``full`` - so the hourly
    job stays at ~60 requests instead of ~500. A second refresh while one is running is skipped."""
    import fcntl
    import json
    import time as _time
    from src.catalysts import news as news_mod
    from src.daily import store
    cfg = settings.get("news") or {}
    lock_path = path(settings, "database").parent / ".news.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(lock_path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Another news refresh is running - skipped.")
        lock.close()
        return
    marker = path(settings, "cache_dir") / "news" / "last_universe_refresh"
    every = float(cfg.get("universe_every_hours", 6)) * 3600
    full = full or not marker.exists() or _time.time() - marker.stat().st_mtime > every
    client = news_mod.NewsClient(path(settings, "cache_dir"), cache_minutes=int(cfg.get("cache_minutes", 50)),
                                 max_age_days=int(cfg.get("max_age_days", 21)))
    cands: set[str] = set()
    reports = path(settings, "reports_dir")
    latest = sorted((p for p in reports.iterdir() if (p / "data.json").exists()), reverse=True) if reports.exists() else []
    if latest:
        try:
            snap = json.loads((latest[0] / "data.json").read_text())
            cands = {c["symbol"] for side in snap.get("candidates", {}).values() for c in side}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            cands = set()                 # unreadable snapshot: market news + (maybe) universe only
    conn = store.connect(path(settings, "database"), path(settings, "backups_dir"))
    try:
        symbols = list(settings["universe"]) if full else sorted(cands)
        per = client.many(symbols, google_for=cands)
        for sym, items in per.items():
            news_mod.save(conn, sym, items)
        market = client.market_news()
        news_mod.save(conn, "_MARKET", market)
        if full:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
    finally:
        conn.close()
        lock.close()
    print(f"News refreshed ({'whole universe' if full else 'candidates + market'}): "
          f"{sum(len(v) for v in per.values())} stock headlines, {len(market)} market headlines.")


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
    failed = latest.parent / "last_failure"
    if failed.exists() and (date.today() - date.fromtimestamp(failed.stat().st_mtime)).days < 3:
        return False                      # a recent attempt failed: do not spend 10 minutes on every run
    try:
        session = date.fromisoformat(json.loads(latest.read_text())["session"])
    except (OSError, ValueError, KeyError):
        return True
    return (date.today() - session).days > max_age_days


if __name__ == "__main__":
    sys.exit(main())

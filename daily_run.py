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
    parser.add_argument("--if-new", action="store_true", help="exit if the session was already processed successfully")
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
                print(f"交易日 {target} 已經處理過，不需再運行。")
                return 0
    try:
        result = run_daily(settings, session=session, offline=args.offline, skip_deep=args.skip_deep, progress=print)
    except DailyRunError as exc:
        print(f"運行失敗：{exc}", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    reg = result["regime"]
    from src.i18n.zh import tr
    print(f"\n交易日 {result['session']['date']}：市場狀態 {tr(reg['label'])}（{reg['composite']:+.0f}）")
    for side in ("long", "short"):
        top = result["candidates"][side][:5]
        print(f"{'做多' if side == 'long' else '做空'}前列：" + ("、".join(f"{c['symbol']}（{c['score']:.0f} 分 {tr(c['tier'])}）" for c in top) or "沒有"))
    for w in result["warnings"]:
        print(f"提示：{w}")
    print(f"報告：{result['output_dir']}/report.html  ·  網頁介面：python serve.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

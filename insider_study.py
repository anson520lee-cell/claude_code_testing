"""Insider buying study (research only) - see src/backtest/insider.py.

    python insider_study.py            # ~10-20 minutes the first time: SEC Form 4 data sets since 2020 (~300 MB,
                                       # cached forever) + Yahoo bars for every company with insider purchases
    python insider_study.py --offline  # cached files only
Needs SEC_USER_AGENT (SEC fair-access policy). Writes reports/backtest/<session>/insider.json (Research page).
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from datetime import date

import pandas as pd

from src.backtest import insider
from src.daily import store
from src.daily.settings import all_symbols, load_terminal_settings, path
from src.data import market_data
from src.data.sec import SecClient
from src.market import calendar, regime


def quarter_urls(client: SecClient, first_year: int) -> list[str]:
    r = client.http.get(insider.DATASET_PAGE, headers=client.headers, timeout=60)
    r.raise_for_status()
    links = sorted(set(re.findall(r'href="(/files/[^"]*?(\d{4})q\d_form345\.zip)"', r.text)))
    return ["https://www.sec.gov" + u for u, y in links if int(y) >= first_year]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Insider buying study (research only)")
    ap.add_argument("--first-year", type=int, default=2020)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--validate-from", type=int, default=2010,
                    help="also check the CEO/CFO style on filings from this year to 2019 (independent period)")
    args = ap.parse_args(argv)
    t0 = time.time()
    settings = load_terminal_settings()
    agent = settings["data"].get("sec_user_agent")
    if not agent:
        print("Set SEC_USER_AGENT=\"Your Name your@email\" in .env first (SEC fair-access policy).", file=sys.stderr)
        return 2
    session = calendar.latest_completed_session()
    cache = path(settings, "cache_dir")
    client = SecClient(agent, cache, offline=args.offline)
    files = sorted((client.cache_dir / "files").glob("*_form345.zip"))
    if not args.offline:
        urls = quarter_urls(client, args.first_year)
        print(f"SEC insider data sets: {len(urls)} quarters (downloaded once, then cached)...")
        for u in urls:
            client.get_file(u, u.rsplit("/", 1)[1])
        files = sorted((client.cache_dir / "files").glob("*_form345.zip"))
    first = min(args.first_year, args.validate_from)
    if not args.offline:
        for u in quarter_urls(client, first):
            if int(u.rsplit("/", 1)[1][:4]) < args.first_year:
                client.get_file(u, u.rsplit("/", 1)[1])
        files = sorted((client.cache_dir / "files").glob("*_form345.zip"))
    old_files = [f for f in files if args.validate_from <= int(f.name[:4]) < args.first_year]
    files = [f for f in files if int(f.name[:4]) >= args.first_year]
    if not files:
        print(f"No SEC insider data ({client.last_error or 'nothing cached'}).", file=sys.stderr)
        return 3
    buys = pd.concat([insider.parse_quarter(f.read_bytes()) for f in files], ignore_index=True).drop_duplicates()
    events = insider.insider_events(buys)
    print(f"{len(buys)} open-market purchases, {len(events)} insider signals in {events['sym'].nunique()} stocks")

    conn = store.connect(path(settings, "database"), path(settings, "backups_dir"))
    prices = market_data.load_universe_prices(conn, all_symbols(settings), session, max_days=2000)
    conn.close()
    hist = regime.compute_regime_history(prices, [s for s in settings["universe"] if s in prices], settings)
    pfile = cache / "insider_prices.pkl"
    other: dict = {}
    if pfile.exists():
        try:
            other = pickle.loads(pfile.read_bytes())
        except Exception:  # noqa: BLE001 - corrupt cache: download again
            other = {}
    need = sorted(set(events["sym"]) - set(prices))
    stale = not other or max((df.index.max() for df in other.values()), default=pd.Timestamp.min) < pd.Timestamp(session) - pd.Timedelta(days=7)
    if not args.offline and stale:
        print(f"Downloading daily bars for {len(need)} more stocks (Yahoo, about 5-10 minutes)...")
        other = market_data.YahooBatchDownloader().download(need, start=date(args.first_year - 1, 1, 1))
        store.write_atomic(pfile, pickle.dumps(other))
    allp = {**other, **{s: p for s, p in prices.items() if s in set(events["sym"])}}
    data = insider.collect(events, allp, prices["SPY"]["adj_close"], hist["label"], set(settings["universe"]))
    slip = float(settings["scanner"].get("slippage_bps", 10)) / 10_000
    res = insider.run(data, slip, progress=print)
    if old_files:
        print(f"Independent check on {len(old_files)} older quarters...")
        ob = pd.concat([insider.parse_quarter(f.read_bytes()) for f in old_files + files], ignore_index=True).drop_duplicates()
        oev = insider.insider_events(ob)
        oev = oev[oev["style"] == "ins_ceo_100k"]
        ofile = cache / "insider_prices_old.pkl"
        oprices: dict = pickle.loads(ofile.read_bytes()) if ofile.exists() else {}
        if not args.offline and not oprices:
            oprices = market_data.YahooBatchDownloader().download(sorted(set(oev["sym"])), start=date(args.validate_from - 1, 1, 1))
            store.write_atomic(ofile, pickle.dumps(oprices))
        odata = insider.collect(oev, {**allp, **oprices}, prices["SPY"]["adj_close"], hist["label"], set(settings["universe"]))
        res["period_check"] = insider.period_check(odata, slip)
        res["verdict"] = insider.verdict(res)
    res.update({"generated_at": store.utc_now(), "session": session.isoformat(), "elapsed_seconds": round(time.time() - t0, 1),
                "n_purchases": int(len(buys)), "quarters": [f.name[:6] for f in files]})
    out = path(settings, "reports_dir").parent / "backtest" / session.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    store.write_atomic(out / "insider.json", json.dumps(store.jsonable(res), ensure_ascii=False, indent=1))
    print("\n".join(["", "= Verdict =", *res["verdict"]]))
    print(f"\nDone in {res['elapsed_seconds']} s -> {out}/insider.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

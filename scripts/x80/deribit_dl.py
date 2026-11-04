"""python -I scripts/x80/deribit_dl.py OUTDIR : Deribit DVOL (30-day implied volatility index) daily OHLC for BTC and ETH.

Public endpoint, no key: /api/v2/public/get_volatility_index_data. History starts 2021-03-24. Writes OUTDIR/dvol.csv.
"""
import csv
import json
import sys
import time
import urllib.request

URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data?currency={c}&start_timestamp={a}&end_timestamp={b}&resolution=1D"


def fetch(cur: str, a: int, b: int) -> list:
    rows, end = [], b
    while True:
        with urllib.request.urlopen(URL.format(c=cur, a=a, b=end), timeout=60) as resp:
            res = json.load(resp)["result"]
        rows += res["data"]
        cont = res.get("continuation")
        if not cont or cont <= a or not res["data"]:
            return rows
        end = cont
        time.sleep(0.3)


def main() -> None:
    out = sys.argv[1]
    a, b = 1609459200000, int(time.time() * 1000)
    with open(f"{out}/dvol.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["currency", "time_ms", "open", "high", "low", "close"])
        for cur in ("BTC", "ETH"):
            data = sorted({r[0]: r for r in fetch(cur, a, b)}.values())
            for r in data:
                w.writerow([cur, *r])
            print(cur, len(data), "days", data[0][0] if data else None, data[-1][0] if data else None)


if __name__ == "__main__":
    main()

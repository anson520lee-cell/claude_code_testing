"""python -I scripts/x80/binance_perp_1d.py OUTDIR FUNDING_PARQUET : daily klines of every Binance USDT-M perpetual that ever
paid funding (delisted ones included), for the months between its first and last funding print. Public archive
data.binance.vision/data/futures/um/monthly/klines/SYM/1d/. Writes OUTDIR/perp_1d.parquet."""
import concurrent.futures as cf
import io
import os
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from binance_dl import get

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"]


def month(sym: str, ym: str) -> pd.DataFrame | None:
    r = get(f"https://data.binance.vision/data/futures/um/monthly/klines/{sym}/1d/{sym}-1d-{ym}.zip")
    if r is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0]).decode()
    hdr = raw.split("\n", 1)[0].startswith("open_time")
    df = pd.read_csv(io.StringIO(raw), header=0 if hdr else None)
    df.columns = COLS[: df.shape[1]]
    t = pd.to_numeric(df["open_time"])
    df["date"] = pd.to_datetime(t, unit="us" if t.iloc[0] > 1e14 else "ms").dt.floor("D")
    df["symbol"] = sym
    return df[["symbol", "date", "open", "high", "low", "close", "quote_volume"]]


def main() -> None:
    out, fpath = sys.argv[1:3]
    f = pd.read_parquet(fpath, columns=["symbol", "time"])
    span = f.groupby("symbol")["time"].agg(["min", "max"])
    jobs = []
    for sym, (a, b) in span.iterrows():
        for p in pd.period_range(pd.Timestamp(a).strftime("%Y-%m"), pd.Timestamp(b).strftime("%Y-%m"), freq="M"):
            jobs.append((sym, p.strftime("%Y-%m")))
    print(len(span), "symbols,", len(jobs), "month files")
    with cf.ThreadPoolExecutor(32) as ex:
        parts = [d for d in ex.map(lambda a: month(*a), jobs) if d is not None]
    df = pd.concat(parts).sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    df.to_parquet(f"{out}/perp_1d.parquet", index=False)
    print(df["symbol"].nunique(), "symbols with klines,", len(df), "rows", df["date"].min(), df["date"].max())


if __name__ == "__main__":
    main()

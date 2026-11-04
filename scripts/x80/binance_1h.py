"""python -I scripts/x80/binance_1h.py OUTDIR SYMBOL... : Binance USDT-M perpetual 1-hour klines from the public archive
(data.binance.vision/data/futures/um/monthly/klines/SYM/1h/), parsed in memory, written to OUTDIR/perp_1h.parquet."""
import concurrent.futures as cf
import io
import sys
import zipfile

import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__file__))
from binance_dl import get

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"]


def month(sym: str, ym: str) -> pd.DataFrame | None:
    r = get(f"https://data.binance.vision/data/futures/um/monthly/klines/{sym}/1h/{sym}-1h-{ym}.zip")
    if r is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0]).decode()
    first = raw.split("\n", 1)[0]
    df = pd.read_csv(io.StringIO(raw), header=0 if first.startswith("open_time") else None, names=None if first.startswith("open_time") else COLS)
    df.columns = COLS[: df.shape[1]]
    t = pd.to_numeric(df["open_time"])
    df["time"] = pd.to_datetime(t, unit="us" if t.iloc[0] > 1e14 else "ms")
    df["symbol"] = sym
    return df[["symbol", "time", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote_volume"]]


def main() -> None:
    out, syms = sys.argv[1], sys.argv[2:]
    months = [p.strftime("%Y-%m") for p in pd.period_range("2019-09", pd.Timestamp.today().strftime("%Y-%m"), freq="M")]
    jobs = [(s, m) for s in syms for m in months]
    with cf.ThreadPoolExecutor(16) as ex:
        parts = [d for d in ex.map(lambda a: month(*a), jobs) if d is not None]
    df = pd.concat(parts).sort_values(["symbol", "time"]).drop_duplicates(["symbol", "time"])
    df.to_parquet(f"{out}/perp_1h.parquet", index=False)
    print(df.groupby("symbol")["time"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()

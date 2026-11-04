"""Download Binance public archive (data.binance.vision): spot 1d klines for every USDT pair (delisted ones included),
USDT-M funding rates and 1d premium-index klines. Parsed in memory (CSV inside zip), written to parquet.
usage: python -I binance_dl.py OUTDIR [spot|funding|premium]..."""
import concurrent.futures as cf
import io
import os
import re
import sys
import time
import zipfile

import pandas as pd
import requests

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DL = "https://data.binance.vision/"
STABLE = {"USDC", "BUSD", "TUSD", "PAX", "USDP", "DAI", "FDUSD", "USDS", "USDSB", "SUSD", "EUR", "GBP", "AUD", "TRY", "BRL",
          "RUB", "UAH", "NGN", "ZAR", "BIDR", "IDRT", "BVND", "VAI", "UST", "AEUR", "EURI", "XUSD", "USD1", "PYUSD", "BFUSD",
          "RLUSD", "USDE", "PAXG", "WBTC", "WBETH", "BKRW", "JPY", "MXN", "PLN", "RON", "ARS", "COP", "CZK"}
sess = requests.Session()
sess.mount("https://", requests.adapters.HTTPAdapter(pool_connections=48, pool_maxsize=48))


def get(url, **kw):
    for i in range(5):
        try:
            r = sess.get(url, timeout=60, **kw)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def listing(prefix, what="Prefix"):
    out, marker = [], ""
    while True:
        r = get(S3, params={"delimiter": "/", "prefix": prefix, "marker": marker})
        txt = r.text
        items = re.findall(rf"<{what}>([^<]*)</{what}>", txt)
        if what == "Prefix":
            items = [x for x in items if x != prefix]
        out += items
        if "<IsTruncated>true</IsTruncated>" not in txt:
            return out
        m = re.search(r"<NextMarker>([^<]*)</NextMarker>", txt)
        marker = m.group(1) if m else items[-1]


def keep(sym):
    if not sym.endswith("USDT"):
        return False
    base = sym[:-4]
    if base in STABLE or not base:
        return False
    return not re.search(r"(UP|DOWN|BULL|BEAR)$", base) or base in {"JUP", "SUP", "PUP"}


def read_zip(content, names):
    z = zipfile.ZipFile(io.BytesIO(content))
    raw = z.read(z.namelist()[0]).decode()
    first = raw.split("\n", 1)[0]
    header = 0 if re.search(r"[a-z]", first) else None
    df = pd.read_csv(io.StringIO(raw), header=header)
    if header is None:
        df.columns = names[: df.shape[1]]
    return df


KL = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume",
      "taker_buy_quote_volume", "ignore"]


def to_dt(x):
    x = pd.to_numeric(x)
    us = x > 1e14                                    # 2025+ spot files use microseconds
    return pd.to_datetime((x.where(~us, x // 1000)), unit="ms")


def job_symbol(kind, sym):
    if kind == "spot":
        pre = f"data/spot/monthly/klines/{sym}/1d/"
    elif kind == "premium":
        pre = f"data/futures/um/monthly/premiumIndexKlines/{sym}/1d/"
    else:
        pre = f"data/futures/um/monthly/fundingRate/{sym}/"
    keys = [k for k in listing(pre, "Key") if k.endswith(".zip")]
    frames = []
    for k in keys:
        r = get(DL + k)
        if r is None:
            continue
        try:
            df = read_zip(r.content, KL if kind != "funding" else ["calc_time", "funding_interval_hours", "last_funding_rate"])
        except Exception:
            continue
        frames.append(df)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    if kind == "funding":
        d = d.rename(columns={"calc_time": "time", "last_funding_rate": "rate"})
        d["time"] = to_dt(d["time"])
        out = d[["time", "rate"]].copy()
    else:
        d["date"] = to_dt(d["open_time"]).dt.normalize()
        cols = ["open", "high", "low", "close", "volume", "quote_volume"] if kind == "spot" else ["open", "high", "low", "close"]
        out = d[["date"] + cols].copy()
        for c in cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["symbol"] = sym
    return out


def run(kind, outdir):
    root = {"spot": "data/spot/monthly/klines/", "funding": "data/futures/um/monthly/fundingRate/",
            "premium": "data/futures/um/monthly/premiumIndexKlines/"}[kind]
    syms = sorted({p[len(root):].strip("/") for p in listing(root)})
    syms = [s for s in syms if keep(s)]
    only = os.environ.get("BINANCE_ONLY")
    if only:
        wanted = set(open(only).read().split())
        syms = [s for s in syms if s in wanted]
    print(kind, len(syms), "symbols", flush=True)
    frames, t0 = [], time.time()
    with cf.ThreadPoolExecutor(12) as ex:
        futs = {ex.submit(job_symbol, kind, s): s for s in syms}
        for i, f in enumerate(cf.as_completed(futs)):
            d = f.result()
            if d is not None and len(d):
                frames.append(d)
            if i % 50 == 0:
                print(kind, i, len(syms), round(time.time() - t0), flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(f"{outdir}/{kind}.parquet", index=False)
    print(kind, "done", out.shape, out["symbol"].nunique(), flush=True)


if __name__ == "__main__":
    for k in sys.argv[2:]:
        run(k, sys.argv[1])

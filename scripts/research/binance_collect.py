"""Bounded, resumable Binance PUBLIC archive collection, including delisted pairs."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import re
import threading
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import requests

from scripts.x80.binance_dl import keep

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DL = "https://data.binance.vision/"
PREFIX = {"spot":"data/spot/monthly/klines/", "funding":"data/futures/um/monthly/fundingRate/",
          "premium":"data/futures/um/monthly/premiumIndexKlines/"}
KL = ["open_time","open","high","low","close","volume","close_time","quote_volume","count",
      "taker_buy_volume","taker_buy_quote_volume","ignore"]


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj,indent=2),encoding="utf-8")
    temp.replace(path)


class Archive:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = self.root/"archive_cache"
        self.cache.mkdir(parents=True,exist_ok=True)

    def get(self, url, optional=False):
        key = hashlib.sha256(url.encode()).hexdigest()
        data, meta = self.cache/(key+".bin"), self.cache/(key+".json")
        if data.exists() and meta.exists():
            raw = data.read_bytes()
            if hashlib.sha256(raw).hexdigest() != json.loads(meta.read_text())["sha256"]:
                raise ValueError("Cached public source hash changed: "+key)
            return raw
        last = None
        for _ in range(2):
            try:
                response = requests.get(url,timeout=(3,6))
                if response.status_code == 404 and optional:
                    return None
                if response.status_code == 429:
                    raise RuntimeError("Archive rate limited; Retry-After="+response.headers.get("Retry-After","unknown"))
                response.raise_for_status()
                raw = response.content
                temp = data.with_suffix(".tmp")
                temp.write_bytes(raw)
                temp.replace(data)
                atomic_json(meta,{"url":url,"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)})
                return raw
            except requests.RequestException as exc:
                last = exc
        raise last

    def listing(self, prefix, tags):
        out, marker = [], ""
        for _ in range(100):
            query = requests.Request("GET",S3,params={"delimiter":"/","prefix":prefix,"marker":marker}).prepare().url
            tree = ET.fromstring(self.get(query))
            values = {}
            for node in tree.iter():
                values.setdefault(node.tag.rsplit("}",1)[-1],[]).append(node.text or "")
            items = values.get(tags,[])
            out.extend(x for x in items if x != prefix)
            if values.get("IsTruncated",["false"])[0] != "true":
                return out
            marker = values.get("NextMarker",items[-1:] or [""])[0]
            if not marker:
                raise ValueError("Truncated listing has no continuation")
        raise ValueError("Public listing page bound exceeded")


def parse_archive(raw, kind, symbol, asof):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Expected exactly one CSV in archive")
        content = z.read(names[0])
    first = content.decode("utf-8-sig").splitlines()[0].split(",")[0].strip()
    try:
        float(first)
        header = None
    except ValueError:
        header = 0
    d = pd.read_csv(io.BytesIO(content),header=header)
    if header is None:
        d.columns = (["calc_time","funding_interval_hours","last_funding_rate"] if kind == "funding" else KL)[:len(d.columns)]
    field = "calc_time" if kind == "funding" else "open_time"
    stamp = pd.to_numeric(d[field],errors="raise")
    stamp = stamp.where(stamp <= 1e14, stamp//1000)
    t = pd.to_datetime(stamp,unit="ms")
    if kind == "funding":
        out = pd.DataFrame({"time":t,"rate":pd.to_numeric(d["last_funding_rate"],errors="raise")})
        # Midnight settlement belongs to the day that just ended.
        out = out[out.time <= pd.Timestamp(asof)]
    else:
        cols = ["open","high","low","close"] + (["volume","quote_volume"] if kind == "spot" else [])
        out = d[cols].apply(pd.to_numeric,errors="raise")
        out.insert(0,"date",t.dt.floor("D"))
        out = out[out.date < pd.Timestamp(asof)]
    out["symbol"] = symbol
    return out


def one_symbol(archive, kind, symbol, asof):
    if kind == "spot":
        prefix = PREFIX[kind]+symbol+"/1d/"
    elif kind == "premium":
        prefix = PREFIX[kind]+symbol+"/1d/"
    else:
        prefix = PREFIX[kind]+symbol+"/"
    keys = archive.listing(prefix,"Key")
    first_current_month = pd.Timestamp(asof).replace(day=1)
    kept = []
    for key in keys:
        found = re.search(r"-(\d{4})-(\d{2})\.zip$",key)
        if found and pd.Timestamp(int(found[1]),int(found[2]),1) < first_current_month:
            kept.append(key)
    frames = [parse_archive(archive.get(DL+key),kind,symbol,asof) for key in kept]
    tail_missing = []
    # Partial-month spot/premium/funding daily archives are attempted explicitly.
    for day in pd.date_range(first_current_month,pd.Timestamp(asof)-pd.Timedelta(days=1)):
        if kind == "spot":
            key = "data/spot/daily/klines/"+symbol+"/1d/"+symbol+"-1d-"+day.strftime("%Y-%m-%d")+".zip"
        elif kind == "premium":
            key = "data/futures/um/daily/premiumIndexKlines/"+symbol+"/1d/"+symbol+"-1d-"+day.strftime("%Y-%m-%d")+".zip"
        else:
            key = "data/futures/um/daily/fundingRate/"+symbol+"/"+symbol+"-fundingRate-"+day.strftime("%Y-%m-%d")+".zip"
        raw = archive.get(DL+key,optional=True)
        if raw is None:
            tail_missing.append(str(day.date()))
        else:
            frames.append(parse_archive(raw,kind,symbol,asof))
    if not frames:
        return {"status":"empty_before_asof","rows":0,"tail_missing":tail_missing}
    datecol = "time" if kind == "funding" else "date"
    frame = pd.concat(frames,ignore_index=True).sort_values(datecol)
    duplicated = int(frame.duplicated([datecol,"symbol"]).sum())
    frame = frame.drop_duplicates([datecol,"symbol"],keep="first")
    directory = archive.root/(kind+"_symbols")
    directory.mkdir(exist_ok=True)
    dest = directory/(symbol+".parquet")
    tmp = dest.with_suffix(".tmp")
    frame.to_parquet(tmp,index=False)
    tmp.replace(dest)
    return {"status":"ok","path":str(dest.relative_to(archive.root)),"rows":len(frame),
            "start":str(frame[datecol].min()),"end":str(frame[datecol].max()),
            "sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
            "monthly_archives":len(kept),"duplicates":duplicated,"tail_missing":tail_missing}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--kind",choices=list(PREFIX),required=True)
    p.add_argument("--asof",required=True)
    p.add_argument("--max-new",type=int,default=20)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--retry-blocked",action="store_true",help="Explicit bounded retry of failed symbols; successful files are reused")
    args = p.parse_args()
    if args.max_new < 1 or not 1 <= args.workers <= 2:
        p.error("max-new positive; workers must be 1 or 2")
    archive = Archive(args.root)
    path = archive.root/("collection_"+args.kind+".json")
    codehash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if path.exists():
        state = json.loads(path.read_text())
        if state["asof"] != args.asof or state["collector_sha256"] != codehash:
            raise ValueError("Changed collection/asof; preserve old output and choose another directory")
    else:
        prefix = PREFIX[args.kind]
        syms = sorted({x[len(prefix):].strip("/") for x in archive.listing(prefix,"Prefix")})
        syms = [s for s in syms if keep(s)]
        priority = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT"]
        syms = [s for s in priority if s in syms] + [s for s in syms if s not in priority]
        state = {"asof":args.asof,"kind":args.kind,"collector_sha256":codehash,"symbols":syms,"files":{}}
        atomic_json(path,state)
    for symbol, row in state["files"].items():
        if row["status"] == "ok":
            saved = archive.root/row["path"]
            if not saved.exists() or hashlib.sha256(saved.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Saved symbol hash changed or file missing: "+symbol)
    todo = [s for s in state["symbols"] if s not in state["files"] or
            (args.retry_blocked and state["files"][s]["status"] == "blocked")][:args.max_new]
    print(args.kind,len(state["files"]),"/",len(state["symbols"]),"recorded;",len(todo),"this batch",flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as workers:
        jobs = {workers.submit(one_symbol,archive,args.kind,s,args.asof):s for s in todo}
        for future in cf.as_completed(jobs):
            symbol = jobs[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"status":"blocked","error":type(exc).__name__+": "+str(exc)[:300]}
            state["files"][symbol] = row
            atomic_json(path,state)
            print(symbol,row["status"],row.get("rows",row.get("error")),flush=True)
    complete = len(state["files"]) == len(state["symbols"])
    blocked = [s for s,v in state["files"].items() if v["status"] == "blocked"]
    state["complete"] = complete and not blocked
    atomic_json(path,state)
    if state["complete"]:
        parts = [pd.read_parquet(archive.root/v["path"]) for v in state["files"].values() if v["status"] == "ok"]
        if parts:
            dest = archive.root/(args.kind+".parquet")
            tmp = dest.with_suffix(".tmp")
            pd.concat(parts,ignore_index=True).to_parquet(tmp,index=False)
            tmp.replace(dest)
            print("Published",dest,flush=True)
    print("Progress",len(state["files"]),"/",len(state["symbols"]),"blocked",len(blocked),
          "complete",state["complete"],flush=True)
    print("tail_missing is NOT imputed to zero; inspect delistings/partial-month gaps before strategy use.",flush=True)


if __name__ == "__main__":
    main()

"""python -m src.backtest.optimize.search POOL_DIR CASH_WEEKLY OUT_DIR [N_RANDOM]

Scores thousands of book configurations on data BEFORE the holdout only (the pool is cut at HOLDOUT before anything
runs, so nothing here can see 2025-26). Every configuration tried is kept, because the overfitting checks need the
full count and every return series, not just the winners.

  structured  x80's book and every one-step change of it (each setting of each dimension, the rest as x80)
  random      N_RANDOM draws over all dimensions; half of them use a random subset of 8-24 of the 30 sleeves

Writes OUT_DIR/configs.csv, returns_tv80.parquet, returns_tv30.parquet (weekly account returns, levered to 80% / 30%
target vol with all costs), books.parquet (unlevered book, crypto risk share) and metrics.csv.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, fields

import numpy as np
import pandas as pd

from src.backtest.optimize import HOLDOUT
from src.backtest.optimize.engine import BASELINE, Config, Pool, lever, run_book, stats
from src.backtest.optimize.pool import CORE, NEW

OPTIONS = {
    "select": ["all", "pos", "sr25", "top10"],
    "method": ["erc", "ivol", "ew", "hrp", "maxdiv", "family", "ens", "sharpe"],
    "cov": ["sample", "shrink", "ewma"],
    "look": [52, 104, 156, 260],
    "cap": [1.0, 0.25, 0.15],
    "crypto_cap": [1.0, 0.5, 0.35],
    "rebal": ["w", "m", "band"],
    "lev_band": [0.0, 0.1, 0.2],
}
N_SUBSETS, SEED = 40, 7
PERIODS = {"2010-24": ("2010-01-01", "2024-12-31"), "2010-17": ("2010-01-01", "2017-12-31"),
           "2018-24": ("2018-01-01", "2024-12-31")}
TVS = (0.3, 0.8)


def universes(cols: list[str]) -> dict[str, list[str]]:
    rng = np.random.default_rng(SEED)
    u = {"core16": CORE, "core+new": CORE + NEW, "all30": cols}
    for k in range(N_SUBSETS):
        size = min(int(rng.integers(8, 25)), len(cols))
        u[f"subset{k:02d}"] = sorted(rng.choice(cols, size=size, replace=False).tolist(), key=cols.index)
    return u


def configs(n_random: int, names: list[str]) -> list[Config]:
    out = [BASELINE]
    for dim, opts in [("universe", ["core+new", "all30"])] + list(OPTIONS.items()):
        for o in opts:
            c = Config(**{**asdict(BASELINE), dim: o})
            if c not in out:
                out.append(c)
    rng = np.random.default_rng(SEED + 1)
    named = ["core16", "core+new", "all30"]
    subsets = [n for n in names if n.startswith("subset")]
    seen = set(out)
    n_struct = len(out)
    while len(out) < n_struct + n_random:
        uni = named[rng.integers(len(named))] if rng.random() < 0.5 else subsets[rng.integers(len(subsets))]
        c = Config(universe=uni, **{k: v[rng.integers(len(v))] for k, v in OPTIONS.items()})
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


_POOL: Pool | None = None


def _init(pool_dir: str, cash_path: str) -> None:
    global _POOL
    w = pd.read_parquet(os.path.join(pool_dir, "pool_weekly.parquet"))
    w = w[w.index < HOLDOUT]
    m = pd.read_csv(os.path.join(pool_dir, "pool_meta.csv"), index_col=0)
    cash = pd.read_parquet(cash_path)["cash"]
    _POOL = Pool(w, cash, m, universes(list(w.columns)))


def _one(cfg: Config) -> tuple[Config, dict, dict]:
    pool = _POOL
    bk = run_book(pool, cfg)
    rets, row = {}, {}
    for tv in TVS:
        r, lev = lever(bk, pool.cash, tv, cfg.lev_band)
        rets[tv] = r.astype(np.float32)
        for lab, (a, z) in PERIODS.items():
            for k, v in stats(r, pool.cash, pool.idx, a, z).items():
                row[f"{k}_{lab}_tv{int(tv * 100)}"] = v
        row[f"lev_tv{int(tv * 100)}"] = float(np.nanmean(lev[pool.idx >= "2010-01-01"]))
    row["turn"] = float(np.nanmean(bk["turn"][pool.idx >= "2010-01-01"]))
    row["crypto_share"] = float(np.nanmean(bk["crypto_share"][pool.idx >= "2010-01-01"]))
    rets["base"] = bk["base"].astype(np.float32)
    return cfg, row, rets


def main() -> None:
    pool_dir, cash_path, out = sys.argv[1:4]
    n_random = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
    os.makedirs(out, exist_ok=True)
    _init(pool_dir, cash_path)
    idx = _POOL.idx
    cfgs = configs(n_random, list(_POOL.universes))
    print(f"{len(cfgs)} configurations; data {idx.min().date()}..{idx.max().date()} (holdout {HOLDOUT} not loaded)", flush=True)
    t0 = time.time()
    rows, r80, r30, base = [], {}, {}, {}
    with mp.Pool(os.cpu_count(), initializer=_init, initargs=(pool_dir, cash_path)) as p:
        for k, (cfg, row, rets) in enumerate(p.imap(_one, cfgs, chunksize=8)):
            key = f"c{k:05d}"
            rows.append({"id": key, **asdict(cfg), **row})
            r80[key], r30[key], base[key] = rets[0.8], rets[0.3], rets["base"]
            if (k + 1) % 250 == 0:
                print(f"{k + 1}/{len(cfgs)}  {time.time() - t0:.0f}s", flush=True)
    met = pd.DataFrame(rows).set_index("id")
    met.to_csv(os.path.join(out, "metrics.csv"))
    met[[f.name for f in fields(Config)]].to_csv(os.path.join(out, "configs.csv"))
    pd.DataFrame(r80, index=idx).to_parquet(os.path.join(out, "returns_tv80.parquet"))
    pd.DataFrame(r30, index=idx).to_parquet(os.path.join(out, "returns_tv30.parquet"))
    pd.DataFrame(base, index=idx).to_parquet(os.path.join(out, "books.parquet"))
    pd.Series({k: v for k, v in _POOL.universes.items()}).map(lambda a: "; ".join(_POOL.cols[i] for i in a)).to_csv(
        os.path.join(out, "universes.csv"))
    best = met.sort_values("sharpe_2010-24_tv80", ascending=False)
    show = ["universe", "select", "method", "cov", "look", "cap", "crypto_cap", "rebal", "lev_band",
            "sharpe_2010-24_tv80", "sharpe_2010-17_tv80", "sharpe_2018-24_tv80", "cagr_2018-24_tv80", "max_dd_2018-24_tv80"]
    print(best[show].head(15).round(3).to_string())
    print("x80 baseline:\n", met.loc["c00000", show].to_string())
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

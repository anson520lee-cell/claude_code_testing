"""A separate preregistered, restartable ETF allocator study; never alters live trading."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sqlite3
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

UNIVERSES = {
    "us": ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE"],
    "global": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"],
    "defensive": ["SPY", "TLT", "IEF", "GLD", "EFA", "DBC"],
}
PERIODS = {"2008-12": ("2008-01-01", "2012-12-31"), "2013-17": ("2013-01-01", "2017-12-31"),
           "2018-21": ("2018-01-01", "2021-12-31"), "2022-24": ("2022-01-01", "2024-12-31"),
           "2008-24": ("2008-01-01", "2024-12-31"), "2018-24": ("2018-01-01", "2024-12-31")}
SELECTION = ("2008-12", "2013-17", "2018-21", "2022-24")


@dataclass(frozen=True)
class Config:
    universe: str
    family: str
    look: int
    trend: int
    top: int
    target_vol: float
    exposure_cap: float
    rebalance: int


def configurations():
    out = []
    for u in UNIVERSES:
        for look, trend, top, tv, cap, rb in itertools.product(
                [21, 63, 126, 252], [0, 200], [1, 2, 3], [0., .3, .6], [1., 2., 3.], [5, 21]):
            out.append(Config(u, "momentum", look, trend, top, tv, cap, rb))
        for look, trend, top, tv, cap, rb in itertools.product(
                [2, 5, 10], [0, 200], [1, 3], [0., .3], [1., 2., 3.], [1, 5]):
            out.append(Config(u, "reversal", look, trend, top, tv, cap, rb))
    assert len(out) == 1728
    return out


@njit(cache=True)
def simulate(returns, targets, rates, days, rebalance, exposure_cap, cost=.001):
    """Targets known from yesterday's close execute at today's close, after today's P&L."""
    T, n = returns.shape
    held = np.zeros(n)
    result = np.zeros(T)
    peak_exposure = 0.
    ruined = False
    for t in range(T):
        if ruined:
            continue
        gross = held.sum()
        interest = max(1-gross, 0.)*rates[t]*days[t]/365.
        interest -= max(gross-1, 0.)*(rates[t]+.01)*days[t]/365.
        growth = 1. + interest
        for j in range(n):
            if not np.isfinite(returns[t, j]):
                if abs(held[j]) > 1e-12:
                    raise ValueError("Missing return for an allocated ETF")
            else:
                growth += held[j]*returns[t, j]
        if growth <= 0:
            result[t] = -1.
            ruined = True
            held[:] = 0.
            continue
        drift = np.zeros(n)
        for j in range(n):
            if np.isfinite(returns[t, j]):
                drift[j] = held[j]*(1+returns[t, j])/growth
        peak_exposure = max(peak_exposure, gross, drift.sum())
        desired = drift.copy()
        if t % rebalance == 0:
            desired = targets[t].copy()
        if desired.sum() > exposure_cap:
            desired *= exposure_cap/desired.sum()
        if t == T-1:
            desired[:] = 0.  # charge the final liquidation, too
        # Self-financing target weights are measured after transaction costs.
        fee = 0.
        for _ in range(12):
            fee = cost*np.abs(desired*(1-fee)-drift).sum()
        net_growth = growth*(1-fee)
        if net_growth <= 0:
            result[t] = -1.
            ruined = True
            held[:] = 0.
        else:
            result[t] = net_growth-1.
            held = desired
            peak_exposure = max(peak_exposure, held.sum())
    return result, peak_exposure, ruined


def statistics(r, idx, cash=None):
    x = np.asarray(r, dtype=float)
    if len(x) < 200:
        return {}
    eq = np.cumprod(1+x)
    years = (idx[-1]-idx[0]).days/365.25
    cagr = eq[-1]**(1/years)-1 if eq[-1] > 0 else -1.
    dd = np.min(eq/np.maximum.accumulate(np.r_[1., eq])[1:]-1)
    excess = x if cash is None else x-np.asarray(cash)
    sd = excess.std(ddof=1)
    annual = pd.Series(x, index=idx).groupby(idx.year).apply(lambda y: (1+y).prod()-1)
    roll = pd.Series(eq, index=idx).pct_change(756).dropna()
    rolling_min = float(((1+roll)**(1/3)-1).min()) if len(roll) else None
    return {"cagr": float(cagr), "max_dd": float(dd), "sharpe": float(excess.mean()/sd*np.sqrt(252)) if sd > 0 else 0.,
            "worst_year": float(annual.min()), "rolling_3y_min": rolling_min, "years": float(years)}


class Data:
    def __init__(self, root):
        self.root = Path(root)
        parts = [pd.read_parquet(self.root/"x80/raw"/name) for name in ["prices.parquet", "prices_extra.parquet"]]
        frame = pd.concat(parts, ignore_index=True)
        frame["date"] = pd.to_datetime(frame["date"])
        self.prices = frame.pivot(index="date", columns="symbol", values="adj_close").sort_index()
        spy_days = self.prices["SPY"].dropna().index
        self.prices = self.prices.reindex(spy_days)
        self.raw_returns = self.prices.pct_change(fill_method=None)
        irx = frame[frame.symbol == "^IRX"].set_index("date")["close"].sort_index()
        self.rates = irx.reindex(spy_days).ffill().shift(1).fillna(0.)/100.
        self.index = spy_days[spy_days >= "2008-01-01"]
        self.training = self.index < "2025-01-01"
        self.r = {}
        self.cov = {}
        self.sigma = {}
        self.features = {}
        for u, cols in UNIVERSES.items():
            ret = self.raw_returns[cols]
            # These fixed funds all existed by the evaluation start.
            if ret.reindex(self.index).isna().any().any():
                raise ValueError("Missing evaluation-period ETF bars: "+u)
            self.r[u] = ret.reindex(self.index).to_numpy(float)
            self.sigma[u] = ret.rolling(60, min_periods=60).std()
            self.cov[u] = np.nan_to_num(ret.rolling(60, min_periods=60).cov().to_numpy().reshape(len(ret), len(cols), len(cols)))
        self.rate_values = self.rates.reindex(self.index).to_numpy(float)
        self.days = self.index.to_series().diff().dt.days.fillna(1).to_numpy(float)
        self.cash = self.rate_values*self.days/365.

    def signals(self, cfg):
        key = (cfg.universe, cfg.family, cfg.look, cfg.trend, cfg.top)
        if key not in self.features:
            cols = UNIVERSES[cfg.universe]
            price = self.prices[cols]
            mom = price.pct_change(cfg.look, fill_method=None)
            score = mom if cfg.family == "momentum" else -mom
            valid = (score > 0) & self.sigma[cfg.universe].gt(0)
            if cfg.trend:
                valid &= price > price.rolling(cfg.trend, min_periods=cfg.trend).mean()
            ranked = score.where(valid).rank(axis=1, ascending=False, method="first")
            chosen = ranked.le(cfg.top).to_numpy()
            vol = self.sigma[cfg.universe].to_numpy()
            inv = np.divide(1., vol, out=np.zeros_like(vol), where=np.isfinite(vol) & (vol > 0))
            base = inv*chosen
            base = np.divide(base, base.sum(axis=1, keepdims=True), out=np.zeros_like(base), where=base.sum(axis=1, keepdims=True) > 0)
            var = np.einsum("ti,tij,tj->t", base, self.cov[cfg.universe], base)
            forecast = np.sqrt(np.maximum(var, 0)*252)
            # Today's close may use only yesterday's signal.
            base = pd.DataFrame(base, index=price.index).shift(1).reindex(self.index).fillna(0.).to_numpy()
            forecast = pd.Series(forecast, index=price.index).shift(1).reindex(self.index).fillna(0.).to_numpy()
            self.features[key] = (base, forecast)
        base, forecast = self.features[key]
        exposure = np.ones(len(base)) if cfg.target_vol == 0 else np.divide(
            cfg.target_vol, forecast, out=np.zeros(len(base)), where=forecast > 1e-12)
        exposure = np.minimum(exposure, cfg.exposure_cap)
        return np.ascontiguousarray(base*exposure[:, None])

    def run(self, cfg, training=True, cost=.001):
        mask = self.training if training else np.ones(len(self.index), bool)
        return simulate(np.ascontiguousarray(self.r[cfg.universe][mask]),
                        np.ascontiguousarray(self.signals(cfg)[mask]),
                        self.rate_values[mask], self.days[mask], cfg.rebalance, cfg.exposure_cap, cost)


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def metrics(data, r, peak, ruined):
    idx = data.index[data.training]
    cash = data.cash[data.training]
    row = {"peak_exposure": float(peak), "ruined": bool(ruined)}
    for label, (start, end) in PERIODS.items():
        mask = (idx >= start) & (idx <= end)
        for key, value in statistics(r[mask], idx[mask], cash[mask]).items():
            row[key+"_"+label] = value
    row["score"] = min(row["cagr_"+lab] for lab in SELECTION) if not ruined else -1.
    row["all_periods_80"] = not ruined and all(row["cagr_"+lab] >= .8 for lab in SELECTION)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--max-new", type=int, default=250)
    args = p.parse_args()
    if args.max_new < 1:
        p.error("max-new must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    data = Data(args.root)
    configs = configurations()
    h = hashlib.sha256(Path(__file__).read_bytes())
    prereg = Path(__file__).parents[2]/"reports/etf_expansion/PREREG.md"
    h.update(prereg.read_bytes())
    h.update(json.dumps([asdict(c) for c in configs], sort_keys=True).encode())
    symbols = sorted({s for cols in UNIVERSES.values() for s in cols})
    h.update(pd.util.hash_pandas_object(data.prices[symbols], index=True).to_numpy().tobytes())
    h.update(pd.util.hash_pandas_object(data.rates, index=True).to_numpy().tobytes())
    signature = h.hexdigest()
    db = sqlite3.connect(args.out/"checkpoint.sqlite3", timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS trials (id INTEGER PRIMARY KEY, config TEXT, metrics TEXT, returns BLOB)")
    old = db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
    if old and old[0] != signature:
        raise ValueError("Changed study fingerprint; use a new output directory")
    with db:
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('fingerprint',?)", (signature,))
    done = {x[0] for x in db.execute("SELECT id FROM trials")}
    print("ETF study:", len(done), "/", len(configs), "checkpointed", flush=True)
    new = 0
    for i, cfg in enumerate(configs):
        if i in done:
            continue
        r, peak, ruined = data.run(cfg)
        row = metrics(data, r, peak, ruined)
        with db:
            db.execute("INSERT INTO trials VALUES (?,?,?,?)",
                       (i, json.dumps(asdict(cfg), sort_keys=True), json.dumps(row),
                        zlib.compress(r.astype(np.float32).tobytes())))
        new += 1
        if new % 25 == 0:
            print(len(done)+new, "/", len(configs), "saved", flush=True)
        atomic_json(args.out/"progress.json", {"completed":len(done)+new, "total":len(configs),
                    "complete":False, "fingerprint":signature, "training_end":"2024-12-31"})
        if new >= args.max_new:
            break
    count = db.execute("SELECT count(*) FROM trials").fetchone()[0]
    if count == len(configs):
        rows = [{"id":i, **json.loads(c), **json.loads(m)} for i,c,m in db.execute("SELECT id,config,metrics FROM trials ORDER BY id")]
        table = pd.DataFrame(rows).sort_values(["score","sharpe_2008-24","max_dd_2008-24"], ascending=[False,False,False])
        winner = table.iloc[0].to_dict()
        # Freeze the winner using pre-2025 metrics before evaluating later returns.
        atomic_json(args.out/"frozen_winner.json", {"fingerprint":signature,"selection_period_end":"2024-12-31","winner":winner})
        cfg = configs[int(winner["id"])]
        r, peak, ruined = data.run(cfg, training=False)
        idx = data.index
        later = idx >= "2025-01-01"
        report = {"trial_count":len(configs),"pre2025_all_subperiods_80":int(table.all_periods_80.sum()),
                  "frozen_winner":winner,"reused_2025_26":statistics(r[later],idx[later],data.cash[later]),
                  "full_path_ruined":bool(ruined),"full_peak_exposure":float(peak),
                  "qualification":"historical screen only; no untouched holdout or executable intraday margin proof",
                  "fully_funded_top":table[table.exposure_cap == 1.].head(5).to_dict(orient="records")}
        table.to_csv(args.out/"metrics.csv",index=False)
        pd.DataFrame({"date":idx,"return":r}).to_parquet(args.out/"frozen_winner_daily.parquet",index=False)
        atomic_json(args.out/"summary.json",report)
        atomic_json(args.out/"progress.json",{"completed":count,"total":len(configs),"complete":True,"fingerprint":signature})
        print(json.dumps(report,indent=2),flush=True)
    else:
        print("Paused; repeat the same command to continue.", count, "saved",flush=True)
    db.close()


if __name__ == "__main__":
    main()

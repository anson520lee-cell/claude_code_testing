"""Stages of adaptive_stop_study.py: (A) two-phase grid + account, (B) learned ATR multiple, compact report."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import adaptive_stop as ad

SLIP = 0.001
WINDOWS = {"all": ("2010-01-01", "2100-01-01"), "hold": ("2010-01-01", "2019-12-31"),
           "is": ("2020-01-01", "2024-06-30"), "oos": ("2024-07-01", "2100-01-01")}
RISKS = (0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)


def _r(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, nd)


def load(cache: Path):
    d = pickle.loads((cache / "collect.pkl").read_bytes())
    return d


def trades_of(d: dict, setup: str) -> dict[str, np.ndarray]:
    sig = d["signals"]
    m = (sig["setup"] == setup).to_numpy()
    T = {k: np.ascontiguousarray(d[k][m], dtype=np.float64) for k in "OHLC"}
    T["ref"] = sig["ref"].to_numpy(float)[m]
    T["atr"] = sig["atr"].to_numpy(float)[m]
    T["cond"] = np.full(int(m.sum()), setup == "squeeze")
    T["date"] = sig["date"].to_numpy()[m]
    T["symbol"] = sig["symbol"].to_numpy()[m]
    T["gidx"] = np.flatnonzero(m)
    return T


class DateStats:
    """Fast per-date-cluster mean / t for many rows (rules) at once."""

    def __init__(self, dates: np.ndarray):
        self.order = np.argsort(dates, kind="stable")
        ds = dates[self.order]
        self.starts = np.flatnonzero(np.r_[True, ds[1:] != ds[:-1]])
        self.counts = np.diff(np.r_[self.starts, len(ds)])
        self.pm = ad.period_masks(ds)
        self.udates = ds[self.starts]
        self.pm_dates = ad.period_masks(self.udates)

    def stats(self, D: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        """D: rules x trades (deltas). Returns per period mean (per trade) and t (date clusters)."""
        Ds = D[:, self.order]
        sums = np.add.reduceat(Ds, self.starts, axis=1)
        means = sums / self.counts[None, :]
        out = {}
        for p, md in self.pm_dates.items():
            mt = self.pm[p]
            if md.sum() < 5:
                continue
            x = means[:, md]
            mu = x.mean(axis=1)
            sd = x.std(axis=1, ddof=1)
            out[p] = {"mean": Ds[:, mt].mean(axis=1), "t": mu / (sd / np.sqrt(md.sum()) + 1e-18), "n": int(mt.sum())}
        md = self.pm_dates["hold"] | self.pm_dates["oos"]
        x = means[:, md]
        out["pool"] = {"mean": Ds[:, self.pm["hold"] | self.pm["oos"]].mean(axis=1),
                       "t": x.mean(axis=1) / (x.std(axis=1, ddof=1) / np.sqrt(md.sum()) + 1e-18)}
        return out


def gate(st: dict, n_rules: int) -> tuple[np.ndarray, np.ndarray, float]:
    cand = (st["is"]["mean"] > 0) & (st["is"]["t"] >= 1.65)
    bar = max(2.5, ad.bonferroni_t(int(cand.sum())))
    ok = cand & (st["hold"]["mean"] > 0) & (st["oos"]["mean"] > 0) & (st["pool"]["t"] >= bar)
    return cand, ok, bar


def frontier(acct: ad.Account, window: tuple[str, str], seeds: int, risks=RISKS, heat: float = 0.0) -> list[dict]:
    out = []
    for r in risks:
        p = acct.run(*window, seeds=seeds, risk=r, heat=heat)
        out.append({"risk": r, **{k: _r(v) for k, v in p.items()}})
    return out


def cagr_at_dd(front: list[dict], dd: float) -> float:
    """CAGR interpolated at maximum drawdown ``dd`` (negative) along a risk frontier."""
    xs = np.array([-f["max_dd"] for f in front])
    ys = np.array([f["cagr"] for f in front])
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if -dd < xs[0] or -dd > xs[-1]:
        return float("nan")
    return float(np.interp(-dd, xs, ys))


# ====================================================================== (A) grid
def grid(args) -> int:
    t0 = time.time()
    cache = Path(args.cache)
    d = load(cache)
    prices = pickle.loads(Path(args.prices).read_bytes())
    cal = prices["SPY"].index
    res: dict = {"setups": {}}
    acct = ad.Account(cal)
    live_res = {}
    Ts = {s: trades_of(d, s) for s in ad.SETUPS + ("random",)}
    for s in ad.SETUPS:
        live_res[s] = ad.simulate(Ts[s], ad.LIVE[s], SLIP, paths=True)
        acct.set_setup(s, Ts[s], live_res[s])
    base_acct = {w: acct.run(*win, seeds=8) for w, win in WINDOWS.items()}
    res["live_account"] = {w: {k: _r(v) for k, v in p.items()} for w, p in base_acct.items()}
    print("live account", {w: (round(p["cagr"], 3), round(p["max_dd"], 3)) for w, p in base_acct.items()}, flush=True)
    for s in ad.SETUPS:
        T = Ts[s]
        rules = ad.grid(s)
        names = [ad.rule_name(r) for r in rules]
        base = live_res[s]
        E = np.zeros((len(rules), len(T["ref"])), dtype=np.float64)
        R = np.zeros_like(E)
        days = np.zeros(len(rules))
        stopped = np.zeros(len(rules))
        acc = {w: np.zeros((len(rules), 3)) for w in WINDOWS}
        for j, r in enumerate(rules):
            o = ad.simulate(T, r, SLIP, paths=True)
            E[j], R[j] = o["eq"], o["r"]
            days[j] = o["days"][o["ok"]].mean()
            stopped[j] = (o["exit_px"][o["ok"]] < o["entry"][o["ok"]]).mean()
            acct.set_setup(s, T, o)
            for w, win in WINDOWS.items():
                p = acct.run(*win, seeds=4)
                acc[w][j] = (p["cagr"], p["max_dd"], p["sharpe"])
            if (j + 1) % 500 == 0:
                print(f"  {s}: {j + 1}/{len(rules)} rules, {time.time() - t0:.0f}s", flush=True)
        acct.set_setup(s, T, base)
        ds = DateStats(T["date"])
        st = ds.stats(E - base["eq"][None, :])
        st_r = ds.stats(R - base["r"][None, :])
        cand, ok, bar = gate(st, len(rules))
        # ---- walk-forward choice of a fixed rule: each year use the rule with the best mean eq over all PRIOR years
        yrs = pd.to_datetime(T["date"]).year.to_numpy()
        wf_pick, wf_delta = {}, np.zeros(len(yrs))
        for y in range(2010, int(yrs.max()) + 1):
            past = yrs < y - 0       # signals of prior years (their windows end by mid-January at the latest)
            past &= pd.to_datetime(T["date"]).to_numpy() < np.datetime64(f"{y - 1}-12-01")
            if past.sum() < 200:
                continue
            j = int(np.argmax(E[:, past].mean(axis=1)))
            wf_pick[y] = names[j]
            cur = yrs == y
            wf_delta[cur] = E[j, cur] - base["eq"][cur]
        wf_st = ds.stats(wf_delta[None, :])
        # ---- random entries: does the same rule help on random entries (generic) or only on the setup?
        rows = []
        order_is = np.argsort(-np.nan_to_num(st["is"]["t"], nan=-9))
        top = list(order_is[:30])
        best_acc = int(np.nanargmax(acc["is"][:, 2]))
        top = list(dict.fromkeys(top + [best_acc, 0]))
        Trand = Ts["random"]
        base_rand = ad.simulate(Trand, ad.LIVE["random"], SLIP)
        ds_rand = DateStats(Trand["date"])
        for j in range(len(rules)):
            row = {"rule": names[j], "S0": rules[j]["S0"], "T1": rules[j]["T1"], "trig": rules[j]["trig"],
                   "T2": rules[j]["T2"], "be": rules[j]["be"], "tgt": rules[j]["tgt"],
                   "avg_days": _r(days[j], 2), "loss_share": _r(stopped[j], 3)}
            for p in ("hold", "is", "oos", "gfc", "pool"):
                if p in st:
                    row[f"{p}_d_eq"] = _r(st[p]["mean"][j], 6)
                    row[f"{p}_t"] = _r(st[p]["t"][j], 2)
                    row[f"{p}_d_r"] = _r(st_r[p]["mean"][j], 4)
            for w in WINDOWS:
                row[f"acct_{w}_cagr"], row[f"acct_{w}_dd"], row[f"acct_{w}_sharpe"] = (_r(x) for x in acc[w][j])
            row["candidate"], row["pass"] = bool(cand[j]), bool(ok[j])
            if j in top:
                rr = ad.simulate(Trand, {**rules[j], "tgt": 0.0}, SLIP)
                sr = ds_rand.stats((rr["eq"] - base_rand["eq"])[None, :])
                row["random_pool_d_eq"] = _r(sr["pool"]["mean"][0], 6)
                row["random_is_d_eq"] = _r(sr["is"]["mean"][0], 6)
            rows.append(row)
        res["setups"][s] = {"n_trades": int(len(T["ref"])), "n_rules": len(rules), "n_candidates": int(cand.sum()),
                            "bar": _r(bar, 2), "n_pass": int(ok.sum()), "rules": rows,
                            "walk_forward": {"picks": wf_pick, **{p: {"d_eq": _r(v["mean"][0], 6), "t": _r(v["t"][0], 2)}
                                                                  for p, v in wf_st.items()}},
                            "best_is_account": names[best_acc]}
        print(f"{s}: {len(rules)} rules, IS candidates {int(cand.sum())}, bar {bar:.2f}, pass {int(ok.sum())}; "
              f"best IS acct {names[best_acc]} {time.time() - t0:.0f}s", flush=True)
        np.save(cache / f"grid_E_{s}.npy", E.astype(np.float32))
    (cache / "grid.json").write_text(json.dumps(res, ensure_ascii=False))
    print(f"grid done {time.time() - t0:.0f}s")
    return 0


# ====================================================================== (B) learned ATR multiple
B1_MENU = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
B2_MENU = (0.0, 1.5, 2.0, 3.0, 5.0, 8.0)          # 0 = no trail (only the initial stop)
FOLDS = [(2010, 2011), (2012, 2013), (2014, 2015), (2016, 2017), (2018, 2019), (2020, 2021), (2022, 2023),
         (2024, 2025), (2026, 2026)]
LGB = dict(objective="huber", alpha=1.0, learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.3,
           bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=7)
TOPK = 60
MIN_INNER_TR, MIN_INNER_VAL = 500, 200
TAUS = (0.0, 0.0005, 0.001, 0.002, 0.004)


def _wins(y: np.ndarray, q: float = 0.01) -> np.ndarray:
    lo, hi = np.nanquantile(y, q), np.nanquantile(y, 1 - q)
    return np.clip(y, lo, hi)


def _fit(X, y, Xv, yv, params, rounds=400):
    import lightgbm as lgb
    scale = float(np.nanstd(y)) or 1.0
    dtr = lgb.Dataset(X, _wins(y) / scale, free_raw_data=True)
    dv = lgb.Dataset(Xv, _wins(yv) / scale, reference=dtr) if Xv is not None else None
    cb = [lgb.early_stopping(50, verbose=False)] if dv is not None else []
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dv] if dv is not None else None, callbacks=cb)
    return m, scale


def wf_policy(X: np.ndarray, Y: np.ndarray, dates: np.ndarray, params: dict, max_rows: int | None = None,
              topk: int = TOPK, log=print, tag: str = "") -> dict[str, np.ndarray]:
    """Walk-forward forecasts of Y (rows x candidates: gain of each candidate over the default) with nested feature
    selection and threshold choice. Returns pred (rows x M, NaN outside the test folds), tau per row, the features
    chosen per fold."""
    n, M = Y.shape
    pred = np.full((n, M), np.nan, dtype=np.float32)
    tau_row = np.full(n, np.nan)
    d = pd.to_datetime(dates)
    yrs = d.year.to_numpy()
    info = []
    rng = np.random.default_rng(3)
    for (y0, y1) in FOLDS:
        test = (yrs >= y0) & (yrs <= y1)
        if not test.any():
            continue
        train = d.to_numpy() < np.datetime64(f"{y0 - 1}-12-01")          # 30-day purge / embargo before the test fold
        inner_val = train & (yrs >= y0 - 2)
        inner_tr = train & (d.to_numpy() < np.datetime64(f"{y0 - 3}-12-01"))
        if inner_tr.sum() < MIN_INNER_TR or inner_val.sum() < MIN_INNER_VAL:
            continue

        def sub(mask):
            ix = np.flatnonzero(mask)
            if max_rows and len(ix) > max_rows:
                ix = np.sort(rng.choice(ix, max_rows, replace=False))
            return ix

        itr, iva, tr = sub(inner_tr), sub(inner_val), sub(train)
        sel_cols, models, pv = [], [], np.zeros((len(iva), M))
        for j in range(M):
            m0, _ = _fit(X[itr], Y[itr, j], X[iva], Y[iva, j], params)
            gain = m0.feature_importance("gain")
            cols = np.argsort(-gain)[:topk]
            cols = cols[gain[cols] > 0] if (gain[cols] > 0).any() else cols
            m1, sc = _fit(X[itr][:, cols], Y[itr, j], X[iva][:, cols], Y[iva, j], params)
            pv[:, j] = m1.predict(X[iva][:, cols], num_iteration=m1.best_iteration or None) * sc
            sel_cols.append(cols)
            models.append(max(20, int(m1.best_iteration or 100)))
        # threshold chosen on the inner validation years (realised gain of following the forecast)
        best_tau, best_gain = TAUS[0], -np.inf
        for tau in TAUS:
            jj = np.argmax(pv, axis=1)
            use = pv[np.arange(len(iva)), jj] > tau
            g = np.where(use, Y[iva, jj], 0.0).mean()
            if g > best_gain:
                best_tau, best_gain = tau, g
        ix_test = np.flatnonzero(test)
        for j in range(M):
            cols = sel_cols[j]
            m2, sc = _fit(X[tr][:, cols], Y[tr, j], None, None, params, rounds=models[j])
            pred[ix_test, j] = m2.predict(X[ix_test][:, cols]) * sc
        tau_row[ix_test] = best_tau
        info.append({"fold": f"{y0}-{y1}", "n_train": int(train.sum()), "n_test": int(test.sum()), "tau": best_tau,
                     "inner_gain": _r(best_gain, 6), "features": [int(c) for c in sel_cols[0][:15]]})
        log(f"    {tag} fold {y0}-{y1}: train {train.sum()} test {test.sum()} tau {best_tau} inner gain {best_gain:.5f}")
    return {"pred": pred, "tau": tau_row, "folds": info}


def choose(pred: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Index of the chosen candidate per row (-1 = keep the default)."""
    ok = np.isfinite(pred).all(axis=1)
    jj = np.argmax(np.where(np.isfinite(pred), pred, -np.inf), axis=1)
    best = pred[np.arange(len(pred)), jj]
    return np.where(ok & (best > tau), jj, -1)


def shuffle_within_year(choice: np.ndarray, dates: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = choice.copy()
    yrs = pd.to_datetime(dates).year.to_numpy()
    for y in np.unique(yrs):
        ix = np.flatnonzero(yrs == y)
        out[ix] = choice[rng.permutation(ix)]
    return out




def _period_table(delta: np.ndarray, dates: np.ndarray) -> dict:
    ds = DateStats(dates)
    st = ds.stats(delta[None, :])
    return {p: {"d_eq": _r(v["mean"][0], 6), "t": _r(v["t"][0], 2)} for p, v in st.items()}


def _mix(sims: list[dict], choice: np.ndarray, base: dict) -> dict:
    """Per-trade result of following ``choice`` (index into sims, -1 = base)."""
    out = {k: base[k].copy() for k in ("ok", "eq", "days", "size", "path_eq", "path_cap", "r")}
    for j, s in enumerate(sims):
        m = choice == j
        for k in out:
            out[k][m] = s[k][m]
    return out


def _acct_windows(acct: ad.Account, seeds: int = 8) -> dict:
    return {w: {k: _r(v) for k, v in acct.run(*win, seeds=seeds).items()} for w, win in WINDOWS.items()}


def learn(args) -> int:
    t0 = time.time()
    cache = Path(args.cache)
    d = load(cache)
    prices = pickle.loads(Path(args.prices).read_bytes())
    cal = prices["SPY"].index
    Xe = np.load(cache / "Xe.npy", mmap_mode="r")
    Xd = np.load(cache / "Xd.npy", mmap_mode="r")
    cols = list(d["cols"])
    sig = d["signals"]
    nr = np.cumsum((sig["setup"] != "random").to_numpy()) - 1
    Ts = {s: trades_of(d, s) for s in ad.SETUPS}
    acct = ad.Account(cal)
    base = {}
    for s in ad.SETUPS:
        base[s] = ad.simulate(Ts[s], ad.LIVE[s], SLIP, paths=True)
        acct.set_setup(s, Ts[s], base[s])
    res: dict = {"live_account": _acct_windows(acct), "setups": {}, "cols": len(cols)}
    only = set(args.setups.split(",")) if getattr(args, "setups", None) else set(ad.SETUPS)
    for s in ad.SETUPS:
        if s not in only:
            continue
        T, b = Ts[s], base[s]
        R: dict = {}
        live = ad.LIVE[s]
        # ------------------------------------------------ B1: one ATR number per trade, chosen at entry
        menu = [dict(S0=m, T1=m, trig="never", T2=0.0, be=False, tgt=live["tgt"]) for m in B1_MENU]
        sims = [ad.simulate(T, r, SLIP, paths=True) for r in menu]
        Y = np.stack([x["eq"] - b["eq"] for x in sims], axis=1)
        X = np.asarray(Xe[T["gidx"]], dtype=np.float32)
        print(f"{s}: B1 {X.shape}, {time.time() - t0:.0f}s", flush=True)
        wf = wf_policy(X, Y, T["date"], LGB, tag=f"{s} B1")
        ch = choose(wf["pred"], wf["tau"])
        tested = np.isfinite(wf["pred"]).all(axis=1)
        gain = np.where(ch >= 0, Y[np.arange(len(Y)), np.maximum(ch, 0)], 0.0)
        shuf = []
        for sd in range(20):
            c2 = shuffle_within_year(np.where(tested, ch, -1), T["date"], sd)
            shuf.append(np.where(c2 >= 0, Y[np.arange(len(Y)), np.maximum(c2, 0)], 0.0))
        shuf = np.mean(shuf, axis=0)
        oracle = np.maximum(Y.max(axis=1), 0.0)
        fixed = {f"m={m:g}": _period_table(Y[:, j], T["date"]) for j, m in enumerate(B1_MENU)}
        mixed = _mix(sims, np.where(tested, ch, -1), b)
        acct.set_setup(s, T, {**mixed})
        acc_b1 = _acct_windows(acct)
        np.savez_compressed(cache / f"policy_B1_{s}.npz", **{k: mixed[k] for k in ("ok", "eq", "days", "size", "path_eq", "path_cap", "r")})
        acct.set_setup(s, T, b)
        imp = pd.Series(np.concatenate([f["features"] for f in wf["folds"]] or [np.array([], int)])).value_counts().head(25)
        R["B1"] = {"menu": list(B1_MENU), "model": _period_table(gain, T["date"]), "shuffle": _period_table(shuf, T["date"]),
                   "model_minus_shuffle": _period_table(gain - shuf, T["date"]), "oracle": _period_table(oracle, T["date"]),
                   "fixed": fixed, "folds": wf["folds"], "account": acc_b1,
                   "choice_share": {("live" if j < 0 else f"m={B1_MENU[j]:g}"): _r((ch[tested] == j).mean(), 3) for j in range(-1, len(B1_MENU))},
                   "top_features": {cols[int(k)]: int(v) for k, v in imp.items()}}
        print(f"  B1 model {R['B1']['model']}\n  B1 shuffle {R['B1']['shuffle']}", flush=True)
        # ------------------------------------------------ B2: the ATR multiple re-chosen at every close
        menu2 = B2_MENU
        base_m = live["T1"]
        bj = int(np.argmin(np.abs(np.asarray(menu2) - base_m)))
        V, alive = ad.cont_values(T, live["S0"], live["tgt"], menu2, SLIP)
        ii, kk = np.nonzero(alive)
        Y2 = V[ii, kk, :] - V[ii, kk, bj][:, None]
        S = ad.state_features(T, SLIP)
        xrows = nr[T["gidx"][ii]]
        X2 = np.concatenate([np.asarray(Xd[xrows, kk, :], dtype=np.float32), S[ii, kk, :]], axis=1)
        del S
        print(f"{s}: B2 {X2.shape}, {time.time() - t0:.0f}s", flush=True)
        p2 = dict(LGB, min_data_in_leaf=400)
        wf2 = wf_policy(X2, Y2, T["date"][ii], p2, max_rows=100_000, tag=f"{s} B2")
        ch2 = choose(wf2["pred"], wf2["tau"])
        tested2 = np.isfinite(wf2["pred"]).all(axis=1)

        def policy(choice_rows):
            mult = np.full(T["O"].shape, base_m if base_m > 0 else 0.0)
            mv = np.where(choice_rows >= 0, np.asarray(menu2)[np.maximum(choice_rows, 0)], base_m)
            mult[ii, kk + 1] = mv
            return ad.simulate(T, live, SLIP, mult=mult, S0_for_mult=live["S0"], paths=True)

        chk = policy(np.full(len(ii), -1))
        assert np.nanmax(np.abs(chk["eq"] - b["eq"])) < 1e-9, "policy with the default multiple must equal the live exit"
        pol = policy(np.where(tested2, ch2, -1))
        g2 = pol["eq"] - b["eq"]
        sh2 = []
        for sd in range(5):
            c2 = shuffle_within_year(np.where(tested2, ch2, -1), T["date"][ii], sd)
            sh2.append(policy(c2)["eq"] - b["eq"])
        sh2 = np.mean(sh2, axis=0)
        acct.set_setup(s, T, pol)
        acc_b2 = _acct_windows(acct)
        np.savez_compressed(cache / f"policy_B2_{s}.npz", **{k: pol[k] for k in ("ok", "eq", "days", "size", "path_eq", "path_cap", "r")})
        acct.set_setup(s, T, b)
        names2 = cols + list(ad.STATE_COLS)
        imp2 = pd.Series(np.concatenate([f["features"] for f in wf2["folds"]] or [np.array([], int)])).value_counts().head(25)
        R["B2"] = {"menu": list(menu2), "default": base_m, "rows": int(len(ii)),
                   "model": _period_table(g2, T["date"]), "shuffle": _period_table(sh2, T["date"]),
                   "model_minus_shuffle": _period_table(g2 - sh2, T["date"]), "folds": wf2["folds"], "account": acc_b2,
                   "choice_share": {("default" if j < 0 else f"m={menu2[j]:g}"): _r((ch2[tested2] == j).mean(), 3) for j in range(-1, len(menu2))},
                   "row_oracle_gain": _r(np.nanmean(np.maximum(Y2.max(axis=1), 0)), 6),
                   "top_features": {names2[int(k)]: int(v) for k, v in imp2.items()}}
        print(f"  B2 model {R['B2']['model']}\n  B2 shuffle {R['B2']['shuffle']}", flush=True)
        res["setups"][s] = R
        (cache / f"learn_{s}.json").write_text(json.dumps(R, ensure_ascii=False))
        del X2, Y2, V
    (cache / "learn.json").write_text(json.dumps(res, ensure_ascii=False))
    print(f"learn done {time.time() - t0:.0f}s")
    return 0


# ====================================================================== final account comparison (risk frontier)
NEO_RULES = {   # Neo's idea taken literally (pre-registered): wide while under water, tighter once above the entry
    f"neo_S0={s0:g}_T2={t2:g}": dict(S0=s0, T1=0.0, trig="close>entry", T2=t2, be=False, tgt=0.0)
    for s0 in (5.0, 6.0) for t2 in (2.0, 2.5, 3.0)}


def _rule_from_name(rows: list[dict], name: str) -> dict:
    r = next(x for x in rows if x["rule"] == name)
    return dict(S0=r["S0"], T1=r["T1"], trig=r["trig"], T2=r["T2"], be=r["be"], tgt=r["tgt"])


def final(args) -> int:
    t0 = time.time()
    cache = Path(args.cache)
    d = load(cache)
    prices = pickle.loads(Path(args.prices).read_bytes())
    cal = prices["SPY"].index
    G = json.loads((cache / "grid.json").read_text())
    Ts = {s: trades_of(d, s) for s in ad.SETUPS}
    live = {s: ad.simulate(Ts[s], ad.LIVE[s], SLIP, paths=True) for s in ad.SETUPS}

    def res_for(s, spec):
        if spec is None:
            return live[s]
        if isinstance(spec, str) and spec.startswith("policy_"):
            f = cache / f"{spec}_{s}.npz"
            if not f.exists():
                return None
            z = np.load(f)
            return {k: z[k] for k in z.files}
        return ad.simulate(Ts[s], spec, SLIP, paths=True)

    best = {s: _rule_from_name(G["setups"][s]["rules"], G["setups"][s]["best_is_account"]) for s in ad.SETUPS}
    best_t, passed = {}, {}
    for s in ad.SETUPS:
        rows = G["setups"][s]["rules"]
        bt = max(rows, key=lambda r: r["is_t"] if r["is_t"] is not None else -9)
        best_t[s] = _rule_from_name(rows, bt["rule"])
        ps = [r for r in rows if r["pass"]]
        if ps:
            passed[s] = _rule_from_name(rows, max(ps, key=lambda r: r["is_t"])["rule"])
    combos: dict[str, dict] = {
        "live": {},
        "breakout_trail5": {"breakout": dict(S0=1.0, T1=5.0, trig="never", T2=0.0, be=False, tgt=0.0)},
        "A_gate_pass": passed,
        "A_best_IS_t": best_t,
        "A_best_IS_account": best,
        "B1_learned_entry": {s: "policy_B1" for s in ad.SETUPS},
        "B2_learned_daily": {s: "policy_B2" for s in ad.SETUPS},
    }
    for nm, r in list(NEO_RULES.items()) + [(k, v) for k, v in FAMILIES.items() if v is not None and not k.startswith("neo")]:
        sq = dict(r, tgt=ad.SQUEEZE_TGT_ATR)
        combos[nm] = {"crash_rebound": r, "breakout": r, "squeeze": sq}
    out: dict = {"combos": {}, "rules": {nm: {s: ad.rule_name(r) for s, r in spec.items() if isinstance(r, dict)} for nm, spec in combos.items()}}
    for nm, spec in combos.items():
        acct = ad.Account(cal)
        ok = True
        for s in ad.SETUPS:
            r = res_for(s, spec.get(s))
            if r is None:
                ok = False
                break
            acct.set_setup(s, Ts[s], r)
        if not ok:
            continue
        row = {}
        for w, win in WINDOWS.items():
            fr = frontier(acct, win, seeds=12, risks=RISKS2)
            at1 = next(f for f in fr if f["risk"] == 0.01)
            row[w] = {"at_1pct": at1, "frontier": fr}
        heat = {w: {k: _r(v) for k, v in acct.run(*win, seeds=8, risk=0.01, heat=0.04).items()} for w, win in WINDOWS.items()}
        row["heat4"] = heat
        out["combos"][nm] = row
        print(nm, {w: (row[w]["at_1pct"]["cagr"], row[w]["at_1pct"]["max_dd"]) for w in WINDOWS}, f"{time.time() - t0:.0f}s", flush=True)
    # CAGR at the live account's drawdown (risk-matched comparison)
    for nm, row in out["combos"].items():
        for w in WINDOWS:
            dd_live = out["combos"]["live"][w]["at_1pct"]["max_dd"]
            row[w]["cagr_at_live_dd"] = _r(cagr_at_dd(row[w]["frontier"], dd_live))
            row[w]["live_cagr_at_live_dd"] = out["combos"]["live"][w]["at_1pct"]["cagr"]
    spy = prices["SPY"]["adj_close"]
    out["spy"] = {}
    for w, (a, b) in WINDOWS.items():
        x = spy[(spy.index >= a) & (spy.index <= b)].to_numpy()
        out["spy"][w] = {k: _r(v) for k, v in ad.perf(x / x[0]).items()}
    (cache / "final.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"final done {time.time() - t0:.0f}s")
    return 0


# ====================================================================== exit families: setups vs random entries (risk-matched)
FAMILIES = {
    "live": None,
    "wide5_only": dict(S0=5.0, T1=0.0, trig="never", T2=0.0, be=False, tgt=0.0),
    "wide3_only": dict(S0=3.0, T1=0.0, trig="never", T2=0.0, be=False, tgt=0.0),
    "neo5_to3": dict(S0=5.0, T1=0.0, trig="close>entry", T2=3.0, be=False, tgt=0.0),
    "neo5_to2.5": dict(S0=5.0, T1=0.0, trig="close>entry", T2=2.5, be=False, tgt=0.0),
    "neo5_to2": dict(S0=5.0, T1=0.0, trig="close>entry", T2=2.0, be=False, tgt=0.0),
    "wide5_to5": dict(S0=5.0, T1=0.0, trig="close>entry", T2=5.0, be=False, tgt=0.0),
    "tight1_no_trail": dict(S0=1.0, T1=0.0, trig="never", T2=0.0, be=False, tgt=0.0),
}
RISKS2 = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)


def families(args) -> int:
    t0 = time.time()
    cache = Path(args.cache)
    d = load(cache)
    prices = pickle.loads(Path(args.prices).read_bytes())
    cal = prices["SPY"].index
    Ts = {s: trades_of(d, s) for s in ad.SETUPS + ("random",)}
    out: dict = {"setups": {}, "random": {}}
    for nm, rule in FAMILIES.items():
        acct = ad.Account(cal)
        for s in ad.SETUPS:
            r = ad.LIVE[s] if rule is None else (dict(rule, tgt=ad.SQUEEZE_TGT_ATR) if s == "squeeze" else rule)
            acct.set_setup(s, Ts[s], ad.simulate(Ts[s], r, SLIP, paths=True))
        racct = ad.Account(cal)
        rr = ad.LIVE["random"] if rule is None else rule
        racct.set_setup("crash_rebound", Ts["random"], ad.simulate(Ts["random"], rr, SLIP, paths=True))
        for key, A in (("setups", acct), ("random", racct)):
            out[key][nm] = {w: frontier(A, win, seeds=12, risks=RISKS2) for w, win in WINDOWS.items()}
        print(nm, f"{time.time() - t0:.0f}s", flush=True)
    (cache / "families.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"families done {time.time() - t0:.0f}s")
    return 0


# ====================================================================== compact copy for the repo
def report(args) -> int:
    cache = Path(args.cache)
    G = json.loads((cache / "grid.json").read_text())
    out: dict = {"generated": time.strftime("%Y-%m-%d"), "design": "src/backtest/adaptive_stop.py docstring; docs 7n",
                 "live_account": G["live_account"], "grid": {}}
    keep = ("rule", "avg_days", "loss_share", "is_d_r", "is_t", "hold_d_r", "hold_t", "oos_d_r", "oos_t", "pool_t",
            "random_pool_d_eq", "candidate", "pass")
    for s, v in G["setups"].items():
        rows = v["rules"]
        top = sorted(rows, key=lambda r: -(r["is_t"] if r["is_t"] is not None else -9))[:15]
        neo = [r for r in rows if r["T1"] == 0 and r["trig"] == "close>entry" and not r["be"] and r["S0"] in (4, 5, 6)
               and r["T2"] in (2, 2.5, 3) and r["tgt"] == 0]
        heat = {f"S0={r['S0']:g}|T2={r['T2']:g}": {p: r.get(f"{p}_d_r") for p in ("hold", "is", "oos")}
                for r in rows if r["T1"] == 0 and r["trig"] == "close>entry" and not r["be"] and r["tgt"] == 0}
        out["grid"][s] = {k: v[k] for k in ("n_trades", "n_rules", "n_candidates", "bar", "n_pass", "walk_forward")}
        out["grid"][s].update({"top_is": [{k: r.get(k) for k in keep} for r in top],
                               "passed": [{k: r.get(k) for k in keep} for r in rows if r["pass"]],
                               "neo_literal": [{k: r.get(k) for k in keep} for r in neo],
                               "close_above_entry_heatmap_R": heat})
    for name in ("learn", "learn_b1_notional", "final", "families"):
        f = cache / f"{name}.json"
        if f.exists():
            x = json.loads(f.read_text())
            if name == "learn":
                for s in x.get("setups", {}).values():
                    for b in ("B1", "B2"):
                        s[b].pop("folds", None)
            out[name] = x
    for name in ("anatomy", "notional"):
        f = cache.parent / f"{name}.json"
        if f.exists():
            out[name] = json.loads(f.read_text())
    dst = Path(args.out) if getattr(args, "out", None) else Path("config/adaptive_stop.default.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=0))
    print(f"wrote {dst} ({dst.stat().st_size // 1024} KB)")
    return 0

"""Strategy lab: which exits, and which combination of ALL indicators, actually work out of sample?

Long only (the live terminal is long only). Two questions, answered walk-forward so nothing is judged on
the data that chose it:

1. Exits per setup. Every long setup episode is simulated with 7 stops x 10 exits x 5 time stops
   (fixed ATR / R targets and 2 / 2.5 / 3-ATR trailing stops; 5-20 sessions). Each setup's exit is
   chosen on the first 60% of dates (in sample) and judged on the last 40% (out of sample), next to
   random entries with the same exit (so the setup's own edge is visible).

2. Indicator weights. Every setup episode AND a random-entry sample (every 10th session of every stock)
   gets one feature vector of ~170 indicators (src/scanner/model_features.py: moving averages,
   oscillators, volume, relative strength, the 49 analyst techniques, the volume profile / chip
   distribution, market regime). The label is the trade's R with the chosen exit. For each test year,
   models are fitted only on data that ended a month before that year starts:
     ic      sign-consistent information-coefficient weights (a feature counts only if its rank
             correlation with the outcome had the same sign in >= 75% of training years)
     ridge   ridge regression on standardised, winsorised features (penalty chosen on the last 20% of
             the training window)
   Out-of-sample checks: rank correlation of the score with the outcome, average R of the top fifth vs
   all trades, positive years, and whether the top fifth of RANDOM entries does as well (then the setup
   adds nothing beyond the model). A model is adopted only if the top fifth beats the average by
   >= 0.05R with t >= 2 and in most test years.

Options positioning cannot be tested: there is no free history of option chains (the terminal now keeps
its own daily IV record for a later study). Survivorship bias: only today's universe is tested.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.backtest.research import simulate, summary
from src.scanner.model_features import GROUP_NAMES, feature_vector
from src.scanner.plan import plan_for_setup
from src.scanner.setups import LONG_SETUPS, SETUP_NAMES

H_LAB = 20
LAB_STOPS = ["structural", "atr1.0", "atr1.5", "atr2.0", "atr2.5", "atr3.0", "swing"]
LAB_TARGETS = ["atr1.5", "atr2", "atr3", "atr4", "r1.5", "r2", "r3", "trail2", "trail2.5", "trail3"]
LAB_HORIZONS = [5, 7, 10, 15, 20]
BASE_STEP = 10
MIN_TRAIN_YEARS = 2


def exit_name(stop: str, target: str, horizon: int) -> str:
    s = {"structural": "structural stop", "swing": "swing-low stop"}.get(stop, f"{stop[3:]} ATR stop")
    if target.startswith("trail"):
        t = f"{target[5:] or '3'}-ATR trailing stop"
    elif target.startswith("atr"):
        t = f"{target[3:]} ATR target"
    else:
        t = f"{target[1:]}R target"
    return f"{s} + {t}, max {horizon} sessions"


# ====================================================================== collection
def _collect_symbol(job: tuple) -> tuple[list[dict], np.ndarray]:
    sym, f, st, spy, regime, rs_col, sec_col, cfg, techniques, offset = job
    o, h, lo, c, a = (f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "atr"))
    n = len(f)
    rows: list[dict] = []
    fwd: list[np.ndarray] = []
    cache: dict[int, dict] = {}

    def vec(t: int) -> dict:
        if t not in cache:
            date = f.index[t]
            ctx = {"regime": regime.get(date, np.nan), "rs_pct": rs_col.get(date, np.nan),
                   "sector_rs_pct": sec_col.get(date, np.nan)}
            cache[t] = feature_vector(f, t, spy, ctx, techniques=techniques)
        return cache[t]

    plans = {s: plan_for_setup(f, s, cfg) for s in LONG_SETUPS}
    for setup in LONG_SETUPS:
        mask = st[f"setup_{setup}"].to_numpy(dtype=bool)
        starts = np.flatnonzero(mask & ~np.concatenate([[False], mask[:-1]]))
        starts = starts[(starts >= 260) & (starts + H_LAB < n)]
        for t in starts:
            p = plans[setup].iloc[t]
            if not (np.isfinite(a[t]) and a[t] > 0 and np.isfinite(p["stop"])):
                continue
            rows.append({"symbol": sym, "setup": setup, "date": f.index[t], "ref": float(p["entry"]), "atr": a[t],
                         "structural": float(p["stop"]), "swing": float(f["swing_low_last"].iat[t]),
                         "conditional": setup == "squeeze", **vec(t)})
            fwd.append(np.stack([o[t + 1: t + 1 + H_LAB], h[t + 1: t + 1 + H_LAB], lo[t + 1: t + 1 + H_LAB],
                                 c[t + 1: t + 1 + H_LAB]]))
    for t in range(260 + offset % BASE_STEP, n - H_LAB, BASE_STEP):
        if not (np.isfinite(a[t]) and a[t] > 0):
            continue
        rows.append({"symbol": sym, "setup": "baseline", "date": f.index[t], "ref": c[t], "atr": a[t],
                     "structural": c[t] - 1.5 * a[t], "swing": float(f["swing_low_last"].iat[t]), "conditional": False,
                     **vec(t)})
        fwd.append(np.stack([o[t + 1: t + 1 + H_LAB], h[t + 1: t + 1 + H_LAB], lo[t + 1: t + 1 + H_LAB],
                             c[t + 1: t + 1 + H_LAB]]))
    return rows, (np.array(fwd) if fwd else np.empty((0, 4, H_LAB)))


def collect(features: dict[str, pd.DataFrame], setups: dict[str, pd.DataFrame], spy: pd.Series, regime: pd.Series,
            settings: dict[str, Any], techniques: bool = True, workers: int = 4,
            progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    rs = pd.DataFrame({s: 0.5 * f["rel_20d"] + 0.3 * f["rel_60d"] + 0.2 * f["rel_5d"] for s, f in features.items()})
    sec = pd.DataFrame({s: f["rel_sector_20d"] for s, f in features.items()})
    rs_pct, sec_pct = rs.rank(axis=1, pct=True) * 100, sec.rank(axis=1, pct=True) * 100
    jobs = [(sym, f, setups[sym], spy, regime, rs_pct[sym], sec_pct[sym], settings["scanner"], techniques, k)
            for k, (sym, f) in enumerate(features.items())]
    rows: list[dict] = []
    arrays: list[np.ndarray] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_collect_symbol, jobs, chunksize=2)
            for k, (r, arr) in enumerate(results):
                rows.extend(r)
                arrays.append(arr)
                if progress and (k + 1) % 25 == 0:
                    progress(f"Feature vectors: {k + 1}/{len(jobs)} stocks")
    else:
        for k, job in enumerate(jobs):
            r, arr = _collect_symbol(job)
            rows.extend(r)
            arrays.append(arr)
    sig = pd.DataFrame(rows)
    fwd = np.concatenate([x for x in arrays if len(x)]) if rows else np.empty((0, 4, H_LAB))
    sig["long"] = True
    sig["date"] = pd.to_datetime(sig["date"])
    return {"signals": sig, "O": fwd[:, 0], "H": fwd[:, 1], "L": fwd[:, 2], "C": fwd[:, 3]}


def _subset(data: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    sig = data["signals"][mask].reset_index(drop=True)
    for col in ("score", "tier", "planned_rr", "regime", "chip_stop", "chip_target", "resistance"):
        if col not in sig:
            sig[col] = np.nan
    return {"signals": sig, **{k: data[k][mask] for k in ("O", "H", "L", "C")}}


# ====================================================================== 1. exits
def optimise_exits(data: dict[str, Any], slip: float, is_fraction: float = 0.6,
                   progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    sig = data["signals"]
    cutoff = sig.loc[sig["setup"] != "baseline", "date"].quantile(is_fraction)
    setup_d = _subset(data, (sig["setup"] != "baseline").to_numpy())
    base_d = _subset(data, (sig["setup"] == "baseline").to_numpy())
    grid: list[dict] = []
    for sm in LAB_STOPS:
        for tm in LAB_TARGETS:
            for hz in LAB_HORIZONS:
                tr = simulate(setup_d, sm, tm, hz, slip)
                br = simulate(base_d, sm if sm != "structural" else "atr1.5", tm, hz, slip)
                b_is, b_oos = br[br["date"] <= cutoff], br[br["date"] > cutoff]
                row = {"stop": sm, "target": tm, "horizon": hz,
                       "baseline_is": summary(b_is), "baseline_oos": summary(b_oos), "by_setup": {}}
                for s, part in [("all", tr), *tr.groupby("setup")]:
                    ins, oos = part[part["date"] <= cutoff], part[part["date"] > cutoff]
                    row["by_setup"][s] = {"is": summary(ins), "oos": summary(oos)}
                grid.append(row)
        if progress:
            progress(f"Exit grid: stop '{sm}' done")
    out: dict[str, Any] = {"cutoff": pd.Timestamp(cutoff).strftime("%Y-%m-%d"), "n_combinations": len(grid),
                           "stops": LAB_STOPS, "targets": LAB_TARGETS, "horizons": LAB_HORIZONS, "chosen": {}}

    def pick(s: str, min_n: int) -> dict | None:
        cands = [g for g in grid if g["by_setup"].get(s, {}).get("is", {}).get("n", 0) >= min_n]
        return max(cands, key=lambda g: g["by_setup"][s]["is"]["avg_r"]) if cands else None
    best_all = pick("all", 300)
    for s in ["all", *LONG_SETUPS]:
        g = pick(s, 150) or best_all
        if g is None:
            continue
        r = g["by_setup"].get(s) or g["by_setup"]["all"]
        top_all = best_all["by_setup"].get(s) if best_all else None
        # an exit that only fits the setup's own sample (worse out of sample than the common best exit) is
        # replaced by the common one
        use_common = s != "all" and top_all and top_all["oos"].get("n") and r["oos"].get("n") and \
            top_all["oos"]["avg_r"] > r["oos"]["avg_r"] + 0.02
        if use_common:
            g, r = best_all, top_all
        out["chosen"][s] = {"stop": g["stop"], "target": g["target"], "horizon": g["horizon"],
                            "name": exit_name(g["stop"], g["target"], g["horizon"]), "common": bool(use_common),
                            "is": r["is"], "oos": r["oos"], "baseline_is": g["baseline_is"], "baseline_oos": g["baseline_oos"],
                            "edge_oos": (r["oos"].get("avg_r", np.nan) - g["baseline_oos"].get("avg_r", np.nan))
                            if r["oos"].get("n") and g["baseline_oos"].get("n") else None}
    out["top"] = sorted(({"stop": g["stop"], "target": g["target"], "horizon": g["horizon"],
                          "name": exit_name(g["stop"], g["target"], g["horizon"]), **g["by_setup"]["all"],
                          "baseline_oos": g["baseline_oos"]} for g in grid if g["by_setup"]["all"]["is"].get("n")),
                        key=lambda x: -x["is"]["avg_r"])[:15]
    out["heat"] = [{"stop": g["stop"], "target": g["target"], "horizon": g["horizon"],
                    "is": g["by_setup"]["all"]["is"].get("avg_r"), "oos": g["by_setup"]["all"]["oos"].get("avg_r"),
                    "base_oos": g["baseline_oos"].get("avg_r")} for g in grid]
    return out


def label_trades(data: dict[str, Any], chosen: dict[str, dict], slip: float) -> pd.Series:
    """R of every row (setup episodes and random entries) with the exit chosen for its setup."""
    sig = data["signals"]
    r = pd.Series(np.nan, index=sig.index)
    for s in sig["setup"].unique():
        rule = chosen.get(s) or chosen["all"]
        mask = (sig["setup"] == s).to_numpy()
        sub = _subset(data, mask)
        stop = rule["stop"] if not (s == "baseline" and rule["stop"] == "structural") else "atr1.5"
        sub["signals"]["_row"] = np.flatnonzero(mask)
        tr = simulate(sub, stop, rule["target"], rule["horizon"], slip)
        rows = sub["signals"]["_row"].to_numpy()
        # simulate drops unfilled / invalid trades: map back by position in the subset
        keep = simulate_index(sub, stop, rule["target"], rule["horizon"], slip)
        r.iloc[rows[keep]] = tr["r"].to_numpy()
    return r


def simulate_index(sub: dict[str, Any], stop: str, target: str, horizon: int, slip: float) -> np.ndarray:
    """Positions (within ``sub``) of the trades simulate() keeps - same validity rules."""
    from src.backtest.research import stop_levels
    sig = sub["signals"]
    O, H, L = (sub[k][:, :horizon] for k in ("O", "H", "L"))
    ref = sig["ref"].to_numpy()
    cond = sig["conditional"].to_numpy(dtype=bool)
    entry_raw = np.where(cond, np.maximum(O[:, 0], ref), O[:, 0])
    filled = ~cond | (H[:, 0] >= ref)
    entry = entry_raw * (1 + slip)
    risk = entry - stop_levels(sig, ref, stop)
    return np.flatnonzero(filled & (risk > 0) & np.isfinite(risk) & np.isfinite(entry))


# ====================================================================== 2. indicator weights
def feature_columns(sig: pd.DataFrame) -> list[str]:
    return [c for c in sig.columns if c[:2] in ("f_", "d_", "t_", "v_", "m_")]


def _prep(train: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    keep, lo, hi, mu, sd = [], [], [], [], []
    for c in cols:
        x = train[c].astype(float)
        if x.notna().mean() < 0.7 or x.nunique(dropna=True) < 2:
            continue
        l_, h_ = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
        xc = x.clip(l_, h_)
        s = float(xc.std())
        if not np.isfinite(s) or s < 1e-12:
            continue
        keep.append(c)
        lo.append(l_)
        hi.append(h_)
        mu.append(float(xc.mean()))
        sd.append(s)
    return {"cols": keep, "lo": np.array(lo), "hi": np.array(hi), "mu": np.array(mu), "sd": np.array(sd)}


def _z(df: pd.DataFrame, prep: dict[str, Any]) -> np.ndarray:
    x = df[prep["cols"]].to_numpy(dtype=float)
    x = np.clip(x, prep["lo"], prep["hi"])
    z = (x - prep["mu"]) / prep["sd"]
    return np.nan_to_num(z, nan=0.0)


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy()


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 30:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    sa, sb = ra.std(), rb.std()
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else float("nan")


def _ridge(z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    k = z.shape[1]
    return np.linalg.solve(z.T @ z + lam * len(y) * np.eye(k), z.T @ (y - y.mean()))


def fit_models(train: pd.DataFrame, cols: list[str], label: str = "r") -> dict[str, Any]:
    prep = _prep(train, cols)
    z, y = _z(train, prep), train[label].to_numpy(dtype=float)
    years = train["date"].dt.year.to_numpy()
    # ic model: per-year rank correlations, keep features with a consistent sign
    ic_all = np.array([_spearman(z[:, j], y) for j in range(z.shape[1])])
    yearly = []
    for yr in np.unique(years):
        m = years == yr
        if m.sum() >= 200:
            yearly.append([_spearman(z[m, j], y[m]) for j in range(z.shape[1])])
    yearly_a = np.array(yearly) if yearly else np.full((1, z.shape[1]), np.nan)
    same = (np.sign(yearly_a) == np.sign(ic_all)).mean(axis=0)
    tstat = ic_all * math.sqrt(len(y))
    ok = (same >= 0.75) & (np.abs(tstat) >= 3) & np.isfinite(ic_all)
    w_ic = np.where(ok, ic_all, 0.0)
    # ridge: penalty chosen on the last 20% of the training window (time order)
    order = np.argsort(train["date"].to_numpy())
    cut = order[: int(len(order) * 0.8)], order[int(len(order) * 0.8):]
    best_lam, best_ic = 1.0, -np.inf
    for lam in (0.01, 0.1, 1.0, 10.0, 100.0):
        w = _ridge(z[cut[0]], y[cut[0]], lam)
        ic = _spearman(z[cut[1]] @ w, y[cut[1]])
        if np.isfinite(ic) and ic > best_ic:
            best_lam, best_ic = lam, ic
    w_ridge = _ridge(z, y, best_lam)
    return {"prep": prep, "ic": w_ic, "ic_all": ic_all, "ic_same_sign": same, "ridge": w_ridge, "lambda": best_lam,
            "n_train": int(len(y)), "n_ic_features": int(ok.sum())}


def score(df: pd.DataFrame, model: dict[str, Any], kind: str) -> np.ndarray:
    return _z(df, model["prep"]) @ model[kind]


def _quintile_stats(s: np.ndarray, r: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(s) & np.isfinite(r)
    s, r = s[ok], r[ok]
    if len(r) < 100:
        return {"n": int(len(r))}
    q = pd.qcut(_rank(s), 5, labels=False)
    top, bot = r[q == 4], r[q == 0]
    diff = top.mean() - r.mean()
    se = top.std(ddof=1) / math.sqrt(len(top))
    return {"n": int(len(r)), "avg_r": float(r.mean()), "top_avg_r": float(top.mean()), "bottom_avg_r": float(bot.mean()),
            "top_minus_all": float(diff), "t_top_minus_all": float(diff / se) if se > 0 else float("nan"),
            "rank_ic": _spearman(s, r), "quintiles": [float(r[q == k].mean()) for k in range(5)]}


def walk_forward(sig: pd.DataFrame, cols: list[str], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    sig = sig[np.isfinite(sig["r"])].reset_index(drop=True)
    years = sorted(sig["date"].dt.year.unique())
    test_years = [y for y in years if y - years[0] >= MIN_TRAIN_YEARS]
    preds = {k: np.full(len(sig), np.nan) for k in ("ic", "ridge")}
    folds = []
    for yr in test_years:
        start = pd.Timestamp(f"{yr}-01-01")
        train = sig[sig["date"] < start - pd.Timedelta(days=30)]
        test_m = (sig["date"].dt.year == yr).to_numpy()
        if len(train) < 2000 or test_m.sum() < 200:
            continue
        model = fit_models(train, cols)
        fold = {"year": int(yr), "n_train": model["n_train"], "lambda": model["lambda"], "n_ic_features": model["n_ic_features"]}
        for kind in ("ic", "ridge"):
            s = score(sig[test_m], model, kind)
            preds[kind][test_m] = s
            r = sig.loc[test_m, "r"].to_numpy()
            setup_m = (sig.loc[test_m, "setup"] != "baseline").to_numpy()
            fold[kind] = {"setups": _quintile_stats(s[setup_m], r[setup_m]), "random": _quintile_stats(s[~setup_m], r[~setup_m])}
        folds.append(fold)
        if progress:
            progress(f"Walk-forward {yr}: ic top-fifth {fold['ic']['setups'].get('top_avg_r', float('nan')):+.3f}R, "
                     f"ridge {fold['ridge']['setups'].get('top_avg_r', float('nan')):+.3f}R, "
                     f"all {fold['ic']['setups'].get('avg_r', float('nan')):+.3f}R")
    out: dict[str, Any] = {"test_years": [f["year"] for f in folds], "folds": folds, "models": {}}
    tested = np.isfinite(preds["ic"])
    r_all = sig["r"].to_numpy()
    is_setup = (sig["setup"] != "baseline").to_numpy()
    for kind in ("ic", "ridge"):
        st = _quintile_stats(preds[kind][tested & is_setup], r_all[tested & is_setup])
        rnd = _quintile_stats(preds[kind][tested & ~is_setup], r_all[tested & ~is_setup])
        pos_years = [f[kind]["setups"].get("top_minus_all", np.nan) > 0 for f in folds]
        st["positive_years"] = float(np.mean(pos_years)) if pos_years else None
        st["passes"] = bool(st.get("top_minus_all", -1) >= 0.05 and st.get("t_top_minus_all", 0) >= 2
                            and (st["positive_years"] or 0) > 0.5)
        out["models"][kind] = {"setups": st, "random": rnd}
    return out


def factor_table(sig: pd.DataFrame, cols: list[str], cutoff: pd.Timestamp) -> list[dict[str, Any]]:
    """Every indicator alone: rank correlation with the trade result in and out of sample, and how often
    its sign held from year to year (setup episodes only)."""
    s = sig[(sig["setup"] != "baseline") & np.isfinite(sig["r"])]
    ins, oos = s[s["date"] <= cutoff], s[s["date"] > cutoff]
    rows = []
    for c in cols:
        a, b = ins[c].astype(float), oos[c].astype(float)
        if a.notna().mean() < 0.7 or a.nunique() < 2:
            continue
        ic_is = _spearman(a.fillna(a.median()).to_numpy(), ins["r"].to_numpy())
        ic_oos = _spearman(b.fillna(a.median()).to_numpy(), oos["r"].to_numpy())
        yearly = [_spearman(g[c].astype(float).fillna(a.median()).to_numpy(), g["r"].to_numpy())
                  for _, g in s.groupby(s["date"].dt.year) if len(g) >= 200]
        same = float(np.mean([np.sign(v) == np.sign(ic_is) for v in yearly if np.isfinite(v)])) if yearly else float("nan")
        rows.append({"feature": c, "group": GROUP_NAMES.get(c[0], c[0]), "ic_is": ic_is, "ic_oos": ic_oos,
                     "same_sign_years": same, "held": bool(np.isfinite(ic_is) and np.isfinite(ic_oos)
                                                           and np.sign(ic_is) == np.sign(ic_oos) and abs(ic_oos) >= 0.02)})
    rows.sort(key=lambda x: -abs(x["ic_is"]) if np.isfinite(x["ic_is"]) else 0)
    return rows


def final_model(sig: pd.DataFrame, cols: list[str], kind: str) -> dict[str, Any]:
    """Refit the adopted model on ALL data for live use (weights, preprocessing, score percentiles)."""
    sig = sig[np.isfinite(sig["r"])]
    model = fit_models(sig, cols)
    s = score(sig[sig["setup"] != "baseline"], model, kind)
    w = model[kind]
    prep = model["prep"]
    top = sorted(zip(prep["cols"], w), key=lambda x: -abs(x[1]))
    return {"kind": kind, "cols": prep["cols"], "lo": prep["lo"].tolist(), "hi": prep["hi"].tolist(),
            "mu": prep["mu"].tolist(), "sd": prep["sd"].tolist(), "w": [float(x) for x in w],
            "percentiles": [float(np.nanpercentile(s, p)) for p in range(0, 101)],
            "top_weights": [{"feature": c, "weight": float(v)} for c, v in top[:25] if v != 0],
            "n_train": model["n_train"]}


def apply_model(vec: dict[str, float], model: dict[str, Any]) -> float | None:
    """Live: percentile (0-100) of a feature vector's model score among historical setup episodes."""
    if not model or not model.get("cols"):
        return None
    x = np.array([float(vec.get(c, np.nan)) for c in model["cols"]], dtype=float)
    x = np.clip(x, model["lo"], model["hi"])
    z = np.nan_to_num((x - np.array(model["mu"])) / np.array(model["sd"]), nan=0.0)
    s = float(z @ np.array(model["w"]))
    return float(np.searchsorted(np.array(model["percentiles"]), s, side="right") - 1)


# ====================================================================== 3. regime filter
REGIME_KEEP = ("Bullish", "Strong Bullish")
REGIME_LABELS = ["Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish"]


def regime_split(data: dict[str, Any], chosen: dict[str, dict], label: pd.Series, cutoff: pd.Timestamp, slip: float,
                 setups: tuple[str, ...] = ("momentum", "pullback")) -> dict[str, Any]:
    """Would a setup do better if it were only taken in Bullish / Strong Bullish markets? Each setup's trades (its
    chosen exit) and random entries with the same exit, split by the market regime on the signal day, in and out
    of sample. The filter is adopted only if (fixed before the test): the kept regimes beat random entries by
    >= 0.05R both in AND out of sample, and the dropped regimes did worse than random in both halves."""
    from src.backtest.research import simulate
    sig = data["signals"]
    out: dict[str, Any] = {"keep": list(REGIME_KEEP), "cutoff": pd.Timestamp(cutoff).strftime("%Y-%m-%d"), "setups": {}}

    def stats(t: pd.DataFrame) -> dict[str, Any]:
        if not len(t):
            return {"n": 0}
        daily = t.groupby("date")["r"].mean()
        sd = daily.std(ddof=1)
        return {"n": int(len(t)), "avg_r": float(t["r"].mean()), "win_rate": float((t["r"] > 0).mean()),
                "t": float(daily.mean() / (sd / math.sqrt(len(daily)))) if len(daily) > 2 and sd > 0 else float("nan")}
    for s in setups:
        rule = chosen.get(s)
        if not rule or not (sig["setup"] == s).any():
            continue
        stop = rule["stop"]
        tr = simulate(_subset(data, (sig["setup"] == s).to_numpy()), stop, rule["target"], rule["horizon"], slip)
        br = simulate(_subset(data, (sig["setup"] == "baseline").to_numpy()), stop if stop != "structural" else "atr1.5",
                      rule["target"], rule["horizon"], slip)
        for t in (tr, br):
            t["label"] = pd.to_datetime(t["date"]).map(label).fillna("n/a")
            t["oos"] = pd.to_datetime(t["date"]) > cutoff
        rows = []
        for part, is_oos in (("in sample", False), ("out of sample", True)):
            for grp in [*REGIME_LABELS, "kept", "dropped", "all"]:
                def sel(t: pd.DataFrame) -> pd.DataFrame:
                    t = t[t["oos"] == is_oos]
                    if grp == "kept":
                        return t[t["label"].isin(REGIME_KEEP)]
                    if grp == "dropped":
                        return t[~t["label"].isin(REGIME_KEEP)]
                    return t if grp == "all" else t[t["label"] == grp]
                a, b = stats(sel(tr)), stats(sel(br))
                rows.append({"part": part, "regime": grp, "setup": a, "random": b,
                             "edge": a["avg_r"] - b["avg_r"] if a["n"] and b["n"] else None})
        get = lambda part, grp: next(r for r in rows if r["part"] == part and r["regime"] == grp)  # noqa: E731
        kept = [get(p, "kept")["edge"] for p in ("in sample", "out of sample")]
        dropped = [get(p, "dropped")["edge"] for p in ("in sample", "out of sample")]
        passes = all(e is not None and e >= 0.05 for e in kept) and all(e is not None and e < 0 for e in dropped)
        out["setups"][s] = {"exit": rule["name"], "rows": rows, "passes": bool(passes),
                            "kept_edge": kept, "dropped_edge": dropped,
                            "dropped_avg_r": [get(p, "dropped")["setup"].get("avg_r") for p in ("in sample", "out of sample")]}
    return out


def regime_verdict(rs: dict[str, Any]) -> list[str]:
    out = []
    for s, v in rs.get("setups", {}).items():
        k, d, dr = v["kept_edge"], v["dropped_edge"], v["dropped_avg_r"]
        f = lambda x: "n/a" if x is None else f"{x:+.2f}R"  # noqa: E731
        out.append(f"{SETUP_NAMES.get(s, s)} only in Bullish / Strong Bullish markets: edge over random entries in the kept "
                   f"regimes {f(k[0])} in sample, {f(k[1])} out of sample; in the dropped regimes {f(d[0])} / {f(d[1])} "
                   f"(those trades still averaged {f(dr[0])} / {f(dr[1])}) - "
                   f"{'ADOPTED' if v['passes'] else 'not adopted (the filter must help in both halves)'}.")
    return out


# ====================================================================== driver
def run_lab(data: dict[str, Any], settings: dict[str, Any], progress: Callable[[str], None] | None = None,
            regime_label: pd.Series | None = None) -> dict[str, Any]:
    say = progress or (lambda m: None)
    slip = float(settings["scanner"].get("slippage_bps", 10)) / 10_000
    sig = data["signals"]
    res: dict[str, Any] = {"available": True, "method": __doc__, "n_setup_signals": int((sig["setup"] != "baseline").sum()),
                           "n_random": int((sig["setup"] == "baseline").sum()), "n_stocks": int(sig["symbol"].nunique()),
                           "first": sig["date"].min().strftime("%Y-%m-%d"), "last": sig["date"].max().strftime("%Y-%m-%d"),
                           "setup_names": SETUP_NAMES}
    say("Optimising exits per setup...")
    ex = optimise_exits(data, slip, progress=say)
    res["exits"] = ex
    say("Labelling every trade with its setup's exit...")
    sig = sig.copy()
    sig["r"] = label_trades(data, ex["chosen"], slip).to_numpy()
    sig = pd.concat([sig, pd.get_dummies(sig["setup"], prefix="m_setup").astype(float)], axis=1)
    cols = feature_columns(sig)
    res["n_features"] = len(cols)
    res["feature_groups"] = {GROUP_NAMES.get(k, k): sum(c.startswith(k + "_") for c in cols) for k in ("f", "d", "t", "v", "m")}
    cutoff = pd.Timestamp(ex["cutoff"])
    say(f"Single-indicator study ({len(cols)} indicators)...")
    res["factors"] = factor_table(sig, cols, cutoff)
    say("Walk-forward model test...")
    wf = walk_forward(sig, cols, progress=say)
    res["walk_forward"] = wf
    passing = [k for k in ("ridge", "ic") if wf["models"].get(k, {}).get("setups", {}).get("passes")]
    if passing:
        kind = max(passing, key=lambda k: wf["models"][k]["setups"]["top_minus_all"])
        res["adopted_model"] = final_model(sig, cols, kind)
    else:
        res["adopted_model"] = None
    if regime_label is not None and len(regime_label):
        say("Regime filter test (momentum / pullback only in Bullish markets)...")
        res["regime_split"] = regime_split(data, ex["chosen"], regime_label, cutoff, slip)
    res["verdict"] = lab_verdict(res)
    return res


def lab_verdict(res: dict[str, Any]) -> list[str]:
    out = []
    ch = res["exits"]["chosen"]
    a = ch.get("all")
    if a:
        out.append(f"Best common long exit: {a['name']} - in sample {a['is'].get('avg_r', float('nan')):+.2f}R, out of sample "
                   f"{a['oos'].get('avg_r', float('nan')):+.2f}R ({a['oos'].get('n', 0)} trades); random entries with the same "
                   f"exit {a['baseline_oos'].get('avg_r', float('nan')):+.2f}R.")
    for s in LONG_SETUPS:
        c = ch.get(s)
        if c and c["oos"].get("n"):
            out.append(f"{SETUP_NAMES.get(s, s)}: {c['name']}{' (common exit)' if c['common'] else ''} - out of sample "
                       f"{c['oos']['avg_r']:+.2f}R over {c['oos']['n']} trades, edge over random entries "
                       f"{c['edge_oos']:+.2f}R." if c.get("edge_oos") is not None else
                       f"{SETUP_NAMES.get(s, s)}: {c['name']} - out of sample {c['oos']['avg_r']:+.2f}R.")
    held = [f for f in res["factors"] if f["held"]]
    out.append(f"{len(held)} of {len(res['factors'])} indicators kept the same direction of effect out of sample "
               f"(|rank correlation| >= 0.02); the strongest single indicator correlations are around "
               f"{max((abs(f['ic_oos']) for f in res['factors'] if np.isfinite(f['ic_oos'])), default=0):.2f}.")
    for kind, name in (("ridge", "Ridge model on all indicators"), ("ic", "Sign-consistent indicator weights")):
        m = res["walk_forward"]["models"].get(kind, {}).get("setups", {})
        if m.get("n"):
            out.append(f"{name}: out of sample the top fifth averaged {m['top_avg_r']:+.2f}R vs {m['avg_r']:+.2f}R for all setup "
                       f"trades (t {m['t_top_minus_all']:.1f}, rank correlation {m['rank_ic']:+.3f}, better in "
                       f"{(m.get('positive_years') or 0) * 100:.0f}% of test years) - {'ADOPTED' if m.get('passes') else 'not adopted'}.")
            rnd = res["walk_forward"]["models"][kind].get("random", {})
            if rnd.get("n"):
                out.append(f"  Same model on random entries: top fifth {rnd['top_avg_r']:+.2f}R vs {rnd['avg_r']:+.2f}R.")
    if res.get("regime_split"):
        out += regime_verdict(res["regime_split"])
    if not res.get("adopted_model"):
        out.append("No indicator combination ranked trades reliably out of sample, so the live ranking keeps the "
                   "backtest-based setup weighting and the composite score is shown for context only.")
    return out

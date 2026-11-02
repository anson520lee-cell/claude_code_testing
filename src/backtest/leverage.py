"""Leverage and scaling-in study (research only): do 2x daily-leveraged long ETFs, tighter stops or
splitting the entry into tranches make the validated setups (crash rebound, squeeze, breakout) better?

Everything is measured in R = profit / planned risk, with the SAME risk budget per trade (1R = 1% of the
account in the portfolio part). That matters: with risk-based sizing a 2x ETF does not double the R of a
trade - it halves the capital the position needs. What leverage changes is
  * the path: a 2x daily-reset ETF compounds 2x the DAILY return (volatility decay in choppy stretches,
    extra gain in trends), its stop sits twice as far away in % for the same underlying move;
  * costs: expense ratio + financing of the borrowed half, every day the position is held;
  * gaps: a gap through the stop costs twice as much in %, so the same R stop can lose more than 1R;
  * capital: the same risk needs half the cash, so more signals can be taken at once.

The synthetic ETF is exact for a daily-reset product (open / high / low / close of day j are 2x the
underlying's move from the previous close, compounded at each close) minus the daily cost; it is
checked against real 2x single-stock ETFs (NVDL, TSLL, ...) in ``validate_synthetic``.

Scaling-in: a plan is a list of tranches (kind, parameter, fraction of the risk budget). Each tranche is
sized so that its own risk to the CURRENT stop is its fraction of the budget, so the total risk never
exceeds 1R whatever fills. Unfilled tranches simply leave risk unused (that is counted, not hidden).
Same-bar ambiguity is resolved against the trade (an add and a stop on the same bar = add fills, then
the stop hits).

Selection protocol (same as the strategy lab / discovery): parameters are chosen on the first 60% of
setup dates, judged on the last 40%, next to random entries (every 10th session of every stock) that
use exactly the same plan. Pass = edge over random >= 0.05R in AND out of sample, and out-of-sample
t >= 2.5 on the edge with one observation per signal date.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.scanner.plan import plan_for_setup

H_MAX = 20
STUDY_SETUPS = ("crash_rebound", "squeeze", "breakout")
BASE_STEP = 10

# yearly cash rate used for the financing of the borrowed half (3-month T-bill, rounded yearly averages)
CASH_RATE = {2019: 0.021, 2020: 0.004, 2021: 0.0005, 2022: 0.020, 2023: 0.051, 2024: 0.050, 2025: 0.042,
             2026: 0.039, 2027: 0.039}
ETF_EXPENSE = 0.0105          # typical 2x single-stock ETF expense ratio (NVDL 1.06%, TSLL 0.95%)
SWAP_SPREAD = 0.005           # swap financing above the cash rate


# ====================================================================== collection
def collect(features: dict[str, pd.DataFrame], setups: dict[str, pd.DataFrame], cfg: dict[str, Any],
            calendar_index: pd.DatetimeIndex, step: int = BASE_STEP,
            setups_wanted: tuple[str, ...] = STUDY_SETUPS) -> dict[str, Any]:
    """Every setup episode (first day of each run) and a random-entry sample, with 20 forward bars."""
    cal_pos = pd.Series(np.arange(len(calendar_index)), index=calendar_index)
    rows: list[dict] = []
    fwd: list[np.ndarray] = []
    for k, (sym, f) in enumerate(features.items()):
        o, h, lo, c, a = (f[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close", "atr"))
        n = len(f)
        st = setups[sym]
        pos = cal_pos.reindex(f.index).to_numpy()

        def add(t: int, setup: str, ref: float, structural: float, conditional: bool) -> None:
            if not (np.isfinite(a[t]) and a[t] > 0 and np.isfinite(c[t]) and np.isfinite(pos[t + 1])):
                return
            seg = np.stack([o[t + 1: t + 1 + H_MAX], h[t + 1: t + 1 + H_MAX], lo[t + 1: t + 1 + H_MAX],
                            c[t + 1: t + 1 + H_MAX]])
            if not np.isfinite(seg).all():
                return
            rows.append({"symbol": sym, "setup": setup, "date": f.index[t], "c0": c[t], "ref": ref, "atr": a[t],
                         "structural": structural, "conditional": conditional, "cal0": int(pos[t + 1])})
            fwd.append(seg)

        for setup in setups_wanted:
            plan = plan_for_setup(f, setup, cfg)
            mask = st[f"setup_{setup}"].to_numpy(dtype=bool)
            starts = np.flatnonzero(mask & ~np.concatenate([[False], mask[:-1]]))
            for t in starts[(starts >= 260) & (starts + H_MAX < n)]:
                p = plan.iloc[t]
                if np.isfinite(p["stop"]):
                    add(int(t), setup, float(p["entry"]), float(p["stop"]), setup == "squeeze")
        for t in range(260 + k % step, n - H_MAX, step):
            add(t, "baseline", c[t], c[t] - 1.5 * a[t], False)
    sig = pd.DataFrame(rows)
    sig["date"] = pd.to_datetime(sig["date"])
    arr = np.array(fwd) if fwd else np.empty((0, 4, H_MAX))
    return {"signals": sig, "O": arr[:, 0], "H": arr[:, 1], "L": arr[:, 2], "C": arr[:, 3]}


def subset(data: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {"signals": data["signals"][mask].reset_index(drop=True), **{k: data[k][mask] for k in "OHLC"}}


# ====================================================================== synthetic 2x ETF
def daily_cost(dates: pd.Series, lev: float = 2.0, expense: float = ETF_EXPENSE, spread: float = SWAP_SPREAD,
               scale: float = 1.0, annual: float | None = None) -> np.ndarray:
    """Cost per held session: expense ratio + financing of the (lev - 1) borrowed part, or a flat
    ``annual`` cost (e.g. the drag measured on real 2x ETFs)."""
    if annual is not None:
        return np.full(len(dates), annual / 252.0)
    years = pd.to_datetime(dates).dt.year.to_numpy()
    rate = np.array([CASH_RATE.get(int(y), 0.04) for y in years])
    return scale * (expense + (lev - 1) * (rate + spread)) / 252.0


def to_etf(data: dict[str, Any], lev: float = 2.0, cost: np.ndarray | None = None) -> dict[str, Any]:
    """The same signals on a synthetic daily-reset ``lev``x ETF whose price equals the stock's close on the
    signal day. ATR, the plan's entry / structural stop are mapped with the day-0 move, so a stop of
    'k ATR' on the ETF is the same UNDERLYING distance as 'k ATR' on the stock (2k ATR in ETF %)."""
    sig = data["signals"].copy()
    O, H, L, C = (data[k] for k in "OHLC")
    c0 = sig["c0"].to_numpy()[:, None]
    cost = np.zeros(len(sig)) if cost is None else cost
    prev = np.concatenate([c0, C[:, :-1]], axis=1)
    growth = np.cumprod(1 + lev * (C / prev - 1) - cost[:, None], axis=1)
    growth = np.maximum(growth, 1e-6)
    e_prev = c0 * np.concatenate([np.ones((len(sig), 1)), growth[:, :-1]], axis=1)

    def m(x):
        return np.maximum(e_prev * (1 + lev * (x / prev - 1)), 1e-6 * c0)
    out = {"signals": sig, "O": m(O), "H": m(H), "L": m(L), "C": c0 * growth}
    c0f = c0[:, 0]
    sig["ref"] = c0f * (1 + lev * (sig["ref"].to_numpy() / c0f - 1))
    sig["structural"] = c0f * (1 + lev * (sig["structural"].to_numpy() / c0f - 1))
    sig["atr"] = lev * sig["atr"].to_numpy()
    sig["lev"] = lev
    return out


def validate_synthetic(pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]], lev: float = 2.0) -> list[dict]:
    """Real 2x ETF vs synthetic (2x the underlying's daily close-to-close return, no cost): daily tracking,
    the empirical yearly drag, and the error over 20-session holding periods."""
    out = []
    for etf, (u, e) in pairs.items():
        j = pd.concat({"u": u["close"], "e": e["close"]}, axis=1).dropna()
        if len(j) < 120:
            continue
        ru, re = j["u"].pct_change().dropna(), j["e"].pct_change().dropna()
        syn = lev * ru
        diff = re - syn
        g_e = (1 + re).rolling(20).apply(np.prod, raw=True) - 1
        g_s = (1 + syn).rolling(20).apply(np.prod, raw=True) - 1
        err20 = (g_e - g_s).dropna()
        u20 = (1 + ru).rolling(20).apply(np.prod, raw=True) - 1
        decay = (g_e - lev * u20).dropna()          # what "2x the stock's 20-day return" misses
        out.append({"etf": etf, "first": j.index[0].strftime("%Y-%m-%d"), "sessions": int(len(diff)),
                    "daily_corr": float(np.corrcoef(re, syn)[0, 1]),
                    "beta": float(np.cov(re, ru)[0, 1] / np.var(ru, ddof=1)),
                    "drag_per_year": float(-diff.mean() * 252),
                    "tracking_err_daily": float(diff.std()),
                    "err20_median_abs": float(err20.abs().median()),
                    "decay20_mean": float(decay.mean()), "decay20_median": float(decay.median()),
                    "decay20_p10": float(decay.quantile(0.10))})
    return out


# ====================================================================== simulation engine
def simulate_plan(data: dict[str, Any], stop_atr: float, exit_kind: str, exit_param: float, horizon: int,
                  tranches: list[tuple[str, float, float]] | None = None, slip: float = 0.001,
                  marks: bool = False, sizing: str = "planned") -> dict[str, Any]:
    """Long trades with a (possibly) scaled-in entry.

    stop_atr      initial stop = entry reference - stop_atr x ATR (same for every tranche)
    exit_kind     'trail' (stop = max(initial stop, highest high since entry - exit_param x ATR)),
                  'r' (target = reference + exit_param x planned risk) or 'atr' (target in ATR)
    tranches      [(kind, param, fraction of risk budget)]; kinds: 'now' (next open, or the squeeze's
                  buy-stop), 'up' (buy-stop at first fill + param ATR), 'down' (limit at first fill -
                  param ATR), 'day' (open of session param), 'confirm' (open after the first close above
                  the previous session's high). The first tranche must be 'now'.
    sizing        'planned' (default): the first tranche is sized the evening before from the plan
                  (reference - stop), so a gap at the open changes the real risk - as in live use;
                  'actual': sized from the actual fill (research.simulate's convention; a fill just
                  above the stop gives a tiny risk and an extreme R).
    Returns a dict with a trade table and, with ``marks``, per-session P&L / capital arrays (in R units).
    """
    tranches = tranches or [("now", 0.0, 1.0)]
    sig = data["signals"]
    O, H, L, C = (data[k][:, :horizon] for k in "OHLC")
    n = len(sig)
    a = sig["atr"].to_numpy()
    ref = sig["ref"].to_numpy()
    cond = sig["conditional"].to_numpy(dtype=bool)
    e1_raw = np.where(cond, np.maximum(O[:, 0], ref), O[:, 0])
    filled0 = ~cond | (H[:, 0] >= ref)
    e1 = e1_raw * (1 + slip)
    stop0 = ref - stop_atr * a
    risk1 = e1 - stop0
    ok = filled0 & (risk1 > 0) & np.isfinite(risk1)          # an open at / below the stop = no trade
    if exit_kind == "r":                      # targets from the plan reference, as in research.simulate
        target = ref + exit_param * (ref - stop0)
    elif exit_kind == "atr":
        target = ref + exit_param * a
    else:
        target = np.full(n, np.inf)
    nt = len(tranches)
    shares = np.zeros((n, nt))
    fills = np.full((n, nt), np.nan)
    floor = 0.25 * a
    f0 = tranches[0][2]
    plan_risk = ref - stop0
    shares[:, 0] = np.where(ok, f0 / np.maximum(risk1 if sizing == "actual" else plan_risk, 1e-12), 0.0)
    fills[:, 0] = e1
    open_ = ok.copy()
    exit_px = np.full(n, np.nan)
    exit_day = np.full(n, horizon - 1)
    outcome = np.array(["time"] * n, dtype=object)
    gap_stop = np.zeros(n, dtype=bool)
    run_hi = e1.copy()
    confirmed = np.zeros(n, dtype=bool)
    pnl_m = np.zeros((n, horizon)) if marks else None
    cap_m = np.zeros((n, horizon)) if marks else None
    for j in range(horizon):
        cur_stop = stop0 if exit_kind != "trail" else np.maximum(stop0, run_hi - exit_param * a)
        # opening fills (day / confirm tranches) before anything else happens on the bar
        for i, (kind, p, frac) in enumerate(tranches[1:], start=1):
            if kind == "day" and j == int(p) and j > 0:
                go = open_ & (shares[:, i] == 0) & (O[:, j] > cur_stop)
            elif kind == "confirm" and j > 0:
                go = open_ & (shares[:, i] == 0) & confirmed & (O[:, j] > cur_stop)
            else:
                continue
            px = O[:, j] * (1 + slip)
            shares[:, i] = np.where(go, frac / np.maximum(px - cur_stop, floor), shares[:, i])
            fills[:, i] = np.where(go, px, fills[:, i])
        # intrabar adds (worst case: before the stop on the same bar; a dip-buy never fills on a bar that
        # also reached the target, nor on the squeeze's entry bar where the low may precede the fill)
        for i, (kind, p, frac) in enumerate(tranches[1:], start=1):
            if kind not in ("up", "down"):
                continue
            blocked = cond.copy() if (kind == "down" and j == 0) else np.zeros(n, dtype=bool)
            if kind == "down":
                blocked = blocked | (H[:, j] >= target)
            lvl = e1 + p * a if kind == "up" else e1 - p * a
            if kind == "up":
                hit = H[:, j] >= lvl
                px = np.maximum(O[:, j], lvl) if j > 0 else lvl
            else:
                hit = (L[:, j] <= lvl) & (lvl > cur_stop)
                px = np.minimum(O[:, j], lvl) if j > 0 else lvl
            go = open_ & (shares[:, i] == 0) & hit & ~blocked
            px = px * (1 + slip)
            shares[:, i] = np.where(go, frac / np.maximum(px - cur_stop, floor), shares[:, i])
            fills[:, i] = np.where(go, px, fills[:, i])
        hit_s = open_ & (L[:, j] <= cur_stop)
        hit_t = open_ & ~hit_s & (H[:, j] >= target)
        px_s = np.minimum(O[:, j], cur_stop) if j > 0 else cur_stop
        px_t = np.maximum(O[:, j], target) if j > 0 else target
        if j > 0:
            gap_stop |= hit_s & (O[:, j] < cur_stop)
        exit_px = np.where(hit_s, px_s, np.where(hit_t, px_t, exit_px))
        outcome = np.where(hit_s, "stop", np.where(hit_t, "target", outcome))
        exit_day = np.where(hit_s | hit_t, j, exit_day)
        closing = hit_s | hit_t
        if j == horizon - 1:
            last = open_ & ~closing
            exit_px = np.where(last, C[:, j], exit_px)
            closing = closing | last
        if marks:
            held = np.nansum(shares * np.where(np.isfinite(fills), C[:, [j]] - fills, 0.0), axis=1)
            done = np.nansum(shares * np.where(np.isfinite(fills), (exit_px * (1 - slip))[:, None] - fills, 0.0), axis=1)
            pnl_m[:, j] = np.where(open_ & ~closing, held, np.where(closing, done, pnl_m[:, j - 1] if j else 0.0))
            cap_m[:, j] = np.where(open_ & ~closing, shares.sum(1) * C[:, j], 0.0)
        open_ = open_ & ~closing
        run_hi = np.maximum(run_hi, H[:, j])
        if j > 0:
            confirmed |= C[:, j] > H[:, j - 1]
        else:
            confirmed |= C[:, 0] > e1_raw
    exit_px = exit_px * (1 - slip)
    fills_ok = np.where(np.isfinite(fills), fills, 0.0)
    pnl = (shares * (exit_px[:, None] - fills_ok)).sum(1)
    used = (shares * np.maximum(fills_ok - stop0[:, None], 0)).sum(1)     # risk to the INITIAL stop actually put on
    notional = (shares * fills_ok).sum(1)
    t = pd.DataFrame({"symbol": sig["symbol"], "setup": sig["setup"], "date": sig["date"], "cal0": sig["cal0"],
                      "r": pnl, "ret": pnl / np.where(notional > 0, notional, np.nan), "days": exit_day + 1,
                      "outcome": outcome, "gap_stop": gap_stop, "n_fills": (shares > 0).sum(1),
                      "notional_r": notional})
    res = {"trades": t[ok].reset_index(drop=True), "ok": ok}
    if marks:
        pnl_m[:, -1] = np.where(ok, pnl, 0.0)
        res["pnl"], res["cap"] = pnl_m[ok], cap_m[ok]
    return res


# ====================================================================== statistics
def stats(t: pd.DataFrame) -> dict[str, Any]:
    if t.empty:
        return {"n": 0}
    r = t["r"]
    daily = r.groupby(t["date"]).mean()
    sd = daily.std(ddof=1)
    out = {"n": int(len(r)), "avg_r": float(r.mean()), "median_r": float(r.median()), "win_rate": float((r > 0).mean()),
           "std_r": float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
           "n_dates": int(len(daily)), "se_daily": float(sd / math.sqrt(len(daily))) if len(daily) > 2 else float("nan"),
           "stop_rate": float((t["outcome"] == "stop").mean()), "avg_days": float(t["days"].mean()),
           "gap_stop_rate": float(t["gap_stop"].mean()),
           "worse_than_1_5r": float((r < -1.5).mean()), "worst_r": float(r.min()),
           "avg_stop_r": float(r[t["outcome"] == "stop"].mean()) if (t["outcome"] == "stop").any() else float("nan")}
    return out


def edge(s: dict, b: dict) -> tuple[float, float]:
    """Edge over random entries and its t (one observation per signal date; random mean taken as known)."""
    if not s.get("n") or not b.get("n"):
        return float("nan"), float("nan")
    e = s["avg_r"] - b["avg_r"]
    se = s.get("se_daily", float("nan"))
    return e, (e / se if se and np.isfinite(se) and se > 0 else float("nan"))


def judge(tr: pd.DataFrame, br: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, Any]:
    ins, oos = tr[tr["date"] <= cutoff], tr[tr["date"] > cutoff]
    b_is, b_oos = br[br["date"] <= cutoff], br[br["date"] > cutoff]
    s_is, s_oos, bi, bo = stats(ins), stats(oos), stats(b_is), stats(b_oos)
    e_is, t_is = edge(s_is, bi)
    e_oos, t_oos = edge(s_oos, bo)
    passes = bool(np.isfinite(e_oos) and e_is >= 0.05 and e_oos >= 0.05 and t_oos >= 2.5)
    return {"is": s_is, "oos": s_oos, "base_is": bi, "base_oos": bo, "edge_is": e_is, "edge_oos": e_oos,
            "t_is": t_is, "t_oos": t_oos, "passes": passes,
            "per_year": {str(y): float(g["r"].mean()) for y, g in tr.groupby(tr["date"].dt.year)}}


def paired_diff(a: pd.DataFrame, b: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, Any]:
    """Same trades, two variants (e.g. stock vs ETF, single vs scaled entry): mean R difference and its t,
    clustered by signal date, in and out of sample."""
    j = a.merge(b, on=["symbol", "setup", "date"], suffixes=("_a", "_b"))
    out = {}
    for name, part in (("is", j[j["date"] <= cutoff]), ("oos", j[j["date"] > cutoff])):
        if part.empty:
            out[name] = {"n": 0}
            continue
        d = (part["r_b"] - part["r_a"]).groupby(part["date"]).mean()
        sd = d.std(ddof=1)
        out[name] = {"n": int(len(part)), "diff_r": float((part["r_b"] - part["r_a"]).mean()),
                     "t": float(d.mean() / (sd / math.sqrt(len(d)))) if len(d) > 2 and sd > 0 else float("nan")}
    return out


# ====================================================================== portfolio
def portfolio(trades: pd.DataFrame, pnl: np.ndarray, cap: np.ndarray, calendar_index: pd.DatetimeIndex,
              risk: float = 0.01, max_positions: int = 10, capital_limit: float = 1.0,
              start: pd.Timestamp | None = None, end: pd.Timestamp | None = None, seed: int = 7) -> dict[str, Any]:
    """Daily mark-to-market account. Each accepted trade risks ``risk`` x equity at entry; a trade is skipped
    when the account already has ``max_positions`` open or the cash it needs (``cap`` x risk$) would push
    total invested capital above ``capital_limit`` x equity (1.0 = no margin)."""
    order = np.random.default_rng(seed).permutation(len(trades))
    t = trades.iloc[order].reset_index(drop=True)
    pnl, cap = pnl[order], cap[order]
    lo = calendar_index.searchsorted(start) if start is not None else 0
    hi = calendar_index.searchsorted(end, side="right") if end is not None else len(calendar_index)
    keep = (t["cal0"] >= lo) & (t["cal0"] < hi)
    t, pnl, cap = t[keep.to_numpy()].reset_index(drop=True), pnl[keep.to_numpy()], cap[keep.to_numpy()]
    by_day: dict[int, list[int]] = {}
    for i, d in enumerate(t["cal0"].to_numpy()):
        by_day.setdefault(int(d), []).append(i)
    last_day = (t["days"].to_numpy() - 1).astype(int)
    need = t["notional_r"].to_numpy()
    days = np.arange(lo, hi)
    realised = 1.0
    open_pos: list[tuple[int, int, float]] = []      # (trade index, entry day, risk $)
    equity = np.zeros(len(days))
    invested = np.zeros(len(days))
    taken = skipped = 0
    for k, d in enumerate(days):
        mtm = inv = 0.0
        for (i, d0, r_usd) in open_pos:
            mtm += r_usd * pnl[i, d - d0]
            inv += r_usd * cap[i, d - d0]
        cur_eq = realised + mtm
        for i in by_day.get(int(d), []):
            r_usd = risk * cur_eq
            if len(open_pos) >= max_positions or inv + r_usd * need[i] > capital_limit * cur_eq:
                skipped += 1
                continue
            open_pos.append((i, int(d), r_usd))
            mtm += r_usd * pnl[i, 0]
            inv += r_usd * cap[i, 0]
            taken += 1
        equity[k] = realised + mtm
        invested[k] = inv / max(equity[k], 1e-9)
        still = []
        for (i, d0, r_usd) in open_pos:
            if d - d0 >= last_day[i]:
                realised += r_usd * pnl[i, last_day[i]]
            else:
                still.append((i, d0, r_usd))
        open_pos = still
    eq = pd.Series(equity, index=calendar_index[days])
    return {"equity": eq, **perf(eq), "taken": taken, "skipped": skipped,
            "avg_invested": float(invested.mean()), "max_invested": float(invested.max())}


def perf(eq: pd.Series) -> dict[str, float]:
    if len(eq) < 20:
        return {}
    rets = eq.pct_change().dropna()
    years = len(eq) / 252
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    dd = float((eq / eq.cummax() - 1).min())
    vol = float(rets.std() * math.sqrt(252))
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else float("nan")
    return {"cagr": cagr, "max_dd": dd, "vol": vol, "sharpe": sharpe,
            "calmar": cagr / -dd if dd < 0 else float("nan"), "total": float(eq.iloc[-1] / eq.iloc[0] - 1)}


# ====================================================================== the whole study
# exits chosen by the strategy lab (config/strategy_lab.default.json) for the three setups kept live
LAB_EXITS = {"crash_rebound": (1.5, "trail", 3.0, 20), "squeeze": (2.0, "r", 3.0, 20), "breakout": (1.0, "trail", 3.0, 20)}
GRID_STOPS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
GRID_EXITS = [("trail", 1.5), ("trail", 2.0), ("trail", 3.0), ("r", 2.0), ("r", 3.0)]
GRID_HORIZONS = [10, 20]
# stocks of the universe with a real 2x long single-stock ETF (checked against Yahoo, 2026-11)
ETF_UNDERLYINGS = {"NVDA": "NVDL", "TSLA": "TSLL", "COIN": "CONL", "AMD": "AMDL", "MSTR": "MSTU", "PLTR": "PLTU",
                   "META": "METU", "SMCI": "SMCL", "HOOD": "HOOG", "IONQ": "IONL", "RGTI": "RGTX", "MARA": "MRAL",
                   "AVGO": "AVL", "MU": "MUU", "ARM": "ARMG", "CRWD": "CRWL", "ORCL": "ORCX", "SHOP": "SHPU",
                   "OKLO": "OKLL", "SOFI": "SOFX", "APP": "APPX", "QBTS": "QBTX", "RKLB": "RKLX", "CRCL": "CRCA",
                   "ASTS": "ASTX", "CRWV": "CWVX", "NBIS": "NBIL", "TSM": "TSMX", "UPST": "UPSX", "AFRM": "AFRU",
                   "SNOW": "SNOU"}
SETUP_LABEL = {"crash_rebound": "Crash rebound", "squeeze": "Squeeze", "breakout": "Breakout"}


def scale_schemes() -> dict[str, list[tuple[str, float, float]]]:
    """Pre-registered scaling-in plans (36): pyramids on strength, buying dips, by day, after confirmation."""
    sc: dict[str, list[tuple[str, float, float]]] = {"single": [("now", 0, 1.0)]}
    pct = lambda fr: "/".join(str(round(x * 100)) for x in fr)  # noqa: E731
    for lvl in (0.5, 1.0, 1.5):
        for f1 in (0.33, 0.5, 0.67):
            sc[f"add +{lvl} ATR {pct((f1, 1 - f1))}"] = [("now", 0, f1), ("up", lvl, 1 - f1)]
    for l1, l2 in ((0.5, 1.0), (1.0, 2.0)):
        for fr in ((0.34, 0.33, 0.33), (0.5, 0.25, 0.25), (0.2, 0.3, 0.5)):
            sc[f"add +{l1} / +{l2} ATR {pct(fr)}"] = [("now", 0, fr[0]), ("up", l1, fr[1]), ("up", l2, fr[2])]
    for lvl in (0.25, 0.5, 1.0):
        for f1 in (0.33, 0.5, 0.67):
            sc[f"dip -{lvl} ATR {pct((f1, 1 - f1))}"] = [("now", 0, f1), ("down", lvl, 1 - f1)]
    for fr in ((0.34, 0.33, 0.33), (0.5, 0.25, 0.25), (0.2, 0.3, 0.5)):
        sc[f"dip -0.5 / -1.0 ATR {pct(fr)}"] = [("now", 0, fr[0]), ("down", 0.5, fr[1]), ("down", 1.0, fr[2])]
    for f1 in (0.33, 0.5, 0.67):
        sc[f"next day {pct((f1, 1 - f1))}"] = [("now", 0, f1), ("day", 1, 1 - f1)]
        sc[f"after confirmation {pct((f1, 1 - f1))}"] = [("now", 0, f1), ("confirm", 0, 1 - f1)]
    sc["3 days 34/33/33"] = [("now", 0, 0.34), ("day", 1, 0.33), ("day", 2, 0.33)]
    sc["5 days 5 x 20"] = [("now", 0, 0.2), ("day", 1, 0.2), ("day", 2, 0.2), ("day", 3, 0.2), ("day", 4, 0.2)]
    return sc


def date_matched(t: pd.DataFrame, b: pd.DataFrame) -> dict[str, Any]:
    """Setup minus random entries ON THE SAME SIGNAL DATES (controls for market timing), t by date."""
    j = pd.concat({"s": t.groupby("date")["r"].mean(), "b": b.groupby("date")["r"].mean()}, axis=1, sort=True).dropna()
    x = j["s"] - j["b"]
    sd = x.std(ddof=1)
    return {"n_dates": int(len(x)), "edge": float(x.mean()) if len(x) else float("nan"),
            "t": float(x.mean() / (sd / math.sqrt(len(x)))) if len(x) > 2 and sd > 0 else float("nan")}


def _split(t: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    return t[t["date"] <= cutoff], t[t["date"] > cutoff]


PORTFOLIO_VARIANTS = {
    "A": ("Stock, 1% risk (today)", "stock", 0.01, False, False, 1.0),
    "B": ("2x ETF, 1% risk, same stop (half the cash)", "etf", 0.01, False, False, 1.0),
    "C": ("2x ETF, same cash, same % stop on the ETF (= half the stop distance)", "etf", 0.01, True, False, 1.0),
    "D": ("2x ETF, same cash, same underlying stop (= 2% risk)", "etf", 0.02, False, False, 1.0),
    "F": ("Stock, scaled in (best in-sample plan), 1% risk", "stock", 0.01, False, True, 1.0),
    "G": ("2x ETF + scaled in, same cash, 2% risk", "etf", 0.02, False, True, 1.0),
    "H": ("Stock, half the stop, 1% risk, 2x margin (= C without the ETF)", "stock", 0.01, True, False, 2.0),
}


def run_study(data: dict[str, Any], calendar_index: pd.DatetimeIndex, spy_close: pd.Series,
              validation: list[dict] | None = None, cost: float | None = None, seeds: int = 12,
              max_positions: int = 10, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    sig = data["signals"]
    setups = [s for s in STUDY_SETUPS if (sig["setup"] == s).sum() >= 30]
    cutoff = sig.loc[sig["setup"] != "baseline", "date"].quantile(0.6)
    val = [v for v in (validation or []) if 1.95 <= v.get("beta", 0) <= 2.05]
    if cost is None:
        cost = float(np.median([v["drag_per_year"] for v in val])) if val else 0.15
    groups = {s: subset(data, (sig["setup"] == s).to_numpy()) for s in [*setups, "baseline"]}
    etf = {s: to_etf(g, 2.0, daily_cost(g["signals"]["date"], annual=cost)) for s, g in groups.items()}
    spaces = {"stock": (groups, 0.001), "etf": (etf, 0.0015)}
    cache: dict[tuple, dict] = {}

    def run(space: str, grp: str, cfg: tuple, tr=None, marks: bool = False) -> dict:
        key = (space, grp, cfg, tuple(tr) if tr else None, marks)
        if key not in cache:
            g, slip = spaces[space]
            cache[key] = simulate_plan(g[grp], cfg[0], cfg[1], cfg[2], cfg[3], tr, slip=slip, marks=marks)
        return cache[key]

    out: dict[str, Any] = {"available": True, "cutoff": pd.Timestamp(cutoff).strftime("%Y-%m-%d"),
                           "first": sig["date"].min().strftime("%Y-%m-%d"), "last": sig["date"].max().strftime("%Y-%m-%d"),
                           "n_stocks": int(sig["symbol"].nunique()), "cost_per_year": cost,
                           "n_signals": {s: int((sig["setup"] == s).sum()) for s in setups},
                           "n_random": int((sig["setup"] == "baseline").sum()),
                           "tradable": {s: int(((sig["setup"] == s) & sig["symbol"].isin(ETF_UNDERLYINGS)).sum()) for s in setups},
                           "validation": validation or [],
                           "validation_summary": {"n_etfs": len(val),
                                                  "drag_median": float(np.median([v["drag_per_year"] for v in val])) if val else None,
                                                  "decay20_median": float(np.median([v["decay20_median"] for v in val])) if val else None,
                                                  "corr_median": float(np.median([v["daily_corr"] for v in val])) if val else None,
                                                  "err20_median": float(np.median([v["err20_median_abs"] for v in val])) if val else None}}
    # ---------------------------------------------------------- 1. trades: stock vs 2x ETF, stop sizes
    trades: dict[str, Any] = {}
    for s in setups:
        st, ek, ep, hz = LAB_EXITS[s]
        rows = []
        for stop in GRID_STOPS:
            cfg = (stop, ek, ep, hz)
            row: dict[str, Any] = {"stop": stop}
            for space in spaces:
                j = judge(run(space, s, cfg)["trades"], run(space, "baseline", cfg)["trades"], cutoff)
                m_is = date_matched(*(_split(run(space, s, cfg)["trades"], cutoff)[0], _split(run(space, "baseline", cfg)["trades"], cutoff)[0]))
                m_oos = date_matched(_split(run(space, s, cfg)["trades"], cutoff)[1], _split(run(space, "baseline", cfg)["trades"], cutoff)[1])
                row[space] = {"is_r": j["is"].get("avg_r"), "oos_r": j["oos"].get("avg_r"), "edge_is": j["edge_is"],
                              "edge_oos": j["edge_oos"], "t_oos": j["t_oos"], "passes": j["passes"],
                              "matched_is": m_is, "matched_oos": m_oos, "n_oos": j["oos"].get("n"),
                              "gap_stop_rate": j["oos"].get("gap_stop_rate"), "avg_stop_r": j["oos"].get("avg_stop_r"),
                              "worst_r": j["oos"].get("worst_r"), "worse_than_1_5r": j["oos"].get("worse_than_1_5r")}
            rows.append(row)
        best = {}
        for space in spaces:
            cands = []
            for stop in GRID_STOPS:
                for gek, gep in GRID_EXITS:
                    for ghz in GRID_HORIZONS:
                        cfg = (stop, gek, gep, ghz)
                        tr = run(space, s, cfg)["trades"]
                        ins = tr[tr["date"] <= cutoff]
                        if len(ins) >= 100:
                            cands.append((float(ins["r"].mean()), cfg))
            if cands:
                cfg = max(cands)[1]
                j = judge(run(space, s, cfg)["trades"], run(space, "baseline", cfg)["trades"], cutoff)
                best[space] = {"stop": cfg[0], "exit": f"{cfg[1]}{cfg[2]:g}", "horizon": cfg[3], "is_r": j["is"].get("avg_r"),
                               "oos_r": j["oos"].get("avg_r"), "edge_is": j["edge_is"], "edge_oos": j["edge_oos"],
                               "t_oos": j["t_oos"], "passes": j["passes"]}
        lab = LAB_EXITS[s]
        trades[s] = {"exit": {"stop": st, "exit": f"{ek}{ep:g}", "horizon": hz}, "stops": rows, "best_in_sample": best,
                     "etf_minus_stock": paired_diff(run("stock", s, lab)["trades"], run("etf", s, lab)["trades"], cutoff)}
        say(f"Trades: {s} done")
    out["trades"] = trades
    out["random_etf_minus_stock"] = paired_diff(run("stock", "baseline", LAB_EXITS[setups[0]])["trades"],
                                                run("etf", "baseline", LAB_EXITS[setups[0]])["trades"], cutoff)
    # ---------------------------------------------------------- 2. scaling in
    schemes = scale_schemes()
    scale: dict[str, Any] = {}
    best_plan: dict[str, list] = {}
    for s in setups:
        cfg = LAB_EXITS[s]
        single = run("stock", s, cfg)["trades"]
        rows = []
        for name, tr in schemes.items():
            if any(k == "down" and p >= cfg[0] for k, p, _ in tr):
                continue
            t = run("stock", s, cfg, tr)["trades"]
            j = judge(t, run("stock", "baseline", cfg, tr)["trades"], cutoff)
            pdf = paired_diff(single, t, cutoff)
            rows.append({"scheme": name, "is_r": j["is"].get("avg_r"), "oos_r": j["oos"].get("avg_r"),
                         "edge_is": j["edge_is"], "edge_oos": j["edge_oos"], "t_oos": j["t_oos"],
                         "vs_single_is": pdf["is"].get("diff_r"), "vs_single_oos": pdf["oos"].get("diff_r"),
                         "t_vs_single_oos": pdf["oos"].get("t"), "fills": float(t["n_fills"].mean())})
        rows.sort(key=lambda x: -(x["is_r"] if np.isfinite(x["is_r"]) else -9))
        top = next((r for r in rows if r["scheme"] != "single"), None)
        adopted = bool(top and top["vs_single_is"] >= 0.05 and top["vs_single_oos"] >= 0.05 and (top["t_vs_single_oos"] or 0) >= 2.5)
        best_plan[s] = schemes[top["scheme"]] if top else schemes["single"]
        scale[s] = {"rows": rows, "best": top, "adopted": adopted}
        say(f"Scaling in: {s} done")
    out["scale"] = scale
    # ---------------------------------------------------------- 3. portfolios
    first = sig["date"].min()
    periods = {"full": (first, None), "in_sample": (first, cutoff), "out_of_sample": (cutoff, None)}

    def assemble(code: str, names: list[str], seed: int | None):
        _, space, _, half, sc, _ = PORTFOLIO_VARIANTS[code]
        T, P, Cp = [], [], []
        for s in names:
            st, ek, ep, hz = LAB_EXITS[s]
            cfg = (st / 2 if half else st, ek, ep, hz)
            tr = best_plan[s] if sc else None
            a = run(space, s, cfg, tr, marks=True)
            if seed is None:
                T.append(a["trades"]); P.append(a["pnl"]); Cp.append(a["cap"])
                continue
            b = run(space, "baseline", cfg, tr, marks=True)
            rng = np.random.default_rng(seed)
            by_date = b["trades"].groupby("date").indices
            pick = np.array([rng.choice(by_date[d]) for d in a["trades"]["date"] if d in by_date], dtype=int)
            T.append(b["trades"].iloc[pick]); P.append(b["pnl"][pick]); Cp.append(b["cap"][pick])
        return pd.concat(T, ignore_index=True), np.concatenate(P), np.concatenate(Cp)

    port: dict[str, Any] = {}
    for set_name, names in (("crash_rebound", ["crash_rebound"] if "crash_rebound" in setups else setups[:1]), ("all", setups)):
        rows = []
        for code, (label, _sp, risk, _h, _sc, capl) in PORTFOLIO_VARIANTS.items():
            t, pn, cp = assemble(code, names, None)
            rnd_sets = [assemble(code, names, k) for k in range(seeds)]
            for pname, (lo, hi) in periods.items():
                own = [portfolio(t, pn, cp, calendar_index, risk=risk, max_positions=max_positions, capital_limit=capl,
                                 start=lo, end=hi, seed=100 + k) for k in range(seeds)]
                rnd = [portfolio(*rs, calendar_index, risk=risk, max_positions=max_positions, capital_limit=capl,
                                 start=lo, end=hi, seed=200 + k) for k, rs in enumerate(rnd_sets)]
                med = lambda lst, k: float(np.median([q[k] for q in lst]))  # noqa: E731
                rows.append({"code": code, "label": label, "period": pname, "cagr": med(own, "cagr"),
                             "cagr_p10": float(np.quantile([q["cagr"] for q in own], 0.1)),
                             "cagr_p90": float(np.quantile([q["cagr"] for q in own], 0.9)),
                             "max_dd": med(own, "max_dd"), "worst_dd": float(min(q["max_dd"] for q in own)),
                             "sharpe": med(own, "sharpe"), "vol": med(own, "vol"), "trades": med(own, "taken"),
                             "invested": med(own, "avg_invested"), "random_cagr": med(rnd, "cagr"),
                             "random_max_dd": med(rnd, "max_dd"), "random_sharpe": med(rnd, "sharpe")})
            say(f"Portfolio {set_name} {code} done")
        port[set_name] = rows
    out["portfolio"] = port
    out["portfolio_note"] = (f"Daily marked-to-market cash account, at most {max_positions} positions, each trade risks the stated % "
                             f"of equity at entry; a signal is skipped when the cash it needs is not there. Medians over {seeds} "
                             "random orders of same-day signals; random = a random stock on the SAME signal day with the same plan.")
    spy = spy_close[spy_close.index >= first]
    r = spy.pct_change().fillna(0)
    c = (0.0089 + np.array([CASH_RATE.get(int(y), 0.04) + 0.003 for y in spy.index.year])) / 252
    sso = pd.Series(np.cumprod(1 + 2 * r.to_numpy() - np.r_[0, c[1:]]), index=spy.index)
    out["benchmarks"] = [{"name": nm, "period": pname, **perf(ser[(ser.index >= lo) & ((ser.index <= hi) if hi is not None else True)])}
                         for pname, (lo, hi) in periods.items()
                         for nm, ser in (("SPY buy & hold", spy / spy.iloc[0]), ("2x SPY (SSO-like) buy & hold", sso))]
    out["verdict"] = verdict(out)
    return out


def verdict(res: dict[str, Any]) -> list[str]:
    v = res.get("validation_summary") or {}
    out = []
    if v.get("n_etfs"):
        out.append(f"Synthetic 2x ETF checked on {v['n_etfs']} real 2x single-stock ETFs: daily correlation {v['corr_median']:.3f}; "
                   f"real cost {v['drag_median']:.0%} a year (median, far above the ~1% expense ratio: swap financing); "
                   f"over 20 sessions the ETF returns {abs(v['decay20_median']):.1%} less than 2x the stock (median, decay + cost).")
    for s, t in (res.get("trades") or {}).items():
        lab = next((r for r in t["stops"] if r["stop"] == t["exit"]["stop"]), None)
        half = next((r for r in t["stops"] if r["stop"] == t["exit"]["stop"] / 2), None)
        d = t["etf_minus_stock"]
        if lab:
            out.append(f"{SETUP_LABEL.get(s, s)}: same trades on the 2x ETF {d['is'].get('diff_r', float('nan')):+.2f}R in sample / "
                       f"{d['oos'].get('diff_r', float('nan')):+.2f}R out of sample vs the stock (decay + cost). Edge over random, stock "
                       f"{lab['stock']['edge_is']:+.2f}R / {lab['stock']['edge_oos']:+.2f}R (t {lab['stock']['t_oos']:.1f}); "
                       + (f"half the stop {half['stock']['edge_is']:+.2f}R / {half['stock']['edge_oos']:+.2f}R (t {half['stock']['t_oos']:.1f}). " if half else "")
                       + ("PASS." if lab["stock"]["passes"] or lab["etf"]["passes"] else "Not significant (t < 2.5)."))
    for s, sc in (res.get("scale") or {}).items():
        b = sc.get("best") or {}
        if b:
            out.append(f"{SETUP_LABEL.get(s, s)} scaled in (cross-check of the scaling-in study): best in-sample plan "
                       f"'{b['scheme']}' vs one entry {b['vs_single_is']:+.2f}R in sample / {b['vs_single_oos']:+.2f}R out of "
                       "sample -> " + ("ADOPT." if sc["adopted"] else "keep one entry."))
    return out

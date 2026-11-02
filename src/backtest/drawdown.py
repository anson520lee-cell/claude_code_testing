"""Drawdown-reduction mechanisms for the multi-position account simulation (research only).

Question (Neo, 2026-11-02): the account simulation of the three live setups (10 slots, 1% risk per trade) has a
maximum drawdown of about -66% (2010-2026). Which risk controls cut it without giving up the return?

The trades are the same as in the other account illustrations (scale_out.account_paths: live exits, 20 session
horizon, 10 bps slippage, daily mark-to-market path per 1R of risk). Only the account layer is new:

    risk $ of a new trade = base risk x equity x regime x volatility x drawdown-breaker multiplier

* regime        market filters from SPY / VIX / universe breadth, known at the previous close (never the same day);
* volatility    volatility targeting from SPY realised vol, VIX or the account's own realised vol (reduce-only);
* breaker       account-level drawdown circuit breaker (tiers, hysteresis) or an equity-curve trend filter;
* structure     fewer slots, total open-risk (heat) cap, per-sector cap, correlation cap, lower base risk.

A trade that is skipped is skipped for good (signals are not queued). Same-day signals are taken in a random order
(seeded; medians over several orders, like the other account studies). The 25% position cap is already inside the
per-trade paths (computed at 1% risk); lowering the risk makes it slightly less binding, so lower-risk accounts are
a touch conservative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

WINDOWS = {
    "all": ("2010-01-01", "2100-01-01"),
    "hold": ("2010-01-01", "2019-12-31"),      # never used to design anything
    "is": ("2020-01-01", "2024-06-30"),        # selection window (the project's in-sample)
    "oos": ("2024-07-01", "2100-01-01"),       # after selection
}


# ====================================================================== daily context series
def build_context(prices: dict[str, pd.DataFrame], calendar_index: pd.DatetimeIndex, symbols: list[str]) -> dict[str, np.ndarray]:
    """Daily series aligned to ``calendar_index`` and shifted by one session: the value at index d is what was
    known at the close of d-1, i.e. before the open on which a signal of day d-1 is entered."""
    spy = prices["SPY"]["adj_close"].reindex(calendar_index).ffill()
    vix = prices["^VIX"]["close"].reindex(calendar_index).ffill() if "^VIX" in prices else pd.Series(np.nan, calendar_index)
    ctx: dict[str, pd.Series] = {}
    ctx["spy_vs200"] = spy / spy.rolling(200).mean() - 1
    ctx["spy_vs50"] = spy / spy.rolling(50).mean() - 1
    ctx["spy_dd"] = spy / spy.rolling(252, min_periods=60).max() - 1
    ctx["spy_vol20"] = spy.pct_change().rolling(20).std() * np.sqrt(252)
    ctx["vix"] = vix
    cl = pd.DataFrame({s: prices[s]["adj_close"] for s in symbols if s in prices}).reindex(calendar_index).ffill(limit=5)
    above50 = (cl > cl.rolling(50).mean()).where(cl.notna() & cl.rolling(50).mean().notna())
    above200 = (cl > cl.rolling(200).mean()).where(cl.notna() & cl.rolling(200).mean().notna())
    ctx["breadth50"] = above50.mean(axis=1)
    ctx["breadth200"] = above200.mean(axis=1)
    return {k: v.shift(1).to_numpy(dtype=float) for k, v in ctx.items()}


def returns_matrix(prices: dict[str, pd.DataFrame], calendar_index: pd.DatetimeIndex, symbols: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    cl = pd.DataFrame({s: prices[s]["adj_close"] for s in symbols if s in prices}).reindex(calendar_index).ffill(limit=5)
    r = cl.pct_change().fillna(0.0)
    return r.to_numpy(dtype=np.float32), {s: i for i, s in enumerate(cl.columns)}


# ====================================================================== the variants
@dataclass
class Variant:
    name: str
    risk: float = 0.01
    slots: int = 10
    heat: float | None = None                 # cap on the sum of open initial risk (fraction of equity)
    heat_adapt: tuple | None = None           # (hi, lo, dd_trigger, use_spy200): heat cap hi normally, lo when SPY<200d or account dd<-dd_trigger
    sector_cap: int | None = None             # max open positions per sector ETF
    corr_cap: float | None = None             # skip a candidate whose mean 60d correlation with open positions exceeds this
    regime: tuple | None = None               # ("kind", params...) -> multiplier from ctx
    regime_setups: tuple[str, ...] | None = None   # None = all setups; else only these are subject to the regime filter
    setup_risk: dict | None = None            # per-setup risk multiplier, e.g. {"breakout": 0.5}
    setup_slots: dict | None = None           # max simultaneous open positions per setup, e.g. {"breakout": 4}
    priority: bool = False                    # same-day signals: crash_rebound, squeeze first, breakout last
    vol: tuple | None = None                  # ("spy", target) | ("vix", target) | ("acct", target)
    breaker: tuple | None = None              # ("tiers", ...) | ("trend", n)
    notes: dict[str, Any] = field(default_factory=dict)


def regime_multiplier(spec: tuple, ctx: dict[str, np.ndarray]) -> np.ndarray:
    """Daily multiplier in [0, 1] from a regime spec."""
    kind = spec[0]
    n = len(next(iter(ctx.values())))
    if kind == "spy_ma":                       # ("spy_ma", "spy_vs200"|"spy_vs50", multiplier_when_below)
        x = ctx[spec[1]]
        return np.where(np.isnan(x), 1.0, np.where(x < 0, spec[2], 1.0))
    if kind == "spy_two_ma":                   # below 200 -> m_low; between (below 50 only) -> m_mid
        x2, x5 = ctx["spy_vs200"], ctx["spy_vs50"]
        out = np.ones(n)
        out[(x5 < 0) & ~np.isnan(x5)] = spec[1]
        out[(x2 < 0) & ~np.isnan(x2)] = spec[2]
        return out
    if kind == "vix":                          # ("vix", level, multiplier_above)
        x = ctx["vix"]
        return np.where(np.isnan(x), 1.0, np.where(x > spec[1], spec[2], 1.0))
    if kind == "vix_tiers":                    # ("vix_tiers", l1, m1, l2, m2)
        x = ctx["vix"]
        out = np.ones(n)
        out[x > spec[1]] = spec[2]
        out[x > spec[3]] = spec[4]
        return np.where(np.isnan(x), 1.0, out)
    if kind == "spy_dd":                       # ("spy_dd", threshold(neg), multiplier_when_deeper)
        x = ctx["spy_dd"]
        return np.where(np.isnan(x), 1.0, np.where(x < spec[1], spec[2], 1.0))
    if kind == "breadth":                      # ("breadth", "breadth50"|"breadth200", level, multiplier_below)
        x = ctx[spec[1]]
        return np.where(np.isnan(x), 1.0, np.where(x < spec[2], spec[3], 1.0))
    raise ValueError(spec)


def vol_multiplier(spec: tuple, ctx: dict[str, np.ndarray]) -> np.ndarray | None:
    if spec[0] == "spy":
        v = ctx["spy_vol20"]
        return np.where(np.isnan(v), 1.0, np.minimum(1.0, spec[1] / np.maximum(v, 1e-6)))
    if spec[0] == "vix":
        v = ctx["vix"] / 100.0
        return np.where(np.isnan(v), 1.0, np.minimum(1.0, spec[1] / np.maximum(v, 1e-6)))
    return None                                # "acct": handled inside the loop


# ====================================================================== the account
def run_account(trades: pd.DataFrame, pnl: np.ndarray, cap: np.ndarray, calendar_index: pd.DatetimeIndex, v: Variant,
                ctx: dict[str, np.ndarray], rets: np.ndarray | None = None, sym_idx: dict[str, int] | None = None,
                start: str | None = None, end: str | None = None, seed: int = 0,
                reg_override: np.ndarray | None = None) -> dict[str, Any]:
    """Daily mark-to-market cash account. Returns the equity series and trade counts."""
    order = np.random.default_rng(seed).permutation(len(trades))
    if v.priority:
        rank = trades["setup"].map({"crash_rebound": 0, "squeeze": 0, "breakout": 1}).to_numpy()[order]
        order = order[np.argsort(rank, kind="stable")]
    t = trades.iloc[order].reset_index(drop=True)
    pnl, cap = pnl[order], cap[order]
    lo = calendar_index.searchsorted(pd.Timestamp(start)) if start else 0
    hi = calendar_index.searchsorted(pd.Timestamp(end), side="right") if end else len(calendar_index)
    keep = ((t["cal0"] >= lo) & (t["cal0"] < hi)).to_numpy()
    t, pnl, cap = t[keep].reset_index(drop=True), pnl[keep], cap[keep]
    by_day: dict[int, list[int]] = {}
    for i, d in enumerate(t["cal0"].to_numpy()):
        by_day.setdefault(int(d), []).append(i)
    last_day = (t["days"].to_numpy() - 1).astype(int)
    need = t["notional_r"].to_numpy()
    setups = t["setup"].to_numpy()
    sectors = t["sector"].to_numpy()
    sidx = np.array([sym_idx.get(s, -1) for s in t["sym"].to_numpy()]) if (sym_idx is not None and v.corr_cap) else None

    reg = reg_override if reg_override is not None else (regime_multiplier(v.regime, ctx) if v.regime else None)
    volm = vol_multiplier(v.vol, ctx) if v.vol and v.vol[0] != "acct" else None
    reg_sub = set(v.regime_setups) if v.regime_setups else None

    days = np.arange(lo, hi)
    realised = 1.0
    peak = 1.0
    open_pos: list[tuple[int, int, float]] = []          # (trade, entry day, risk $)
    equity = np.zeros(len(days))
    eq_hist: list[float] = []
    state_paused = False
    taken = skipped = paused_days = 0
    for k, d in enumerate(days):
        mtm = inv = 0.0
        for (i, d0, r_usd) in open_pos:
            mtm += r_usd * pnl[i, d - d0]
            inv += r_usd * cap[i, d - d0]
        cur_eq = realised + mtm
        peak = max(peak, cur_eq)
        dd = cur_eq / peak - 1.0
        # ---- account-level multipliers (known at the start of the day)
        m_brk = 1.0
        if v.breaker is not None:
            kind = v.breaker[0]
            if kind == "tiers":                  # ("tiers", dd1, m1, dd2, m2, pause_dd, resume_dd)
                _, d1, m1, d2, m2, dp, dr = v.breaker
                if dp is not None:
                    if dd <= dp:
                        state_paused = True
                    elif state_paused and dd >= dr:
                        state_paused = False
                else:
                    state_paused = False
                m_brk = 0.0 if state_paused else (m2 if dd <= d2 else (m1 if dd <= d1 else 1.0))
            elif kind == "trend":                # ("trend", n, m_below): equity below its n-day mean
                if len(eq_hist) >= v.breaker[1]:
                    ma = float(np.mean(eq_hist[-v.breaker[1]:]))
                    m_brk = v.breaker[2] if cur_eq < ma else 1.0
        m_vol = 1.0
        if v.vol and v.vol[0] == "acct":
            if len(eq_hist) >= 21:
                e = np.asarray(eq_hist[-21:])
                rv = float(np.std(e[1:] / e[:-1] - 1) * np.sqrt(252))
                m_vol = min(1.0, v.vol[1] / max(rv, 1e-6))
        elif volm is not None:
            m_vol = float(volm[d])
        for i in by_day.get(int(d), []):
            m_reg = 1.0
            if reg is not None and (reg_sub is None or setups[i] in reg_sub):
                m_reg = float(reg[d])
            mult = m_reg * m_vol * m_brk
            if mult <= 0:
                skipped += 1
                continue
            if v.setup_risk:
                mult *= v.setup_risk.get(setups[i], 1.0)
            r_usd = v.risk * mult * cur_eq
            if r_usd <= 0:
                skipped += 1
                continue
            if v.setup_slots and setups[i] in v.setup_slots and sum(1 for p in open_pos if setups[p[0]] == setups[i]) >= v.setup_slots[setups[i]]:
                skipped += 1
                continue
            if len(open_pos) >= v.slots or inv + r_usd * need[i] > cur_eq:
                skipped += 1
                continue
            heat_now = v.heat
            if v.heat_adapt is not None:
                hi_, lo_, trig, use200 = v.heat_adapt
                bad = dd <= -trig or (use200 and not np.isnan(ctx["spy_vs200"][d]) and ctx["spy_vs200"][d] < 0)
                heat_now = lo_ if bad else hi_
            if heat_now is not None and sum(p[2] for p in open_pos) + r_usd > heat_now * cur_eq:
                skipped += 1
                continue
            if v.sector_cap is not None and sum(1 for p in open_pos if sectors[p[0]] == sectors[i]) >= v.sector_cap:
                skipped += 1
                continue
            if v.corr_cap is not None and open_pos and sidx is not None and sidx[i] >= 0:
                lo_c = max(0, d - 60)
                x = rets[lo_c:d, sidx[i]]
                cs = []
                for (j, _d0, _r) in open_pos:
                    if sidx[j] >= 0:
                        y = rets[lo_c:d, sidx[j]]
                        if x.std() > 0 and y.std() > 0:
                            cs.append(float(np.corrcoef(x, y)[0, 1]))
                if cs and float(np.mean(cs)) > v.corr_cap:
                    skipped += 1
                    continue
            open_pos.append((i, int(d), r_usd))
            mtm += r_usd * pnl[i, 0]
            inv += r_usd * cap[i, 0]
            taken += 1
        if m_brk == 0.0 or (reg is not None and reg_sub is None and reg[d] == 0.0):
            paused_days += 1
        equity[k] = realised + mtm
        eq_hist.append(equity[k])
        still = []
        for (i, d0, r_usd) in open_pos:
            if d - d0 >= last_day[i]:
                realised += r_usd * pnl[i, last_day[i]]
            else:
                still.append((i, d0, r_usd))
        open_pos = still
    return {"equity": pd.Series(equity, index=calendar_index[days]), "taken": taken, "skipped": skipped, "paused_days": paused_days}


def perf(eq: pd.Series) -> dict[str, float]:
    if len(eq) < 40:
        return {}
    rets = eq.pct_change().dropna()
    years = len(eq) / 252
    end = float(eq.iloc[-1])
    cagr = float(end ** (1 / years) - 1) if end > 0 else -1.0
    dd = float((eq / eq.cummax() - 1).min())
    under = eq / eq.cummax() - 1
    ulcer = float(np.sqrt((under ** 2).mean()))
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else float("nan")
    return {"cagr": cagr, "max_dd": dd, "calmar": cagr / -dd if dd < 0 else float("nan"), "sharpe": sharpe, "ulcer": ulcer,
            "total": end - 1.0 if len(eq) else float("nan")}


def evaluate(trades, pnl, cap, calendar_index, v: Variant, ctx, rets, sym_idx, seeds: int = 8, windows: dict | None = None,
             reg_override: np.ndarray | None = None) -> dict[str, dict[str, float]]:
    """Median over ``seeds`` random same-day orders of each window's metrics."""
    out: dict[str, dict[str, float]] = {}
    for w, (lo, hi) in (windows or WINDOWS).items():
        runs = []
        for s in range(seeds):
            r = run_account(trades, pnl, cap, calendar_index, v, ctx, rets, sym_idx, start=lo, end=hi, seed=s, reg_override=reg_override)
            p = perf(r["equity"])
            if p:
                p["taken"] = float(r["taken"])
                p["paused_days"] = float(r["paused_days"])
                runs.append(p)
        out[w] = {k: float(np.nanmedian([x[k] for x in runs])) for k in runs[0]} if runs else {}
        if runs:
            out[w]["dd_sd"] = float(np.std([x["max_dd"] for x in runs]))
            out[w]["cagr_sd"] = float(np.std([x["cagr"] for x in runs]))
    return out


# ====================================================================== trades (live exits, daily paths)
def build_trades(data: dict[str, Any], slip: float, calendar_index: pd.DatetimeIndex, sector_of: dict[str, str],
                 breakout_trail: float | None = None) -> pd.DataFrame:
    """Same as scale_out.account_paths(choice=live) plus the symbol and sector. ``breakout_trail`` overrides the 3 ATR
    trail of the breakout setup (the earlier wide-trail lead)."""
    from src.backtest import scale_out as so
    so.prepare(data)
    sig = data["signals"]
    saved = so.BASE_EXIT["breakout"]["trail"]
    if breakout_trail is not None:
        so.BASE_EXIT["breakout"]["trail"] = breakout_trail
    try:
        rows, pnl_l, cap_l = [], [], []
        for setup in so.SETUPS:
            ms = (sig["setup"] == setup).to_numpy()
            Ds = so.subset(data, ms)
            t, _ = so.run_method(Ds, setup, {"name": "live"}, slip, {}, paths=True)
            ok, days = t["ok"], t["days"]
            pe, pc = t["path_eq"].copy(), t["path_cap"].copy()
            for i in np.flatnonzero(ok):
                d = int(days[i])
                pe[i, d - 1:] = t["eq"][i]
                pc[i, d - 1:] = 0.0
            cal0 = calendar_index.searchsorted(Ds["signals"]["date"].to_numpy()) + 1
            keep = ok & (cal0 + so.H < len(calendar_index))
            syms = Ds["signals"]["symbol"].to_numpy()[keep]
            rows.append(pd.DataFrame({"setup": setup, "sym": syms, "sector": [sector_of.get(s, "?") for s in syms],
                                      "date": Ds["signals"]["date"].to_numpy()[keep], "cal0": cal0[keep],
                                      "days": days[keep], "notional_r": t["size"][keep] / 0.01}))
            pnl_l.append(pe[keep] / 0.01)
            cap_l.append(pc[keep] / 0.01)
    finally:
        so.BASE_EXIT["breakout"]["trail"] = saved
    trades = pd.concat(rows, ignore_index=True)
    trades.attrs["pnl"] = np.concatenate(pnl_l)
    trades.attrs["cap"] = np.concatenate(cap_l)
    return trades

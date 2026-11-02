"""Adaptive ATR stops: a rule-based two-phase stop and a learned stop multiple (research only, docs 7n).

Question (Neo, 2026-11-02): a wide ATR stop is rarely hit, a tight one locks more profit. Can the two be combined -
a wide stop (5-6 ATR) while the trade is under water, a tighter one once it is back above the entry - and can a model
fed with every variable the terminal has (indicators, price / volume, chip distribution, market regime, earnings
calendar, ...) pick the best ATR multiple per trade or per day?

Pre-registered design (written before the first run)
* Trades: every episode start of the three live setups (crash_rebound, squeeze, breakout) plus random entries (every
  20th session of every stock) as a market baseline. Entry at the next open +10 bps (squeeze: buy-stop at the 20-day
  high + 0.1 ATR, only if it trades). 20 sessions max (Neo's decision 2026-11-01). Daily bars 2004-2026 (Yahoo),
  today's universe after the live focus filter (survivorship: today's names; absolute returns are optimistic, paired
  deltas much less so).
* Exit family (A), one rule = (S0, T1, trigger, T2, BE, target):
    S0   initial stop, ATR below the reference (signal close; squeeze: the buy-stop)  1 1.5 2 3 4 5 6
    T1   phase-1 trail (ATR from the highest high of the previous bars)              none 3 5
    trig switch to phase 2 at the close of a day when: never | close > entry | high >= entry + 0.5/1/2/3 ATR |
         close >= entry + 1/2 ATR | the trade is 3/5/10 sessions old                (decided at the close, acts the next day)
    T2   phase-2 trail                                                                1 1.5 2 2.5 3 4 5 6
    BE   at the switch the stop is also raised to the entry (break-even)             no / yes
    target  squeeze only: none or the live 3R target (6 ATR above the buy-stop)
  The live exits are members: crash (1.5, 3, never), breakout (1, 3, never), squeeze (2, none, never, target).
  A bar that reaches the stop and the target counts as a stop; a stop gapped through fills at the open.
* Sizing: 1% of equity at risk to the rule's OWN initial stop, position <= 25% of equity. A wider stop therefore buys
  fewer shares; keeping the live share count with a wider stop is the same rule at a higher risk per trade, which the
  account study covers with a risk frontier (CAGR at equal maximum drawdown).
* Learned stop (B): (B1) at entry a gradient-boosting model per candidate rule predicts its gain over the live exit
  for this trade and the trade gets the candidate with the best forecast (if above a threshold); (B2) at the close of
  every held day the model predicts the gain of switching to an m-ATR trail from the next day on and the stop follows
  the forecast. Features = the terminal's ~210 scanner columns + the 300-series library + candles + weekly context +
  chip distribution (volume-by-price over 60 / 120 days) + earnings calendar + market regime (SPY, VIX, breadth,
  rates, credit, dollar, gold) + trade state; about 600 columns, all known at that close. Walk-forward by calendar
  year (train on everything that ENDED more than 30 days before the test year, purge + embargo), feature selection
  and the threshold chosen inside the training years only (nested), so every prediction for 2010-2026 is out of
  sample. Controls: the same model with the forecasts shuffled within the year (luck), the best fixed rule, and an
  oracle (perfect hindsight) that shows how much there is to find at all.
* Periods (signal date): hold 2010-2019 (never used to choose), is 2020-01..2024-06 (the only window used to choose a
  fixed rule), oos 2024-07.. ; gfc 2005-2009 reported only.
* Gate for a fixed rule (same as 7j-7m): in-sample paired delta > 0 with t >= 1.65, then delta > 0 in hold AND oos and
  the hold+oos pooled t >= max(2.5, Bonferroni over the in-sample candidates of that setup). t on signal-date means.
  The account (10 slots, 1% risk, cash, random order of same-day signals, median of seeds) must also improve in all
  three windows at equal drawdown. Nothing changes in the live terminal unless a rule passes AND the user agrees.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # noqa: BLE001 - numba is optional (slow pure-python fallback, tests only)
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

H = 20
SETUPS = ("crash_rebound", "squeeze", "breakout")
RANDOM_STEP = 20
MIN_BARS = 260
PERIODS = {"gfc": ("2004-01-01", "2009-12-31"), "hold": ("2010-01-01", "2019-12-31"),
           "is": ("2020-01-01", "2024-06-30"), "oos": ("2024-07-01", "2100-01-01")}
# trigger kinds (numba side)
TRIG_NEVER, TRIG_CLOSE_ABOVE, TRIG_HIGH_ATR, TRIG_CLOSE_ATR, TRIG_DAY = 0, 1, 2, 3, 4
TRIGGERS = {"never": (0, 0.0), "close>entry": (1, 0.0), "high+0.5": (2, 0.5), "high+1": (2, 1.0), "high+2": (2, 2.0),
            "high+3": (2, 3.0), "close+1": (3, 1.0), "close+2": (3, 2.0), "day3": (4, 3.0), "day5": (4, 5.0),
            "day10": (4, 10.0)}
S0_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
T1_GRID = (0.0, 3.0, 5.0)              # 0 = no phase-1 trail
T2_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
SQUEEZE_TGT_ATR = 6.0                  # live squeeze target = 3R on a 2 ATR stop
LIVE = {"crash_rebound": dict(S0=1.5, T1=3.0, trig="never", T2=0.0, be=False, tgt=0.0),
        "breakout": dict(S0=1.0, T1=3.0, trig="never", T2=0.0, be=False, tgt=0.0),
        "squeeze": dict(S0=2.0, T1=0.0, trig="never", T2=0.0, be=False, tgt=SQUEEZE_TGT_ATR),
        "random": dict(S0=1.5, T1=3.0, trig="never", T2=0.0, be=False, tgt=0.0)}


def rule_name(r: dict) -> str:
    t1 = "none" if r["T1"] <= 0 else f"{r['T1']:g}"
    s = f"S0={r['S0']:g}|T1={t1}"
    if r["trig"] != "never":
        s += f"|{r['trig']}->T2={r['T2']:g}" + ("|BE" if r["be"] else "")
    if r["tgt"] > 0:
        s += f"|tgt{r['tgt']:g}"
    return s


def grid(setup: str) -> list[dict]:
    """Every rule of family (A) for one setup (the live rule is always included)."""
    tgts = (0.0, SQUEEZE_TGT_ATR) if setup == "squeeze" else (0.0,)
    out, seen = [], set()
    for s0 in S0_GRID:
        for t1 in T1_GRID:
            for tg in tgts:
                cands = [dict(S0=s0, T1=t1, trig="never", T2=0.0, be=False, tgt=tg)]
                for trig in TRIGGERS:
                    if trig == "never":
                        continue
                    for t2 in T2_GRID:
                        for be in (False, True):
                            cands.append(dict(S0=s0, T1=t1, trig=trig, T2=t2, be=be, tgt=tg))
                for r in cands:
                    nm = rule_name(r)
                    if nm not in seen:
                        seen.add(nm)
                        out.append(r)
    live = LIVE[setup]
    if rule_name(live) not in seen:
        out.insert(0, dict(live))
    return out


# ====================================================================== the trade simulator
@njit(cache=True)
def _sim(O, Hh, L, C, ref, a, cond, slip, S0, T1, tkind, tval, T2, be, tgt_atr, mult, use_mult, risk_frac, cap_frac,
         want_paths):
    n, hz = O.shape
    ok = np.zeros(n, dtype=np.bool_)
    entry = np.full(n, np.nan)
    stop0 = np.full(n, np.nan)
    exit_px = np.full(n, np.nan)
    last = np.full(n, hz - 1, dtype=np.int64)
    size = np.zeros(n)
    sw_day = np.full(n, -1, dtype=np.int64)
    if want_paths:
        pe = np.zeros((n, hz))
        pc = np.zeros((n, hz))
    else:
        pe = np.zeros((1, 1))
        pc = np.zeros((1, 1))
    for i in range(n):
        ai = a[i]
        if not (ai > 0) or not np.isfinite(ref[i]):
            continue
        if cond[i]:
            if Hh[i, 0] < ref[i]:
                continue
            first = max(O[i, 0], ref[i])
        else:
            first = O[i, 0]
        e = first * (1.0 + slip)
        s0 = ref[i] - S0 * ai
        if not (e - s0 > 0) or not np.isfinite(e):
            continue
        ok[i] = True
        entry[i] = e
        stop0[i] = s0
        risk = e - s0
        sz = min(risk_frac / max(risk / e, 1e-6), cap_frac)
        size[i] = sz
        tgt = ref[i] + tgt_atr * ai if tgt_atr > 0 else np.inf
        hh = e
        switched = False
        done = False
        px = np.nan
        for k in range(hz):
            if use_mult:
                m = mult[i, k]
                st = s0 if not (m > 0) else max(s0, hh - m * ai)
            else:
                tr = T2 if switched else T1
                st = s0 if tr <= 0 else max(s0, hh - tr * ai)
                if switched and be:
                    st = max(st, e)
            if L[i, k] <= st:
                px = min(O[i, k], st)
                done = True
            elif Hh[i, k] >= tgt:
                px = max(O[i, k], tgt) if k > 0 else tgt
                done = True
            if done:
                last[i] = k
                exit_px[i] = px * (1.0 - slip)
                if want_paths:
                    v = (exit_px[i] / e - 1.0) * sz
                    for j in range(k, hz):
                        pe[i, j] = v
                        pc[i, j] = 0.0
                break
            if want_paths:
                pe[i, k] = (C[i, k] / e - 1.0) * sz
                pc[i, k] = C[i, k] / e * sz
            hh = max(hh, Hh[i, k])
            if not switched and tkind > 0:
                if tkind == 1:
                    switched = C[i, k] > e
                elif tkind == 2:
                    switched = hh >= e + tval * ai
                elif tkind == 3:
                    switched = C[i, k] >= e + tval * ai
                elif tkind == 4:
                    switched = (k + 1) >= tval
                if switched:
                    sw_day[i] = k
        if not done:
            exit_px[i] = C[i, hz - 1] * (1.0 - slip)
            if want_paths:
                pe[i, hz - 1] = (exit_px[i] / e - 1.0) * sz
                pc[i, hz - 1] = 0.0
    return ok, entry, stop0, exit_px, last, size, sw_day, pe, pc


def simulate(T: dict[str, np.ndarray], rule: dict | None, slip: float, mult: np.ndarray | None = None,
             S0_for_mult: float = 1.5, paths: bool = False, risk_frac: float = 0.01, cap_frac: float = 0.25) -> dict[str, np.ndarray]:
    """Run one rule (or a per-day multiple matrix ``mult`` n x H, 0 = only the initial stop) on the trades in T
    (O/H/L/C n x H forward windows, ref, atr, cond). Returns per-trade results; eq = account return per trade at
    ``risk_frac`` of equity at risk to the rule's own initial stop."""
    if mult is not None:
        S0, T1, tk, tv, T2, be, tg = S0_for_mult, 0.0, 0, 0.0, 0.0, False, (rule or {}).get("tgt", 0.0)
        use = True
        mm = np.ascontiguousarray(mult, dtype=np.float64)
    else:
        tk, tv = TRIGGERS[rule["trig"]]
        S0, T1, T2, be, tg = rule["S0"], rule["T1"], rule["T2"], rule["be"], rule["tgt"]
        use = False
        mm = np.zeros((1, 1))
    ok, entry, stop0, exit_px, last, size, sw, pe, pc = _sim(
        T["O"], T["H"], T["L"], T["C"], T["ref"], T["atr"], T["cond"], float(slip), float(S0), float(T1), int(tk),
        float(tv), float(T2), bool(be), float(tg), mm, use, float(risk_frac), float(cap_frac), bool(paths))
    risk = entry - stop0
    ret = np.where(ok, exit_px / entry - 1.0, 0.0)
    out = {"ok": ok, "entry": entry, "stop0": stop0, "exit_px": exit_px, "days": last + 1, "size": size,
           "ret": ret, "eq": ret * size, "r": np.where(ok, (exit_px - entry) / np.where(ok, risk, 1.0), 0.0),
           "r_atr": np.where(ok, (exit_px - entry) / T["atr"], 0.0), "switch_day": sw}
    if paths:
        out["path_eq"], out["path_cap"] = pe, pc
    return out


# ====================================================================== the account
@njit(cache=True)
def _account(cal0, lastk, need, pnl, cap, idx, lo, hi, risk, slots, heat):
    """Cash account, daily mark-to-market. Trades idx are processed in this order (sorted by cal0, then by the
    random order of same-day signals). pnl / cap = per 1R of risk (n x H)."""
    nd = hi - lo
    equity = np.zeros(nd)
    realised = 1.0
    op_i = np.full(slots, -1, dtype=np.int64)
    op_d = np.zeros(slots, dtype=np.int64)
    op_r = np.zeros(slots)
    p = 0
    m = len(idx)
    while p < m and cal0[idx[p]] < lo:
        p += 1
    taken = 0
    for d in range(lo, hi):
        mtm = 0.0
        inv = 0.0
        heat_now = 0.0
        nopen = 0
        for s in range(slots):
            i = op_i[s]
            if i >= 0:
                mtm += op_r[s] * pnl[i, d - op_d[s]]
                inv += op_r[s] * cap[i, d - op_d[s]]
                heat_now += op_r[s]
                nopen += 1
        cur = realised + mtm
        while p < m and cal0[idx[p]] == d:
            i = idx[p]
            p += 1
            r_usd = risk * cur
            if nopen >= slots or inv + r_usd * need[i] > cur:
                continue
            if heat > 0 and heat_now + r_usd > heat * cur:
                continue
            for s in range(slots):
                if op_i[s] < 0:
                    op_i[s] = i
                    op_d[s] = d
                    op_r[s] = r_usd
                    break
            nopen += 1
            mtm += r_usd * pnl[i, 0]
            inv += r_usd * cap[i, 0]
            heat_now += r_usd
            taken += 1
        equity[d - lo] = realised + mtm
        for s in range(slots):
            i = op_i[s]
            if i >= 0 and d - op_d[s] >= lastk[i]:
                realised += op_r[s] * pnl[i, lastk[i]]
                op_i[s] = -1
    return equity, taken


class Account:
    """The three setups' trades with their per-1R daily paths; any setup's exit can be swapped."""

    def __init__(self, calendar_index: pd.DatetimeIndex, slots: int = 10):
        self.cal = calendar_index
        self.slots = slots
        self.parts: dict[str, dict[str, np.ndarray]] = {}
        self._cat: dict | None = None

    def set_setup(self, setup: str, T: dict[str, np.ndarray], res: dict[str, np.ndarray]) -> None:
        ok = res["ok"]
        cal0 = self.cal.searchsorted(T["date"]) + 1
        keep = ok & (cal0 + H < len(self.cal))
        self.parts[setup] = {"cal0": cal0[keep].astype(np.int64), "lastk": (res["days"][keep] - 1).astype(np.int64),
                             "need": res["size"][keep] / 0.01, "pnl": res["path_eq"][keep] / 0.01,
                             "cap": res["path_cap"][keep] / 0.01}
        self._cat = None

    def run(self, start: str, end: str, seeds: int = 8, risk: float = 0.01, heat: float = 0.0) -> dict[str, float]:
        if self._cat is None:
            P = [self.parts[s] for s in SETUPS if s in self.parts]
            cal0 = np.concatenate([p["cal0"] for p in P])
            self._cat = {"cal0": cal0, "lastk": np.concatenate([p["lastk"] for p in P]),
                         "need": np.concatenate([p["need"] for p in P]),
                         "pnl": np.ascontiguousarray(np.concatenate([p["pnl"] for p in P])),
                         "cap": np.ascontiguousarray(np.concatenate([p["cap"] for p in P])), "idx": {}}
        cat = self._cat
        cal0, lastk, need, pnl, cap = cat["cal0"], cat["lastk"], cat["need"], cat["pnl"], cat["cap"]
        lo = int(self.cal.searchsorted(pd.Timestamp(start)))
        hi = int(min(self.cal.searchsorted(pd.Timestamp(end), side="right"), len(self.cal) - H - 1))
        runs = []
        for s in range(seeds):
            idx = cat["idx"].get(s)
            if idx is None:
                rnd = np.random.default_rng(s).random(len(cal0))
                idx = cat["idx"][s] = np.lexsort((rnd, cal0)).astype(np.int64)
            eq, taken = _account(cal0, lastk, need, pnl, cap, idx, lo, hi, float(risk), int(self.slots), float(heat))
            runs.append(perf(eq) | {"taken": float(taken)})
        out = {k: float(np.median([r[k] for r in runs])) for k in runs[0]}
        out["dd_sd"] = float(np.std([r["max_dd"] for r in runs]))
        out["cagr_sd"] = float(np.std([r["cagr"] for r in runs]))
        return out


def perf(eq: np.ndarray) -> dict[str, float]:
    eq = np.asarray(eq, dtype=float)
    years = len(eq) / 252
    end = float(eq[-1]) if len(eq) else float("nan")
    cagr = float(end ** (1 / years) - 1) if end > 0 and years > 0 else -1.0
    dd = float(np.min(eq / np.maximum.accumulate(eq) - 1)) if len(eq) else float("nan")
    r = eq[1:] / eq[:-1] - 1
    sharpe = float(r.mean() / r.std() * math.sqrt(252)) if len(r) > 2 and r.std() > 0 else float("nan")
    return {"cagr": cagr, "max_dd": dd, "calmar": cagr / -dd if dd < 0 else float("nan"), "sharpe": sharpe}


# ====================================================================== statistics
def date_t(x: np.ndarray, dates: np.ndarray) -> tuple[float, float, int]:
    """Mean of x and its t-statistic on per-signal-date means (clusters = dates)."""
    m = np.isfinite(x)
    if m.sum() < 5:
        return float("nan"), float("nan"), int(m.sum())
    s = pd.Series(x[m]).groupby(dates[m]).mean()
    if len(s) < 5 or s.std() == 0:
        return float(np.mean(x[m])), float("nan"), int(m.sum())
    return float(np.mean(x[m])), float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))), int(m.sum())


def bonferroni_t(n_tests: int, alpha: float = 0.05) -> float:
    return float(NormalDist().inv_cdf(1 - alpha / max(1, n_tests)))


def period_masks(dates: np.ndarray, periods: dict[str, tuple[str, str]] = PERIODS) -> dict[str, np.ndarray]:
    d = pd.to_datetime(dates)
    return {k: np.asarray((d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))) for k, (a, b) in periods.items()}


# ====================================================================== features
def chip_features(h: np.ndarray, lo: np.ndarray, c: np.ndarray, v: np.ndarray, a: np.ndarray,
                  windows: tuple[int, ...] = (60, 120)) -> dict[str, np.ndarray]:
    """Chip distribution (volume by price) from daily bars: each day's volume is spread uniformly over its high-low
    range (24 price bins over the window's range). Known at the close of each day (uses bars up to that day only).
    profit = share of the window's volume traded below today's close (獲利盤), poc = distance of the close to the
    highest-volume bin in ATR, cost = distance to the volume-weighted average price, width = 70% value-area width in
    ATR (concentration), over / under = share of volume within 2 ATR above / below the close (nearby supply / support)."""
    n = len(c)
    out: dict[str, np.ndarray] = {}
    tp = (h + lo + c) / 3.0
    for w in windows:
        prof, poc, cost, width, over, under = (np.full(n, np.nan) for _ in range(6))
        for t in range(w, n):
            hh, ll, vv, aa, cc = h[t - w + 1:t + 1], lo[t - w + 1:t + 1], v[t - w + 1:t + 1], a[t], c[t]
            if not (aa > 0) or not np.all(np.isfinite(hh)) or vv.sum() <= 0:
                continue
            top, bot = hh.max(), ll.min()
            if top <= bot:
                continue
            edges = np.linspace(bot, top, 25)
            mid = (edges[:-1] + edges[1:]) / 2
            # overlap of each bar's [low, high] with each bin, volume spread uniformly
            ov = np.clip(np.minimum(hh[:, None], edges[None, 1:]) - np.maximum(ll[:, None], edges[None, :-1]), 0, None)
            rngb = np.maximum(hh - ll, 1e-12)[:, None]
            wv = (ov / rngb * vv[:, None]).sum(axis=0)
            tot = wv.sum()
            if tot <= 0:
                continue
            p = wv / tot
            prof[t] = p[mid < cc].sum()
            poc[t] = (cc - mid[np.argmax(p)]) / aa
            cost[t] = (cc - (tp[t - w + 1:t + 1] * vv).sum() / vv.sum()) / aa
            order = np.argsort(-p)
            cum = np.cumsum(p[order])
            sel = order[: int(np.searchsorted(cum, 0.7)) + 1]
            width[t] = (mid[sel].max() - mid[sel].min()) / aa
            over[t] = p[(mid > cc) & (mid <= cc + 2 * aa)].sum()
            under[t] = p[(mid < cc) & (mid >= cc - 2 * aa)].sum()
        for k, arr in (("profit", prof), ("poc_atr", poc), ("cost_atr", cost), ("va_width_atr", width),
                       ("supply2", over), ("support2", under)):
            out[f"chip{w}_{k}"] = arr.astype(np.float32)
    return out


def earnings_features(index: pd.DatetimeIndex, earn: list[pd.Timestamp] | None) -> dict[str, np.ndarray]:
    """Days since the last report (sessions, causal) and calendar days to the next scheduled report (companies
    announce the date weeks ahead; Yahoo's list is used as the schedule), and a flag for a report inside the
    20-session holding window."""
    n = len(index)
    if not earn:
        return {k: np.full(n, np.nan, dtype=np.float32) for k in ("earn_since", "earn_to_cal", "earn_in_window")}
    e = np.array(sorted(pd.to_datetime(earn).normalize().unique()), dtype="datetime64[ns]")
    d = index.normalize().to_numpy()
    j = np.searchsorted(e, d, side="right")          # reports strictly after today -> next
    prev_ix = j - 1
    has_prev = prev_ix >= 0
    pos = np.arange(n)
    prev_pos = np.where(has_prev, np.searchsorted(d, e[np.clip(prev_ix, 0, len(e) - 1)]), 0)
    since = np.where(has_prev, pos - prev_pos, np.nan).astype(np.float32)
    has_next = j < len(e)
    to = np.where(has_next, (e[np.clip(j, 0, len(e) - 1)] - d) / np.timedelta64(1, "D"), np.nan).astype(np.float32)
    # the list only covers ~100 reports; outside its span the values are unknown
    span = (d >= e[0] - np.timedelta64(120, "D")) & (d <= e[-1])
    since[~span] = np.nan
    to[~span] = np.nan
    return {"earn_since": since, "earn_to_cal": to, "earn_in_window": np.where(np.isfinite(to), (to <= 28).astype(np.float32), np.nan)}


def market_features(prices: dict[str, pd.DataFrame], symbols: list[str]) -> pd.DataFrame:
    """Market regime per date (known at that close)."""
    spy = prices["SPY"]["adj_close"].astype(float)
    idx = spy.index
    out = pd.DataFrame(index=idx)
    for n_ in (1, 5, 20, 60):
        out[f"m_spy_ret{n_}"] = spy.pct_change(n_)
    for n_ in (20, 50, 200):
        out[f"m_spy_vs_sma{n_}"] = spy / spy.rolling(n_).mean() - 1
    out["m_spy_dd252"] = spy / spy.rolling(252, min_periods=60).max() - 1
    out["m_spy_vol20"] = spy.pct_change().rolling(20).std() * math.sqrt(252)
    out["m_spy_vol_ratio"] = out["m_spy_vol20"] / (spy.pct_change().rolling(120).std() * math.sqrt(252))
    if "^VIX" in prices:
        v = prices["^VIX"]["close"].astype(float).reindex(idx).ffill()
        out["m_vix"] = v
        out["m_vix_chg5"] = v - v.shift(5)
        out["m_vix_pct252"] = v.rolling(252, min_periods=60).rank(pct=True)
    for sym, nm in (("QQQ", "qqq"), ("IWM", "iwm"), ("TLT", "tlt"), ("HYG", "hyg"), ("GLD", "gld"), ("UUP", "uup")):
        if sym in prices:
            x = prices[sym]["adj_close"].astype(float).reindex(idx).ffill(limit=5)
            out[f"m_{nm}_ret20"] = x.pct_change(20)
            if sym in ("QQQ", "IWM"):
                out[f"m_{nm}_vs_spy20"] = x.pct_change(20) - spy.pct_change(20)
    if "HYG" in prices and "TLT" in prices:
        cr = (prices["HYG"]["adj_close"] / prices["TLT"]["adj_close"]).reindex(idx).ffill(limit=5)
        out["m_credit_chg20"] = cr.pct_change(20)
    if "^TNX" in prices:
        y = prices["^TNX"]["close"].astype(float).reindex(idx).ffill(limit=5)
        out["m_tnx"] = y
        out["m_tnx_chg20"] = y - y.shift(20)
    cl = pd.DataFrame({s: prices[s]["adj_close"] for s in symbols if s in prices}).reindex(idx).ffill(limit=5)
    for n_ in (20, 50, 200):
        ma = cl.rolling(n_).mean()
        out[f"m_breadth{n_}"] = (cl > ma).where(cl.notna() & ma.notna()).mean(axis=1)
    hi52 = cl.rolling(252, min_periods=120).max()
    lo52 = cl.rolling(252, min_periods=120).min()
    out["m_new_high_share"] = (cl >= hi52).where(hi52.notna()).mean(axis=1)
    out["m_new_low_share"] = (cl <= lo52).where(lo52.notna()).mean(axis=1)
    out["m_xs_disp20"] = cl.pct_change(20).std(axis=1)
    return out.astype(np.float32)


def stock_feature_frame(f: pd.DataFrame, earn: list | None, mkt: pd.DataFrame) -> pd.DataFrame:
    """All features of one stock per day (~600 columns), known at that day's close."""
    import warnings

    from src.backtest.holding_study import numeric_frame
    from src.technical import candles
    from src.technical import indicator_library as il
    warnings.simplefilter("ignore")
    o, h, lo, c, v, a = (f[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close", "volume", "atr"))
    parts = [numeric_frame(f)]
    lib = il.compute(f[["open", "high", "low", "close", "volume"]])
    parts.append(lib.astype(np.float32))
    bars = candles.bar_features(o, h, lo, c, v, a)
    flags = candles.detect(o, h, lo, c, a)
    extra = {f"b_{k}": np.asarray(x, dtype=np.float32) for k, x in bars.items()}
    extra.update({f"p_{k}": flags[k].astype(np.float32) for k in candles.PATTERN_IDS})
    extra.update(chip_features(h, lo, c, v, a))
    extra.update(earnings_features(f.index, earn))
    parts.append(pd.DataFrame(extra, index=f.index))
    parts.append(candles.weekly_context(f).astype(np.float32))
    parts.append(mkt.reindex(f.index).ffill(limit=5))
    X = pd.concat(parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return X.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def collect_stock(sym: str, f: pd.DataFrame, st: pd.DataFrame, settings: dict[str, Any], X: pd.DataFrame,
                  si: int, trade_days: bool = True) -> dict[str, Any]:
    """Signals of one stock with forward windows, entry-day features and (setup trades only) the features at the
    close of every day of the 20-session window."""
    from src.scanner.plan import plan_for_setup
    cfg = settings["scanner"]
    n = len(f)
    o, h, lo, c, a = (f[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close", "atr"))
    sig: list[tuple[int, str, float]] = []
    for s in SETUPS:
        mask = st[f"setup_{s}"].to_numpy(dtype=bool)
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        plan = plan_for_setup(f, s, cfg)["entry"].to_numpy(float)
        for t in starts[(starts >= MIN_BARS) & (starts + H + 1 < n)]:
            if np.isfinite(a[t]) and a[t] > 0 and np.isfinite(plan[t]):
                sig.append((int(t), s, float(plan[t])))
    for t in range(MIN_BARS + (si * 7) % RANDOM_STEP, n - H - 1, RANDOM_STEP):
        if np.isfinite(a[t]) and a[t] > 0:
            sig.append((t, "random", float(c[t])))
    if not sig:
        return {"symbol": sym}
    ts = np.array([x[0] for x in sig])
    kinds = np.array([x[1] for x in sig])
    ref = np.array([x[2] for x in sig])
    fw = ts[:, None] + np.arange(1, H + 1)[None, :]
    Xv = X.to_numpy(np.float32)
    out = {"symbol": sym, "date": f.index[ts].to_numpy(), "setup": kinds, "ref": ref, "atr": a[ts], "t": ts,
           "O": o[fw].astype(np.float32), "H": h[fw].astype(np.float32), "L": lo[fw].astype(np.float32),
           "C": c[fw].astype(np.float32), "Xe": Xv[ts]}
    if trade_days:
        m = kinds != "random"
        rows = fw[m]                                   # close of day k (k = 0 .. H-1) of each setup trade
        out["Xd"] = Xv[rows[:, : H - 1].ravel()].reshape(int(m.sum()), H - 1, Xv.shape[1])
    return out


# ====================================================================== (B2) continuation values for a daily stop choice
@njit(cache=True)
def _cont_values(O, Hh, L, C, ref, a, cond, slip, S0, tgt_atr, menu, risk_frac, cap_frac):
    """V[i, k, j] = account result of trade i if it is still held at the close of day k (0-based) and from day k+1
    on the stop is max(initial stop, highest high so far - menu[j] x ATR) (menu 0 = initial stop only).
    alive[i, k] = the trade is open at the close of day k under EVERY member of the menu (it never touched the
    initial stop or the target), i.e. the decision at that close is real whatever was chosen before."""
    n, hz = O.shape
    M = len(menu)
    V = np.full((n, hz - 1, M), np.nan)
    alive = np.zeros((n, hz - 1), dtype=np.bool_)
    for i in range(n):
        ai = a[i]
        if not (ai > 0):
            continue
        if cond[i]:
            if Hh[i, 0] < ref[i]:
                continue
            first = max(O[i, 0], ref[i])
        else:
            first = O[i, 0]
        e = first * (1.0 + slip)
        s0 = ref[i] - S0 * ai
        if not (e - s0 > 0):
            continue
        sz = min(risk_frac / max((e - s0) / e, 1e-6), cap_frac)
        tgt = ref[i] + tgt_atr * ai if tgt_atr > 0 else np.inf
        hh_k = e
        for k in range(hz - 1):
            if L[i, k] <= s0 or Hh[i, k] >= tgt:
                break
            hh_k = max(hh_k, Hh[i, k])
            alive[i, k] = True
            for j in range(M):
                m = menu[j]
                hh = hh_k
                px = np.nan
                for q in range(k + 1, hz):
                    st = s0 if not (m > 0) else max(s0, hh - m * ai)
                    if L[i, q] <= st:
                        px = min(O[i, q], st)
                        break
                    if Hh[i, q] >= tgt:
                        px = max(O[i, q], tgt)
                        break
                    hh = max(hh, Hh[i, q])
                if not np.isfinite(px):
                    px = C[i, hz - 1]
                V[i, k, j] = (px * (1.0 - slip) / e - 1.0) * sz
    return V, alive


def cont_values(T: dict[str, np.ndarray], S0: float, tgt_atr: float, menu: tuple[float, ...], slip: float):
    return _cont_values(T["O"], T["H"], T["L"], T["C"], T["ref"], T["atr"], T["cond"], float(slip), float(S0),
                        float(tgt_atr), np.asarray(menu, dtype=np.float64), 0.01, 0.25)


def state_features(T: dict[str, np.ndarray], slip: float) -> np.ndarray:
    """Policy-free trade state at the close of each day k (n x H-1 x 8): day, close vs entry, MFE, MAE, give-back
    from the high, sessions since the high (all in ATR), return since entry, today's range."""
    O, Hh, L, C, a = T["O"], T["H"], T["L"], T["C"], T["atr"][:, None]
    first = np.where(T["cond"], np.maximum(O[:, 0], T["ref"]), O[:, 0])
    e = (first * (1 + slip))[:, None]
    hz = O.shape[1] - 1
    hi = np.maximum.accumulate(Hh, axis=1)[:, :hz]
    lo_ = np.minimum.accumulate(L, axis=1)[:, :hz]
    Ck = C[:, :hz]
    pos_hi = np.zeros_like(Ck)
    best = np.full(len(O), -np.inf)
    last = np.zeros(len(O))
    for k in range(hz):
        new = Hh[:, k] >= best
        best = np.maximum(best, Hh[:, k])
        last = np.where(new, k, last)
        pos_hi[:, k] = k - last
    S = np.stack([np.broadcast_to(np.arange(hz, dtype=float), Ck.shape), (Ck - e) / a, (hi - e) / a, (lo_ - e) / a,
                  (hi - Ck) / a, pos_hi, Ck / e - 1, (Hh[:, :hz] - L[:, :hz]) / a], axis=2)
    return S.astype(np.float32)


STATE_COLS = ("s_day", "s_close_vs_entry_atr", "s_mfe_atr", "s_mae_atr", "s_giveback_atr", "s_days_since_high",
              "s_ret_since_entry", "s_range_atr")

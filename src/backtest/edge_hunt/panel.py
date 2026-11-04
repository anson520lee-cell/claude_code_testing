"""Signals and trades for the three live setups (+ random entries) on a point-in-time S&P 500 universe.

Uses the LIVE code (src.scanner.features.build_features, src.scanner.setups.detect_setups) so the setups are
exactly what the terminal lists, and the LIVE exits (strategy lab choices, config/strategy_lab.default.json):

    crash_rebound  signal day only (scanner.fresh_only_setups), next open, stop close - 1.5 ATR, 3-ATR trail, 20 sessions
    breakout       first day of the episode, next open, stop close - 1 ATR, 3-ATR trail, 20 sessions
    squeeze        first day of the episode, buy-stop at 20-day high + 0.1 ATR next session only,
                   stop entry ref - 2 ATR, target 3R, 20 sessions
    random         every 5th eligible session of every stock, next open, each of the three exits
                   (squeeze exit unconditional: stop close - 2 ATR, 3R target)

Slippage 10 bps per side (config/terminal.yaml backtest.slippage_bps). A stock is eligible on a date when it is an
S&P 500 member that day (fja05680/sp500 change log), its unadjusted close is >= $5 and it has >= 260 sessions.
"""

from __future__ import annotations

import math
import zlib
from typing import Any

import numpy as np
import pandas as pd

from src.scanner.features import build_features
from src.scanner.setups import detect_setups

H = 20
SLIP = 0.001
RANDOM_STEP = 5


def membership(hist_csv: str, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """ticker -> bool array over ``dates``: S&P 500 member on that date (Yahoo-style tickers: BRK.B -> BRK-B)."""
    h = pd.read_csv(hist_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    pos = np.searchsorted(dates.values, h["date"].values, side="left")
    ends = np.append(pos[1:], len(dates))
    out: dict[str, np.ndarray] = {}
    for i, row in h.iterrows():
        a, b = int(pos[i]), int(ends[i])
        if b <= a:
            continue
        for t in str(row["tickers"]).split(","):
            t = t.strip().replace(".", "-")
            if not t:
                continue
            if t not in out:
                out[t] = np.zeros(len(dates), dtype=bool)
            out[t][a:b] = True
    return out


def trade(O: np.ndarray, Hh: np.ndarray, L: np.ndarray, C: np.ndarray, t: int, ref: float, atr: float,
          stop_atr: float, trail: float | None, target_r: float | None, conditional: bool = False) -> tuple[float, float, int]:
    """(R, % return, sessions held) of one long trade signalled at the close of ``t``; nan if not filled."""
    o, hi, lo, c = O[t + 1:t + 1 + H], Hh[t + 1:t + 1 + H], L[t + 1:t + 1 + H], C[t + 1:t + 1 + H]
    if len(o) < H or not (np.isfinite(atr) and atr > 0 and np.isfinite(o[0])):
        return math.nan, math.nan, 0
    if conditional:
        if hi[0] < ref:
            return math.nan, math.nan, 0
        entry = max(o[0], ref) * (1 + SLIP)
    else:
        entry = o[0] * (1 + SLIP)
    stop = ref - stop_atr * atr
    risk = entry - stop
    if not (risk > 0):
        return math.nan, math.nan, 0
    target = entry + target_r * risk if target_r else math.inf
    run_hi = entry
    for k in range(H):
        level = max(stop, run_hi - trail * atr) if trail else stop
        if lo[k] <= level:
            px = min(o[k], level) if k > 0 else level
            px *= 1 - SLIP
            return (px - entry) / risk, px / entry - 1, k + 1
        if hi[k] >= target:
            px = max(o[k], target) if k > 0 else target
            px *= 1 - SLIP
            return (px - entry) / risk, px / entry - 1, k + 1
        run_hi = max(run_hi, hi[k])
    px = c[H - 1] * (1 - SLIP)
    return (px - entry) / risk, px / entry - 1, H


BORROW = 0.02     # short borrow cost per year (general collateral; hard-to-borrow names cost more)


def trade_short(O: np.ndarray, Hh: np.ndarray, L: np.ndarray, C: np.ndarray, t: int, ref: float, atr: float,
                stop_atr: float = 1.5, trail: float = 3.0) -> tuple[float, float, int]:
    """Mirror of the live long exit: short at the next open, stop ref + 1.5 ATR, trailing stop = lowest low since
    entry + 3 ATR, 20 sessions; 10 bps per side plus 2%/yr borrow. Returns (R, % return, sessions)."""
    o, hi, lo, c = O[t + 1:t + 1 + H], Hh[t + 1:t + 1 + H], L[t + 1:t + 1 + H], C[t + 1:t + 1 + H]
    if len(o) < H or not (np.isfinite(atr) and atr > 0 and np.isfinite(o[0])):
        return math.nan, math.nan, 0
    entry = o[0] * (1 - SLIP)
    stop = ref + stop_atr * atr
    risk = stop - entry
    if not (risk > 0):
        return math.nan, math.nan, 0
    run_lo = entry
    for k in range(H):
        level = min(stop, run_lo + trail * atr)
        if hi[k] >= level:
            px = (max(o[k], level) if k > 0 else level) * (1 + SLIP)
            pnl = entry - px - entry * BORROW * (k + 1) / 252
            return pnl / risk, pnl / entry, k + 1
        run_lo = min(run_lo, lo[k])
    px = c[H - 1] * (1 + SLIP)
    pnl = entry - px - entry * BORROW * H / 252
    return pnl / risk, pnl / entry, H


EXITS = {  # name -> (stop_atr, trail, target_r)
    "crash": (1.5, 3.0, None),
    "bo": (1.0, 3.0, None),
    "sq": (2.0, None, 3.0),
}
SETUP_EXIT = {"crash_rebound": "crash", "breakout": "bo", "squeeze": "sq"}


def _monthly_seasonality(close: pd.Series) -> pd.Series:
    """For each date: mean return of the same calendar month over the previous 1..10 years (>= 5 years needed)."""
    m = close.resample("ME").last().pct_change()
    out = pd.Series(np.nan, index=m.index)
    for i, d in enumerate(m.index):
        prev = [m.get(pd.Timestamp(year=d.year - k, month=d.month, day=1) + pd.offsets.MonthEnd(0)) for k in range(1, 11)]
        prev = [x for x in prev if x is not None and np.isfinite(x)]
        if len(prev) >= 5:
            out.iloc[i] = float(np.mean(prev))
    # value known at the start of month d: same-month history only uses earlier years
    return out


def membership_all(prices: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, min_dv: float) -> dict[str, np.ndarray]:
    """Replication universe: every ticker in ``prices`` counts as a member on each date where its 20-session average
    dollar volume (unadjusted close x volume) is >= ``min_dv`` (the $5 price and 260-session rules apply later)."""
    out: dict[str, np.ndarray] = {}
    for s, x in prices.items():
        if s.startswith("^"):
            continue
        x = x.reindex(dates)
        dv = (x["close"] * x["volume"]).rolling(20, min_periods=15).mean()
        out[s] = (dv >= min_dv).to_numpy(bool)
    return out


def universe(hist: str, prices: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """``hist`` is either the S&P 500 change-log CSV or ``ALL:<min dollar volume>`` (replication universe)."""
    if hist.startswith("ALL"):
        return membership_all(prices, dates, float(hist.split(":")[1]) if ":" in hist else 0.0)
    return membership(hist, dates)


def run_symbol(job: tuple) -> pd.DataFrame | None:
    sym, px, spy, member, live_list = job
    if len(px) < 300:
        return None
    try:
        f = build_features(px, spy, None)
        s = detect_setups(f)
    except Exception:  # noqa: BLE001
        return None
    n = len(f)
    O, Hh, L, C, A = (f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "atr"))
    raw_close = px["close"].reindex(f.index).to_numpy(dtype=float)
    mem = member if member is not None else np.zeros(n, dtype=bool)
    idx = np.arange(n)
    elig = (mem | live_list) & (raw_close >= 5) & (idx >= 260) & (idx + H < n)
    ret1 = f["return_1d"].to_numpy(dtype=float)
    ret5 = f["return_5d"].to_numpy(dtype=float)
    ret3 = f["close"].to_numpy() / f["close"].shift(3).to_numpy() - 1
    ext = f["extension_atr"].to_numpy(dtype=float)
    event = ((ret5 <= -0.20) | (ret3 <= -0.15) | (ext <= -3)) & np.isfinite(ext)
    ev_s = pd.Series(event, index=f.index)
    recent = ev_s.shift(1).rolling(10, min_periods=1).max().fillna(0).astype(bool).to_numpy()
    first = event & ~recent
    bo = s["setup_breakout"].to_numpy(bool)
    sq = s["setup_squeeze"].to_numpy(bool)
    bo_first = bo & ~np.r_[False, bo[:-1]]
    sq_first = sq & ~np.r_[False, sq[:-1]]
    shorts = {}
    for name in ("breakdown", "rally_fail", "ema21_loss"):
        m = s[f"setup_{name}"].to_numpy(bool)
        shorts[name] = m & ~np.r_[False, m[:-1]]
    # extra causal features (new constructs; not in the strategy lab's 183)
    spy_c = spy.reindex(f.index).ffill().to_numpy(dtype=float)
    spy_r1 = np.r_[np.nan, spy_c[1:] / spy_c[:-1] - 1]
    spy_r5 = np.r_[[np.nan] * 5, spy_c[5:] / spy_c[:-5] - 1]
    beta = f["beta120"].to_numpy(dtype=float)
    logO = np.log(O)
    logC = np.log(C)
    on = np.r_[np.nan, logO[1:] - logC[:-1]]                      # overnight log return
    on5 = pd.Series(on).rolling(5).sum().to_numpy()
    tot5 = np.r_[[np.nan] * 5, logC[5:] - logC[:-5]]
    resid = pd.Series(ret1 - np.nan_to_num(beta, nan=1.0) * spy_r1)
    ivol60 = resid.rolling(60, min_periods=50).std().to_numpy() * math.sqrt(252)
    max20 = pd.Series(ret1).shift(1).rolling(20, min_periods=15).max().to_numpy()
    crash_cnt = pd.Series(first.astype(float)).shift(11).rolling(241, min_periods=100).sum().to_numpy()
    mom12_1 = np.r_[[np.nan] * 252, C[231:-21] / C[:-252] - 1] if n > 252 else np.full(n, np.nan)
    seas = _monthly_seasonality(f["close"])
    seas_d = seas.reindex(f.index, method="ffill")
    # seasonality value for the month we are IN: the month-end label of the current month
    cur_me = f.index + pd.offsets.MonthEnd(0)
    seas_cur = seas.reindex(cur_me).to_numpy(dtype=float)
    hv60 = f["hv60"].to_numpy(dtype=float)
    dv20 = f["dollar_volume20"].to_numpy(dtype=float)
    vol = f["volume"].to_numpy(dtype=float)
    vr = f["volume_ratio"].to_numpy(dtype=float)
    clv = f["clv"].to_numpy(dtype=float)
    e200 = f["ema200"].to_numpy(dtype=float)
    hi20 = f["high20"].to_numpy(dtype=float)
    from52 = f["from_52w_high"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    rnd_off = zlib.crc32(sym.encode()) % RANDOM_STEP

    def base(t: int, setup: str) -> dict[str, Any]:
        return {"symbol": sym, "date": f.index[t], "setup": setup, "member": bool(mem[t]), "live_list": bool(live_list),
                "live_like": bool(beta[t] >= 1.3 and hv60[t] >= 0.35), "beta": beta[t], "hv60": hv60[t], "dv20": dv20[t],
                "close": C[t], "raw_close": raw_close[t], "atr": A[t], "volume": vol[t], "ret1": ret1[t], "ret5": ret5[t],
                "ext_atr": ext[t], "spy_r5": spy_r5[t], "on_share5": on5[t] / tot5[t] if tot5[t] < -0.02 else np.nan,
                "speed": (logC[t] - logC[t - 1]) / tot5[t] if tot5[t] < -0.02 else np.nan,
                "sys_share": beta[t] * spy_r5[t] / ret5[t] if ret5[t] < -0.02 else np.nan,
                "ivol60": ivol60[t], "max20": max20[t], "crash_cnt": crash_cnt[t], "mom12_1": mom12_1[t],
                "seas": seas_cur[t], "vr": vr[t], "clv": clv[t], "above200": bool(C[t] > e200[t]), "from52": from52[t]}

    for t in np.flatnonzero(elig & (first | bo_first | sq_first)):
        for setup, flag in (("crash_rebound", first[t]), ("breakout", bo_first[t]), ("squeeze", sq_first[t])):
            if not flag:
                continue
            ex = EXITS[SETUP_EXIT[setup]]
            cond = setup == "squeeze"
            ref = hi20[t] + 0.1 * A[t] if cond else C[t]
            r, ret, days = trade(O, Hh, L, C, t, ref, A[t], *ex, conditional=cond)
            if not np.isfinite(r):
                continue
            row = base(t, setup)
            row.update({"r": r, "ret": ret, "days": days})
            rows.append(row)
    for name, m in shorts.items():
        for t in np.flatnonzero(elig & m):
            r, ret, days = trade_short(O, Hh, L, C, t, C[t], A[t])
            if not np.isfinite(r):
                continue
            row = base(t, name)
            row.update({"r": r, "ret": ret, "days": days})
            rows.append(row)
    for t in np.flatnonzero(elig & ((idx + rnd_off) % RANDOM_STEP == 0)):
        row = base(t, "random")
        for name, ex in EXITS.items():
            r, ret, _ = trade(O, Hh, L, C, t, C[t], A[t], *ex)
            row[f"r_{name}"], row[f"ret_{name}"] = r, ret
        row["r_short"], row["ret_short"], _ = trade_short(O, Hh, L, C, t, C[t], A[t])
        if np.isfinite(row["r_crash"]):
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


def load_prices(path: str) -> dict[str, pd.DataFrame]:
    p = pd.read_parquet(path)
    p["date"] = pd.to_datetime(p["date"]).dt.tz_localize(None) if getattr(p["date"].dt, "tz", None) else pd.to_datetime(p["date"])
    out = {}
    for sym, g in p.groupby("symbol"):
        x = g.set_index("date").sort_index()[["open", "high", "low", "close", "adj_close", "volume"]]
        x = x[~x.index.duplicated()].dropna(subset=["close"])
        x = x[(x["close"] > 0) & (x["adj_close"] > 0)]
        out[sym] = x
    return out

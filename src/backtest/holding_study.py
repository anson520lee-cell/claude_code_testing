"""What happens AFTER entry: candles, volume, indicators and trade state as exit signals for a held position.

Question (Neo, 2026-11-02): the live exits are fixed at entry (ATR stop + trailing stop) and ignore what the market
does after the entry, so profits are given back and stops are taken in full. Once a position is held, does what is
visible each day - a hammer or a pin bar that recovers on the hourly chart, volume expansion, the day-1 / day-2 /
day-3 state of 300+ indicators, extreme overbought / oversold readings - tell whether to keep holding, sell, or sell
half? Measured on real trades, not on textbook claims (a hammer may well be followed by FASTER falls).

Unit: a TRADE-DAY = the close of the k-th session of a trade that the live exit has not closed yet (k = 1 is the
entry day). For each trade-day the CONTINUATION VALUE is

    CV = result of holding on with the live exit  -  result of selling at the next open (10 bps slippage)

in account % (live sizing: 1% risk, position capped at 25%) and in R. CV > 0: holding was better. An exit signal is
useful only where CV is reliably NEGATIVE. Selling half and letting the rest run on the live exit changes the trade by
exactly half of a full exit (linear), so a 50% partial has the same sign and t-statistic as the full exit: partial
exits cannot turn an exit with negative value into a positive one; they only halve its size and its variance.

Design (fixed before the first run; same data, periods and gate as 7j / 7k)
* Trades: every episode start of the three live setups (crash_rebound, squeeze, breakout), entry at the next open
  (squeeze: buy-stop at the 20-day high + 0.1 ATR), live exit (crash 1.5 ATR stop + 3-ATR trail; breakout 1 ATR + 3-ATR
  trail; squeeze 2 ATR + 3R target), 20 sessions; plus random entries (every 20th session of every stock) with the
  crash-rebound exit as a market baseline. Daily bars 2004-2026 (Yahoo), 233 stocks after the live focus filter.
* A daily signal seen at the close of day k is acted on at the open of day k+1. An hourly signal (2024-01.. only,
  Yahoo keeps ~2.9 years of 60-minute bars) is acted on at the open of the next hour.
* Rule = "the FIRST time signal S appears (optionally only while the trade is >= +0.5R or <= 0R), sell at the next
  open" (or: raise the stop to that bar's low - 0.1 ATR). Paired per trade: delta = rule - live (= -CV at the first
  firing; trades where S never fires have delta 0).
* Signals (all computed with data up to the signal bar only):
    candles  46 daily patterns x {any volume, volume >= 1.5x, <= 0.7x} (src/technical/candles.py), 25 hourly patterns
             x 3 volume states (incl. the pin bar through yesterday's low that recovers, 打針收回), 30 multi-timeframe
             combinations (daily pattern x weekly trend, hourly pattern x daily trend);
    indicators  every numeric column the terminal computes (build_features, distances in ATR, the analyst techniques and
             volume profile on days 1-5, categorical states one-hot, weekly context, market, change since the signal
             day, trade state: day, open R, MFE, MAE, give-back, distance to the stop); extreme-tail rules with
             thresholds fixed on the in-sample trade-days (10% / 90%, 5% / 95%);
    classic  overbought / oversold exits (RSI6 >= 80 / 90, RSI14 >= 70 / 80, close above the upper band, >= 3 / 4 ATR
             above EMA21, volume climax, MACD histogram turning down, RSI6 <= 20 while held, ...);
    state    early failure (open R <= -0.25 / -0.5 / -0.75 at the close of day 1 / 2 / 3 / 5), not working by day
             3 / 5 / 7 / 10, give-back after a +1 / 1.5 / 2 / 3R run-up, stalling (no new high for 5 sessions);
    model    one ridge regression of CV on all features, fitted on in-sample trade-days only; exit when the forecast
             falls below a threshold chosen in sample (one pre-registered test per setup).
* Periods (signal date): is 2020-01..2024-06 (the only data used to choose), hold 2010-2019 (never used to design
  anything), oos 2024-07.., gfc 2005-2009 (report only). Hourly: is_h 2024-01..2025-03, oos_h 2025-04.. .
* Gate: in-sample paired delta > 0 with t >= 1.65 -> candidate; then delta > 0 in hold AND oos and the hold+oos pooled
  t >= max(2.5, Bonferroni over the number of in-sample candidates of that setup); a max-t sign-flip permutation over
  all rules of a setup on the pooled confirmation sample is reported as a selection-free guard. Hourly rules: the
  same with oos_h alone as confirmation (exploratory: short sample). t-statistics are on per-date means (clusters =
  signal dates). Nothing changes in the live terminal unless a rule passes AND the user agrees.
* Descriptive answers (no gate, for the holdings page): what each pattern does over the next 1 / 3 / 5 / 10 sessions
  across ALL stock-days (next open to close, in ATR, minus the same day's universe average), and inside held trades by
  day of the trade.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.scanner.plan import plan_for_setup
from src.technical import candles

H = 20
SETUPS = ("crash_rebound", "squeeze", "breakout")
LIVE_EXIT = {
    "crash_rebound": {"stop_atr": 1.5, "trail": 3.0, "tgt_r": None},
    "breakout": {"stop_atr": 1.0, "trail": 3.0, "tgt_r": None},
    "squeeze": {"stop_atr": 2.0, "trail": None, "tgt_r": 3.0},
    "random": {"stop_atr": 1.5, "trail": 3.0, "tgt_r": None},
}
RANDOM_STEP = 20
MIN_BARS = 260
PERIODS = {"gfc": ("2004-01-01", "2009-12-31"), "hold": ("2010-01-01", "2019-12-31"),
           "is": ("2020-01-01", "2024-06-30"), "oos": ("2024-07-01", "2100-01-01")}
HPERIODS = {"is_h": ("2024-01-01", "2025-03-31"), "oos_h": ("2025-04-01", "2100-01-01")}
FWD = (1, 3, 5, 10)
TECH_MAX_DAY = 5
PROFIT_R, LOSS_R = 0.5, 0.0

CAT_COLS = ("trend", "ema50_state", "rsi_zone", "macd_state", "macd_hist_trend", "macd_zero_context", "momentum",
            "volume_level", "day_direction", "vwap_position", "bb_position", "band_state", "band_context", "volatility",
            "relative_strength", "adx_state", "structure", "swing_condition", "swing_structure")
LEVELS = {"open", "high", "low", "close", "volume", "ema9", "ema21", "ema50", "ema200", "ema250", "macd", "macd_signal",
          "macd_hist", "atr", "bb_mid", "bb_upper", "bb_lower", "mavol20", "avg_volume_5d", "obv", "vwap20", "high20",
          "low20", "high60", "low60", "prior_high20", "prior_low20", "high252", "low252", "prior_high55", "prior_low55",
          "dollar_volume20", "low3", "low5", "low10", "high3", "high5", "high10", "swing_high_last", "swing_high_prev",
          "swing_low_last", "swing_low_prev", "macd_hist_3d"}
DELTA_COLS = ("rsi6", "rsi14", "bb_pct_b", "adx", "plus_di", "minus_di", "extension_atr", "volume_ratio", "obv_slope10",
              "clv", "atr_pct", "rel_5d", "swing_score", "macd_hist")
MTF_DAILY = ("hammer", "shooting_star", "bull_engulfing", "bear_engulfing", "long_lower_wick", "long_upper_wick",
             "gap_up_faded", "big_down", "big_up", "doji")
MTF_HOURLY = ("spring", "hammer", "shooting_star", "bear_engulfing", "vol_spike_down")


# ====================================================================== per-stock numeric frame (causal)
def numeric_frame(f: pd.DataFrame) -> pd.DataFrame:
    """Every indicator of build_features in scale-free numeric form (f_ = raw stationary columns, d_ = levels as
    distances in ATR, c_ = categorical states one-hot)."""
    cols: dict[str, np.ndarray] = {}
    for col in f.columns:
        if col in LEVELS or col in CAT_COLS:
            continue
        s = f[col]
        if s.dtype == bool:
            cols[f"f_{col}"] = s.to_numpy(dtype=float)
        elif pd.api.types.is_numeric_dtype(s):
            cols[f"f_{col}"] = s.to_numpy(dtype=float)
    c, a = f["close"].to_numpy(float), f["atr"].to_numpy(float)
    a = np.where(a > 0, a, np.nan)
    for col in ("ema9", "ema21", "ema50", "ema200", "ema250", "vwap20", "bb_upper", "bb_lower", "high20", "low20",
                "high60", "low60", "high252", "low252"):
        if col in f.columns:
            cols[f"d_{col}"] = (c - f[col].to_numpy(float)) / a
    for col in ("macd", "macd_signal", "macd_hist", "macd_hist_3d"):
        if col in f.columns:
            cols[f"d_{col}"] = f[col].to_numpy(float) / a
    e9, e21, e50, e200 = (f[k].to_numpy(float) for k in ("ema9", "ema21", "ema50", "ema200"))
    cols["d_stack_bull"] = ((c > e9) & (e9 > e21) & (e21 > e50) & (e50 > e200)).astype(float)
    cols["d_stack_bear"] = ((c < e9) & (e9 < e21) & (e21 < e50) & (e50 < e200)).astype(float)
    cols["d_above_ema200"] = (c > e200).astype(float)
    for col in CAT_COLS:
        if col not in f.columns:
            continue
        vals = f[col].astype(str).to_numpy()
        for v in sorted(set(vals) - {"n/a", "", "nan", "None"}):
            cols[f"c_{col}={v}"] = (vals == v).astype(float)
    return pd.DataFrame(cols, index=f.index).astype(np.float32)


def market_frame(spy: pd.DataFrame, vix: pd.DataFrame | None) -> pd.DataFrame:
    """Market context per date: SPY 1 / 5-day return, SPY vs EMA50, VIX level and 5-day change."""
    c = spy["adj_close"].astype(float)
    out = pd.DataFrame({"m_spy_ret1": c.pct_change(), "m_spy_ret5": c.pct_change(5),
                        "m_spy_above_ema50": (c > c.ewm(span=50, adjust=False).mean()).astype(float)}, index=spy.index)
    if vix is not None and not vix.empty:
        v = vix["close"].astype(float).reindex(spy.index).ffill()
        out["m_vix"] = v
        out["m_vix_chg5"] = v - v.shift(5)
    return out


# ====================================================================== live exit, path by path
def live_paths(O: np.ndarray, Hh: np.ndarray, L: np.ndarray, C: np.ndarray, ref: np.ndarray, a: np.ndarray,
               kind: str, slip: float) -> dict[str, np.ndarray]:
    """Vectorised live exit on forward windows (m x H, day 1 = column 0). Same rules as structure.simulate."""
    p = LIVE_EXIT[kind]
    m, hz = O.shape
    cond = np.full(m, kind == "squeeze")
    first_px = np.where(cond, np.maximum(O[:, 0], ref), O[:, 0])
    filled = ~cond | (Hh[:, 0] >= ref)
    entry = first_px * (1 + slip)
    stop0 = ref - p["stop_atr"] * a
    ok = filled & np.isfinite(entry) & np.isfinite(stop0) & (entry - stop0 > 0)
    risk = np.where(ok, entry - stop0, np.nan)
    tgt = ref + p["tgt_r"] * (ref - stop0) if p["tgt_r"] else np.full(m, np.inf)
    live = ok.copy()
    exit_px = np.full(m, np.nan)
    exit_day = np.full(m, hz - 1)
    hi_prev = entry.copy()
    stop_day = np.full((m, hz), np.nan)
    open_after = np.zeros((m, hz), dtype=bool)
    for k in range(hz):
        st = stop0 if p["trail"] is None else np.maximum(stop0, hi_prev - p["trail"] * a)
        stop_day[:, k] = st
        hit_s = live & (L[:, k] <= st)
        px_s = np.minimum(O[:, k], st)
        hit_t = live & ~hit_s & (Hh[:, k] >= tgt)
        px_t = np.maximum(O[:, k], tgt) if k > 0 else tgt
        exit_px = np.where(hit_s, px_s, np.where(hit_t, px_t, exit_px))
        exit_day = np.where(hit_s | hit_t, k, exit_day)
        live &= ~(hit_s | hit_t)
        open_after[:, k] = live
        hi_prev = np.maximum(hi_prev, Hh[:, k])
    exit_px = np.where(live, C[:, hz - 1], exit_px) * (1 - slip)
    size = np.where(ok, np.minimum(0.01 / np.maximum(risk / entry, 1e-6), 0.25), 0.0)
    return {"entry": entry, "stop0": stop0, "risk": risk, "tgt": tgt, "ok": ok, "exit_px": exit_px, "exit_day": exit_day,
            "stop_day": stop_day, "open_after": open_after, "size": size,
            "r": (exit_px - entry) / risk, "ret": exit_px / entry - 1}


def resim_from(O: np.ndarray, Hh: np.ndarray, L: np.ndarray, C: np.ndarray, base: dict[str, np.ndarray], a: np.ndarray,
               kind: str, k_fire: np.ndarray, new_stop: np.ndarray, slip: float, new_trail: float | None = None) -> np.ndarray:
    """Account result per trade when, at the close of day k_fire (0-based; -1 = never), the stop is raised to
    new_stop (and / or the trail tightened to new_trail ATR). Trades that never fire keep the live result."""
    p = LIVE_EXIT[kind]
    m, hz = O.shape
    entry, stop0, tgt = base["entry"], base["stop0"], base["tgt"]
    live = base["ok"].copy()
    exit_px = np.full(m, np.nan)
    hi_prev = entry.copy()
    raised = np.full(m, -np.inf)
    for k in range(hz):
        st = stop0 if p["trail"] is None else np.maximum(stop0, hi_prev - p["trail"] * a)
        active = (k_fire >= 0) & (k > k_fire)
        if new_trail is not None:
            st = np.where(active, np.maximum(st, hi_prev - new_trail * a), st)
        st = np.where(active, np.maximum(st, raised), st)
        hit_s = live & (L[:, k] <= st)
        px_s = np.minimum(O[:, k], st)
        hit_t = live & ~hit_s & (Hh[:, k] >= tgt)
        px_t = np.maximum(O[:, k], tgt) if k > 0 else tgt
        exit_px = np.where(hit_s, px_s, np.where(hit_t, px_t, exit_px))
        live &= ~(hit_s | hit_t)
        hi_prev = np.maximum(hi_prev, Hh[:, k])
        raised = np.where(k_fire == k, new_stop, raised)
    exit_px = np.where(live, C[:, hz - 1], exit_px) * (1 - slip)
    return (exit_px / entry - 1) * base["size"]


# ====================================================================== collection (one stock)
def collect_stock(sym: str, f: pd.DataFrame, st: pd.DataFrame, settings: dict[str, Any], mkt: pd.DataFrame,
                  si: int, slip: float, hourly: pd.DataFrame | None = None, raw: pd.DataFrame | None = None) -> dict[str, Any]:
    """Trades, trade-days and universe pattern rows of one stock."""
    cfg = settings["scanner"]
    n = len(f)
    o, h, lo, c, v, a = (f[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close", "volume", "atr"))
    X = numeric_frame(f)
    bars = candles.bar_features(o, h, lo, c, v, a)
    flags = candles.detect(o, h, lo, c, a)
    wk = candles.weekly_context(f)
    mk = mkt.reindex(f.index).ffill()
    dates = f.index
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
    out: dict[str, Any] = {"symbol": sym}
    # ---------------- universe-wide pattern rows (every stock-day)
    tt = np.arange(MIN_BARS, n - max(FWD) - 1)
    if tt.size:
        a_t = a[tt]
        uni = {"date": dates[tt].to_numpy(), "vol_ratio": bars["vol_ratio"][tt].astype(np.float32),
               "w_up": wk["w_up_trend"].to_numpy(float)[tt].astype(np.float32)}
        for hz in FWD:
            uni[f"fwd{hz}"] = ((c[tt + hz] - o[tt + 1]) / a_t).astype(np.float32)
        uni["flags"] = np.vstack([flags[k][tt] for k in candles.PATTERN_IDS]).T
        out["uni"] = uni
    if not sig:
        return out
    ts = np.array([x[0] for x in sig])
    kinds = np.array([x[1] for x in sig])
    ref = np.array([x[2] for x in sig])
    at = a[ts]
    fw = ts[:, None] + np.arange(1, H + 1)[None, :]
    O, Hh, L, C = o[fw], h[fw], lo[fw], c[fw]
    res = {k: np.empty(len(ts)) for k in ("entry", "stop0", "risk", "size", "r", "ret", "exit_day")}
    res["ok"] = np.zeros(len(ts), dtype=bool)
    stop_day = np.full((len(ts), H), np.nan)
    open_after = np.zeros((len(ts), H), dtype=bool)
    tgt = np.full(len(ts), np.inf)
    for kind in set(kinds):
        m = kinds == kind
        p = live_paths(O[m], Hh[m], L[m], C[m], ref[m], at[m], kind, slip)
        for k in res:
            res[k][m] = p[k]
        stop_day[m], open_after[m], tgt[m] = p["stop_day"], p["open_after"], p["tgt"]
    trades = pd.DataFrame({"symbol": sym, "setup": kinds, "date": dates[ts], "t": ts, "ref": ref, "atr": at, **res})
    out["trades"] = trades
    out["fwd_ohlc"] = np.stack([O, Hh, L, C], axis=1).astype(np.float32)   # (m, 4, H)
    out["stop_day"] = stop_day.astype(np.float32)
    out["tgt"] = tgt
    # ---------------- trade-days (k = 0..H-2 in 0-based days; open at the close of day k)
    ii, kk = np.nonzero(open_after[:, : H - 1] & res["ok"][:, None])
    if ii.size == 0:
        return out
    entry, risk, size = res["entry"][ii], res["risk"][ii], res["size"][ii]
    sell = O[ii, kk + 1] * (1 - slip)
    cv_r = res["r"][ii] - (sell - entry) / risk
    cv_eq = (res["ret"][ii] - (sell / entry - 1)) * size
    hi_run = np.maximum.accumulate(Hh, axis=1)
    lo_run = np.minimum.accumulate(L, axis=1)
    day_ix = ts[ii] + 1 + kk                                   # bar index of day k's close
    close_k = c[day_ix]
    st_next = stop_day[ii, kk + 1]
    stop0 = res["stop0"][ii]
    pos_hi = np.array([int(np.argmax(Hh[i, : k + 1])) for i, k in zip(ii, kk)])
    td = {"sig": ii.astype(np.int32), "k": (kk + 1).astype(np.int16), "cv_r": cv_r.astype(np.float32),
          "cv_eq": cv_eq.astype(np.float32),
          "s_day": (kk + 1).astype(np.float32), "s_open_r": ((close_k - entry) / risk).astype(np.float32),
          "s_mfe_r": ((hi_run[ii, kk] - entry) / risk).astype(np.float32),
          "s_mae_r": ((lo_run[ii, kk] - entry) / risk).astype(np.float32),
          "s_dist_stop_atr": ((close_k - st_next) / at[ii]).astype(np.float32),
          "s_locked_r": ((st_next - entry) / risk).astype(np.float32),
          "s_trail_on": (st_next > stop0 + 1e-9).astype(np.float32),
          "s_days_since_high": (kk - pos_hi).astype(np.float32),
          "s_ret_since_entry_atr": ((close_k - entry) / at[ii]).astype(np.float32)}
    td["s_giveback_r"] = td["s_mfe_r"] - td["s_open_r"]
    for hz in (1, 3, 5):
        j = np.minimum(day_ix + hz, n - 1)
        td[f"fwd{hz}_atr"] = np.where(day_ix + hz < n, (c[j] - close_k) / at[ii], np.nan).astype(np.float32)
    Xv = X.to_numpy()
    feat = {col: Xv[day_ix, j] for j, col in enumerate(X.columns)}
    for col in DELTA_COLS:
        if f"f_{col}" in X.columns:
            jx = X.columns.get_loc(f"f_{col}")
            feat[f"x_{col}"] = (Xv[day_ix, jx] - Xv[ts[ii], jx]).astype(np.float32)
    for k_, arr in bars.items():
        feat[f"b_{k_}"] = np.asarray(arr, dtype=np.float32)[day_ix]
    for k_ in candles.PATTERN_IDS:
        feat[f"p_{k_}"] = flags[k_][day_ix].astype(np.float32)
    for col in wk.columns:
        feat[col] = wk[col].to_numpy(float)[day_ix].astype(np.float32)
    for col in mk.columns:
        feat[col] = mk[col].to_numpy(float)[day_ix].astype(np.float32)
    td.update(feat)
    out["td"] = pd.DataFrame(td)
    out["td_t"] = day_ix
    if hourly is not None and not hourly.empty and raw is not None:
        out["uni_h"] = universe_hourly(f, raw, hourly)
    # ---------------- hourly (2024-01.. only)
    if hourly is not None and not hourly.empty and raw is not None:
        out["hourly"] = hourly_trade_days(f, raw, hourly, trades, out["td"], day_ix, O, stop_day, tgt, slip)
    return out


def universe_hourly(f: pd.DataFrame, raw: pd.DataFrame, hb: pd.DataFrame) -> dict[str, np.ndarray] | None:
    """Every hourly bar of one stock: pattern flags + the move from the NEXT bar's open to the close of that session,
    of the next session and of 3 sessions later, in the stock's daily ATR (of the previous day). Dividend-adjusted scale."""
    hf = candles.hourly_flags(hb)
    if hf.empty:
        return None
    hday = pd.DatetimeIndex(pd.to_datetime(np.asarray(hb.index.date)))
    di = f.index.get_indexer(hday)
    ok = di >= 1
    factor = (raw["adj_close"] / raw["close"]).reindex(hday).to_numpy(float)
    ho = hb["open"].to_numpy(float) * factor
    c, a, o = f["close"].to_numpy(float), f["atr"].to_numpy(float), f["open"].to_numpy(float)
    n = len(hb)
    nxt = np.r_[ho[1:], np.nan]
    last = hf["last_of_day"].to_numpy(bool)
    nxt_open = np.where(last, np.where(di + 1 < len(o), o[np.minimum(di + 1, len(o) - 1)], np.nan), nxt)
    base_i = np.where(last, di + 1, di)         # the session the entry falls in
    out = {"ts": hb.index.tz_convert("America/New_York").tz_localize(None).to_numpy(),
           "flags": hf[candles.HOURLY_IDS].to_numpy(bool), "vol_ratio": hf["vol_ratio"].to_numpy(np.float32)}
    a_prev = np.where(ok, a[np.maximum(di - 1, 0)], np.nan)
    for tag, plus in (("d0", 0), ("d1", 1), ("d3", 3)):
        j = base_i + plus
        val = np.where(ok & (j < len(c)), (c[np.minimum(j, len(c) - 1)] - nxt_open) / a_prev, np.nan)
        out[tag] = val.astype(np.float32)
    up = np.where(ok, c[np.maximum(di - 1, 0)] > f["ema21"].to_numpy(float)[np.maximum(di - 1, 0)], False)
    out["daily_up"] = up
    return out if n else None


def hourly_trade_days(f: pd.DataFrame, raw: pd.DataFrame, hb: pd.DataFrame, trades: pd.DataFrame, td: pd.DataFrame,
                      day_ix: np.ndarray, O: np.ndarray, stop_day: np.ndarray, tgt: np.ndarray, slip: float) -> dict[str, Any]:
    """Hourly patterns inside held trades. For each trade-day: which hourly patterns fired during that session while
    the trade was open (columns hd_<id>[@hv|@lv], for the daily-decision mode), plus end-of-day hourly features; and
    per trade the first hourly firing of each pattern with the continuation value of selling at the next hour's open
    (intraday mode). Hourly prices are put on the daily (dividend-adjusted) scale with that day's adj_close / close."""
    hf = candles.hourly_flags(hb)
    if hf.empty:
        return {}
    hday = pd.DatetimeIndex(pd.to_datetime(np.asarray(hb.index.date)))
    factor = (raw["adj_close"] / raw["close"]).reindex(hday).to_numpy(float)
    ho, hh, hl, hc = (hb[k].to_numpy(float) * factor for k in ("open", "high", "low", "close"))
    ids = candles.HOURLY_IDS
    flagm = hf[ids].to_numpy(bool)
    vr = hf["vol_ratio"].to_numpy(float)
    hv, lv = vr >= candles.HV, vr <= candles.LV
    # day -> positions of its hourly bars
    day_keys = hday.to_numpy()
    uniq, first = np.unique(day_keys, return_index=True)
    order = np.argsort(first)
    uniq, first = uniq[order], first[order]
    last = np.r_[first[1:], len(day_keys)]
    pos = {pd.Timestamp(d): (s, e) for d, s, e in zip(uniq, first, last)}
    dates = f.index
    ema21 = f["ema21"].to_numpy(float)
    c = f["close"].to_numpy(float)
    hcols = [f"hd_{k}{tag}" for k in ids for tag in ("", "@hv", "@lv")]
    HD = np.zeros((len(td), len(hcols)), dtype=np.float32)
    HD[:] = np.nan
    extra = {k: np.full(len(td), np.nan, dtype=np.float32) for k in ("h_last_clv", "h_close_vs_vwap_atr", "h_max_vol_ratio",
                                                                       "h_low_vs_prevlow_atr", "h_rsi14")}
    hr = pd.Series(hc)
    delta = hr.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    hrsi = (100 - 100 / (1 + gain / loss.where(loss > 0))).to_numpy()
    sig = td["sig"].to_numpy()
    kk = td["k"].to_numpy() - 1
    setups = trades["setup"].to_numpy()
    entry = trades["entry"].to_numpy()
    risk = trades["risk"].to_numpy()
    size = trades["size"].to_numpy()
    ret_final = trades["ret"].to_numpy()
    r_final = trades["r"].to_numpy()
    ref = trades["ref"].to_numpy()
    first_fire = {k: np.full(len(trades), np.nan) for k in [f"{i}{t}" for i in ids for t in ("", "@hv", "@lv")]}
    first_fire.update({f"{i}|{ctx}": np.full(len(trades), np.nan) for i in MTF_HOURLY for ctx in ("up", "dn")})
    fire_day = {k: np.full(len(trades), -1) for k in first_fire}
    fire_cv_r = {k: np.full(len(trades), np.nan) for k in first_fire}
    # walk the trade-days in order (sorted by sig, k) so the first firing per trade is found
    for row in range(len(td)):
        d = pd.Timestamp(dates[day_ix[row]])
        se = pos.get(d)
        if se is None:
            continue
        s_, e_ = se
        i = sig[row]
        k = kk[row]
        st_k = stop_day[i, k]
        # bars of this session while the trade was open: before any intraday stop / target touch
        lows, highs = hl[s_:e_], hh[s_:e_]
        hit = (lows <= st_k) | (highs >= tgt[i])
        # the trade is open at the close (it is a trade-day), so hits can only be mismatches between hourly and daily
        # bars (rare); ignore them
        valid = np.ones(e_ - s_, dtype=bool)
        if k == 0 and setups[i] == "squeeze":
            filled_at = np.flatnonzero(highs >= ref[i])
            valid[: filled_at[0] if filled_at.size else len(valid)] = False
        fm = flagm[s_:e_] & valid[:, None]
        anyf = fm.any(0)
        anyhv = (fm & hv[s_:e_, None]).any(0)
        anylv = (fm & lv[s_:e_, None]).any(0)
        HD[row] = np.column_stack([anyf, anyhv, anylv]).ravel().astype(np.float32)
        j = e_ - 1
        extra["h_last_clv"][row] = (hc[j] - hl[j]) / (hh[j] - hl[j]) if hh[j] > hl[j] else np.nan
        hatr = hf["atr"].to_numpy()[j]
        extra["h_close_vs_vwap_atr"][row] = (hc[j] - hf["vwap"].to_numpy()[j] * factor[j]) / hatr if hatr > 0 else np.nan
        extra["h_max_vol_ratio"][row] = np.nanmax(vr[s_:e_]) if np.isfinite(vr[s_:e_]).any() else np.nan
        prev_i = day_ix[row] - 1
        extra["h_low_vs_prevlow_atr"][row] = (np.min(hl[s_:e_]) - f["low"].to_numpy(float)[prev_i]) / f["atr"].to_numpy(float)[prev_i]
        extra["h_rsi14"][row] = hrsi[j]
        # intraday mode: first firing -> sell at the next hour's open (last hour -> next session's open)
        up_ctx = c[day_ix[row] - 1] > ema21[day_ix[row] - 1]
        for jj in np.flatnonzero(fm.any(1)):
            nxt = ho[s_ + jj + 1] if jj + 1 < e_ - s_ else (O[i, k + 1] if k + 1 < O.shape[1] else np.nan)
            if not np.isfinite(nxt):
                continue
            sell = nxt * (1 - slip)
            cvr = r_final[i] - (sell - entry[i]) / risk[i]
            cve = (ret_final[i] - (sell / entry[i] - 1)) * size[i]
            for q in np.flatnonzero(fm[jj]):
                pid = ids[q]
                keys = [pid] + ([f"{pid}@hv"] if hv[s_ + jj] else []) + ([f"{pid}@lv"] if lv[s_ + jj] else [])
                if pid in MTF_HOURLY:
                    keys.append(f"{pid}|{'up' if up_ctx else 'dn'}")
                for key in keys:
                    if not np.isfinite(first_fire[key][i]):
                        first_fire[key][i] = cve
                        fire_cv_r[key][i] = cvr
                        fire_day[key][i] = k
    return {"hd": pd.DataFrame(HD, columns=hcols), "extra": pd.DataFrame(extra), "first_cv_eq": first_fire,
            "first_cv_r": fire_cv_r, "first_day": fire_day}


# ====================================================================== statistics
def _t_by_date(x: np.ndarray, dcode: np.ndarray, nd: int) -> tuple[float, float, int]:
    """(mean of per-date means, t, number of dates) with dates as clusters."""
    if x.size < 6:
        return float("nan"), float("nan"), 0
    s = np.bincount(dcode, weights=x, minlength=nd)
    cnt = np.bincount(dcode, minlength=nd)
    ok = cnt > 0
    m = s[ok] / cnt[ok]
    if m.size < 4:
        return float(x.mean()), float("nan"), int(m.size)
    sd = m.std(ddof=1)
    return float(x.mean()), float(m.mean() / (sd / math.sqrt(m.size))) if sd > 0 else float("nan"), int(m.size)


def first_fire(mask: np.ndarray, sig: np.ndarray, n_trades: int) -> np.ndarray:
    """Row index of the first True per trade (rows sorted by sig, k), -1 if none."""
    out = np.full(n_trades, -1)
    idx = np.flatnonzero(mask)
    if idx.size:
        s = sig[idx]
        keep = np.r_[True, s[1:] != s[:-1]]
        out[s[keep]] = idx[keep]
    return out


class Book:
    """Trades + trade-days of ONE setup, with period masks and fast paired statistics."""

    def __init__(self, trades: pd.DataFrame, td: pd.DataFrame, periods: dict[str, tuple[str, str]] = PERIODS):
        self.trades = trades.reset_index(drop=True)
        self.td = td.reset_index(drop=True)
        self.n = len(self.trades)
        d = pd.to_datetime(self.trades["date"])
        codes, uniq = pd.factorize(d, sort=True)
        self.dcode, self.nd = codes.astype(np.int64), len(uniq)
        self.pmask = {p: ((d >= pd.Timestamp(lo_)) & (d <= pd.Timestamp(hi_))).to_numpy() for p, (lo_, hi_) in periods.items()}
        self.sig = self.td["sig"].to_numpy()
        self.cv_eq = self.td["cv_eq"].to_numpy(np.float64)
        self.cv_r = self.td["cv_r"].to_numpy(np.float64)

    def delta_exit(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Paired delta (account fraction, R) of 'sell at the next open at the first firing' + fired flag."""
        ff = first_fire(mask, self.sig, self.n)
        fired = ff >= 0
        d_eq = np.zeros(self.n)
        d_r = np.zeros(self.n)
        d_eq[fired] = -self.cv_eq[ff[fired]]
        d_r[fired] = -self.cv_r[ff[fired]]
        return d_eq, d_r, fired

    def stats(self, d_eq: np.ndarray, d_r: np.ndarray, fired: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p, m in self.pmask.items():
            if m.sum() < 10:
                out[p] = {"n": int(m.sum())}
                continue
            mean, t, nd = _t_by_date(d_eq[m], self.dcode[m], self.nd)
            out[p] = {"n": int(m.sum()), "d_eq": mean * 100, "t": t, "d_r": float(d_r[m].mean()),
                      "fired": float(fired[m].mean()),
                      "d_r_fired": float(d_r[m & fired].mean()) if (m & fired).sum() >= 5 else None}
        conf = self.pmask.get("hold", np.zeros(self.n, bool)) | self.pmask.get("oos", np.zeros(self.n, bool))
        if conf.sum() > 10:
            mean, t, _ = _t_by_date(d_eq[conf], self.dcode[conf], self.nd)
            out["pooled"] = {"d_eq": mean * 100, "t": t}
        return out

    def date_means(self, d_eq: np.ndarray, which: np.ndarray) -> np.ndarray:
        s = np.bincount(self.dcode[which], weights=d_eq[which], minlength=self.nd)
        cnt = np.bincount(self.dcode[which], minlength=self.nd)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def gate(rules: dict[str, dict[str, Any]], is_key: str = "is", conf_keys: tuple[str, ...] = ("hold", "oos"),
         pooled_key: str = "pooled") -> dict[str, Any]:
    """Apply the pre-registered gate to a {rule: stats} dict of one setup."""
    from statistics import NormalDist
    cand = [k for k, v in rules.items() if (v.get(is_key, {}).get("d_eq") or -1) > 0 and (v.get(is_key, {}).get("t") or 0) >= 1.65]
    bar = max(2.5, float(NormalDist().inv_cdf(1 - 0.05 / max(1, len(cand)))))
    passed = []
    for k in cand:
        v = rules[k]
        if all((v.get(c, {}).get("d_eq") or -1) > 0 for c in conf_keys) and (v.get(pooled_key, {}).get("t") or 0) >= bar:
            passed.append(k)
    return {"n_rules": len(rules), "n_is_candidates": len(cand), "bar_t": bar, "validated": passed}


def maxt_permutation(cols: list[np.ndarray], n_perm: int = 500, seed: int = 11) -> tuple[np.ndarray, float]:
    """Max-t sign-flip permutation over rules on per-date mean deltas (dates x rules). Returns per-rule p and the 95%
    null quantile of the max t."""
    M = np.column_stack(cols).astype(np.float32)
    keep = np.isfinite(M).any(1)
    M = np.nan_to_num(M[keep])
    nd = M.shape[0]
    if nd < 10:
        return np.full(M.shape[1], np.nan), float("nan")
    sq = np.sqrt(nd)
    obs = M.mean(0) / (M.std(0, ddof=1) / sq + 1e-12)
    rng = np.random.default_rng(seed)
    mx = np.empty(n_perm)
    for i in range(n_perm):
        s = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=nd)[:, None]
        X = M * s
        mx[i] = (X.mean(0) / (X.std(0, ddof=1) / sq + 1e-12)).max()
    return (mx[None, :] >= obs[:, None]).mean(1), float(np.quantile(mx, 0.95))


# ====================================================================== rule catalogue
def pattern_masks(td: pd.DataFrame) -> dict[str, np.ndarray]:
    """Daily candle variants (46 x 3 volume states) + daily x weekly multi-timeframe (10 x 2)."""
    vr = td["b_vol_ratio"].to_numpy(float)
    flags = {k: td[f"p_{k}"].to_numpy() > 0.5 for k in candles.PATTERN_IDS}
    out = {f"candle:{k}": m for k, m in candles.with_volume(flags, vr).items()}
    wu = td["w_up_trend"].to_numpy(float)
    for k in MTF_DAILY:
        out[f"mtf:{k}|weekly_up"] = flags[k] & (wu == 1)
        out[f"mtf:{k}|weekly_down"] = flags[k] & (wu == 0)
    return out


def hourly_masks(td: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for col in td.columns:
        if col.startswith("hd_"):
            out[f"hourly_eod:{col[3:]}"] = td[col].to_numpy(float) > 0.5
    return out


def classic_masks(td: pd.DataFrame) -> dict[str, np.ndarray]:
    g = lambda k: td[k].to_numpy(float) if k in td else np.full(len(td), np.nan)  # noqa: E731
    up_day = g("b_ret_atr") > 0
    base = {
        "rsi6>=80": g("f_rsi6") >= 80, "rsi6>=90": g("f_rsi6") >= 90, "rsi14>=70": g("f_rsi14") >= 70,
        "rsi14>=80": g("f_rsi14") >= 80, "close_above_upper_band": g("f_bb_pct_b") >= 1.0,
        "far_above_upper_band": g("f_bb_pct_b") >= 1.2, "ext>=3atr_above_ema21": g("f_extension_atr") >= 3,
        "ext>=4atr_above_ema21": g("f_extension_atr") >= 4, "volume_climax_up": (g("b_vol_ratio") >= 3) & up_day,
        "macd_hist_turning_down": (g("d_macd_hist") > 0) & (g("d_macd_hist_3d") < 0),
        "rsi6<=20_while_held": g("f_rsi6") <= 20, "rsi14<=30_while_held": g("f_rsi14") <= 30,
        "close_below_lower_band": g("f_bb_pct_b") <= 0.0, "adx>=40": g("f_adx") >= 40,
    }
    out = {}
    pr = g("s_open_r") >= PROFIT_R
    for k, m in base.items():
        out[f"classic:{k}"] = m
        out[f"classic:{k}|in_profit"] = m & pr
    return out


def state_masks(td: pd.DataFrame) -> dict[str, np.ndarray]:
    k, r, mfe = td["k"].to_numpy(), td["s_open_r"].to_numpy(float), td["s_mfe_r"].to_numpy(float)
    dsh = td["s_days_since_high"].to_numpy(float)
    out = {}
    for day in (1, 2, 3, 5):
        for x in (-0.25, -0.5, -0.75):
            out[f"state:day{day}_open_r<={x}"] = (k == day) & (r <= x)
    for day in (3, 5, 7, 10):
        out[f"state:not_working_day{day}(<+0.25R)"] = (k == day) & (r < 0.25)
    for m_, g_ in ((1.0, 0.75), (1.5, 1.0), (2.0, 1.0), (2.0, 1.5), (3.0, 1.5)):
        out[f"state:giveback_after_{m_}R_to_{m_ - g_:g}R"] = (mfe >= m_) & (r <= m_ - g_)
    for d_ in (4, 5, 7):
        out[f"state:no_new_high_{d_}d(k>={d_})"] = (dsh >= d_) & (k >= d_)
    return out


# ====================================================================== study
def evaluate_masks(book: Book, masks: dict[str, np.ndarray], contexts: bool = True) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Exit-at-first-firing for every mask (and in-profit / in-loss variants). Returns stats and per-rule deltas."""
    r = book.td["s_open_r"].to_numpy(float)
    ctx = {"": None}
    if contexts:
        ctx = {"": None, "|in_profit": r >= PROFIT_R, "|in_loss": r <= LOSS_R}
    stats, deltas = {}, {}
    for name, m in masks.items():
        for tag, cm in ctx.items():
            mm = m if cm is None else (m & cm)
            if mm.sum() < 20:
                continue
            d_eq, d_r, fired = book.delta_exit(mm)
            stats[name + tag] = book.stats(d_eq, d_r, fired)
            stats[name + tag]["rows"] = int(mm.sum())
            deltas[name + tag] = d_eq
    return stats, deltas


def conditional_cv(book: Book, masks: dict[str, np.ndarray], periods: tuple[str, ...] = ("is", "hold", "oos")) -> dict[str, Any]:
    """Descriptive: mean CV (R) on trade-days where the mask holds vs the same day-of-trade without it."""
    k = book.td["k"].to_numpy()
    kb = np.select([k == 1, k == 2, k == 3, k <= 5, k <= 10], [1, 2, 3, 4, 6], 11)
    trade_p = {p: book.pmask[p][book.sig] for p in periods}
    out = {}
    cv = book.cv_r
    for name, m in masks.items():
        row = {}
        for p in periods:
            pm = trade_p[p]
            a_ = m & pm
            if a_.sum() < 30:
                continue
            # day-of-trade matched contrast: weight the without-mask mean by the mask's day distribution
            w = np.bincount(kb[a_], minlength=12).astype(float)
            mean_no = np.bincount(kb[pm & ~m], weights=cv[pm & ~m], minlength=12) / np.maximum(np.bincount(kb[pm & ~m], minlength=12), 1)
            row[p] = {"n": int(a_.sum()), "cv_r": float(cv[a_].mean()), "matched_cv_r": float((w * mean_no).sum() / w.sum()),
                      "fwd1": float(np.nanmean(book.td["fwd1_atr"].to_numpy(float)[a_])),
                      "fwd5": float(np.nanmean(book.td["fwd5_atr"].to_numpy(float)[a_]))}
        out[name] = row
    return out


def day_profile(book: Book) -> list[dict[str, Any]]:
    """Day 1 / 2 / 3 / 5 / 10 x open-R bucket: how many trades, the rest of the trade if held, chance to end red."""
    td = book.td
    k, r = td["k"].to_numpy(), td["s_open_r"].to_numpy(float)
    final_r = book.trades["r"].to_numpy(float)[book.sig]
    bins = [(-99, -0.75), (-0.75, -0.25), (-0.25, 0.25), (0.25, 1.0), (1.0, 2.0), (2.0, 99)]
    rows = []
    for day in (1, 2, 3, 5, 10):
        for lo_, hi_ in bins:
            m = (k == day) & (r > lo_) & (r <= hi_)
            if m.sum() < 30:
                continue
            rows.append({"day": day, "open_r": f"{lo_:g}..{hi_:g}", "n": int(m.sum()), "cv_r": float(book.cv_r[m].mean()),
                         "final_r": float(final_r[m].mean()), "p_final_below_0": float((final_r[m] < 0).mean()),
                         "share_of_day": float(m.sum() / max(1, (k == day).sum()))})
    return rows


def indicator_scan(book: Book, cols: list[str]) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """IC of every feature with CV by period + extreme-tail exit rules with in-sample thresholds."""
    td = book.td
    trade_p = {p: book.pmask[p][book.sig] for p in book.pmask}
    is_rows = trade_p["is"]
    ic: dict[str, Any] = {}
    masks: dict[str, np.ndarray] = {}
    thresholds: dict[str, dict[str, float]] = {}
    cv = book.cv_r
    for col in cols:
        x = td[col].to_numpy(np.float64)
        fin = np.isfinite(x)
        row = {}
        for p, pm in trade_p.items():
            m = pm & fin
            if m.sum() < 200 or np.nanstd(x[m]) == 0:
                continue
            row[p] = float(pd.Series(x[m]).rank().corr(pd.Series(cv[m]).rank()))
        ic[col] = row
        xi = x[is_rows & fin]
        if xi.size < 500 or np.unique(xi).size < 3:
            if xi.size >= 500 and np.unique(xi).size == 2 and set(np.unique(xi)) <= {0.0, 1.0} and xi.mean() < 0.5:
                masks[f"ind:{col}==1"] = x == 1
            continue
        q = {qq: float(np.quantile(xi, qq)) for qq in (0.05, 0.10, 0.90, 0.95)}
        thresholds[col] = q
        masks[f"ind:{col}>=q90"] = x >= q[0.90]
        masks[f"ind:{col}>=q95"] = x >= q[0.95]
        masks[f"ind:{col}<=q10"] = x <= q[0.10]
        masks[f"ind:{col}<=q05"] = x <= q[0.05]
    return ic, masks, thresholds


def ridge_policy(book: Book, cols: list[str], lam: float = 50.0) -> dict[str, Any]:
    """One pre-registered model per setup: ridge on rank-transformed features (in-sample trade-days only), exit at the
    first trade-day whose forecast CV is below a threshold chosen in sample."""
    td = book.td
    is_rows = book.pmask["is"][book.sig]
    X = td[cols].to_numpy(np.float64)
    med = np.nanmedian(X[is_rows], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med)
    # rank-transform with in-sample quantiles (robust to fat tails)
    Z = np.empty_like(X)
    for j in range(X.shape[1]):
        qs = np.quantile(X[is_rows, j], np.linspace(0, 1, 21))
        Z[:, j] = np.interp(X[:, j], qs, np.linspace(-1, 1, 21)) if np.ptp(qs) > 0 else 0.0
    y = np.clip(book.cv_eq, *np.quantile(book.cv_eq[is_rows], [0.01, 0.99]))
    Zi, yi = Z[is_rows], y[is_rows]
    mu = Zi.mean(0)
    A = (Zi - mu).T @ (Zi - mu) + lam * len(yi) / 1000 * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, (Zi - mu).T @ (yi - yi.mean()))
    pred = (Z - mu) @ beta + yi.mean()
    best, best_q = -np.inf, None
    for q in (0.01, 0.02, 0.05, 0.10, 0.20):
        th = np.quantile(pred[is_rows], q)
        d_eq, d_r, fired = book.delta_exit(pred <= th)
        mis = book.pmask["is"]
        if d_eq[mis].mean() > best:
            best, best_q = d_eq[mis].mean(), (q, th)
    q, th = best_q
    d_eq, d_r, fired = book.delta_exit(pred <= th)
    stats = book.stats(d_eq, d_r, fired)
    top = np.argsort(-np.abs(beta))[:15]
    return {"quantile": q, "threshold": float(th), "stats": stats, "lambda": lam,
            "top_weights": [{"feature": cols[j], "beta": float(beta[j])} for j in top], "deltas": d_eq}


def universe_patterns(uni: dict[str, np.ndarray]) -> dict[str, Any]:
    """What each daily pattern does next across ALL stock-days: forward move from the next open in ATR minus the same
    day's average over all stocks (market removed), by period, with t on per-date means."""
    dates = pd.to_datetime(uni["date"])
    codes, uq = pd.factorize(dates, sort=True)
    nd = len(uq)
    pm = {p: np.asarray((dates >= pd.Timestamp(lo_)) & (dates <= pd.Timestamp(hi_))) for p, (lo_, hi_) in PERIODS.items()}
    fwd = {}
    for hz in FWD:
        x = uni[f"fwd{hz}"].astype(np.float64)
        x = np.clip(x, -15, 15)
        fin = np.isfinite(x)
        s = np.bincount(codes[fin], weights=x[fin], minlength=nd)
        cnt = np.bincount(codes[fin], minlength=nd)
        fwd[hz] = np.where(fin, x - (s / np.maximum(cnt, 1))[codes], np.nan)
    flags = {k: uni["flags"][:, j] for j, k in enumerate(candles.PATTERN_IDS)}
    variants = candles.with_volume(flags, uni["vol_ratio"])
    wu = uni["w_up"]
    for k in MTF_DAILY:
        variants[f"{k}|weekly_up"] = flags[k] & (wu == 1)
        variants[f"{k}|weekly_down"] = flags[k] & (wu == 0)
    out = {}
    for name, m in variants.items():
        row = {"freq": float(m.mean())}
        for p, pmask in pm.items():
            mm = m & pmask
            if mm.sum() < 50:
                continue
            cell = {"n": int(mm.sum())}
            for hz in FWD:
                x = fwd[hz]
                ok = mm & np.isfinite(x)
                mean, t, _ = _t_by_date(x[ok], codes[ok], nd)
                cell[f"x{hz}"] = mean
                cell[f"t{hz}"] = t
            row[p] = cell
        out[name] = row
    return out


def universe_hourly_patterns(U: dict[str, np.ndarray]) -> dict[str, Any]:
    """What each hourly pattern does next across all stocks' hourly bars (2024-01..): move from the next bar's open to
    the close of that session (d0), of the next session (d1) and 3 sessions later (d3), in daily ATR, minus the average of all stocks at the
    same hour (market removed); by is_h / oos_h with t on per-hour means."""
    ts = pd.to_datetime(U["ts"])
    codes, uq = pd.factorize(ts, sort=True)
    nd = len(uq)
    pm = {p: np.asarray((ts >= pd.Timestamp(lo_)) & (ts <= pd.Timestamp(hi_) + pd.Timedelta(days=1))) for p, (lo_, hi_) in HPERIODS.items()}
    fwd = {}
    for k in ("d0", "d1", "d3"):
        x = np.clip(U[k].astype(np.float64), -15, 15)
        fin = np.isfinite(x)
        s_ = np.bincount(codes[fin], weights=x[fin], minlength=nd)
        cnt = np.bincount(codes[fin], minlength=nd)
        fwd[k] = np.where(fin, x - (s_ / np.maximum(cnt, 1))[codes], np.nan)
    flags = {k: U["flags"][:, j] for j, k in enumerate(candles.HOURLY_IDS)}
    var = candles.with_volume(flags, U["vol_ratio"])
    for k in MTF_HOURLY:
        var[f"{k}|daily_up"] = flags[k] & U["daily_up"]
        var[f"{k}|daily_down"] = flags[k] & ~U["daily_up"]
    out = {}
    for name, m in var.items():
        row = {"freq": float(m.mean())}
        for p, pmask in pm.items():
            mm = m & pmask
            if mm.sum() < 50:
                continue
            cell = {"n": int(mm.sum())}
            for k in ("d0", "d1", "d3"):
                ok = mm & np.isfinite(fwd[k])
                mean, t, _ = _t_by_date(fwd[k][ok], codes[ok], nd)
                cell[f"x_{k}"], cell[f"t_{k}"] = mean, t
            row[p] = cell
        out[name] = row
    return out


def hourly_rules(book: Book, H_first: dict[str, np.ndarray], periods: dict[str, tuple[str, str]] = HPERIODS) -> dict[str, Any]:
    """Intraday mode: first hourly firing -> sell at the next hour's open. Paired delta per trade = -CV."""
    d = pd.to_datetime(book.trades["date"])
    pmask = {p: ((d >= pd.Timestamp(lo_)) & (d <= pd.Timestamp(hi_))).to_numpy() for p, (lo_, hi_) in periods.items()}
    covered = pmask["is_h"] | pmask["oos_h"]
    out = {}
    for key, cv in H_first.items():
        fired = np.isfinite(cv)
        if (fired & covered).sum() < 20:
            continue
        d_eq = np.where(fired, -np.nan_to_num(cv), 0.0)
        row = {}
        for p, m in pmask.items():
            if m.sum() < 10:
                continue
            mean, t, _ = _t_by_date(d_eq[m], book.dcode[m], book.nd)
            row[p] = {"n": int(m.sum()), "d_eq": mean * 100, "t": t, "fired": float(fired[m].mean())}
        row["pooled"] = row.get("oos_h", {})
        out[f"hourly:{key}"] = row
    return out

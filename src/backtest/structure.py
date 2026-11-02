"""Structure-based stops / targets / entry filters versus the live ATR exits (research only).

Question (Neo, 2026-11-02): the live exits are ATR-based. Do exits built from market structure - swing
pivots, moving-average support, volume-profile nodes, Fibonacci levels, gaps, round numbers, channel and
trend-following stops, per-stock bounce habits - beat them for the three live setups (crash_rebound,
squeeze, breakout)?

Design (same rules as the earlier studies)
* Signals: every setup episode (first day active) plus a random-entry sample (every 10th session of every
  stock). Entry at the first open after the signal (squeeze: the buy-stop at 20d high + 0.1 ATR, only if
  it trades). Slippage 10 bps per side. A bar that touches stop and target counts as a stop; gaps fill
  at the open. 20 session horizon.
* Every level is computed from data up to the signal close (no look-ahead). Indicators that need a
  forward path (trailing stops, Parabolic SAR, entry-anchored VWAP) use only bars before the one tested.
* Reference exit = the exit the strategy lab chose for that setup (crash_rebound 1.5 ATR stop + 3 ATR
  trail; breakout 1.0 ATR stop + 3 ATR trail; squeeze 2.0 ATR stop + 3R target). Every method is compared
  PAIRED with that reference on the same trades, so the entry (and the market) cancels out.
* Unit: account return per trade with the live sizing (1% risk, position capped at 25% of the account):
  eq = return x min(0.01 / stop-distance-%, 0.25). R is reported too.
* Periods (by signal date): is = 2020-01..2024-06 (selection), oos = 2024-07.. (after selection),
  hold = 2010..2019 (never used to design anything; survivorship-biased universe, but the comparison is
  paired), gfc = 2005..2009 (report only).
* Multiple testing: ~80 methods x 3 setups. A method is only considered if it was chosen on the in-sample
  period alone (paired delta > 0, t >= 1.65); it then has to be positive in hold AND oos and its pooled
  confirmation t (hold + oos) has to clear a Bonferroni bar over the number of IS candidates (and >= 2.5).
  A max-t sign-flip permutation (White's reality check style) on the pooled confirmation is reported as a
  second, selection-free guard.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

from src.scanner.plan import plan_for_setup
from src.technical.structure import swing_points

H = 20
SETUPS = ("crash_rebound", "squeeze", "breakout")
BASE_STEP = 10
MIN_BARS = 260
MAS = (21, 50, 100, 200, 250)
BASE_EXIT = {
    "crash_rebound": {"stop_atr": 1.5, "trail": 3.0, "tgt_r": None},
    "breakout": {"stop_atr": 1.0, "trail": 3.0, "tgt_r": None},
    "squeeze": {"stop_atr": 2.0, "trail": None, "tgt_r": 3.0},
}
PERIODS = {"gfc": ("2000-01-01", "2009-12-31"), "hold": ("2010-01-01", "2019-12-31"),
           "is": ("2020-01-01", "2024-06-30"), "oos": ("2024-07-01", "2100-01-01")}


# ====================================================================== level helpers (per stock, causal)
def _psar_free_levels(f: pd.DataFrame) -> dict[str, np.ndarray]:
    """Series known at each close (index j = value known at the close of bar j)."""
    o, h, lo, c, v = (f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
    a = f["atr"].to_numpy(dtype=float)
    cs = pd.Series(c, index=f.index)
    ema = {m: cs.ewm(span=m, adjust=False, min_periods=m).mean().to_numpy() for m in (9, 21, 50, 100, 200, 250)}
    sma20 = cs.rolling(20).mean()
    sd20 = cs.rolling(20).std(ddof=0)
    ls, hs = pd.Series(lo, index=f.index), pd.Series(h, index=f.index)
    lv: dict[str, np.ndarray] = {f"ema{m}": ema[m] for m in ema}
    lv["sma20"] = sma20.to_numpy()
    lv["bb_lo"] = (sma20 - 2 * sd20).to_numpy()
    lv["bb_up"] = (sma20 + 2 * sd20).to_numpy()
    lv["kelt_lo"] = ema[21] - 2 * a
    lv["kelt_up"] = ema[21] + 2 * a
    lv["don10_low"] = ls.rolling(10).min().to_numpy()
    lv["don20_low"] = ls.rolling(20).min().to_numpy()
    lv["don55_low"] = ls.rolling(55).min().to_numpy()
    lv["low5"] = ls.rolling(5).min().to_numpy()
    lv["chand"] = hs.rolling(22).max().to_numpy() - 3 * a
    lv["st_lower"] = (h + lo) / 2 - 3 * a                      # Supertrend(10, 3) lower band (ratcheted per trade)
    lv["prev_low"] = lo
    lv["swing_low"] = f["swing_low_last"].to_numpy(dtype=float)
    lv["swing_low_prev"] = f["swing_low_prev"].to_numpy(dtype=float)
    lv["swing_high"] = f["swing_high_last"].to_numpy(dtype=float)
    lv["high20"] = f["high20"].to_numpy(dtype=float)
    lv["high60"] = f["high60"].to_numpy(dtype=float)
    lv["high252"] = f["high252"].to_numpy(dtype=float)
    lv["prev_high"] = h
    lv["high5"] = hs.rolling(5).max().to_numpy()
    # anchored VWAP from the last confirmed swing low / from the 252-session low
    tp = (h + lo + c) / 3
    pv = np.nan_to_num(tp * v)
    vv = np.nan_to_num(v)
    cpv, cv = np.cumsum(pv), np.cumsum(vv)
    n = len(f)
    is_high, is_low = swing_points(f["high"], f["low"], 3)
    piv = np.where(is_low.to_numpy(dtype=bool))[0]
    anchor = np.full(n, -1)
    if piv.size:
        conf = piv + 3
        pos = np.searchsorted(conf, np.arange(n), side="right") - 1
        anchor = np.where(pos >= 0, piv[np.maximum(pos, 0)], -1)

    def avwap(anc: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan)
        ok = anc >= 0
        j = np.flatnonzero(ok)
        a0 = anc[j]
        base_pv = np.where(a0 > 0, cpv[np.maximum(a0 - 1, 0)], 0.0)
        base_v = np.where(a0 > 0, cv[np.maximum(a0 - 1, 0)], 0.0)
        den = cv[j] - base_v
        out[j] = np.where(den > 0, (cpv[j] - base_pv) / np.where(den > 0, den, 1), np.nan)
        return out
    lv["avwap_swing"] = avwap(anchor)
    anc252 = np.full(n, -1)
    if n > 252:
        win = swv(lo, 252)
        anc252[251:] = np.arange(n - 251) + win.argmin(1)
    lv["avwap_52wl"] = avwap(anc252)
    # recent up-gap bottom (gap-fill support) and last down-gap top (gap-fill resistance)
    gap_up = np.full(n, np.nan)
    gap_dn = np.full(n, np.nan)
    gu = np.r_[False, lo[1:] > h[:-1]]
    gd = np.r_[False, h[1:] < lo[:-1]]
    idx = np.arange(n)
    last_gu = np.maximum.accumulate(np.where(gu, idx, -1))
    last_gd = np.maximum.accumulate(np.where(gd, idx, -1))
    okg = (last_gu >= 0) & (idx - last_gu <= 20)
    gap_up[okg] = h[np.maximum(last_gu[okg] - 1, 0)]
    okd = (last_gd >= 0) & (idx - last_gd <= 60)
    gap_dn[okd] = lo[np.maximum(last_gd[okd] - 1, 0)]
    lv["gap_up_bottom"] = gap_up
    lv["gap_dn_top"] = gap_dn
    lv["gap_up_age"] = np.where(okg, idx - last_gu, np.nan)
    return lv


def _bounce_stats(f: pd.DataFrame, ema: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-stock, per-MA bounce record known at each close: shrunk bounce rate and event count.
    Event at u: price came from above, the low reached MA + 0.25 ATR and the close held >= MA - 0.5 ATR.
    Success: the next 5 lows stayed above MA - 0.5 ATR and the close 5 sessions later is above the MA.
    Only events with u + 5 <= j are counted at j (no look-ahead)."""
    lo, c = f["low"].to_numpy(dtype=float), f["close"].to_numpy(dtype=float)
    a = f["atr"].to_numpy(dtype=float)
    n = len(f)
    out: dict[str, np.ndarray] = {}
    for m in MAS:
        ma = ema[f"ema{m}"]
        prev_above = np.r_[False, c[:-1] > ma[:-1]]
        ev = prev_above & (lo <= ma + 0.25 * a) & (c >= ma - 0.5 * a) & np.isfinite(ma) & np.isfinite(a)
        succ = np.zeros(n, dtype=bool)
        if n > 6:
            fut_min = np.full(n, np.nan)
            fut_min[: n - 5] = swv(lo[1:], 5)[: n - 5].min(1)
            fut_c = np.full(n, np.nan)
            fut_c[: n - 5] = c[5:]
            succ = ev & (fut_min > ma - 0.5 * a) & (fut_c > ma)
        e = np.zeros(n)
        s = np.zeros(n)
        e[5:] = np.cumsum(ev.astype(float))[: n - 5]
        s[5:] = np.cumsum(succ.astype(float))[: n - 5]
        out[f"bn{m}"] = e
        out[f"br{m}"] = (s + 2.0) / (e + 4.0)
    return out


def _vp(h: np.ndarray, lo: np.ndarray, v: np.ndarray, bins: int = 48) -> dict[str, Any] | None:
    """Volume profile of a window (same bin logic as analyst.techniques.volume_profile, vectorised)."""
    top, bot = float(np.nanmax(h)), float(np.nanmin(lo))
    if not top > bot:
        return None
    edges = np.linspace(bot, top, bins + 1)
    span = np.maximum(h - lo, 1e-9)[:, None]
    ov = np.clip(np.minimum(edges[None, 1:], h[:, None]) - np.maximum(edges[None, :-1], lo[:, None]), 0, None) / span
    vol = (ov * np.nan_to_num(v)[:, None]).sum(0)
    mids = (edges[:-1] + edges[1:]) / 2
    poc_i = int(vol.argmax())
    total, acc = vol.sum(), vol[poc_i]
    lo_i = hi_i = poc_i
    while acc < 0.7 * total and (lo_i > 0 or hi_i < bins - 1):
        down = vol[lo_i - 1] if lo_i > 0 else -1
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1
        if up >= down:
            hi_i += 1
            acc += vol[hi_i]
        else:
            lo_i -= 1
            acc += vol[lo_i]
    mean = vol.mean()
    mid = vol[1:-1]
    hvn_i = np.flatnonzero((mid >= 1.3 * mean) & (mid >= vol[:-2]) & (mid >= vol[2:])) + 1
    lvn = np.flatnonzero(vol <= 0.45 * mean)
    zones, cur = [], []
    for i in lvn:
        if cur and i == cur[-1] + 1:
            cur.append(i)
        else:
            if len(cur) >= 2:
                zones.append(cur)
            cur = [i]
    if len(cur) >= 2:
        zones.append(cur)
    return {"poc": float(mids[poc_i]), "val": float(edges[lo_i]), "vah": float(edges[hi_i + 1]),
            "hvn": mids[hvn_i], "vac": [(float(edges[z[0]]), float(edges[z[-1] + 1])) for z in zones]}


def _round_levels(px: float) -> tuple[float, float]:
    """(nearest round number strictly below, strictly above) a price: step = one tenth of the price's power of ten."""
    step = 10 ** (math.floor(math.log10(px)) - 1) if px > 0 else 1.0
    below = math.floor(px / step - 1e-9) * step
    return below, below + step


# ====================================================================== collection
STOP_KEYS = ["swing_low", "swing_low2", "low_prev", "low5", "don10", "don20", "don55", "ema21", "ema50", "ema100", "ema200",
             "ema250", "near_ma", "bb_lo", "kelt_lo", "avwap_swing", "avwap_52wl", "vp_poc120", "vp_hvn120", "vp_val120",
             "vp_hvn250", "vp_poc250", "fib382", "fib500", "fib618", "gap_up", "round", "pct5", "pct8", "chand", "st_lower",
             "zone", "bounce_ma", "structural"]
TGT_KEYS = ["high20", "high60", "high252", "swing_high", "fib1272", "fib1618", "hvn_above", "poc_above", "vac_top", "bb_up",
            "kelt_up", "round_up", "gap_dn_fill", "ema200_up", "meas_move", "avwap_52wh", "prev_high", "high5"]
SER_KEYS = ["ema9", "ema21", "ema50", "sma20", "don10_low", "don20_low", "chand", "st_lower", "kelt_lo", "bb_lo", "swing_low",
            "avwap_swing"]


def collect(features: dict[str, pd.DataFrame], setups: dict[str, pd.DataFrame], settings: dict[str, Any],
            progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    cfg = settings["scanner"]
    meta, S, T, SER, FL = [], {k: [] for k in STOP_KEYS}, {k: [] for k in TGT_KEYS}, {k: [] for k in SER_KEYS}, {}
    OHLCV = {k: [] for k in "OHLCV"}
    FILT: dict[str, list] = {}
    for si, (sym, f) in enumerate(features.items()):
        st = setups[sym]
        n = len(f)
        o, h, lo, c, v, a = (f[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close", "volume", "atr"))
        lv = _psar_free_levels(f)
        bs = _bounce_stats(f, lv)
        plans = {s: plan_for_setup(f, s, cfg) for s in SETUPS}
        sig_t: list[tuple[int, str, float]] = []
        for s in SETUPS:
            mask = st[f"setup_{s}"].to_numpy(dtype=bool)
            starts = np.flatnonzero(mask & ~np.concatenate([[False], mask[:-1]]))
            for t in starts[(starts >= MIN_BARS) & (starts + H + 1 < n)]:
                ent = float(plans[s]["entry"].iat[t])
                if np.isfinite(a[t]) and a[t] > 0 and np.isfinite(ent):
                    sig_t.append((int(t), s, ent))
        for t in range(MIN_BARS + si % BASE_STEP, n - H - 1, BASE_STEP):
            if np.isfinite(a[t]) and a[t] > 0:
                sig_t.append((t, "baseline", float(c[t])))
        if not sig_t:
            continue
        ts = np.array([x[0] for x in sig_t])
        piv_low = np.flatnonzero(swing_points(f["high"], f["low"], 3)[1].to_numpy(dtype=bool))
        vp_cache: dict[tuple[int, int], dict | None] = {}
        for t, setup, ref in sig_t:
            at = a[t]
            row: dict[str, float] = {}
            # ---- stop candidates (price levels)
            sl, sl2 = lv["swing_low"][t], lv["swing_low_prev"][t]
            row["swing_low"] = sl - 0.1 * at
            row["swing_low2"] = np.nanmin([sl, sl2]) - 0.1 * at if np.isfinite(sl) else np.nan
            row["low_prev"] = lo[t] - 0.1 * at
            row["low5"] = lv["low5"][t] - 0.25 * at
            row["don10"] = lv["don10_low"][t] - 0.1 * at
            row["don20"] = lv["don20_low"][t] - 0.1 * at
            row["don55"] = lv["don55_low"][t] - 0.1 * at
            for m in (21, 50, 100, 200, 250):
                row[f"ema{m}"] = lv[f"ema{m}"][t] - 0.5 * at
            below = [lv[f"ema{m}"][t] for m in MAS if lv[f"ema{m}"][t] < ref - 0.5 * at]
            row["near_ma"] = (max(below) - 0.5 * at) if below else np.nan
            row["bb_lo"] = lv["bb_lo"][t]
            row["kelt_lo"] = lv["kelt_lo"][t]
            row["avwap_swing"] = lv["avwap_swing"][t] - 0.25 * at
            row["avwap_52wl"] = lv["avwap_52wl"][t] - 0.25 * at
            row["pct5"], row["pct8"] = ref * 0.95, ref * 0.92
            row["chand"], row["st_lower"] = lv["chand"][t], lv["st_lower"][t]
            row["gap_up"] = lv["gap_up_bottom"][t] - 0.1 * at
            rb, ru = _round_levels(ref)
            row["round"] = rb - 0.1 * at
            row["round_up"] = ru
            row["structural"] = float(plans[setup]["stop"].iat[t]) if setup != "baseline" else np.nan
            # ---- volume profiles
            for w in (120, 250):
                key = (t, w)
                vp_cache[key] = _vp(h[t - w + 1:t + 1], lo[t - w + 1:t + 1], v[t - w + 1:t + 1]) if t >= w - 1 else None
            for w, tag in ((120, "120"), (250, "250")):
                vp = vp_cache[(t, w)]
                if vp is None:
                    for k in (f"vp_poc{tag}", f"vp_hvn{tag}", f"vp_val{tag}"):
                        row[k] = np.nan
                    continue
                row[f"vp_poc{tag}"] = vp["poc"] - 0.25 * at if vp["poc"] < ref - 0.5 * at else np.nan
                hb = [x for x in vp["hvn"] if x < ref - 0.5 * at]
                row[f"vp_hvn{tag}"] = (max(hb) - 0.25 * at) if hb else np.nan
                row[f"vp_val{tag}"] = vp["val"] - 0.25 * at if vp["val"] < ref - 0.5 * at else np.nan
            vp = vp_cache[(t, 120)]
            # ---- Fibonacci on the last confirmed swing leg
            sh = lv["swing_high"][t]
            leg = sh - sl if (np.isfinite(sh) and np.isfinite(sl) and sh > sl) else np.nan
            for r, k in ((0.382, "fib382"), (0.5, "fib500"), (0.618, "fib618")):
                row[k] = (sh - r * leg - 0.1 * at) if np.isfinite(leg) else np.nan
            row["fib1272"] = (sl + 1.272 * leg) if np.isfinite(leg) else np.nan
            row["fib1618"] = (sl + 1.618 * leg) if np.isfinite(leg) else np.nan
            # ---- multi-touch support zone (>= 3 pivot lows within 0.5 ATR, last 120 sessions)
            lo_p = piv_low[(piv_low + 3 <= t) & (piv_low > t - 120)]
            zone = np.nan
            if lo_p.size >= 3:
                vals = lo[lo_p]
                best = -np.inf
                for x in vals:
                    if x >= ref - 0.5 * at:
                        continue
                    near = vals[np.abs(vals - x) <= 0.5 * at]
                    if near.size >= 3 and near.mean() > best:
                        best = near.mean()
                if np.isfinite(best):
                    zone = best - 0.25 * at
            row["zone"] = zone
            # ---- stock-specific favourite MA (best shrunk bounce rate among MAs below price, >= 4 past events)
            best_m, best_r = None, -1.0
            for m in MAS:
                lvl = lv[f"ema{m}"][t]
                if np.isfinite(lvl) and lvl < ref - 0.5 * at and bs[f"bn{m}"][t] >= 4 and bs[f"br{m}"][t] > best_r:
                    best_m, best_r = m, bs[f"br{m}"][t]
            row["bounce_ma"] = (lv[f"ema{best_m}"][t] - 0.5 * at) if best_m else np.nan
            for k in STOP_KEYS:
                S[k].append(row.get(k, np.nan))
            # ---- target candidates
            trow: dict[str, float] = {"high20": lv["high20"][t], "high60": lv["high60"][t], "high252": lv["high252"][t],
                                       "swing_high": sh, "fib1272": row["fib1272"], "fib1618": row["fib1618"],
                                       "bb_up": lv["bb_up"][t], "kelt_up": lv["kelt_up"][t], "round_up": ru,
                                       "gap_dn_fill": lv["gap_dn_top"][t], "prev_high": h[t], "high5": lv["high5"][t]}
            e200, e250 = lv["ema200"][t], lv["ema250"][t]
            ups = [x for x in (e200, e250) if np.isfinite(x) and x > ref + 0.5 * at]
            trow["ema200_up"] = min(ups) if ups else np.nan
            rng20 = lv["high20"][t] - lv["don20_low"][t]
            trow["meas_move"] = (ref + rng20) if (setup in ("squeeze", "breakout") and np.isfinite(rng20)) else np.nan
            if vp is not None:
                ha = [x for x in vp["hvn"] if x > ref + 0.75 * at]
                trow["hvn_above"] = min(ha) if ha else np.nan
                trow["poc_above"] = vp["poc"] if vp["poc"] > ref + 0.75 * at else np.nan
                va = [hi_ for lo_, hi_ in vp["vac"] if lo_ >= ref and lo_ - ref <= 1.5 * at]
                trow["vac_top"] = min(va) if va else np.nan
            else:
                trow["hvn_above"] = trow["poc_above"] = trow["vac_top"] = np.nan
            anc_h = int(np.argmax(h[t - 251:t + 1])) + t - 251 if t >= 251 else -1
            if anc_h >= 0:
                tpv = (h[anc_h:t + 1] + lo[anc_h:t + 1] + c[anc_h:t + 1]) / 3
                vs = np.nan_to_num(v[anc_h:t + 1])
                trow["avwap_52wh"] = float((tpv * vs).sum() / vs.sum()) if vs.sum() > 0 else np.nan
            else:
                trow["avwap_52wh"] = np.nan
            for k in TGT_KEYS:
                T[k].append(trow.get(k, np.nan))
            # ---- filters (structure context at the signal close)
            fr: dict[str, bool] = {}
            touch = False
            for m in (50, 100, 200, 250):
                mm = lv[f"ema{m}"][t - 2:t + 1]
                touch |= bool(np.any((lo[t - 2:t + 1] <= mm + 0.25 * at) & (lo[t - 2:t + 1] >= mm - 0.5 * at)))
            fr["touch_ma_support"] = touch
            fr["near_zone"] = bool(np.isfinite(zone) and (ref - (zone + 0.25 * at)) <= 1.5 * at)
            fr["above_poc"] = bool(vp is not None and ref >= vp["poc"])
            fr["in_value_area"] = bool(vp is not None and vp["val"] <= ref <= vp["vah"])
            fr["clear_path_3atr"] = bool(vp is not None and not np.any((vp["hvn"] > ref) & (vp["hvn"] < ref + 3 * at)))
            fr["near_52w_high"] = bool(c[t] >= 0.85 * lv["high252"][t]) if np.isfinite(lv["high252"][t]) else False
            fr["above_ema200"] = bool(c[t] > lv["ema200"][t]) if np.isfinite(lv["ema200"][t]) else False
            fr["ema200_rising"] = bool(f["ema200_slope"].iat[t] > 0) if np.isfinite(f["ema200_slope"].iat[t]) else False
            good_b = False
            for m in MAS:
                lvl = lv[f"ema{m}"][t]
                if np.isfinite(lvl) and 0 < ref - lvl <= 3 * at and bs[f"bn{m}"][t] >= 4 and bs[f"br{m}"][t] >= 0.6:
                    good_b = True
            fr["stock_bounce_habit"] = good_b
            fr["higher_low"] = bool(sl > sl2) if (np.isfinite(sl) and np.isfinite(sl2)) else False
            fr["fib_zone_38_62"] = bool(np.isfinite(leg) and sh - 0.618 * leg - 0.25 * at <= ref <= sh - 0.382 * leg + 0.25 * at)
            fr["above_avwap_swing"] = bool(c[t] > lv["avwap_swing"][t]) if np.isfinite(lv["avwap_swing"][t]) else False
            fr["recent_gap_up"] = bool(np.isfinite(lv["gap_up_age"][t]) and lv["gap_up_age"][t] <= 10)
            fr["on_round_support"] = bool(0 <= ref - rb <= 1.0 * at)
            fr["no_swing_high_2atr"] = bool(not (np.isfinite(sh) and 0 < sh - ref < 2 * at))
            for k, val in fr.items():
                FILT.setdefault(k, []).append(val)
            meta.append((sym, setup, f.index[t], ref, at))
        # ---- forward windows
        tt = ts[:, None]
        fw = np.arange(1, H + 1)[None, :]
        for k, arr in zip("OHLCV", (o, h, lo, c, v)):
            OHLCV[k].append(arr[tt + fw])
        sw = np.arange(0, H + 1)[None, :]
        for k in SER_KEYS:
            SER[k].append(lv[k if k != "swing_low" else "swing_low"][tt + sw])
        if progress and (si + 1) % 10 == 0:
            progress(f"collected {si + 1}/{len(features)} stocks, {len(meta)} signals")
    sig = pd.DataFrame(meta, columns=["symbol", "setup", "date", "ref", "atr"])
    sig["date"] = pd.to_datetime(sig["date"])
    out = {"signals": sig, "S": {k: np.array(v, dtype=float) for k, v in S.items()},
           "T": {k: np.array(v, dtype=float) for k, v in T.items()},
           "F": {k: np.array(v, dtype=bool) for k, v in FILT.items()},
           "SER": {k: np.concatenate(v) for k, v in SER.items()}}
    for k in "OHLCV":
        out[k] = np.concatenate(OHLCV[k])
    return out


def subset(data: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    out = {"signals": data["signals"][mask].reset_index(drop=True)}
    for k in "OHLCV":
        out[k] = data[k][mask]
    for g in ("S", "T", "F", "SER"):
        out[g] = {k: v[mask] for k, v in data[g].items()}
    return out


# ====================================================================== simulation
def simulate(D: dict[str, Any], slip: float, stop: np.ndarray, tgt_r: float | None = None, trail: float | None = None,
             cap_tgt: np.ndarray | None = None, sser: str | None = None, sbuf: float = 0.0, cser: str | None = None,
             avwap_exit: bool = False, psar: bool = False, be: float | None = None, hz: int = H) -> pd.DataFrame:
    """One exit rule on every trade of D. Returns eq (account fraction), r, ret, days for the trades that fill."""
    sig = D["signals"]
    O, Hh, L, C, V = (D[k][:, :hz] for k in "OHLCV")
    n = len(sig)
    ref, a = sig["ref"].to_numpy(), sig["atr"].to_numpy()
    cond = (sig["setup"] == "squeeze").to_numpy()
    first_px = np.where(cond, np.maximum(O[:, 0], ref), O[:, 0])
    filled = ~cond | (Hh[:, 0] >= ref)
    entry = first_px * (1 + slip)
    ok = filled & (entry - stop > 0) & np.isfinite(entry) & np.isfinite(stop)
    risk = np.where(ok, entry - stop, 1.0)
    tgt = np.full(n, np.inf)
    if tgt_r is not None:
        tgt = ref + tgt_r * (ref - stop)
    if cap_tgt is not None:
        tgt = np.fmin(tgt, np.where(np.isfinite(cap_tgt), cap_tgt, np.inf))
    exit_px = np.full(n, np.nan)
    days = np.full(n, hz)
    live = ok.copy()
    hi_prev = entry.copy()
    ser = None
    if sser or cser:
        ser = D["SER"][sser or cser]
    sser_run = np.full(n, -np.inf)
    sar = stop.copy()
    af = np.full(n, 0.02)
    ep = entry.copy()
    be_armed = np.zeros(n, dtype=bool)
    cpv = np.zeros(n)
    cv = np.zeros(n)
    tpf = (Hh + L + C) / 3
    for k in range(hz):
        st = stop.copy()
        if trail is not None:
            st = np.maximum(st, hi_prev - trail * a)
        if sser:
            sser_run = np.maximum(sser_run, ser[:, k] - sbuf * a)
            st = np.maximum(st, sser_run)
        if psar:
            st = np.maximum(st, sar)
        if be is not None:
            st = np.where(be_armed, np.maximum(st, entry * 1.001), st)
        hit_s = live & (L[:, k] <= st)
        # a stop level above the open (e.g. a trailing indicator that sits above price after a crash) sells at the open:
        # never at a level the market did not trade at
        px_s = np.minimum(O[:, k], st)
        hit_t = live & ~hit_s & (Hh[:, k] >= tgt)
        px_t = np.maximum(O[:, k], tgt) if k > 0 else tgt
        exit_px = np.where(hit_s, px_s, np.where(hit_t, px_t, exit_px))
        days = np.where(hit_s | hit_t, k + 1, days)
        live &= ~(hit_s | hit_t)
        # close-basis exits (executed at the next open, or at this close on the last bar)
        if cser is not None or avwap_exit:
            if avwap_exit:
                cpv += np.nan_to_num(tpf[:, k] * V[:, k])
                cv += np.nan_to_num(V[:, k])
                lvl = np.where(cv > 0, cpv / np.where(cv > 0, cv, 1), np.nan)
                cond_c = (C[:, k] < lvl) & (k >= 1)
            else:
                cond_c = C[:, k] < ser[:, k + 1]
            hit_c = live & cond_c & np.isfinite(cond_c)
            if k < hz - 1:
                exit_px = np.where(hit_c, O[:, k + 1], exit_px)
                days = np.where(hit_c, k + 2, days)
                live &= ~hit_c
        hi_prev = np.maximum(hi_prev, Hh[:, k])
        if psar:
            newhi = Hh[:, k] > ep
            af = np.where(newhi, np.minimum(af + 0.02, 0.2), af)
            ep = np.maximum(ep, Hh[:, k])
            sar = np.minimum(sar + af * (ep - sar), L[:, k])
            sar = np.maximum(sar, st * 0 + stop)  # never below the initial stop
        if be is not None:
            be_armed |= (hi_prev >= entry + be * risk)
    exit_px = np.where(live, C[:, hz - 1], exit_px) * (1 - slip)
    pnl = exit_px - entry
    ret = pnl / entry
    risk_pct = risk / entry
    eq = ret * np.minimum(0.01 / np.maximum(risk_pct, 1e-6), 0.25)
    return pd.DataFrame({"eq": eq, "r": pnl / risk, "ret": ret, "days": days, "ok": ok})


# ====================================================================== methods
def stop_from_level(D: dict[str, Any], level: np.ndarray, base_atr: float) -> tuple[np.ndarray, np.ndarray]:
    ref, a = D["signals"]["ref"].to_numpy(), D["signals"]["atr"].to_numpy()
    dist = (ref - level) / a
    valid = np.isfinite(dist) & (dist >= 0.5)
    d = np.clip(np.where(valid, dist, base_atr), 0.75, 4.0)
    return ref - d * a, valid


def target_cap(D: dict[str, Any], level: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref, a = D["signals"]["ref"].to_numpy(), D["signals"]["atr"].to_numpy()
    dist = (level - ref) / a
    valid = np.isfinite(dist) & (dist >= 1.0) & (dist <= 8.0)
    return np.where(valid, level, np.nan), valid


def catalog() -> list[dict[str, Any]]:
    """name, family, params. Families: stop (initial stop at a structure level), exit (stop management),
    target (take-profit at a structure level on top of the live exit), filter (entry filter)."""
    m: list[dict[str, Any]] = []
    for k in ("atr1.0", "atr1.5", "atr2.0", "atr2.5", "atr3.0"):
        m.append({"name": f"stop: {k.upper()} (ATR variant)", "family": "stop", "kind": "atr", "k": float(k[3:])})
    for k in STOP_KEYS:
        m.append({"name": f"stop: {k}", "family": "stop", "kind": "level", "key": k})
    for hz in (5, 10, 15):
        m.append({"name": f"exit: time stop {hz} sessions", "family": "exit", "kind": "hz", "hz": hz})
    for b in (1.0, 1.5):
        m.append({"name": f"exit: break-even after +{b}R", "family": "exit", "kind": "be", "be": b})
    for t in (2.0, 2.5, 4.0, 5.0):
        m.append({"name": f"exit: ATR trail {t}", "family": "exit", "kind": "trail", "t": t})
    for key, buf in (("don10_low", 0.1), ("don20_low", 0.1), ("chand", 0.0), ("st_lower", 0.0), ("kelt_lo", 0.0), ("bb_lo", 0.0),
                     ("swing_low", 0.1)):
        m.append({"name": f"exit: trail on {key}", "family": "exit", "kind": "sser", "key": key, "buf": buf})
    for key in ("ema9", "ema21", "ema50", "sma20"):
        m.append({"name": f"exit: close below {key}", "family": "exit", "kind": "cser", "key": key})
    m.append({"name": "exit: close below entry-anchored VWAP", "family": "exit", "kind": "avwap"})
    m.append({"name": "exit: close below swing-low-anchored VWAP", "family": "exit", "kind": "cser", "key": "avwap_swing"})
    m.append({"name": "exit: Parabolic SAR trail", "family": "exit", "kind": "psar"})
    for k in TGT_KEYS:
        m.append({"name": f"target: {k}", "family": "target", "kind": "tcap", "key": k})
    return m


FILTERS = ["touch_ma_support", "near_zone", "above_poc", "in_value_area", "clear_path_3atr", "near_52w_high", "above_ema200",
           "ema200_rising", "stock_bounce_habit", "higher_low", "fib_zone_38_62", "above_avwap_swing", "recent_gap_up",
           "on_round_support", "no_swing_high_2atr"]


def base_run(D: dict[str, Any], setup: str, slip: float, **over) -> pd.DataFrame:
    b = BASE_EXIT[setup]
    ref, a = D["signals"]["ref"].to_numpy(), D["signals"]["atr"].to_numpy()
    kw = {"stop": ref - b["stop_atr"] * a, "tgt_r": b["tgt_r"], "trail": b["trail"]}
    kw.update(over)
    return simulate(D, slip, **kw)


def run_method(D: dict[str, Any], setup: str, meth: dict[str, Any], slip: float) -> tuple[pd.DataFrame, np.ndarray]:
    """(trades, applied mask) for one catalog method on the trades of D."""
    b = BASE_EXIT[setup]
    ref, a = D["signals"]["ref"].to_numpy(), D["signals"]["atr"].to_numpy()
    applied = np.ones(len(ref), dtype=bool)
    kind = meth["kind"]
    if kind == "atr":
        return base_run(D, setup, slip, stop=ref - meth["k"] * a), applied
    if kind == "level":
        st, applied = stop_from_level(D, D["S"][meth["key"]], b["stop_atr"])
        return base_run(D, setup, slip, stop=st), applied
    if kind == "hz":
        return base_run(D, setup, slip, hz=meth["hz"]), applied
    if kind == "be":
        return base_run(D, setup, slip, be=meth["be"]), applied
    if kind == "trail":
        return base_run(D, setup, slip, trail=meth["t"]), applied
    if kind == "sser":
        return base_run(D, setup, slip, sser=meth["key"], sbuf=meth["buf"]), applied
    if kind == "cser":
        return base_run(D, setup, slip, cser=meth["key"]), applied
    if kind == "avwap":
        return base_run(D, setup, slip, avwap_exit=True), applied
    if kind == "psar":
        return base_run(D, setup, slip, psar=True), applied
    if kind == "tcap":
        cap, applied = target_cap(D, D["T"][meth["key"]])
        return base_run(D, setup, slip, cap_tgt=cap), applied
    raise ValueError(kind)


# ====================================================================== statistics
def _date_means(x: np.ndarray, dates: np.ndarray) -> np.ndarray:
    return pd.Series(x).groupby(dates).mean().to_numpy()


def t_paired(d: np.ndarray, dates: np.ndarray) -> float:
    if len(d) < 6:
        return float("nan")
    m = _date_means(d, dates)
    if len(m) < 4 or m.std(ddof=1) == 0:
        return float("nan")
    return float(m.mean() / (m.std(ddof=1) / math.sqrt(len(m))))


def t_diff_groups(y: np.ndarray, flag: np.ndarray, dates: np.ndarray) -> float:
    """Cluster-robust t of mean(y | flag) - mean(y | ~flag) (clusters = signal dates)."""
    if flag.sum() < 5 or (~flag).sum() < 5:
        return float("nan")
    x = flag.astype(float)
    xc, yc = x - x.mean(), y - y.mean()
    sxx = (xc ** 2).sum()
    beta = (xc * yc).sum() / sxx
    resid = yc - beta * xc
    g = pd.Series(xc * resid).groupby(dates).sum().to_numpy()
    se = math.sqrt((g ** 2).sum()) / sxx
    return float(beta / se) if se > 0 else float("nan")


def period_masks(sig: pd.DataFrame) -> dict[str, np.ndarray]:
    return {p: ((sig["date"] >= pd.Timestamp(lo)) & (sig["date"] <= pd.Timestamp(hi))).to_numpy() for p, (lo, hi) in PERIODS.items()}


# ====================================================================== study
def _pm(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def _period_stats(d_eq: np.ndarray, d_r: np.ndarray, dates: np.ndarray, masks: dict[str, np.ndarray], applied: np.ndarray) -> dict:
    out = {}
    for p, m in masks.items():
        if m.sum() < 5:
            out[p] = {"n": int(m.sum())}
            continue
        out[p] = {"n": int(m.sum()), "d_eq": _pm(d_eq[m]) * 100, "t": t_paired(d_eq[m], dates[m]), "d_r": _pm(d_r[m]),
                  "applied": float(applied[m].mean()),
                  "d_eq_applied": (_pm(d_eq[m & applied]) * 100) if (m & applied).sum() >= 5 else None}
    return out


def run(data: dict[str, Any], slip: float, progress: Callable[[str], None] | None = None, n_perm: int = 2000) -> dict[str, Any]:
    from scipy.stats import norm
    sig = data["signals"]
    meths = catalog()
    res: dict[str, Any] = {"setups": {}, "n_methods": len(meths), "n_filters": len(FILTERS), "periods": PERIODS}
    rng = np.random.default_rng(7)
    for setup in SETUPS:
        ms = (sig["setup"] == setup).to_numpy()
        mb = (sig["setup"] == "baseline").to_numpy()
        Ds, Db = subset(data, ms), subset(data, mb)
        masks_s, masks_b = period_masks(Ds["signals"]), period_masks(Db["signals"])
        dates_s = Ds["signals"]["date"].to_numpy()
        dates_b = Db["signals"]["date"].to_numpy()
        bs, bb = base_run(Ds, setup, slip), base_run(Db, setup, slip)
        okm_s, okm_b = bs["ok"].to_numpy(), bb["ok"].to_numpy()
        entry = {"n_signals": {p: int(m.sum()) for p, m in masks_s.items()},
                 "base": {p: {"avg_eq": _pm(bs["eq"].to_numpy()[masks_s[p] & okm_s]) * 100, "avg_r": _pm(bs["r"].to_numpy()[masks_s[p] & okm_s]),
                              "random_avg_eq": _pm(bb["eq"].to_numpy()[masks_b[p] & okm_b]) * 100,
                              "random_avg_r": _pm(bb["r"].to_numpy()[masks_b[p] & okm_b])} for p in masks_s},
                 "methods": {}, "filters": {}}
        conf_cols, conf_names = [], []
        conf_mask = masks_s["hold"] | masks_s["oos"]
        for meth in meths:
            ts, app_s = run_method(Ds, setup, meth, slip)
            tb, app_b = run_method(Db, setup, meth, slip)
            if not (np.array_equal(ts["ok"].to_numpy(), okm_s) and np.array_equal(tb["ok"].to_numpy(), okm_b)):
                # a method that changes which trades fill is not paired: count the common trades only
                common_s, common_b = ts["ok"].to_numpy() & okm_s, tb["ok"].to_numpy() & okm_b
            else:
                common_s, common_b = okm_s, okm_b
            d_eq = np.where(common_s, ts["eq"].to_numpy() - bs["eq"].to_numpy(), 0.0)
            d_r = np.where(common_s, ts["r"].to_numpy() - bs["r"].to_numpy(), 0.0)
            e_eq = np.where(common_b, tb["eq"].to_numpy() - bb["eq"].to_numpy(), 0.0)
            e_r = np.where(common_b, tb["r"].to_numpy() - bb["r"].to_numpy(), 0.0)
            ms_ = {p: m & common_s for p, m in masks_s.items()}
            mb_ = {p: m & common_b for p, m in masks_b.items()}
            st_s = _period_stats(d_eq, d_r, dates_s, ms_, app_s)
            st_b = _period_stats(e_eq, e_r, dates_b, mb_, app_b)
            pooled = conf_mask & common_s
            pt = t_paired(d_eq[pooled], dates_s[pooled]) if pooled.sum() > 5 else float("nan")
            avg_m = {p: _pm(ts["eq"].to_numpy()[ms_[p]]) * 100 for p in ms_}
            avg_mr = {p: _pm(tb["eq"].to_numpy()[mb_[p]]) * 100 for p in mb_}
            entry["methods"][meth["name"]] = {"family": meth["family"], "setup": st_s, "random": st_b, "pooled_conf_t": pt,
                                              "pooled_conf_d": _pm(d_eq[pooled]) * 100,
                                              "avg_eq": avg_m, "random_avg_eq": avg_mr}
            # date-mean column for the max-t permutation (pooled confirmation sample)
            conf_cols.append(pd.Series(np.where(pooled, d_eq, np.nan)).groupby(dates_s).mean())
            conf_names.append(meth["name"])
        # ---- entry filters (pass vs fail under the live exit)
        y_s, y_b = bs["eq"].to_numpy(), bb["eq"].to_numpy()
        for fk in FILTERS:
            fs, fb = Ds["F"][fk], Db["F"][fk]
            row = {}
            for p in masks_s:
                a_, b_ = masks_s[p] & okm_s, masks_b[p] & okm_b
                if a_.sum() < 10:
                    row[p] = {"n": int(a_.sum())}
                    continue
                pas, fail = a_ & fs, a_ & ~fs
                rp, rf = b_ & fb, b_ & ~fb
                row[p] = {"n": int(a_.sum()), "pass_frac": float(fs[a_].mean()), "avg_eq_pass": _pm(y_s[pas]) * 100,
                          "avg_eq_fail": _pm(y_s[fail]) * 100, "diff": (_pm(y_s[pas]) - _pm(y_s[fail])) * 100 if pas.sum() and fail.sum() else None,
                          "t": t_diff_groups(y_s[a_], fs[a_], dates_s[a_]),
                          "random_diff": (_pm(y_b[rp]) - _pm(y_b[rf])) * 100 if rp.sum() and rf.sum() else None,
                          "random_t": t_diff_groups(y_b[b_], fb[b_], dates_b[b_])}
            a_ = (masks_s["hold"] | masks_s["oos"]) & okm_s
            row["pooled_t"] = t_diff_groups(y_s[a_], fs[a_], dates_s[a_]) if a_.sum() > 20 else float("nan")
            entry["filters"][fk] = row
        # ---- max-t sign-flip permutation on the pooled confirmation sample (selection-free guard)
        A = pd.concat(conf_cols, axis=1)
        A.columns = conf_names
        A = A.dropna(how="all").fillna(0.0)
        M = A.to_numpy()
        nd = M.shape[0]
        obs = M.mean(0) / (M.std(0, ddof=1) / math.sqrt(nd) + 1e-12)
        mx = np.empty(n_perm)
        for i in range(n_perm):
            sgn = rng.choice([-1.0, 1.0], size=nd)
            X = M * sgn[:, None]
            mx[i] = (X.mean(0) / (X.std(0, ddof=1) / math.sqrt(nd) + 1e-12)).max()
        for nm, t_ in zip(conf_names, obs):
            entry["methods"][nm]["maxt_p"] = float((mx >= t_).mean())
        entry["maxt_null_95"] = float(np.quantile(mx, 0.95))
        # ---- verdict per method
        cand = [nm for nm, v in entry["methods"].items() if v["setup"]["is"].get("d_eq", -1) > 0 and (v["setup"]["is"].get("t") or 0) >= 1.65]
        k = max(1, len(cand))
        zb = float(norm.isf(0.05 / k))
        entry["n_is_candidates"] = len(cand)
        entry["bonferroni_t"] = max(2.5, zb)
        passed = []
        for nm in cand:
            v = entry["methods"][nm]
            h_, o_ = v["setup"]["hold"], v["setup"]["oos"]
            if (h_.get("d_eq") or -1) > 0 and (o_.get("d_eq") or -1) > 0 and (v["pooled_conf_t"] or 0) >= entry["bonferroni_t"]:
                passed.append(nm)
        entry["validated"] = passed
        for fk, row in entry["filters"].items():
            ok_is = (row["is"].get("diff") or -1) > 0 and (row["is"].get("t") or 0) >= 1.65
            row["is_candidate"] = bool(ok_is)
            row["validated"] = bool(ok_is and (row["hold"].get("diff") or -1) > 0 and (row["oos"].get("diff") or -1) > 0
                                    and (row["pooled_t"] or 0) >= max(2.5, float(norm.isf(0.05 / max(1, len(FILTERS))))))
        # ---- post-hoc scan: how the trail multiplier behaves (the lab's grid stopped at 3; not part of the gate)
        scan = {}
        for tr in (3.5, 4.0, 5.0, 6.0, 8.0, 12.0):
            t1 = base_run(Ds, setup, slip, trail=tr)
            t2 = base_run(Db, setup, slip, trail=tr)
            dd = np.where(okm_s, t1["eq"].to_numpy() - bs["eq"].to_numpy(), 0.0)
            db = np.where(okm_b, t2["eq"].to_numpy() - bb["eq"].to_numpy(), 0.0)
            scan[str(tr)] = {p: {"d_eq": _pm(dd[masks_s[p] & okm_s]) * 100, "t": t_paired(dd[masks_s[p] & okm_s], dates_s[masks_s[p] & okm_s]),
                                 "random_d_eq": _pm(db[masks_b[p] & okm_b]) * 100} for p in masks_s}
        entry["trail_scan_post_hoc"] = scan
        res["setups"][setup] = entry
        if progress:
            progress(f"{setup}: IS candidates {len(cand)}, Bonferroni t {entry['bonferroni_t']:.2f}, validated {passed}")
    return res


def compact(res: dict[str, Any], top: int = 8) -> dict[str, Any]:
    """Small copy for config/structure_study.default.json (the full tables stay in the report folder)."""
    nz = lambda x: -9.0 if x is None or x != x else x  # noqa: E731
    out = {k: res[k] for k in ("session", "generated_at", "counts", "survivorship_note", "periods", "n_methods", "n_filters") if k in res}
    out["setups"] = {}
    for setup, e in res["setups"].items():
        ms = e["methods"]

        def brief(nm: str) -> dict[str, Any]:
            v = ms[nm]
            return {"name": nm, "family": v["family"], "is": v["setup"]["is"], "hold": v["setup"]["hold"], "oos": v["setup"]["oos"],
                    "pooled_conf_t": v["pooled_conf_t"], "maxt_p": v.get("maxt_p"), "random_is_d_eq": v["random"]["is"].get("d_eq")}
        fam: dict[str, dict[str, float]] = {}
        for nm, v in ms.items():
            f = fam.setdefault(v["family"], {"n": 0, "mean_pooled_d_eq": 0.0, "n_pooled_positive": 0})
            f["n"] += 1
            f["mean_pooled_d_eq"] += v["pooled_conf_d"]
            f["n_pooled_positive"] += int(nz(v["pooled_conf_t"]) > 0)
        for f in fam.values():
            f["mean_pooled_d_eq"] /= f["n"]
        out["setups"][setup] = {
            "n_signals": e["n_signals"], "base": e["base"], "n_is_candidates": e["n_is_candidates"],
            "bonferroni_t": e["bonferroni_t"], "maxt_null_95": e["maxt_null_95"], "validated": e["validated"],
            "top_by_pooled_t": [brief(nm) for nm in sorted(ms, key=lambda k: -nz(ms[k]["pooled_conf_t"]))[:top]],
            "top_by_is_t": [brief(nm) for nm in sorted(ms, key=lambda k: -nz(ms[k]["setup"]["is"].get("t")))[:top]],
            "worst_by_pooled_t": [brief(nm) for nm in sorted(ms, key=lambda k: nz(ms[k]["pooled_conf_t"]))[:3]],
            "families": fam, "trail_scan_post_hoc": e.get("trail_scan_post_hoc"),
            "filters": {k: {"is": v["is"], "hold": v["hold"], "oos": v["oos"], "pooled_t": v["pooled_t"], "validated": v["validated"]}
                        for k, v in e["filters"].items()}}
    return out

"""Partial profit-taking (scale-OUT) and stock-specific profit targets versus the live exits (research only).

Question (Neo, 2026-11-02): the live exits (crash_rebound and breakout: ATR trailing stop; squeeze: 2 ATR stop +
fixed 3R target) seem to take the full stop on losers and, on winners, only sell after the price has already fallen
back - "profit given back". Would (a) a stock-specific estimate of how far the rebound / move will run (from the
stock's own history, indicators and context) used as a take-profit, or (b) selling a fixed PROPORTION at a
take-profit level and letting the rest run, earn more and give back less?

Entry is unchanged: ONE full entry at the first open after the signal (squeeze: the buy-stop). Scaling IN is not
part of this study (rejected 2026-11-01, section 7h).

Pre-registered design (written before the first run; same rules as structure.py / section 7j)
* Signals: every setup episode (first day active) of crash_rebound / squeeze / breakout plus random entries
  (every 10th session of every stock). Entry at the next open (+10 bps), squeeze at max(open, buy-stop) only if
  it trades. 20 sessions. A bar that reaches the stop and a take-profit counts as a stop (conservative for every
  take-profit rule). A take-profit gapped through sells at the open.
* Reference = the live exit (crash_rebound 1.5 ATR stop + 3 ATR trail; breakout 1.0 ATR + 3 ATR trail; squeeze
  2.0 ATR + 3R target). Every method is compared PAIRED with it on the same trades.
* Unit: account return per trade with the live sizing (1% risk to the initial stop, position <= 25% of the
  account), so a partial exit changes the profit, never the size. R is reported too.
* Stock-specific estimates use only data known at the signal close: a stock's "run-up" history is the move from
  the next open to the highest high before the setup's ATR stop would have been hit (20 sessions), in ATR, for
  windows that ENDED before the signal; the model is fitted on in-sample signals only.
* Periods (signal date): is 2020-01..2024-06 (the only data used to choose), oos 2024-07.., hold 2010-2019 (never
  used to design anything), gfc 2005-2009 (reported only).
* Gate (fixed in advance): in-sample paired delta > 0 with t >= 1.65 (t on signal-date means), then delta > 0 in
  hold AND oos, and the hold+oos pooled t >= max(2.5, Bonferroni over the in-sample candidates). A max-t sign-flip
  permutation over all methods on the pooled sample is a second, selection-free guard. If nothing passes, the live
  logic is not changed.
* Give-back is measured on every method: for trades that were at least +1R up at some point, the average R given
  back from the best point (MFE) to the realised result, and the share that still closed at a loss ("round trip").
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.scanner.plan import plan_for_setup

H = 20
SETUPS = ("crash_rebound", "squeeze", "breakout")
BASE_STEP = 10
MIN_BARS = 260
HIST_WIN = 756            # 3 years of the stock's own run-up history
HIST_MIN = 200            # at least this many past windows, else no stock-specific estimate
STOP_KS = (1.0, 1.5, 2.0)
QS = (0.4, 0.5, 0.6, 0.7)
BASE_EXIT = {
    "crash_rebound": {"stop_atr": 1.5, "trail": 3.0, "tgt_r": None},
    "breakout": {"stop_atr": 1.0, "trail": 3.0, "tgt_r": None},
    "squeeze": {"stop_atr": 2.0, "trail": None, "tgt_r": 3.0},
}
PERIODS = {"gfc": ("2000-01-01", "2009-12-31"), "hold": ("2010-01-01", "2019-12-31"),
           "is": ("2020-01-01", "2024-06-30"), "oos": ("2024-07-01", "2100-01-01")}
MODEL_FEATURES = ("log_atr_pct", "hist_med", "setup_prior", "return_5d", "return_20d", "from_52w_high", "extension_atr",
                  "rsi14", "log_volume_ratio", "bb_width_percentile", "beta120", "log_hv_ratio", "adx", "spy_ret20",
                  "crash_leg_atr")


# ====================================================================== per-stock history (causal)
def runup_series(o: np.ndarray, h: np.ndarray, lo: np.ndarray, a: np.ndarray, k: float, hz: int = H) -> np.ndarray:
    """runup[u] = (highest high before a k-ATR stop from the open of u+1 is hit, within hz bars) - open(u+1), in ATR(u).
    Needs bars u+1..u+hz, so the value is KNOWN only at the close of bar u+hz. A stop on the first bar gives 0."""
    n = len(o)
    out = np.full(n, np.nan)
    m = n - hz - 1
    if m <= 0:
        return out
    idx = np.arange(m)[:, None] + 1 + np.arange(hz)[None, :]
    ent = o[1:m + 1]
    st = ent - k * a[:m]
    Hw, Lw = h[idx], lo[idx]
    hit = Lw <= st[:, None]
    first = np.where(hit.any(1), hit.argmax(1), hz)
    alive = np.arange(hz)[None, :] < first[:, None]          # bars strictly before the stop bar
    best = np.where(alive, Hw, -np.inf).max(1)
    ru = np.where(np.isfinite(best), np.maximum(best - ent, 0.0), 0.0) / a[:m]
    ok = np.isfinite(a[:m]) & (a[:m] > 0) & np.isfinite(ent)
    out[:m] = np.where(ok, ru, np.nan)
    return out


def hist_quantiles(ru: np.ndarray, t: int, qs: tuple[float, ...] = QS, hz: int = H) -> np.ndarray:
    """Quantiles of the stock's own run-ups whose windows ended before the close of t (u + hz <= t - 1)."""
    hi = t - hz - 1                      # last usable u
    if hi < 0:
        return np.full(len(qs), np.nan)
    w = ru[max(0, hi - HIST_WIN + 1):hi + 1]
    w = w[np.isfinite(w)]
    if w.size < HIST_MIN:
        return np.full(len(qs), np.nan)
    return np.quantile(w, qs)


def crash_leg(h: np.ndarray, lo: np.ndarray, t: int) -> tuple[float, float]:
    """(low of the crash, high before it): lowest low of the last 3 bars and the highest high of the 10 bars before."""
    low_ = float(np.min(lo[max(0, t - 2):t + 1]))
    pre = float(np.max(h[max(0, t - 12):max(1, t - 2)]))
    return low_, pre


def collect(features: dict[str, pd.DataFrame], setups: dict[str, pd.DataFrame], settings: dict[str, Any],
            spy_close: pd.Series | None = None, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    cfg = settings["scanner"]
    meta: list[tuple] = []
    OHLC = {k: [] for k in "OHLC"}
    E: dict[str, list] = {}
    spy_r20 = spy_close.pct_change(20) if spy_close is not None else None
    for si, (sym, f) in enumerate(features.items()):
        st = setups[sym]
        n = len(f)
        o, h, lo, c, a = (f[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close", "atr"))
        ru = {k: runup_series(o, h, lo, a, k) for k in STOP_KS}
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
        sig_t.sort()
        # same-setup history of this stock: (t, run-up at the setup's stop, crash retracement)
        prior: dict[str, list[tuple[int, float, float]]] = {s: [] for s in SETUPS}
        spy_al = spy_r20.reindex(f.index).to_numpy(dtype=float) if spy_r20 is not None else np.full(n, np.nan)
        cols = {k: f[k].to_numpy(dtype=float) for k in ("atr_pct", "return_5d", "return_20d", "from_52w_high",
                                                         "extension_atr", "rsi14", "volume_ratio", "bb_width_percentile",
                                                         "beta120", "hv20", "hv120", "adx")}
        for t, setup, ref in sig_t:
            k_stop = BASE_EXIT.get(setup, BASE_EXIT["crash_rebound"])["stop_atr"]
            row: dict[str, float] = {}
            for k in STOP_KS:
                qv = hist_quantiles(ru[k], t)
                for q, v in zip(QS, qv):
                    row[f"hq{int(q * 100)}_k{k}"] = float(v)
                row[f"real_ru_k{k}"] = float(ru[k][t])              # realised (for evaluation only)
            low_, pre = crash_leg(h, lo, t)
            leg = pre - low_
            row["crash_low"], row["crash_leg"] = low_, leg
            row["crash_leg_atr"] = leg / a[t] if a[t] > 0 else np.nan
            # stock's own past same-setup signals (windows finished before t)
            if setup in prior:
                past = [x for x in prior[setup] if x[0] + H + 1 <= t]
                row["prior_n"] = float(len(past))
                row["prior_med"] = float(np.median([x[1] for x in past])) if past else np.nan
                rr = [x[2] for x in past if np.isfinite(x[2])]
                row["prior_retr_n"] = float(len(rr))
                row["prior_retr_med"] = float(np.median(rr)) if rr else np.nan
                kk = BASE_EXIT[setup]["stop_atr"]
                real_hi = o[t + 1] + ru[kk][t] * a[t] if np.isfinite(ru[kk][t]) else np.nan
                retr = (real_hi - low_) / leg if leg > 0 and np.isfinite(real_hi) else np.nan
                row["real_retr"] = float(retr)
                prior[setup].append((t, float(ru[kk][t]), float(retr)))
            else:
                row["prior_n"], row["prior_med"], row["prior_retr_n"], row["prior_retr_med"], row["real_retr"] = 0.0, np.nan, 0.0, np.nan, np.nan
            row["log_atr_pct"] = math.log(cols["atr_pct"][t]) if cols["atr_pct"][t] > 0 else np.nan
            for k in ("return_5d", "return_20d", "from_52w_high", "extension_atr", "rsi14", "bb_width_percentile", "beta120", "adx"):
                row[k] = cols[k][t]
            row["log_volume_ratio"] = math.log(cols["volume_ratio"][t]) if cols["volume_ratio"][t] > 0 else np.nan
            row["log_hv_ratio"] = math.log(cols["hv20"][t] / cols["hv120"][t]) if cols["hv120"][t] > 0 and cols["hv20"][t] > 0 else np.nan
            row["spy_ret20"] = spy_al[t]
            hm = row[f"hq50_k{k_stop}"]
            row["hist_med"] = math.log(hm + 0.25) if np.isfinite(hm) else np.nan
            for key, val in row.items():
                E.setdefault(key, []).append(val)
            meta.append((sym, setup, f.index[t], ref, a[t]))
        ts = np.array([x[0] for x in sig_t])
        fw = ts[:, None] + np.arange(1, H + 1)[None, :]
        for k, arr in zip("OHLC", (o, h, lo, c)):
            OHLC[k].append(arr[fw])
        if progress and (si + 1) % 25 == 0:
            progress(f"collected {si + 1}/{len(features)} stocks, {len(meta)} signals")
    sig = pd.DataFrame(meta, columns=["symbol", "setup", "date", "ref", "atr"])
    sig["date"] = pd.to_datetime(sig["date"])
    out: dict[str, Any] = {"signals": sig, "E": {k: np.array(v, dtype=float) for k, v in E.items()}}
    for k in "OHLC":
        out[k] = np.concatenate(OHLC[k])
    return out


def subset(data: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    out = {"signals": data["signals"][mask].reset_index(drop=True), "E": {k: v[mask] for k, v in data["E"].items()}}
    for k in "OHLC":
        out[k] = data[k][mask]
    for k in ("X", "P"):
        if k in data:
            out[k] = {kk: vv[mask] for kk, vv in data[k].items()}
    return out


# ====================================================================== simulation with partial exits
def simulate(D: dict[str, Any], slip: float, stop: np.ndarray, trail: float | None = None,
             final_tgt: np.ndarray | None = None, partials: list[tuple[np.ndarray, float]] | None = None,
             be_after_partial: bool = False, tighten: tuple[float, float] | None = None,
             lock: tuple[float, float] | None = None, time_partials: list[tuple[int, float]] | None = None,
             rest_trail: float | None = None, hz: int = H, paths: bool = False) -> dict[str, np.ndarray]:
    """One exit rule (with optional partial take-profits) on every trade of D.

    stop: initial stop price. trail: k-ATR trail from the highest high of the previous bars. final_tgt: price at which
    everything still open is sold. partials: [(price, fraction of the ORIGINAL position)] sold when the high reaches
    the price (ascending order). be_after_partial: after the first partial, the stop of the rest is the entry.
    tighten (trigger_r, k2): once the high has reached entry + trigger_r x risk, the trail becomes k2 ATR.
    lock (trigger_r, frac): once +trigger_r was reached, the stop is >= entry + frac x (highest high - entry).
    time_partials: [(day, fraction)] sold at that day's close. rest_trail: once the first partial is taken, the rest
    switches to a rest_trail ATR trail and its final target is dropped (squeeze "let the rest run").
    Same-bar rule: stop first (the whole remainder), then partials in ascending price, then the final target.
    """
    sig = D["signals"]
    O, Hh, L, C = (D[k][:, :hz] for k in "OHLC")
    n = len(sig)
    ref, a = sig["ref"].to_numpy(dtype=float), sig["atr"].to_numpy(dtype=float)
    cond = (sig["setup"] == "squeeze").to_numpy()
    first_px = np.where(cond, np.maximum(O[:, 0], ref), O[:, 0])
    filled = ~cond | (Hh[:, 0] >= ref)
    entry = first_px * (1 + slip)
    ok = filled & (entry - stop > 0) & np.isfinite(entry) & np.isfinite(stop)
    risk = np.where(ok, entry - stop, 1.0)
    tgt = np.full(n, np.inf) if final_tgt is None else np.where(np.isfinite(final_tgt), final_tgt, np.inf)
    parts = partials or []
    taken = [np.zeros(n, dtype=bool) for _ in parts]
    rem = np.where(ok, 1.0, 0.0)
    pnl = np.zeros(n)                     # realised price-pnl per ORIGINAL share
    live = ok.copy()
    hi_prev = entry.copy()
    mfe_hi = entry.copy()
    any_partial = np.zeros(n, dtype=bool)
    tight = np.zeros(n, dtype=bool)
    locked = np.zeros(n, dtype=bool)
    last_day = np.full(n, hz)
    exit_kind = np.full(n, "time", dtype=object)
    path_pnl = np.zeros((n, hz)) if paths else None
    path_rem = np.zeros((n, hz)) if paths else None
    for k in range(hz):
        st = stop.copy()
        tr_k = np.full(n, np.inf)
        if trail is not None:
            tr_k = np.full(n, trail)
        if tighten is not None:
            tr_k = np.where(tight, np.minimum(tr_k, tighten[1]), tr_k)
        if rest_trail is not None:
            tr_k = np.where(any_partial, rest_trail, tr_k)
        st = np.where(np.isfinite(tr_k), np.maximum(st, hi_prev - np.where(np.isfinite(tr_k), tr_k, 0) * a), st)
        if be_after_partial:
            st = np.where(any_partial, np.maximum(st, entry * 1.001), st)
        if lock is not None:
            st = np.where(locked, np.maximum(st, entry + lock[1] * (hi_prev - entry)), st)
        mfe_hi = np.where(live, np.maximum(mfe_hi, Hh[:, k]), mfe_hi)
        hit_s = live & (L[:, k] <= st)
        px_s = np.minimum(O[:, k], st) * (1 - slip)
        pnl = np.where(hit_s, pnl + rem * (px_s - entry), pnl)
        exit_kind = np.where(hit_s, np.where(st > stop + 1e-12, np.where(px_s > entry, "trail_win", "trail_loss"), "stop"), exit_kind)
        rem = np.where(hit_s, 0.0, rem)
        last_day = np.where(hit_s, k + 1, last_day)
        live &= ~hit_s
        for i, (tp, frac) in enumerate(parts):
            h_i = live & ~taken[i] & (Hh[:, k] >= tp)
            sell = np.minimum(frac, rem)
            px = np.maximum(O[:, k], tp) * (1 - slip)
            pnl = np.where(h_i, pnl + sell * (px - entry), pnl)
            rem = np.where(h_i, rem - sell, rem)
            taken[i] |= h_i
            any_partial |= h_i
        done = live & (rem <= 1e-9)
        last_day = np.where(done, k + 1, last_day)
        exit_kind = np.where(done, "partials", exit_kind)
        live &= ~done
        tgt_k = np.where(any_partial, np.inf, tgt) if rest_trail is not None else tgt
        hit_t = live & (Hh[:, k] >= tgt_k)
        px_t = np.maximum(O[:, k], np.where(np.isfinite(tgt_k), tgt_k, 0)) * (1 - slip)
        pnl = np.where(hit_t, pnl + rem * (px_t - entry), pnl)
        rem = np.where(hit_t, 0.0, rem)
        last_day = np.where(hit_t, k + 1, last_day)
        exit_kind = np.where(hit_t, "target", exit_kind)
        live &= ~hit_t
        for day, frac in (time_partials or []):
            if k == day - 1:
                sell = np.where(live, np.minimum(frac, rem), 0.0)
                pnl = pnl + sell * (C[:, k] * (1 - slip) - entry)
                rem = rem - sell
                any_partial |= sell > 0
        hi_prev = np.maximum(hi_prev, Hh[:, k])
        if tighten is not None:
            tight |= hi_prev >= entry + tighten[0] * risk
        if lock is not None:
            locked |= hi_prev >= entry + lock[0] * risk
        if paths:
            path_pnl[:, k] = pnl + rem * (C[:, k] - entry)
            path_rem[:, k] = rem * C[:, k] / entry
    pnl = np.where(live, pnl + rem * (C[:, hz - 1] * (1 - slip) - entry), pnl)
    ret = pnl / entry
    risk_pct = risk / entry
    size = np.minimum(0.01 / np.maximum(risk_pct, 1e-6), 0.25)
    out = {"eq": ret * size, "r": pnl / risk, "ret": ret, "days": last_day, "ok": ok, "mfe_r": (mfe_hi - entry) / risk,
           "kind": exit_kind, "size": size}
    if paths:
        out["path_eq"] = path_pnl / entry[:, None] * size[:, None]
        out["path_cap"] = path_rem * size[:, None]
    return out


# ====================================================================== stock-specific estimates
def fit_model(Dis: dict[str, Any], k: float) -> dict[str, Any]:
    """OLS of log(run-up + 0.25) on the signal-time features, in-sample signals only (clipped, standardised)."""
    X = np.column_stack([Dis["E"][c] for c in MODEL_FEATURES])
    y = np.log(Dis["E"][f"real_ru_k{k}"] + 0.25)
    lo_, hi_ = np.nanpercentile(X, 1, axis=0), np.nanpercentile(X, 99, axis=0)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xc = np.where(np.isfinite(X), np.clip(X, lo_, hi_), med)
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(Xc)), (Xc - mu) / sd])
    m = np.isfinite(y)
    beta, *_ = np.linalg.lstsq(Z[m], y[m], rcond=None)
    res = y[m] - Z[m] @ beta
    return {"beta": beta, "lo": lo_, "hi": hi_, "med": med, "mu": mu, "sd": sd,
            "res_q": {q: float(np.quantile(res, q)) for q in QS}, "r2_is": float(1 - res.var() / y[m].var())}


def predict_model(D: dict[str, Any], mdl: dict[str, Any], q: float) -> np.ndarray:
    X = np.column_stack([D["E"][c] for c in MODEL_FEATURES])
    Xc = np.where(np.isfinite(X), np.clip(X, mdl["lo"], mdl["hi"]), mdl["med"])
    Z = np.column_stack([np.ones(len(Xc)), (Xc - mdl["mu"]) / mdl["sd"]])
    return np.exp(Z @ mdl["beta"] + mdl["res_q"][q]) - 0.25


def prepare(data: dict[str, Any]) -> None:
    """Derived per-signal features that need the in-sample universe (setup_prior shrinkage). In place."""
    sig, E = data["signals"], data["E"]
    is_m = ((sig["date"] >= PERIODS["is"][0]) & (sig["date"] <= PERIODS["is"][1])).to_numpy()
    setup_prior = np.full(len(sig), np.nan)
    E["uni_med"] = np.full(len(sig), np.nan)
    E["uni_retr"] = np.full(len(sig), np.nan)
    for s in SETUPS + ("baseline",):
        m = (sig["setup"] == s).to_numpy()
        k = BASE_EXIT.get(s, BASE_EXIT["crash_rebound"])["stop_atr"]
        uni = float(np.nanmedian(E[f"real_ru_k{k}"][m & is_m])) if (m & is_m).any() else np.nan
        uni_r = float(np.nanmedian(E["real_retr"][m & is_m])) if (m & is_m).any() and s in SETUPS else np.nan
        E["uni_med"][m] = uni
        E["uni_retr"][m] = uni_r
        n_ = E["prior_n"][m]
        pm = np.where(np.isfinite(E["prior_med"][m]), E["prior_med"][m], uni)
        setup_prior[m] = (n_ * pm + 5 * uni) / (n_ + 5)
    E["setup_prior_ru"] = setup_prior
    E["setup_prior"] = np.log(setup_prior + 0.25)


# ====================================================================== methods
def catalog(setup: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sq = setup == "squeeze"
    # A. one partial take-profit at x R, rest follows the live exit (optionally with the stop moved to the entry)
    for x in ((1.0, 1.5, 2.0) if sq else (1.0, 1.5, 2.0, 3.0)):
        for p in (0.25, 0.33, 0.5, 0.75):
            for be in (False, True):
                out.append({"name": f"partial_{x:g}R_{int(p * 100)}%{'_be' if be else ''}", "family": "partial_R",
                            "parts": [(x, p)], "be": be})
    # A2. ladders
    ladders = ([(1.0, 1 / 3), (2.0, 1 / 3)], [(1.0, 0.25), (1.5, 0.25), (2.0, 0.25)], [(1.5, 0.5), (2.5, 0.25)]) if sq else \
        ([(1.0, 1 / 3), (2.0, 1 / 3)], [(1.5, 1 / 3), (3.0, 1 / 3)], [(1.0, 0.25), (2.0, 0.25), (3.0, 0.25)])
    for lad in ladders:
        for be in (False, True):
            nm = "+".join(f"{x:g}R" for x, _ in lad)
            out.append({"name": f"ladder_{nm}{'_be' if be else ''}", "family": "ladder", "parts": lad, "be": be})
    # A3. squeeze: sell part at the target, let the rest run on a 3-ATR trail (no fixed target)
    if sq:
        for x in (2.0, 3.0):
            for p in (0.5, 0.67, 0.75):
                out.append({"name": f"run_rest_{x:g}R_{int(p * 100)}%", "family": "let_run", "parts": [(x, p)],
                            "be": False, "rest_trail": 3.0})
    # B. stock-specific take-profit = q-quantile of the stock's own past run-ups (3 years), full or half
    for q in QS:
        for use in ("full", "half"):
            out.append({"name": f"stock_hist_q{int(q * 100)}_{use}", "family": "stock_hist", "est": f"hist_q{int(q * 100)}", "use": use})
            # control: one fixed ATR distance for every stock = the in-sample median of the same estimate
            out.append({"name": f"fixed_ctrl_q{int(q * 100)}_{use}", "family": "fixed_control", "est": f"fixed_q{int(q * 100)}", "use": use})
    # B2. the stock's own past signals of the SAME setup (shrunk to the universe median with weight 5)
    for sc in (0.8, 1.0, 1.25):
        for use in ("full", "half"):
            out.append({"name": f"stock_setup_x{sc:g}_{use}", "family": "stock_setup", "est": f"setup_{sc:g}", "use": use})
    # C. model forecast (indicators + stock history + market), q-quantile of the predicted run-up
    for q in (0.4, 0.5, 0.6):
        for use in ("full", "half"):
            out.append({"name": f"model_q{int(q * 100)}_{use}", "family": "model", "est": f"model_q{int(q * 100)}", "use": use})
    # D. crash rebound: retracement of the crash leg (fixed share, or the stock's own past share)
    if setup == "crash_rebound":
        for fr in (0.5, 0.618, 0.786, 1.0):
            for use in ("full", "half"):
                out.append({"name": f"retrace_{fr:g}_{use}", "family": "retrace", "est": f"retr_{fr:g}", "use": use})
        for use in ("full", "half"):
            out.append({"name": f"retrace_stock_{use}", "family": "retrace_stock", "est": "retr_stock", "use": use})
    # E. give-back control without selling: tighter trail / profit lock once in profit
    for trig in (1.5, 2.0, 3.0):
        for k2 in (1.5, 2.0):
            out.append({"name": f"tighten_{trig:g}R_to_{k2:g}atr", "family": "tighten", "tighten": (trig, k2)})
    for trig in (1.0, 2.0):
        out.append({"name": f"lock50_after_{trig:g}R", "family": "lock", "lock": (trig, 0.5)})
    # F. time-based partial: sell half at the close of day d
    for d in (3, 5, 10):
        out.append({"name": f"time_half_day{d}", "family": "time_partial", "time": [(d, 0.5)]})
    return out


def estimate_price(D: dict[str, Any], setup: str, est: str, fixed: dict[str, float]) -> np.ndarray:
    """Take-profit price for a stock-specific estimate (NaN = no estimate -> live exit only)."""
    sig, E = D["signals"], D["E"]
    ref, a = sig["ref"].to_numpy(dtype=float), sig["atr"].to_numpy(dtype=float)
    k = BASE_EXIT[setup]["stop_atr"]
    if est.startswith("hist_q"):
        dist = E[f"hq{est[6:]}_k{k}"]
    elif est.startswith("fixed_q"):
        dist = np.full(len(sig), fixed[est])
    elif est.startswith("setup_"):
        dist = E["setup_prior_ru"] * float(est[6:])
    elif est.startswith("model_q"):
        dist = D["P"][est]
    elif est.startswith("retr_"):
        low_, leg = E["crash_low"], E["crash_leg"]
        fr = E["prior_retr_shrunk"] if est == "retr_stock" else float(est[5:])
        px = low_ + fr * leg
        dist = (px - ref) / a
    else:
        raise ValueError(est)
    dist = np.clip(dist, 0.75, 10.0)
    return np.where(np.isfinite(dist), ref + dist * a, np.nan)


def run_method(D: dict[str, Any], setup: str, meth: dict[str, Any], slip: float, fixed: dict[str, float],
               paths: bool = False) -> tuple[dict[str, np.ndarray], np.ndarray]:
    b = BASE_EXIT[setup]
    ref, a = D["signals"]["ref"].to_numpy(dtype=float), D["signals"]["atr"].to_numpy(dtype=float)
    stop = ref - b["stop_atr"] * a
    risk = ref - stop
    base_tgt = ref + b["tgt_r"] * risk if b["tgt_r"] else None
    kw: dict[str, Any] = {"stop": stop, "trail": b["trail"], "final_tgt": base_tgt, "paths": paths}
    applied = np.ones(len(ref), dtype=bool)
    if "parts" in meth:
        kw["partials"] = [(ref + x * risk, p) for x, p in meth["parts"]]
        kw["be_after_partial"] = meth.get("be", False)
        if meth.get("rest_trail"):
            kw["rest_trail"] = meth["rest_trail"]
    elif "est" in meth:
        tp = estimate_price(D, setup, meth["est"], fixed)
        applied = np.isfinite(tp)
        if meth["use"] == "full":
            kw["final_tgt"] = tp if base_tgt is None else np.fmin(np.where(np.isfinite(tp), tp, np.inf), base_tgt)
            if base_tgt is not None:            # squeeze: the stock-specific target REPLACES the 3R target
                kw["final_tgt"] = np.where(np.isfinite(tp), tp, base_tgt)
        else:
            kw["partials"] = [(np.where(np.isfinite(tp), tp, np.inf), 0.5)]
    elif "tighten" in meth:              # squeeze (no trail): the k2 trail starts only once the trigger is reached
        kw["tighten"] = meth["tighten"]
    elif "lock" in meth:
        kw["lock"] = meth["lock"]
    elif "time" in meth:
        kw["time_partials"] = meth["time"]
    return simulate(D, slip, **kw), applied


def base(D: dict[str, Any], setup: str, slip: float, paths: bool = False) -> dict[str, np.ndarray]:
    return run_method(D, setup, {"name": "live"}, slip, {}, paths=paths)[0]


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


def period_masks(sig: pd.DataFrame) -> dict[str, np.ndarray]:
    return {p: ((sig["date"] >= pd.Timestamp(lo)) & (sig["date"] <= pd.Timestamp(hi))).to_numpy() for p, (lo, hi) in PERIODS.items()}


def _pm(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return float("nan")
    rx, ry = pd.Series(x[m]).rank().to_numpy(), pd.Series(y[m]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def giveback_stats(t: dict[str, np.ndarray], m: np.ndarray) -> dict[str, float]:
    """Outcome shape on the trades of mask m: average, win rate, and how much of the best open profit was given back."""
    r, mfe, eq = t["r"][m], t["mfe_r"][m], t["eq"][m]
    if len(r) < 10:
        return {"n": int(len(r))}
    up1 = mfe >= 1.0
    up2 = mfe >= 2.0
    kinds = pd.Series(t["kind"][m]).value_counts(normalize=True).to_dict()
    return {"n": int(len(r)), "avg_eq": _pm(eq) * 100, "avg_r": _pm(r), "median_r": float(np.median(r)),
            "win": float((r > 0).mean()), "sd_eq": float(eq.std() * 100),
            "avg_win_r": _pm(r[r > 0]), "avg_loss_r": _pm(r[r <= 0]),
            "full_stop_rate": float(kinds.get("stop", 0.0)), "reached_1r": float(up1.mean()), "reached_2r": float(up2.mean()),
            "giveback_r_if_1r": _pm(mfe[up1] - r[up1]) if up1.any() else None,
            "capture_if_1r": float(r[up1].mean() / mfe[up1].mean()) if up1.any() else None,
            "round_trip_if_1r": float((r[up1] <= 0).mean()) if up1.any() else None,
            "round_trip_if_2r": float((r[up2] <= 0).mean()) if up2.any() else None,
            "exit_kinds": {k: float(v) for k, v in kinds.items()}}


def _period_stats(d_eq: np.ndarray, d_r: np.ndarray, dates: np.ndarray, masks: dict[str, np.ndarray], applied: np.ndarray) -> dict:
    out = {}
    for p, m in masks.items():
        if m.sum() < 5:
            out[p] = {"n": int(m.sum())}
            continue
        out[p] = {"n": int(m.sum()), "d_eq": _pm(d_eq[m]) * 100, "t": t_paired(d_eq[m], dates[m]), "d_r": _pm(d_r[m]),
                  "applied": float(applied[m].mean())}
    return out


def run(data: dict[str, Any], slip: float, progress: Callable[[str], None] | None = None, n_perm: int = 2000) -> dict[str, Any]:
    prepare(data)
    sig = data["signals"]
    res: dict[str, Any] = {"setups": {}, "periods": PERIODS}
    rng = np.random.default_rng(7)
    for setup in SETUPS:
        ms = (sig["setup"] == setup).to_numpy()
        mb = (sig["setup"] == "baseline").to_numpy()
        Ds, Db = subset(data, ms), subset(data, mb)
        masks_s, masks_b = period_masks(Ds["signals"]), period_masks(Db["signals"])
        dates_s, dates_b = Ds["signals"]["date"].to_numpy(), Db["signals"]["date"].to_numpy()
        k = BASE_EXIT[setup]["stop_atr"]
        # ---- stock-specific estimates that need the in-sample data
        for D_ in (Ds, Db):
            E = D_["E"]
            n_ = E["prior_retr_n"]
            uni_r = np.nanmedian(Ds["E"]["real_retr"][masks_s["is"]]) if setup == "crash_rebound" else np.nan
            pr = np.where(np.isfinite(E["prior_retr_med"]), E["prior_retr_med"], uni_r)
            E["prior_retr_shrunk"] = (n_ * pr + 3 * uni_r) / (n_ + 3)
        mdl = fit_model(subset(Ds, masks_s["is"]), k)
        for D_ in (Ds, Db):
            D_["P"] = {f"model_q{int(q * 100)}": predict_model(D_, mdl, q) for q in (0.4, 0.5, 0.6)}
        fixed = {f"fixed_q{int(q * 100)}": float(np.nanmedian(np.clip(Ds["E"][f"hq{int(q * 100)}_k{k}"][masks_s["is"]], 0.75, 10)))
                 for q in QS}
        # ---- can a stock's history / the model predict how far the move runs? (rank correlation with the realised run-up)
        real = Ds["E"][f"real_ru_k{k}"]
        pred_diag = {}
        for nm, x in (("stock_hist_median", Ds["E"][f"hq50_k{k}"]), ("stock_setup_prior", Ds["E"]["setup_prior_ru"]),
                      ("model_median", Ds["P"]["model_q50"]), ("atr_pct_only", Ds["E"]["log_atr_pct"])):
            pred_diag[nm] = {p: spearman(x[m], real[m]) for p, m in masks_s.items()}
        pred_diag["realised_runup_quantiles_atr"] = {p: [float(v) for v in np.nanquantile(real[m], [0.25, 0.5, 0.75, 0.9])]
                                                     for p, m in masks_s.items() if m.sum() > 20}
        pred_diag["model_r2_is"] = mdl["r2_is"]
        pred_diag["model_beta"] = dict(zip(("const",) + MODEL_FEATURES, [float(b) for b in mdl["beta"]]))
        bs, bb = base(Ds, setup, slip), base(Db, setup, slip)
        okm_s, okm_b = bs["ok"], bb["ok"]
        entry: dict[str, Any] = {
            "n_signals": {p: int(m.sum()) for p, m in masks_s.items()},
            "base": {p: giveback_stats(bs, masks_s[p] & okm_s) for p in masks_s},
            "base_random": {p: giveback_stats(bb, masks_b[p] & okm_b) for p in masks_b},
            "fixed_targets_atr": fixed, "prediction": pred_diag, "methods": {}}
        conf_mask = masks_s["hold"] | masks_s["oos"]
        conf_cols, conf_names = [], []
        for meth in catalog(setup):
            ts, app_s = run_method(Ds, setup, meth, slip, fixed)
            tb, app_b = run_method(Db, setup, meth, slip, fixed)
            d_eq = np.where(okm_s, ts["eq"] - bs["eq"], 0.0)
            d_r = np.where(okm_s, ts["r"] - bs["r"], 0.0)
            e_eq = np.where(okm_b, tb["eq"] - bb["eq"], 0.0)
            e_r = np.where(okm_b, tb["r"] - bb["r"], 0.0)
            ms_ = {p: m & okm_s for p, m in masks_s.items()}
            mb_ = {p: m & okm_b for p, m in masks_b.items()}
            pooled = conf_mask & okm_s
            entry["methods"][meth["name"]] = {
                "family": meth["family"], "setup": _period_stats(d_eq, d_r, dates_s, ms_, app_s),
                "random": _period_stats(e_eq, e_r, dates_b, mb_, app_b),
                "pooled_conf_t": t_paired(d_eq[pooled], dates_s[pooled]), "pooled_conf_d": _pm(d_eq[pooled]) * 100,
                "shape": {p: giveback_stats(ts, ms_[p]) for p in ("is", "hold", "oos")},
                "shape_pooled": giveback_stats(ts, pooled)}
            conf_cols.append(pd.Series(np.where(pooled, d_eq, np.nan)).groupby(dates_s).mean())
            conf_names.append(meth["name"])
        entry["base_pooled"] = giveback_stats(bs, conf_mask & okm_s)
        # ---- max-t sign-flip permutation (selection-free guard)
        A = pd.concat(conf_cols, axis=1)
        A.columns = conf_names
        M = A.dropna(how="all").fillna(0.0).to_numpy()
        nd = M.shape[0]
        obs = M.mean(0) / (M.std(0, ddof=1) / math.sqrt(nd) + 1e-12)
        mx = np.empty(n_perm)
        for i in range(n_perm):
            X = M * rng.choice([-1.0, 1.0], size=nd)[:, None]
            mx[i] = (X.mean(0) / (X.std(0, ddof=1) / math.sqrt(nd) + 1e-12)).max()
        for nm, t_ in zip(conf_names, obs):
            entry["methods"][nm]["maxt_p"] = float((mx >= t_).mean())
        entry["maxt_null_95"] = float(np.quantile(mx, 0.95))
        # ---- verdict
        cand = [nm for nm, v in entry["methods"].items() if (v["setup"]["is"].get("d_eq") or -1) > 0 and (v["setup"]["is"].get("t") or 0) >= 1.65]
        kc = max(1, len(cand))
        entry["n_methods"] = len(entry["methods"])
        entry["n_is_candidates"] = len(cand)
        entry["is_candidates"] = cand
        entry["bonferroni_t"] = max(2.5, NormalDist().inv_cdf(1 - 0.05 / kc))
        entry["validated"] = [nm for nm in cand if (entry["methods"][nm]["setup"]["hold"].get("d_eq") or -1) > 0
                              and (entry["methods"][nm]["setup"]["oos"].get("d_eq") or -1) > 0
                              and (entry["methods"][nm]["pooled_conf_t"] or 0) >= entry["bonferroni_t"]]
        res["setups"][setup] = entry
        if progress:
            progress(f"{setup}: {len(entry['methods'])} methods, IS candidates {len(cand)}, validated {entry['validated']}")
    return res


# ====================================================================== account simulation (illustration)
def account_paths(data: dict[str, Any], choice: dict[str, dict[str, Any] | None], slip: float,
                  calendar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Trades of the three live setups with the chosen exit per setup (None = live exit), with daily account paths
    in the format of leverage.portfolio (pnl and capital per 1R of risk)."""
    prepare(data)
    sig = data["signals"]
    rows, pnl_l, cap_l = [], [], []
    for setup in SETUPS:
        ms = (sig["setup"] == setup).to_numpy()
        Ds = subset(data, ms)
        masks = period_masks(Ds["signals"])
        k = BASE_EXIT[setup]["stop_atr"]
        fixed = {f"fixed_q{int(q * 100)}": float(np.nanmedian(np.clip(Ds["E"][f"hq{int(q * 100)}_k{k}"][masks["is"]], 0.75, 10)))
                 for q in QS}
        meth = choice.get(setup)
        if meth is not None and meth.get("est", "").startswith("model"):
            mdl = fit_model(subset(Ds, masks["is"]), k)
            Ds["P"] = {f"model_q{int(q * 100)}": predict_model(Ds, mdl, q) for q in (0.4, 0.5, 0.6)}
        if meth is not None and meth.get("est") == "retr_stock":
            raise ValueError("retr_stock not supported in the account illustration")
        t, _ = run_method(Ds, setup, meth or {"name": "live"}, slip, fixed, paths=True)
        ok = t["ok"]
        days = t["days"]
        pe = t["path_eq"].copy()
        pc = t["path_cap"].copy()
        for i in np.flatnonzero(ok):           # after the exit the value is the final result and no capital is used
            d = int(days[i])
            pe[i, d - 1:] = t["eq"][i]
            pc[i, d - 1:] = 0.0
        cal0 = calendar_index.searchsorted(Ds["signals"]["date"].to_numpy()) + 1
        keep = ok & (cal0 + H < len(calendar_index))
        rows.append(pd.DataFrame({"setup": setup, "date": Ds["signals"]["date"].to_numpy()[keep], "cal0": cal0[keep],
                                  "days": days[keep], "notional_r": t["size"][keep] / 0.01}))
        pnl_l.append(pe[keep] / 0.01)
        cap_l.append(pc[keep] / 0.01)
    trades = pd.concat(rows, ignore_index=True)
    trades.attrs["pnl"] = np.concatenate(pnl_l)
    trades.attrs["cap"] = np.concatenate(cap_l)
    return trades


def compact(res: dict[str, Any], top: int = 8) -> dict[str, Any]:
    nz = lambda x: -9.0 if x is None or x != x else x  # noqa: E731
    out = {k: res[k] for k in ("session", "generated_at", "counts", "survivorship_note", "periods") if k in res}
    out["setups"] = {}
    for setup, e in res["setups"].items():
        ms = e["methods"]

        def brief(nm: str) -> dict[str, Any]:
            v = ms[nm]
            sp = v["shape_pooled"]
            return {"name": nm, "family": v["family"], "is": v["setup"]["is"], "hold": v["setup"]["hold"], "oos": v["setup"]["oos"],
                    "pooled_conf_t": v["pooled_conf_t"], "pooled_conf_d": v["pooled_conf_d"], "maxt_p": v.get("maxt_p"),
                    "random_is_d_eq": v["random"]["is"].get("d_eq"),
                    "pooled_giveback_r": sp.get("giveback_r_if_1r"), "pooled_round_trip": sp.get("round_trip_if_1r"),
                    "pooled_win": sp.get("win"), "pooled_sd_eq": sp.get("sd_eq")}
        fam: dict[str, dict[str, float]] = {}
        for nm, v in ms.items():
            f = fam.setdefault(v["family"], {"n": 0, "mean_pooled_d_eq": 0.0, "n_pooled_positive": 0})
            f["n"] += 1
            f["mean_pooled_d_eq"] += v["pooled_conf_d"]
            f["n_pooled_positive"] += int(nz(v["pooled_conf_t"]) > 0)
        for f in fam.values():
            f["mean_pooled_d_eq"] /= f["n"]
        out["setups"][setup] = {
            "n_signals": e["n_signals"], "base": e["base"], "base_pooled": e["base_pooled"], "n_methods": e["n_methods"],
            "n_is_candidates": e["n_is_candidates"], "bonferroni_t": e["bonferroni_t"], "maxt_null_95": e["maxt_null_95"],
            "validated": e["validated"], "prediction": {k: v for k, v in e["prediction"].items() if k != "model_beta"},
            "top_by_pooled_t": [brief(nm) for nm in sorted(ms, key=lambda k: -nz(ms[k]["pooled_conf_t"]))[:top]],
            "top_by_is_t": [brief(nm) for nm in sorted(ms, key=lambda k: -nz(ms[k]["setup"]["is"].get("t")))[:top]],
            "worst_by_pooled_t": [brief(nm) for nm in sorted(ms, key=lambda k: nz(ms[k]["pooled_conf_t"]))[:3]],
            "families": fam}
    return out

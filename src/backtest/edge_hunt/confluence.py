"""Multi-strategy confluence (Neo, 2026-11-03: combine the strengths of the online swing strategies).

The 32 long strategies of the swing42 study (src/backtest/swing42, Neo's list of online swing styles) are run on
the same point-in-time S&P 500 panel with the parameters swing42 chose on 2000-2014 (reports/swing42/results_A.json),
so nothing here is tuned on this study's out-of-sample years. Per stock-day:

    conf_n1   number of distinct strategies firing today
    conf_n5   number of distinct strategies that fired in the last 5 sessions
    fam_*5    a strategy of that family fired in the last 5 sessions:
              mr  mean reversion (U&R spring, sweep, RSI(2), Z-score, double bottom)
              bo  breakout / momentum (VCP, HTF, pocket pivot, flat base, cup, 3WT, TTM squeeze, EP gap,
                  short-squeeze proxy, RS leaders, holy grail)
              smc smart-money concepts (order block, FVG, BOS/CHoCH, breaker, OTE, Wyckoff)
              vp  volume profile / VWAP (AVWAP bounce, AVWAP pinch, POC shift, LVN, value area, HVB, UOA proxy)
    n_fam5    number of families with a signal in the last 5 sessions
    ens5      in-sample-weighted ensemble: sum over strategies fired in the last 5 sessions of the strategy's
              in-sample (2010-2017) t-statistic vs same-day random entries with the live crash exit, positive t only
The live crash reference (swing42 id 0) is not counted.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.backtest.swing42 import core, strategies as st

FAMILY = {"mr": [9, 10, 17, 26, 27], "bo": [1, 2, 3, 4, 5, 7, 8, 25, 30, 31, 38],
          "smc": [11, 12, 13, 14, 15, 16], "vp": [18, 19, 20, 21, 22, 23, 33]}
FN = {1: st.s01_vcp, 2: st.s02_htf, 3: st.s03_pocket_pivot, 4: st.s04_flat_base, 5: st.s05_cup_handle, 7: st.s07_3wt,
      8: st.s08_holy_grail, 9: st.s09_double_bottom, 10: st.s10_undercut_rally, 11: st.s11_order_block, 12: st.s12_fvg,
      13: st.s13_bos_choch, 14: st.s14_breaker, 15: st.s15_ote, 16: st.s16_wyckoff, 17: st.s17_sweep,
      18: st.s18_avwap_bounce, 19: st.s19_avwap_pinch, 20: st.s20_poc_shift, 21: st.s21_lvn, 22: st.s22_value_area,
      23: st.s23_hvb_flip, 25: st.s25_squeeze, 26: st.s26_rsi2, 27: st.s27_zscore, 30: st.s30_episodic_pivot,
      31: st.s31_squeeze_proxy, 33: st.s33_uoa_proxy, 38: st.s38_rs_leaders}
COOLDOWN = {20: 20, 26: 5, 27: 5, 30: 20, 31: 20, 38: 1}


def panel_from(prices: dict[str, pd.DataFrame], member: dict[str, np.ndarray], cal: pd.DatetimeIndex) -> core.Panel:
    syms = sorted(s for s in prices if s in member)
    D, S = len(cal), len(syms)
    M = {k: np.full((D, S), np.nan) for k in "OHLCV"}
    raw = np.full((D, S), np.nan)
    elig = np.zeros((D, S), dtype=bool)
    for j, s in enumerate(syms):
        x = prices[s].reindex(cal)
        fac = x["adj_close"] / x["close"]
        for k, col in zip("OHLC", ("open", "high", "low", "close")):
            M[k][:, j] = x[col].to_numpy(float) * fac.to_numpy(float)
        M["V"][:, j] = x["volume"].to_numpy(float)
        raw[:, j] = x["close"].to_numpy(float)
        elig[:, j] = member[s]
    H = np.fmax(M["H"], np.fmax(M["O"], M["C"]))
    L = np.fmin(M["L"], np.fmin(M["O"], M["C"]))
    elig &= np.isfinite(M["O"]) & (np.nan_to_num(raw) >= 5)
    return core.Panel(cal, syms, M["O"], H, L, M["C"], M["V"], elig)


def signals(p: core.Panel, chosen: dict[int, dict]) -> dict[int, np.ndarray]:
    out = {}
    for sid, fn in FN.items():
        prm = (chosen.get(sid) or {}).get("params", {})
        sig = fn(p, None, **prm)
        sig = np.asarray(sig, dtype=bool) & p.elig
        out[sid] = core.fresh(sig, COOLDOWN.get(sid, 10))
        print("  strategy", sid, int(out[sid].sum()), flush=True)
    return out


def features(p: core.Panel, sig: dict[int, np.ndarray], weights: dict[int, float] | None = None) -> dict[str, np.ndarray]:
    def last5(a):
        return core.roll(a.astype(float), 5, "max", minp=1) > 0
    recent = {sid: last5(s) for sid, s in sig.items()}
    f = {"conf_n1": sum(s.astype(int) for s in sig.values()), "conf_n5": sum(r.astype(int) for r in recent.values())}
    for fam, ids in FAMILY.items():
        f[f"fam_{fam}5"] = np.any([recent[i] for i in ids if i in recent], axis=0)
    f["n_fam5"] = sum(f[f"fam_{fam}5"].astype(int) for fam in FAMILY)
    if weights:
        f["ens5"] = sum(recent[i].astype(float) * w for i, w in weights.items() if w > 0 and i in recent)
    return f


def chosen_params(path: str) -> dict[int, dict]:
    d = json.load(open(path))
    return {int(k): v["chosen"] for k, v in d.items() if k.isdigit() and isinstance(v, dict) and "chosen" in v}

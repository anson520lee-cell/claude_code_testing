"""Indicator-triggered changes to the ATR trailing stop, single indicators and combinations (docs 7m).

Neo (2026-11-02), after 7l: "what if the indicators are combined with the ATR trailing stop? Test every indicator,
popular or obscure, and their combinations - maybe a good combo turns up by surprise."

Question per rule: when condition X first appears while a trade is open (close of day k), is it better to change the
exit from the next session on than to keep the live exit? Actions (per setup):
    trail1 / trail1.5 / trail2   tighten the trail to 1 / 1.5 / 2 ATR below the highest high since entry
    trail5                       widen it to 5 ATR (squeeze: drop the 3R target and trail 3 ATR = "let it run")
    breakeven                    raise the stop to the entry price
    sell                         sell at the next open (= 7l's rule, here for the new indicators and the combos)
Paired effect per trade = account result with the action from day k+1 - live result (0 when X never fires).

Speed: for every action and every possible firing day k the whole path is re-simulated ONCE (table E[action][trade, k]);
a rule is then just "first day where X is true" per trade (bit masks of the 20 trade-days), so millions of
combinations cost a gather each. Combinations are SEARCHED in the selection period only (2020-01..2024-06); the
shortlist is then judged on 2010-2019 and 2024-07.. exactly like 7j / 7l (both > 0, pooled t >= Bonferroni bar over the
shortlist, max-t permutation guard). The selection period is used up by the search, so its t is not evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.backtest import holding_study as hs

ACTIONS: dict[str, dict[str, dict[str, Any]]] = {
    "crash_rebound": {"trail1": {"trail": 1.0}, "trail1.5": {"trail": 1.5}, "trail2": {"trail": 2.0}, "trail5": {"trail": 5.0},
                      "breakeven": {"be": True}, "sell": {"sell": True}},
    "breakout": {"trail1": {"trail": 1.0}, "trail1.5": {"trail": 1.5}, "trail2": {"trail": 2.0}, "trail5": {"trail": 5.0},
                 "breakeven": {"be": True}, "sell": {"sell": True}},
    "squeeze": {"trail1": {"trail": 1.0}, "trail1.5": {"trail": 1.5}, "trail2": {"trail": 2.0},
                "trail5": {"trail": 3.0, "drop_target": True}, "breakeven": {"be": True}, "sell": {"sell": True}},
}
ACTION_TEXT = {"trail1": "tighten the trail to 1 ATR", "trail1.5": "tighten the trail to 1.5 ATR", "trail2": "tighten the trail to 2 ATR",
               "trail5": "widen the trail to 5 ATR (squeeze: drop the 3R target, trail 3 ATR)", "breakeven": "stop to break-even",
               "sell": "sell at the next open"}


def simulate(O: np.ndarray, Hh: np.ndarray, L: np.ndarray, C: np.ndarray, base: dict[str, np.ndarray], a: np.ndarray,
             kind: str, k_fire: np.ndarray, act: dict[str, Any], slip: float) -> np.ndarray:
    """Account result per trade when ``act`` applies from the session after day k_fire (0-based, -1 = never)."""
    p = hs.LIVE_EXIT[kind]
    m, hz = O.shape
    entry, stop0, tgt = base["entry"], base["stop0"], base["tgt"]
    live = base["ok"].copy()
    exit_px = np.full(m, np.nan)
    hi_prev = entry.copy()
    for k in range(hz):
        on = (k_fire >= 0) & (k > k_fire)
        st = stop0 if p["trail"] is None else np.maximum(stop0, hi_prev - p["trail"] * a)
        if "trail" in act:
            st = np.where(on, np.maximum(stop0, hi_prev - act["trail"] * a), st)
        if act.get("be"):
            st = np.where(on, np.maximum(st, entry), st)
        if act.get("sell"):
            st = np.where(on, np.inf, st)
        hit_s = live & (L[:, k] <= st)
        px_s = np.minimum(O[:, k], st)
        tg = np.where(on, np.inf, tgt) if act.get("drop_target") else tgt
        hit_t = live & ~hit_s & (Hh[:, k] >= tg)
        px_t = np.maximum(O[:, k], tg) if k > 0 else tg
        exit_px = np.where(hit_s, px_s, np.where(hit_t, px_t, exit_px))
        live &= ~(hit_s | hit_t)
        hi_prev = np.maximum(hi_prev, Hh[:, k])
    exit_px = np.where(live, C[:, hz - 1], exit_px) * (1 - slip)
    return np.where(base["ok"], (exit_px / entry - 1) * base["size"], 0.0)


def action_tables(O, Hh, L, C, base, a, kind: str, slip: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """E[action] (n, hz) = account result if the action starts after day k; plus the live result (n,)."""
    m, hz = O.shape
    live = simulate(O, Hh, L, C, base, a, kind, np.full(m, -1), {}, slip)
    E = {}
    for name, act in ACTIONS[kind].items():
        tab = np.repeat(live[:, None], hz, axis=1)
        for k in range(hz - 1):
            tab[:, k] = simulate(O, Hh, L, C, base, a, kind, np.full(m, k), act, slip)
        E[name] = tab
    return E, live


def bitmasks(masks: dict[str, np.ndarray], sig: np.ndarray, k1: np.ndarray, n_trades: int) -> tuple[list[str], np.ndarray]:
    """(names, B) with B[i, j] = bit (k-1) set when mask j is true on trade i's day k (k = 1..20)."""
    w = (2.0 ** (k1 - 1)).astype(np.float64)
    names = list(masks)
    B = np.zeros((n_trades, len(names)), dtype=np.uint32)
    for j, nm in enumerate(names):
        m = masks[nm]
        B[:, j] = np.bincount(sig[m], weights=w[m], minlength=n_trades).astype(np.uint32)
    return names, B


_LOG2 = {1 << i: i for i in range(32)}


def first_day(x: np.ndarray) -> np.ndarray:
    """0-based index of the lowest set bit (-1 for 0), works on any shape of uint32."""
    x = x.astype(np.uint32)
    lsb = x & (~x + np.uint32(1))
    out = np.full(x.shape, -1, dtype=np.int64)
    nz = lsb > 0
    out[nz] = np.log2(lsb[nz].astype(np.float64)).round().astype(np.int64)
    return out


def deltas(x: np.ndarray, E: np.ndarray, live: np.ndarray, hz: int) -> tuple[np.ndarray, np.ndarray]:
    """Paired account delta per trade and fired flag for bit-mask vector x (n,)."""
    k0 = first_day(x)
    fired = (k0 >= 0) & (k0 < hz - 1)
    d = np.zeros(len(x))
    i = np.flatnonzero(fired)
    d[i] = E[i, k0[i]] - live[i]
    return d, fired


def screen_pairs(B: np.ndarray, pool: list[int], E: dict[str, np.ndarray], live: np.ndarray, rows: np.ndarray, hz: int,
                 min_fired: int = 30, keep: int = 60) -> dict[str, list[tuple[float, int, int, float, int]]]:
    """Every pair (i < j) of pool masks, AND-combined, on the selection rows only. Returns per action the best
    ``keep`` pairs by trade-level t: (t, i, j, mean delta, n fired)."""
    Bs = B[rows][:, pool]
    Es = {a: e[rows] for a, e in E.items()}
    ls = live[rows]
    n = len(rows)
    best: dict[str, list] = {a: [] for a in E}
    ar = np.arange(n)[:, None]
    for ii in range(len(pool) - 1):
        X = Bs[:, ii:ii + 1] & Bs[:, ii + 1:]
        k0 = first_day(X)
        fired = (k0 >= 0) & (k0 < hz - 1)
        nf = fired.sum(axis=0)
        ok = np.flatnonzero(nf >= min_fired)
        if ok.size == 0:
            continue
        k0, fired, nf = k0[:, ok], fired[:, ok], nf[ok]
        kk = np.where(fired, k0, 0)
        for a, e in Es.items():
            d = np.where(fired, e[ar, kk] - ls[:, None], 0.0)
            mu = d.mean(axis=0)
            sd = d.std(axis=0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(sd > 0, mu / (sd / np.sqrt(n)), 0.0)
            top = np.argsort(-t)[:keep]
            best[a].extend((float(t[q]), pool[ii], pool[ii + 1 + ok[q]], float(mu[q]), int(nf[q])) for q in top)
        if ii % 50 == 49:
            for a in best:
                best[a] = sorted(best[a], reverse=True)[:keep]
    return {a: sorted(v, reverse=True)[:keep] for a, v in best.items()}


def extend_triples(B: np.ndarray, pairs: list[tuple[int, int]], pool: list[int], E: np.ndarray, live: np.ndarray,
                   rows: np.ndarray, hz: int, min_fired: int = 30, keep: int = 30) -> list[tuple[float, int, int, int, float, int]]:
    """Greedy third condition for the given pairs (one action), selection rows only."""
    Bs = B[rows]
    es, ls = E[rows], live[rows]
    n = len(rows)
    ar = np.arange(n)[:, None]
    out = []
    P = Bs[:, pool]
    for i, j in pairs:
        X = (Bs[:, i] & Bs[:, j])[:, None] & P
        k0 = first_day(X)
        fired = (k0 >= 0) & (k0 < hz - 1)
        nf = fired.sum(axis=0)
        okc = (nf >= min_fired) & np.isin(np.array(pool), [i, j], invert=True)
        if not okc.any():
            continue
        d = np.where(fired, es[ar, np.where(fired, k0, 0)] - ls[:, None], 0.0)
        mu, sd = d.mean(axis=0), d.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(okc & (sd > 0), mu / (sd / np.sqrt(n)), -np.inf)
        for q in np.argsort(-t)[:5]:
            if np.isfinite(t[q]):
                out.append((float(t[q]), i, j, pool[q], float(mu[q]), int(nf[q])))
    out = sorted(set(out), reverse=True)
    return out[:keep]

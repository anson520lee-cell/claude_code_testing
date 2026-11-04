"""A fast, configurable x80-style book.

Each week t (W-FRI) the book's target weights come only from sleeve returns up to week t-1:
  universe    which sleeves may be used (a named set)
  select      all | pos (trailing Sharpe > 0) | sr25 (> 0.25) | top10 (best 10 by trailing Sharpe); trailing = every
              week the sleeve has existed before t (at least 52), Sharpe of the economic excess return
  method      erc (equal risk contribution) | ivol | ew | hrp (hierarchical risk parity) | maxdiv (maximum
              diversification) | family (equal risk across the six families, equal risk inside each) |
              ens (average of erc, ivol and hrp) | sharpe (mean / variance, means shrunk half way to their average)
  cov         sample | shrink (correlations pulled 30% toward zero) | ewma (half-life = lookback / 3)
  look        weeks of history for the covariance (52, 104, 156, 260)
  cap         largest weight one sleeve may have; crypto_cap: largest share of the book's risk crypto may carry
  rebal       w (trade to target weekly, x80) | m (every 4 weeks, weights drift in between) |
              band (trade only when the drifted weights are more than 10% away from target in total)
  lev_band    the leverage is only reset when the drifted exposure is more than this far from target (0 = weekly, x80)
Costs and financing are x80 meta's: 0.10% of notional traded, funded sleeves borrow at T-bill + 1% (crypto + 4%),
unfunded long/short books earn the T-bill on their margin. Leverage = target vol / the book's vol over all weeks so far.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

COST = 0.001
CAP_LEV = 10.0
FAMILIES = ("crypto trend", "crypto relative value", "volatility selling", "equity long", "equity neutral", "macro")


@dataclass(frozen=True)
class Config:
    universe: str = "core16"
    select: str = "all"
    method: str = "erc"
    cov: str = "sample"
    look: int = 104
    cap: float = 1.0
    crypto_cap: float = 1.0
    rebal: str = "w"
    lev_band: float = 0.0

    def key(self) -> str:
        return "|".join(f"{k}={v}" for k, v in asdict(self).items())


BASELINE = Config()                                                    # x80 round 2: ERC 104 weeks on the 16 sleeves


# ---------------------------------------------------------------------------------------------- weighting methods
def erc(S: np.ndarray, budget: np.ndarray | None = None, iters: int = 60) -> np.ndarray:
    """Long-only equal-risk (or risk-budget) weights: Newton on 0.5 y'Sy - b'log y (Spinu 2013)."""
    n = len(S)
    b = np.full(n, 1.0 / n) if budget is None else budget / budget.sum()
    S = S + np.eye(n) * 1e-10 * np.trace(S) / n
    y = 1 / np.sqrt(np.maximum(np.diag(S), 1e-16))
    y *= np.sqrt(1.0 / (y @ S @ y))
    for _ in range(iters):
        g = S @ y - b / y
        step = np.linalg.solve(S + np.diag(b / y ** 2), g)
        t = 1.0
        while np.any(y - t * step <= 0):
            t *= 0.5
        y_new = y - t * step
        done = np.max(np.abs(y_new - y)) < 1e-10 * np.max(y)
        y = y_new
        if done:
            break
    return y / y.sum()


def _cluster_var(S: np.ndarray, idx: np.ndarray) -> float:
    s = S[np.ix_(idx, idx)]
    iv = 1 / np.maximum(np.diag(s), 1e-16)
    iv /= iv.sum()
    return float(iv @ s @ iv)


def hrp(S: np.ndarray) -> np.ndarray:
    """Hierarchical risk parity (Lopez de Prado 2016): single-linkage tree on correlation distance, bisection."""
    n = len(S)
    if n == 1:
        return np.ones(1)
    sd = np.sqrt(np.maximum(np.diag(S), 1e-16))
    corr = np.clip(S / np.outer(sd, sd), -1, 1)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), "single"))
    w = np.ones(n)
    stack = [order]
    while stack:
        c = stack.pop()
        if len(c) < 2:
            continue
        h = len(c) // 2
        a, b = c[:h], c[h:]
        va, vb = _cluster_var(S, a), _cluster_var(S, b)
        alpha = 1 - va / (va + vb) if va + vb > 0 else 0.5
        w[a] *= alpha
        w[b] *= 1 - alpha
        stack += [a, b]
    return w / w.sum()


def _project_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    return np.maximum(v - (css[k] - 1) / (k + 1), 0)


def min_var(S: np.ndarray, iters: int = 300) -> np.ndarray:
    """Long-only minimum variance by projected gradient on the simplex."""
    n = len(S)
    w = np.full(n, 1.0 / n)
    lr = 1.0 / max(np.linalg.eigvalsh(S)[-1], 1e-16)
    for _ in range(iters):
        w_new = _project_simplex(w - lr * (S @ w))
        if np.max(np.abs(w_new - w)) < 1e-9:
            w = w_new
            break
        w = w_new
    return w


def max_div(S: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.maximum(np.diag(S), 1e-16))
    v = min_var(S / np.outer(sd, sd))
    w = v / sd
    return w / w.sum()


def cap_weights(w: np.ndarray, cap: float) -> np.ndarray:
    if cap >= 1.0 or len(w) * cap < 1.0:
        return w
    w = w.copy()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        extra = (w[over] - cap).sum()
        w[over] = cap
        under = ~over & (w > 0)
        if not under.any():
            break
        w[under] += extra * w[under] / w[under].sum()
    return w


def crypto_risk_cap(w: np.ndarray, S: np.ndarray, is_c: np.ndarray, cap: float) -> np.ndarray:
    """Scale the crypto sleeves down until their share of the book's variance is at most ``cap``."""
    if cap >= 1.0 or not is_c.any() or is_c.all() or w[~is_c].sum() <= 0:
        return w                                                        # nothing else to hand the risk to

    def share(k: float) -> float:
        x = w.copy()
        x[is_c] *= k
        v = x @ S @ x
        return float((x[is_c] * (S @ x)[is_c]).sum() / v) if v > 0 else 0.0

    if share(1.0) <= cap:
        return w
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if share(mid) > cap:
            hi = mid
        else:
            lo = mid
    x = w.copy()
    x[is_c] *= lo
    return x / x.sum()


def weights(method: str, S: np.ndarray, mu: np.ndarray, fam: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.maximum(np.diag(S), 1e-16))
    if method == "erc":
        return erc(S)
    if method == "ivol":
        return (1 / sd) / (1 / sd).sum()
    if method == "ew":
        return np.full(len(S), 1.0 / len(S))
    if method == "hrp":
        return hrp(S)
    if method == "maxdiv":
        return max_div(S)
    if method == "ens":
        return (erc(S) + (1 / sd) / (1 / sd).sum() + hrp(S)) / 3
    if method == "sharpe":
        m = 0.5 * mu + 0.5 * mu.mean()
        v = np.clip(m, 0, None) / sd ** 2
        return v / v.sum() if v.sum() > 0 else (1 / sd) / (1 / sd).sum()
    if method == "family":
        fams = np.unique(fam)
        A = np.zeros((len(S), len(fams)))
        for k, f in enumerate(fams):
            idx = np.flatnonzero(fam == f)
            A[idx, k] = erc(S[np.ix_(idx, idx)]) if len(idx) > 1 else 1.0
        F = A.T @ S @ A
        return A @ erc(F)
    raise ValueError(method)


def covariance(y: np.ndarray, how: str, look: int) -> np.ndarray:
    if how == "ewma":
        hl = max(look / 3, 4)
        a = 0.5 ** (1 / hl)
        k = np.arange(len(y))[::-1]
        wt = a ** k
        wt /= wt.sum()
        m = wt @ y
        z = y - m
        return (z * wt[:, None]).T @ z / (1 - (wt ** 2).sum())
    S = np.cov(y, rowvar=False)
    S = np.atleast_2d(S)
    if how == "shrink":
        sd = np.sqrt(np.maximum(np.diag(S), 1e-16))
        R = S / np.outer(sd, sd)
        R = 0.7 * R + 0.3 * np.eye(len(S))
        S = R * np.outer(sd, sd)
    return S


# ---------------------------------------------------------------------------------------------- the book
class Pool:
    """Weekly excess returns of the candidate sleeves plus what the engine needs about each."""

    def __init__(self, w_ex: pd.DataFrame, cash_w: pd.Series, meta: pd.DataFrame, universes: dict[str, list[str]]):
        self.w_ex = w_ex
        self.idx = w_ex.index
        self.cols = list(w_ex.columns)
        self.X = w_ex.to_numpy(float)
        self.cash = cash_w.reindex(self.idx).fillna(0.0).to_numpy()
        m = meta.reindex(self.cols)
        self.unf = m["unfunded"].to_numpy(bool)
        self.spread = m["spread"].to_numpy(float)
        self.is_c = m["crypto"].to_numpy(bool)
        self.fam = m["family"].to_numpy(str)
        self.universes = {k: np.array([self.cols.index(c) for c in v]) for k, v in universes.items()}
        econ = self.X + np.where(self.unf[None, :], self.cash[:, None], 0.0)    # unfunded books' margin earns T-bills
        n_obs = np.cumsum(np.isfinite(econ), axis=0)
        s1 = np.nancumsum(econ, axis=0)
        s2 = np.nancumsum(econ ** 2, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s1 / n_obs
            var = (s2 - n_obs * mean ** 2) / (n_obs - 1)
            sr = mean / np.sqrt(var) * np.sqrt(52)
        sr[n_obs < 52] = np.nan
        self.trail_sr = np.vstack([np.full(len(self.cols), np.nan), sr[:-1]])     # known at the start of week t
        self.econ = econ


def run_book(pool: Pool, cfg: Config, start: int = 26, keep_weights: bool = False) -> dict[str, np.ndarray]:
    """Unlevered weekly book (x80 conventions) plus the financing mix, turnover and crypto risk share
    (and the weights held each week, with ``keep_weights``)."""
    X, T = pool.X, len(pool.idx)
    held_w = np.zeros((T, len(pool.cols))) if keep_weights else None
    uni = pool.universes[cfg.universe]
    n = len(pool.cols)
    base = np.full(T, np.nan)
    funded = np.zeros(T)
    spread = np.zeros(T)
    unfund = np.zeros(T)
    turn = np.zeros(T)
    crypto_share = np.full(T, np.nan)
    invested = np.zeros(T, bool)
    drifted = np.zeros(n)
    target = np.zeros(n)
    for i in range(start, T):
        retarget = cfg.rebal != "m" or (i - start) % 4 == 0 or not target.any()
        if retarget:
            lo = max(0, i - cfg.look)
            win = X[lo:i][:, uni]
            ok = np.isfinite(win).sum(axis=0) >= max(13, (i - lo) // 2)      # x80: half the window (13+ weeks)
            if cfg.select != "all":
                sr = pool.trail_sr[i, uni]
                good = np.isfinite(sr)
                if cfg.select == "pos":
                    good &= sr > 0
                elif cfg.select == "sr25":
                    good &= sr > 0.25
                elif cfg.select == "top10":
                    rank = np.where(good, sr, -np.inf)
                    keep = np.argsort(-rank)[:10]
                    mask = np.zeros(len(uni), bool)
                    mask[keep] = True
                    good &= mask
                ok &= good
            target = np.zeros(n)
            if ok.any():
                sel = uni[ok]
                y = np.nan_to_num(win[:, ok])
                S = covariance(y, cfg.cov, cfg.look) if len(sel) > 1 else np.atleast_2d(np.var(y, ddof=1))
                live_sd = np.diag(S) > 1e-14
                sel, S = sel[live_sd], S[np.ix_(live_sd, live_sd)]
                if len(sel):
                    mu = np.nanmean(pool.econ[lo:i][:, sel], axis=0)
                    w = weights(cfg.method, S, np.nan_to_num(mu), pool.fam[sel])
                    w = cap_weights(w, cfg.cap)
                    w = crypto_risk_cap(w, S, pool.is_c[sel], cfg.crypto_cap)
                    target[sel] = w
                    v = w @ S @ w
                    crypto_share[i] = float((w[pool.is_c[sel]] * (S @ w)[pool.is_c[sel]]).sum() / v) if v > 0 else 0.0
        row = X[i]
        live = np.isfinite(row)
        if np.any((target > 0) & ~live):
            missing = [pool.cols[j] for j in np.flatnonzero((target > 0) & ~live)]
            raise ValueError(f"Missing return for allocated sleeves at {pool.idx[i]}: {missing}; supply explicit delisting/closure returns")
        tw = target.copy()
        if tw.sum() <= 0:
            base[i] = 0.0
            turn[i] = np.abs(drifted).sum()
            drifted[:] = 0.0
            continue
        tw = tw / tw.sum()                                              # allocations are decided using past data only
        invested[i] = True
        if cfg.rebal == "band" and drifted.sum() > 0:
            dw = np.where(live, drifted, 0.0)
            dw = dw / dw.sum() if dw.sum() > 0 else dw
            if np.abs(dw - tw).sum() <= 0.10 and np.all((dw > 0) <= (tw > 0)):
                tw = dw
        elif cfg.rebal == "m" and not retarget and drifted.sum() > 0:
            dw = np.where(live, drifted, 0.0)
            if dw.sum() > 0:
                tw = dw / dw.sum()
        r = np.where(live, row, 0.0)
        b = float(tw @ r)
        base[i] = b
        f = tw[~pool.unf].sum()
        funded[i] = f
        spread[i] = float((tw[~pool.unf] * pool.spread[~pool.unf]).sum() / f) if f > 0 else 0.0
        unfund[i] = tw[pool.unf].sum()
        turn[i] = np.abs(tw - drifted).sum()
        drifted = tw * (1 + r) / (1 + b)
        if held_w is not None:
            held_w[i] = tw
    out_w = {"weights": held_w} if held_w is not None else {}
    return {**out_w, "base": base, "funded": funded, "spread": spread, "unfunded": unfund, "turn": turn, "crypto_share": crypto_share,
            "invested": invested}


def lever(book: dict[str, np.ndarray], cash: np.ndarray, tv: float, lev_band: float = 0.0, cap: float = CAP_LEV,
          trough: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Weekly account return and leverage. Target leverage = tv / the book's vol over all weeks so far (52+); the
    exposure drifts with the week's result and is only reset when it is more than ``lev_band`` from target.
    ``trough``: the unlevered book's worst cumulative return inside each week (a week that hits -100% is a wipe-out)."""
    base = book["base"]
    T = len(base)
    inv = book.get("invested", np.isfinite(base))
    s = pd.Series(np.where(inv, base, np.nan))                          # weeks with nothing to hold do not count as calm weeks
    bvol = s.expanding(min_periods=52).std().shift(1).ffill().to_numpy() * np.sqrt(52)
    with np.errstate(divide="ignore", invalid="ignore"):
        target = np.minimum(tv / bvol, cap)
    ret = np.full(T, np.nan)
    lev = np.full(T, np.nan)
    held = 0.0
    dead = False
    for i in range(T):
        if not np.isfinite(base[i]):
            continue
        if dead:
            ret[i], lev[i] = 0.0, 0.0
            continue
        if not inv[i] or not np.isfinite(target[i]) or target[i] <= 0:
            ret[i], lev[i], held = cash[i], 0.0, 0.0                   # nothing to hold yet: the account sits in T-bills
            continue
        L = target[i]
        if lev_band > 0 and held > 0 and abs(held / L - 1) <= lev_band:
            L = held
        trade = abs(L - held)
        c = cash[i]
        r = (L * base[i] + c + L * book["unfunded"][i] * c - max(L * book["funded"][i] - 1, 0.0) * book["spread"][i] / 52
             - COST * (L * book["turn"][i] + trade))
        if trough is not None and np.isfinite(trough[i]) and 1 + L * trough[i] <= 0:
            r = -1.0
        r = max(r, -1.0)
        ret[i], lev[i] = r, L
        if r <= -1.0:
            dead = True
            held = 0.0
            continue
        held = L * (1 + base[i]) / (1 + r)
    return ret, lev


def stats(r: np.ndarray, cash: np.ndarray, idx: pd.DatetimeIndex, a: str, z: str) -> dict:
    m = (idx >= a) & (idx <= z) & np.isfinite(r)
    x = r[m]
    if len(x) < 52:
        return {}
    ex = x - cash[m]
    eq = np.cumprod(1 + x)
    yrs = len(x) / 52
    sd = ex.std(ddof=1)
    return {"sharpe": float(ex.mean() / sd * np.sqrt(52)) if sd > 0 else np.nan,
            "cagr": float(eq[-1] ** (1 / yrs) - 1) if eq[-1] > 0 else -1.0,
            "max_dd": float((eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min()), "vol": float(x.std(ddof=1) * np.sqrt(52))}

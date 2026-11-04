"""Joint allocation constraints. Infeasible/unsolved allocations are explicit."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class AllocationBlocked(ValueError):
    """code 1 = proven infeasible; code 2 = numerical solve unresolved."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


def cap_weights(w: np.ndarray, cap: float) -> np.ndarray:
    """Proportional water filling; no silent disabling of infeasible limits."""
    if not np.isfinite(cap) or not 0 < cap <= 1:
        raise ValueError("asset cap must be in (0, 1]")
    if len(w) * cap < 1 - 1e-12:
        raise AllocationBlocked("asset cap cannot support full investment")
    if cap == 1 or np.max(w) <= cap + 1e-12:
        return w.copy()
    out = np.zeros_like(w, dtype=float)
    free = np.ones(len(w), dtype=bool)
    for _ in range(len(w) + 1):
        remaining = 1 - out.sum()
        if not free.any():
            if abs(remaining) <= 1e-12:
                return out
            break
        pref = np.maximum(w[free], 0)
        pref = pref / pref.sum() if pref.sum() > 0 else np.full(free.sum(), 1 / free.sum())
        proposal = remaining * pref
        over = proposal > cap + 1e-12
        if not over.any():
            out[free] = proposal
            return out
        fixed = np.flatnonzero(free)[over]
        out[fixed] = cap
        free[fixed] = False
    raise AllocationBlocked("asset cap allocation did not converge", code=2)


def crypto_share_of(w, S, is_c):
    variance = float(w @ S @ w)
    return float(w[is_c] @ (S @ w)[is_c] / variance) if variance > 1e-18 else 0.0


def within_caps(w, S, is_c, cap, crypto_cap):
    return bool(np.isfinite(w).all() and w.min() >= -1e-10 and abs(w.sum() - 1) <= 1e-8
                and w.max() <= cap + 1e-10 and (crypto_cap >= 1 or crypto_share_of(w, S, is_c) <= crypto_cap + 1e-8))


def crypto_risk_cap(w, S, is_c, cap):
    """Legacy scaling primitive; joint_caps checks feasibility of both limits."""
    if cap >= 1 or not is_c.any() or is_c.all() or w[~is_c].sum() <= 0:
        return w
    if crypto_share_of(w, S, is_c) <= cap:
        return w
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        x = w.copy()
        x[is_c] *= mid
        if crypto_share_of(x, S, is_c) > cap:
            hi = mid
        else:
            lo = mid
    x = w.copy()
    x[is_c] *= lo
    return x / x.sum()


def joint_caps(w, S, is_c, cap, crypto_cap):
    """Verify both limits; failed numerical solves never emit violating weights.

    Fast-path the unchanged baseline and valid legacy scaling. Otherwise solve
    a bounded closest-weight problem with analytic derivatives and two
    deterministic starts. Non-convergence is NOT proof of infeasibility.
    """
    if not np.isfinite(crypto_cap) or not 0 <= crypto_cap <= 1:
        raise ValueError("crypto cap must be in [0, 1]")
    bounded = cap_weights(w, cap)
    if within_caps(bounded, S, is_c, cap, crypto_cap):
        return bounded
    if is_c.all():
        raise AllocationBlocked("all eligible sleeves are crypto and crypto risk cap is below one")
    if abs(len(w) * cap - 1) <= 1e-12:
        raise AllocationBlocked("unique asset-capped allocation violates crypto risk cap")
    scaled = crypto_risk_cap(bounded, S, is_c, crypto_cap)
    if within_caps(scaled, S, is_c, cap, crypto_cap):
        return scaled
    A = (S + S.T) / (2 * max(float(np.trace(S)), 1e-18))
    mask = is_c.astype(float)

    def constraint(x):
        return float(crypto_cap * (x @ A @ x) - (x * mask) @ A @ x)

    def gradient(x):
        return 2 * crypto_cap * (A @ x) - mask * (A @ x) - A @ (x * mask)

    defensive = cap_weights((~is_c).astype(float) / (~is_c).sum(), cap)
    candidates = []
    for start in (bounded, defensive):
        if within_caps(start, S, is_c, cap, crypto_cap):
            candidates.append(start)
        fit = minimize(lambda x: 0.5 * np.sum((x - bounded) ** 2), start,
                       jac=lambda x: x - bounded, method="SLSQP", bounds=[(0, cap)] * len(w),
                       constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1,
                                     "jac": lambda x: np.ones_like(x)},
                                    {"type": "ineq", "fun": constraint, "jac": gradient}],
                       options={"maxiter": 80, "ftol": 1e-12})
        x = np.clip(fit.x, 0, cap)
        if within_caps(x, S, is_c, cap, crypto_cap):
            candidates.append(x)
    if not candidates:
        raise AllocationBlocked("joint-cap solve unresolved; no claim of mathematical infeasibility", code=2)
    return min(candidates, key=lambda x: float(np.sum((x - bounded) ** 2)))

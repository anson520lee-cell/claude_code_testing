"""Small daily backtest engine (no look-ahead: signal at close of day t, trade at open of t+1)."""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / "data"
CASH_YIELD = 0.04      # assumed annual yield on uninvested cash (T-bill like)
MARGIN_RATE = 0.10     # assumed annual cost on borrowed money (weight > 1)
COST_BPS = 5           # one-way slippage + spread, in basis points of traded notional


def load(sym: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{sym}.csv", parse_dates=["date"], index_col="date")


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def run(df: pd.DataFrame, weight: pd.Series, cost_bps: float = COST_BPS) -> pd.Series:
    """weight[t] is the target exposure decided with the close of day t. It is filled at the OPEN of t+1
    and held until the open of t+2 (so it earns open[t+2]/open[t+1]-1). The result is indexed by t."""
    o = df["open"]
    ret = o.shift(-2) / o.shift(-1) - 1
    w = weight.reindex(df.index).fillna(0.0)
    turnover = w.diff().abs().fillna(w.abs())
    cost = turnover * cost_bps / 1e4
    cash = (1 - w.clip(upper=1)).clip(lower=0) * CASH_YIELD / 252
    borrow = (w - 1).clip(lower=0) * MARGIN_RATE / 252
    net = w * ret + cash - borrow - cost
    return net.dropna()


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * 252 ** 0.5
    dd = (eq / eq.cummax() - 1).min()
    return dict(total=eq.iloc[-1] - 1, cagr=cagr, vol=vol, sharpe=(r.mean() * 252 - CASH_YIELD) / vol if vol else np.nan,
                maxdd=dd, calmar=cagr / abs(dd) if dd else np.nan)

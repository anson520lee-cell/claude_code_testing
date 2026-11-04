"""Candidate inputs for a leverage score, each known at the close of week t (W-FRI) and used for week t+1.

Market state (daily closes sampled at the week's last session):
  vix, vix_rel (VIX / its 1-year mean), vix_ts (VIX / VIX3M, > 1 = stress), vix_chg (4-week change),
  spy_trend (SPY / 200-day mean - 1), spy_mom12, spy_dd (from 1-year high), spy_rv20 (20-day realised vol),
  credit (HYG / IEF vs its 50-day mean), rates (13-week T-bill), rates_chg (13-week change),
  btc_trend, btc_rv (30-day), usd_trend (UUP vs 200-day mean).
State of the book itself (unlevered weekly excess returns; nothing after week t):
  b_mom4 / 13 / 26 / 52 (trailing return), b_hit13 / 26 / 52 (share of winning weeks = trailing win rate),
  b_sr52 (trailing Sharpe), b_dd (drawdown from the running peak), b_vr4 / b_vr13 (recent vol / vol of all history
  so far), s_corr13 (mean pairwise correlation of the sleeves, 13 weeks), s_breadth13 (weight share of sleeves
  whose last 13 weeks were positive).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKET = ["vix", "vix_rel", "vix_ts", "vix_chg", "spy_trend", "spy_mom12", "spy_dd", "spy_rv20", "credit", "rates",
          "rates_chg", "btc_trend", "btc_rv", "usd_trend"]
BOOK = ["b_mom4", "b_mom13", "b_mom26", "b_mom52", "b_hit13", "b_hit26", "b_hit52", "b_sr52", "b_dd", "b_vr4", "b_vr13",
        "s_corr13", "s_breadth13"]
LABEL = {"vix": "VIX 水平", "vix_rel": "VIX / 一年平均", "vix_ts": "VIX/VIX3M（>1 恐慌）", "vix_chg": "VIX 4 週變化",
         "spy_trend": "SPY 對 200 日線", "spy_mom12": "SPY 12 個月回報", "spy_dd": "SPY 距一年高位", "spy_rv20": "SPY 20 日波動",
         "credit": "高息債/國債 對 50 日線", "rates": "3 個月國債息率", "rates_chg": "國債息率 13 週變化", "btc_trend": "BTC 對 200 日線",
         "btc_rv": "BTC 30 日波動", "usd_trend": "美元 對 200 日線",
         "b_mom4": "組合近 4 週回報", "b_mom13": "組合近 13 週回報", "b_mom26": "組合近 26 週回報", "b_mom52": "組合近 52 週回報",
         "b_hit13": "組合近 13 週勝率", "b_hit26": "組合近 26 週勝率", "b_hit52": "組合近 52 週勝率", "b_sr52": "組合近 52 週 Sharpe",
         "b_dd": "組合回撤（未加槓桿）", "b_vr4": "近 4 週波動 / 長期波動", "b_vr13": "近 13 週波動 / 長期波動",
         "s_corr13": "策略之間相關性（13 週）", "s_breadth13": "近 13 週賺錢策略佔比"}


def _weekly(s: pd.Series) -> pd.Series:
    return s.dropna().resample("W-FRI").last()


def market(px: dict) -> pd.DataFrame:
    def c(sym, col="close"):
        return px[sym][col].dropna() if sym in px else pd.Series(dtype=float)

    vix, vix3m, spy, irx = c("^VIX"), c("^VIX3M"), c("SPY", "adj_close"), c("^IRX")
    f = {
        "vix": _weekly(vix),
        "vix_rel": _weekly(vix / vix.rolling(252, min_periods=200).mean()),
        "vix_ts": _weekly(vix / vix3m.reindex(vix.index)),
        "vix_chg": _weekly(vix / vix.shift(20) - 1),
        "spy_trend": _weekly(spy / spy.rolling(200).mean() - 1),
        "spy_mom12": _weekly(spy / spy.shift(252) - 1),
        "spy_dd": _weekly(spy / spy.rolling(252, min_periods=200).max() - 1),
        "spy_rv20": _weekly(spy.pct_change().rolling(20).std() * np.sqrt(252)),
        "rates": _weekly(irx / 100),
        "rates_chg": _weekly((irx - irx.shift(63)) / 100),
    }
    if "HYG" in px and "IEF" in px:
        ratio = (c("HYG", "adj_close") / c("IEF", "adj_close")).dropna()
        f["credit"] = _weekly(ratio / ratio.rolling(50).mean() - 1)
    if "BTC-USD" in px:
        b = c("BTC-USD")
        f["btc_trend"] = _weekly(b / b.rolling(200).mean() - 1)
        f["btc_rv"] = _weekly(b.pct_change().rolling(30).std() * np.sqrt(365))
    if "UUP" in px:
        u = c("UUP", "adj_close")
        f["usd_trend"] = _weekly(u / u.rolling(200).mean() - 1)
    return pd.DataFrame(f)


def book_state(base: pd.Series, w_ex: pd.DataFrame, W: pd.DataFrame) -> pd.DataFrame:
    b = base.copy()
    f = {}
    for k in (4, 13, 26, 52):
        f[f"b_mom{k}"] = b.rolling(k, min_periods=k).sum()
    for k in (13, 26, 52):
        f[f"b_hit{k}"] = (b > 0).astype(float).where(b.notna()).rolling(k, min_periods=k).mean()
    f["b_sr52"] = b.rolling(52).mean() / b.rolling(52).std() * np.sqrt(52)
    eq = (1 + b.fillna(0.0)).cumprod()
    f["b_dd"] = (eq / eq.cummax() - 1).where(b.notna())
    allv = b.expanding(min_periods=52).std()
    f["b_vr4"] = b.rolling(4).std() / allv
    f["b_vr13"] = b.rolling(13).std() / allv
    corr, breadth = {}, {}
    for i in range(13, len(w_ex)):
        t = w_ex.index[i]
        win = w_ex.iloc[i - 12:i + 1]
        win = win.loc[:, win.notna().sum() >= 10]
        if win.shape[1] >= 3:
            cm = win.corr().to_numpy()
            iu = np.triu_indices_from(cm, 1)
            corr[t] = float(np.nanmean(cm[iu]))
        if t in W.index:
            w = W.loc[t].reindex(win.columns).fillna(0.0)
            if w.sum() > 0:
                breadth[t] = float((w * (win.sum() > 0)).sum() / w.sum())
    f["s_corr13"] = pd.Series(corr)
    f["s_breadth13"] = pd.Series(breadth)
    return pd.DataFrame(f).reindex(base.index)


def all_features(px: dict, base: pd.Series, w_ex: pd.DataFrame, W: pd.DataFrame) -> pd.DataFrame:
    """Every candidate input at the close of week t (rows = the book's W-FRI weeks)."""
    m = market(px).reindex(base.index, method="ffill", limit=1)
    return pd.concat([m, book_state(base, w_ex, W)], axis=1)

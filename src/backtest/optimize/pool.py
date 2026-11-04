"""python -m src.backtest.optimize.pool X80_OUT RAW_X80 NEW_SLEEVES OUT_DIR : the candidate sleeves.

Writes OUT_DIR/pool_weekly.parquet (W-FRI excess returns over T-bills), pool_daily.parquet (daily excess, for the
intraweek wipe-out check) and pool_meta.csv (family, crypto, unfunded, borrowing spread).

  x80's 16 sleeves                exactly as levscore.sleeves rebuilt them (sleeves_weekly/daily.parquet)
  new (newsleeves.py)             RSI(2) mean reversion, G10 FX carry, turn of the month
  earlier variants, no re-picking ETH short vol (the rule picked for BTC on 2021-22), BTC vs alts beta-neutral,
                                  the walk-forward crypto factor mix, six classic beta-neutral S&P 500 factors,
                                  defensive momentum, top-50 multi-coin trend
Conventions follow x80 meta: crypto sleeves borrow at T-bill + 4%, others T-bill + 1%; long/short books that only
post margin are "unfunded" (their margin earns the T-bill, nothing is borrowed).
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from src.backtest.x80 import meta
from src.backtest.x80.common import cash_daily, load

FAMILY = {
    "BTC trend": "crypto trend", "ETH trend": "crypto trend", "BTC trend (28-rule avg)": "crypto trend",
    "ETH trend (28-rule avg)": "crypto trend", "multi-coin trend": "crypto trend", "multi-coin trend top50": "crypto trend",
    "crypto XS momentum": "crypto trend",
    "BTC vs alts": "crypto relative value", "BTC vs alts (beta neutral)": "crypto relative value",
    "crypto funding L/S": "crypto relative value", "crypto factor mix (walk-forward)": "crypto relative value",
    "short VIX + term filter": "volatility selling", "PutWrite": "volatility selling",
    "BTC short vol (DVOL)": "volatility selling", "ETH short vol (DVOL)": "volatility selling",
    "QQQ vol target": "equity long", "S&P momentum top20": "equity long", "live swing (crash+squeeze)": "equity long",
    "RSI(2) mean reversion": "equity long", "turn of the month": "equity long",
    "momentum top10 defensive": "equity long",
    "S&P 500 long/short": "equity neutral", "low volatility": "equity neutral", "momentum 12-1": "equity neutral",
    "profitability": "equity neutral", "value (book/market)": "equity neutral", "1-week reversal": "equity neutral",
    "low short volume": "equity neutral",
    "multi-asset trend": "macro", "G10 FX carry": "macro",
}
CORE = ["BTC trend", "ETH trend", "crypto XS momentum", "BTC vs alts", "short VIX + term filter", "QQQ vol target",
        "multi-asset trend", "S&P 500 long/short", "PutWrite", "live swing (crash+squeeze)", "crypto funding L/S",
        "BTC trend (28-rule avg)", "ETH trend (28-rule avg)", "multi-coin trend", "S&P momentum top20", "BTC short vol (DVOL)"]
NEW = ["RSI(2) mean reversion", "G10 FX carry", "turn of the month"]
CRYPTO_FAMILIES = {"crypto trend", "crypto relative value"}
CRYPTO_EXTRA = {"BTC short vol (DVOL)", "ETH short vol (DVOL)"}
UNFUNDED = {"BTC vs alts", "BTC vs alts (beta neutral)", "crypto funding L/S", "crypto factor mix (walk-forward)", "G10 FX carry"}
LS_FACTORS = {"low volatility": "low_ivol|beta", "momentum 12-1": "mom12_1|beta", "profitability": "gross_prof|beta",
              "value (book/market)": "value_bm|beta", "1-week reversal": "reversal_1w|beta", "low short volume": "low_svr5|beta"}


def is_crypto(name: str) -> bool:
    return FAMILY[name] in CRYPTO_FAMILIES or name in CRYPTO_EXTRA


def _from_first_trade(x: pd.Series) -> pd.Series:
    x = x.dropna()
    return x[x.index >= x.ne(0).idxmax()]


def build(x80_out: str, raw_x80: str, new_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    px = load(os.path.join(raw_x80, "prices.parquet"))
    daily = {}                                                         # name -> (daily TOTAL or pure L/S return, periods per year)
    cf = pd.read_parquet(f"{x80_out}/crypto_factors_series.parquet")
    daily["crypto factor mix (walk-forward)"] = (_from_first_trade(cf["walk-forward factor mix|beta"]), 365)
    vr = pd.read_parquet(f"{x80_out}/crypto_vrp_series.parquet")
    daily["ETH short vol (DVOL)"] = (_from_first_trade(vr["ETH short vol, iv>rv10, size 0.5, haircut 2"]), 365)
    cl = pd.read_parquet(f"{x80_out}/crypto_ls_series.parquet")
    daily["BTC vs alts (beta neutral)"] = (_from_first_trade(cl["long BTC / short top50 alts (beta neutral)"]), 365)
    ls = pd.read_parquet(f"{x80_out}/ls_neutral_series.parquet")
    for name, col in LS_FACTORS.items():
        daily[name] = (_from_first_trade(ls[col]), 252)
    im = pd.read_parquet(f"{x80_out}/improve_series.parquet")
    daily["momentum top10 defensive"] = (_from_first_trade(im["momentum 6-1 top10 + SPY>200d + vol 20%"]), 252)
    daily["multi-coin trend top50"] = (_from_first_trade(im["multi-coin trend top50"]), 365)
    ex_daily = {}
    for name, (x, ppy) in daily.items():
        ex_daily[name] = x - cash_daily(px, x.index, ppy)
    new = pd.read_parquet(new_path)                                    # already excess of T-bills (FX carry: pure L/S minus T-bill below)
    for name in NEW:
        x = new[name].dropna()
        if name == "G10 FX carry":
            x = x - cash_daily(px, x.index, 260)
        ex_daily[name] = x
    core_d = pd.read_parquet(f"{x80_out}/sleeves_daily.parquet")
    core_w = pd.read_parquet(f"{x80_out}/sleeves_weekly.parquet")
    cal = pd.date_range("2004-01-01", core_d.index.max(), freq="D")
    d = core_d.reindex(cal)
    for name, x in ex_daily.items():
        x = x[x.index >= "2004-01-01"]
        s = x.reindex(cal)
        live = (cal >= x.index.min()) & (cal <= x.index.max())
        s[live] = s[live].fillna(0.0)
        d[name] = s
    w = core_w.copy()
    for name, x in ex_daily.items():
        wk = (1 + x).resample("W-FRI").prod() - 1
        w[name] = wk.reindex(w.index)
    cols = CORE + NEW + [c for c in ex_daily if c not in NEW]
    meta_df = pd.DataFrame({"family": [FAMILY[c] for c in cols], "crypto": [is_crypto(c) for c in cols],
                            "unfunded": [c in UNFUNDED for c in cols],
                            "spread": [0.04 if is_crypto(c) else 0.01 for c in cols],
                            "core": [c in CORE for c in cols], "new": [c in NEW for c in cols]}, index=cols)
    assert all(meta_df.loc[c, "spread"] == meta.spread_of(c) for c in CORE), "core spreads must match x80 meta"
    assert {c for c in CORE if meta_df.loc[c, "unfunded"]} == set(meta.UNFUNDED)
    return w[cols], d[cols], meta_df


def main() -> None:
    x80_out, raw_x80, new_path, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    w, d, m = build(x80_out, raw_x80, new_path)
    w.to_parquet(os.path.join(out, "pool_weekly.parquet"))
    d.to_parquet(os.path.join(out, "pool_daily.parquet"))
    m.to_csv(os.path.join(out, "pool_meta.csv"))
    pre = w[w.index < "2025-01-01"]
    for c in w:
        x = pre[c].dropna()
        print(f"{c:36s} {m.loc[c, 'family']:22s} from {x.index.min().date()}  Sharpe(<2025) {x.mean() / x.std() * 52 ** 0.5:5.2f}")


if __name__ == "__main__":
    main()

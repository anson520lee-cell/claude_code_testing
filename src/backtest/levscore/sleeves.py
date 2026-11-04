"""python -m src.backtest.levscore.sleeves OUT_DIR : the 16 sleeves of the x80 round-2 combination, daily and weekly.

The sleeves are built by the x80 code itself (src.backtest.x80.frontier.sleeves and src.backtest.x80.meta.new_sleeves,
which read the *_series.parquet files that the x80 studies write into OUT_DIR); this module only records, next to
the weekly excess returns that meta.py combines, the daily excess return series each weekly series is made from
(needed to see the losses inside a week at 5-10x leverage). Writes
  OUT_DIR/sleeves_daily.parquet   daily excess returns over T-bills (calendar days; NaN before a sleeve starts)
  OUT_DIR/sleeves_weekly.parquet  weekly (W-FRI) excess returns, identical to meta.py's meta_sleeves_weekly
  OUT_DIR/cash_weekly.parquet     weekly T-bill return used by meta.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from src.backtest.x80 import frontier, meta
from src.backtest.x80.common import cash_daily, load

_CAPTURED: list[pd.Series] = []
_ORIG = frontier.weekly_excess


def _recording_weekly_excess(r: pd.Series, cash: pd.Series) -> pd.Series:
    ex = (r - cash.reindex(r.index).fillna(0.0)).dropna()
    _CAPTURED.append(ex)
    return _ORIG(r, cash)


def build(out_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(daily excess, weekly excess, weekly cash) for the 16 sleeves meta.py combines."""
    frontier.OUT = out_dir
    frontier.weekly_excess = _recording_weekly_excess
    meta.weekly_excess = _recording_weekly_excess
    try:
        px = load()
        _CAPTURED.clear()
        old = frontier.sleeves(px)
        daily_old = dict(zip(old.columns, _CAPTURED))
        _CAPTURED.clear()
        new = meta.new_sleeves(px, out_dir)
        daily_new = dict(zip(new.columns, _CAPTURED))
    finally:
        frontier.weekly_excess = _ORIG
        meta.weekly_excess = _ORIG
    w_ex = pd.concat([old, new], axis=1)
    w_ex = w_ex.drop(columns=[c for c in meta.EXCLUDE if c in w_ex])
    w_ex = w_ex[w_ex.index >= "2004-01-01"]
    daily = {**daily_old, **daily_new}
    daily = {k: v for k, v in daily.items() if k in w_ex.columns}
    cal = pd.date_range("2004-01-01", max(v.index.max() for v in daily.values()), freq="D")
    d = pd.DataFrame({k: v.reindex(cal) for k, v in daily.items()})
    for c in d:                                                           # inside a sleeve's life a non-trading day earns 0 excess
        x = daily[c]
        live = (d.index >= x.index.min()) & (d.index <= x.index.max())
        d.loc[live, c] = d.loc[live, c].fillna(0.0)
    cash_w = (1 + cash_daily(px, pd.date_range("2004-01-01", "2026-11-04"), 365)).resample("W-FRI").prod().reindex(w_ex.index).fillna(0) - 1
    return d[list(w_ex.columns)], w_ex, cash_w


def main() -> None:
    out_dir = sys.argv[1]
    d, w, c = build(out_dir)
    d.to_parquet(os.path.join(out_dir, "sleeves_daily.parquet"))
    w.to_parquet(os.path.join(out_dir, "sleeves_weekly.parquet"))
    c.to_frame("cash").to_parquet(os.path.join(out_dir, "cash_weekly.parquet"))
    chk = (1 + d).resample("W-FRI").prod().sub(1).reindex(w.index)
    err = (chk - w).abs().max()
    print("sleeves", list(w.columns))
    print("max |weekly from daily - weekly|:", err.round(6).to_dict())
    print("first week:", {c_: str(w[c_].first_valid_index().date()) for c_ in w})


if __name__ == "__main__":
    main()

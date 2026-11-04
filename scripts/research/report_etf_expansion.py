"""Publish the completed separate ETF study and cost stress without selecting again."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from scripts.research import etf_expansion as ex

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"research_data/frozen_20261104"
OUT = DATA/"etf_expansion_v1"
REPORT = ROOT/"reports/etf_expansion"


def main():
    progress = json.loads((OUT/"progress.json").read_text())
    if not progress["complete"] or progress["completed"] != 1728:
        raise ValueError("Study is incomplete")
    summary = json.loads((OUT/"summary.json").read_text())
    cfg = ex.Config(**{k:summary["frozen_winner"][k] for k in ex.Config.__dataclass_fields__})
    data = ex.Data(DATA)
    stressed, peak, ruined = data.run(cfg, training=False, cost=.002)
    validation = {"frozen_configuration": summary["frozen_winner"]["id"],
                  "cost_stress_one_way":.002, "peak_exposure":float(peak), "ruined":bool(ruined)}
    for name, mask in [("2008-24",data.training),("reused_2025_26",~data.training)]:
        validation[name] = ex.statistics(stressed[mask],data.index[mask],data.cash[mask])
    benchmarks = {}
    for symbol in ["SPY","QQQ"]:
        j = ex.UNIVERSES["us"].index(symbol)
        r, _, _ = ex.simulate(np.ascontiguousarray(data.r["us"][:,[j]]),
                             np.ones((len(data.index),1)),data.rate_values,data.days,
                             len(data.index)+1,1.,.001)
        benchmarks[symbol] = {
            "2008-24":ex.statistics(r[data.training],data.index[data.training],data.cash[data.training]),
            "reused_2025_26":ex.statistics(r[~data.training],data.index[~data.training],data.cash[~data.training])}
    validation["benchmarks"] = benchmarks
    ex.atomic_json(REPORT/"validation.json",validation)
    for name in ["metrics.csv","summary.json","frozen_winner.json"]:
        source, dest = OUT/name, REPORT/name
        if dest.exists() and dest.read_bytes() != source.read_bytes():
            raise ValueError("Existing published result differs: "+str(dest))
        if not dest.exists():
            shutil.copy2(source,dest)
    w = summary["frozen_winner"]
    later = summary["reused_2025_26"]
    text = f"""# ETF expansion: completed historical screen

1728 of 1728 preregistered trials completed, each transactionally checkpointed.
All four selection subperiods at CAGR >=80%: **{summary['pre2025_all_subperiods_80']}**.
No strategy from this study qualifies for stable 80-90% annual returns.

The frozen pre-2025 winner is trial {w['id']}: universe {w['universe']},
{w['look']}-session positive momentum, top {w['top']}, inverse 60-session
volatility sizing, target vol {w['target_vol']:.0%}, exposure target capped at
{w['exposure_cap']:.1f}x, rebalance every {w['rebalance']} sessions.
Signals execute at the following close and first earn returns one session later.

| Period | CAGR | Daily maximum drawdown |
|---|---:|---:|
| 2008-2012 | {w['cagr_2008-12']:.2%} | {w['max_dd_2008-12']:.2%} |
| 2013-2017 | {w['cagr_2013-17']:.2%} | {w['max_dd_2013-17']:.2%} |
| 2018-2021 | {w['cagr_2018-21']:.2%} | {w['max_dd_2018-21']:.2%} |
| 2022-2024 | {w['cagr_2022-24']:.2%} | {w['max_dd_2022-24']:.2%} |
| 2008-2024 | {w['cagr_2008-24']:.2%} | {w['max_dd_2008-24']:.2%} |
| Reused 2025-2026 | {later['cagr']:.2%} | {later['max_dd']:.2%} |

The 2008-24 minimum rolling three-year CAGR is {w['rolling_3y_min_2008-24']:.2%}.
Worst calendar-year return, including partial years, is {w['worst_year_2008-24']:.2%}.
Peak marked-to-market exposure before the closing risk adjustment was
{summary['full_peak_exposure']:.3f}x; a closing target cap does not prove intraday margin headroom.

One-way trading/slippage costs were 10 bp; the same frozen rule was then checked
at 20 bp (validation.json), without selecting or tuning again. Borrowing uses
the lagged T-bill proxy plus 1%, for actual calendar days. Individual taxes are excluded.

This is a separate expansion, not completion of the original 10026-configuration
30-sleeve study. The original inputs remain incomplete. Prior project research has
already observed 2025-26; it is reused historical validation, never a fresh holdout.
Fixed ETF fund selection has design/survivorship limitations, and neither tradable
intraday margin nor a future return guarantee is established.
"""
    (REPORT/"RESULTS.md").write_text(text,encoding="utf-8")
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/"checkpoint.sqlite3",OUT/"frozen_winner_daily.parquet",DATA/"collection.json"]}
    ex.atomic_json(REPORT/"artifact_hashes.json",{"source_directory":str(OUT),"hashes":hashes,
                 "note":"Full checkpoint/daily artifacts remain on the Windows research host; public summary and all metrics are in Git."})
    print(text,flush=True)
    print(json.dumps(validation,indent=2),flush=True)


if __name__ == "__main__":
    main()

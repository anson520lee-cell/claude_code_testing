"""python -m src.backtest.levscore.trade_check SIGNALS FLAGS PRICES X80_RAW RESULT_DIR [score|squeeze] : is the live
per-trade sizing real? 'score' rebuilds the score trades.py chose on 2010-17 (RESULT_DIR/trades.json) for the live set;
'squeeze' halves every squeeze trade (in 2010-17 squeeze trades lost 0.41 R each on average, crash rebounds made 0.22 R).

1. Out of sample (2018-26), mean R and win rate of the score's thirds, and a month-clustered t of R on the score
   (continuous) and on 'bottom third' (the trades the rule halves).
2. Placebo: the rule's multipliers shuffled among the out-of-sample trades 200 times (the account then holds the same
   mix of sizes, only on the wrong trades); p = share of shuffles whose CAGR beats the real rule's.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from src.backtest.levscore import trades as tr

SEEDS = 3
N_PLACEBO = 200


def main() -> None:
    sig, flags, prices, x80_raw, res = sys.argv[1:6]
    rule = sys.argv[6] if len(sys.argv) > 6 else "score"
    with open(os.path.join(res, "trades.json")) as fh:
        info = json.load(fh)
    signs = {k: int(v) for k, v in info["live (live-like + FINRA filter)"]["signs"].items()}
    s = tr.load_trades(sig, flags, x80_raw)
    cal, A, cols = tr.price_matrix(prices)
    t = s[tr.live_mask(s)].copy()
    t["score"] = tr.score_column(t, signs, 100)
    t["pct"] = tr.expanding_pct_by_date(t, "score", 100)
    t["m"] = tr.tiers(t["pct"])
    if rule == "squeeze":
        t["m"] = np.where(t["setup"] == "squeeze", 0.5, 1.0)
    else:
        t.to_parquet(os.path.join(res, "trade_live_scored.parquet"))
    rows = []
    for lab, (a, b) in (("2018-26", tr.PERIODS["2018-26"]), ("2022-26", tr.PERIODS["2022-26"])):
        o = t[(t["date"] >= a) & (t["date"] <= b)]
        bottom = (o["m"] == 0.5).astype(float)
        row = {"rule": rule, "period": lab, "n": len(o), "t_score": tr.cluster_t(o["r"], o["score"], o["month"]),
               "t_bottom": tr.cluster_t(o["r"], bottom, o["month"])}
        for m_, g in o.groupby("m"):
            row[f"n_{m_}"], row[f"win_{m_}"], row[f"meanR_{m_}"], row[f"medR_{m_}"] = len(g), float(g["win"].mean()), float(g["r"].mean()), float(g["r"].median())
        oos = (t["date"] >= a) & (t["date"] <= b)

        def cagr(mult: pd.Series, seeds=range(SEEDS), a=a, b=b) -> float:
            t["mm"] = mult
            return float(np.mean([tr.curve_stats(tr.simulate(t, cal, A, cols, seed=sd, mult_col="mm"), a, b)["cagr"] for sd in seeds]))

        t0 = time.time()
        real = cagr(t["m"])
        cm = float(t.loc[oos, "m"].mean())
        matched = cagr(pd.Series(np.where(oos, cm, t["m"]), index=t.index))
        rng = np.random.default_rng(0)
        sims = []
        for _ in range(N_PLACEBO):
            mm = t["m"].copy()
            mm[oos] = rng.permutation(mm[oos].to_numpy())
            sims.append(cagr(mm, seeds=(0,)))
        sims = np.array(sims)
        row.update({"cagr_rule": real, "cagr_matched": matched, "placebo_median": float(np.median(sims)),
                    "placebo_p95": float(np.quantile(sims, 0.95)), "p": float((sims >= real).mean()), "secs": time.time() - t0})
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(res, "trade_check.csv" if rule == "score" else f"trade_check_{rule}.csv"), index=False)


if __name__ == "__main__":
    main()

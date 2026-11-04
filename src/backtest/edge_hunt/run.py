"""python -m src.backtest.edge_hunt.run --features features.parquet --out reports/edge_hunt"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.backtest.edge_hunt.gate import IS, OOS, SETUPS, _rows, prepare, test_all


def baseline(df: pd.DataFrame) -> list[dict]:
    from src.backtest.edge_hunt.gate import cluster_t
    out = []
    for pool in ("P", "live_like", "B"):
        for s in SETUPS[:3]:
            x = _rows(df, s, pool)
            for per, (a, b) in (("is", IS), ("oos", OOS)):
                z = x[(x["date"] >= a) & (x["date"] <= b)].dropna(subset=["exc"])
                g = z.groupby(z["date"].dt.to_period("M"))["exc"].mean()
                t = float(g.mean() / g.std() * np.sqrt(len(g))) if len(g) > 2 else float("nan")
                out.append({"pool": pool, "setup": s, "period": per, "n": int(len(z)), "avg_r": float(z["y"].mean()),
                            "random_r": float(z["y"].mean() - z["exc"].mean()), "edge": float(z["exc"].mean()), "t": t})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="", help="comma-separated parquet files keyed by symbol, date")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_parquet(a.features)
    df = df[df["date"] >= "2009-06-01"].reset_index(drop=True)
    for extra in filter(None, a.extra.split(",")):
        x = pd.read_parquet(extra).drop_duplicates(["symbol", "date"])
        df = df.merge(x, on=["symbol", "date"], how="left")
    df = prepare(df)
    df.to_parquet(os.path.join(a.out, "prepared.parquet"))
    res = test_all(df)
    res.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    base = baseline(df)
    summ = {"n_tests": int(len(res)), "n_is_pass": int(res["is_pass"].sum()), "z_needed": float(res["z_needed"].iloc[0]),
            "n_oos_pass": int(res["oos_pass"].sum()), "n_pass": int(res["pass"].sum()), "baseline": base,
            "passed": res[res["pass"]].to_dict("records"), "is_survivors": res[res["is_pass"]].to_dict("records")}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=str)
    cols = ["setup", "feature", "kind", "is_n1", "is_d", "is_t", "oos_n1", "oos_d", "oos_t", "oos_te", "all_t",
            "live_like_all_d", "B_all_d", "oos_edge_c", "pass"]
    print(f"tests {len(res)}  IS survivors {summ['n_is_pass']}  z needed {summ['z_needed']:.2f}  OOS pass {summ['n_oos_pass']}  PASS {summ['n_pass']}")
    print(res[res["is_pass"]][cols].sort_values("oos_t").to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()

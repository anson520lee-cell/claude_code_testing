"""Edge hunt v2: every strategy (live long / live short / online long / online short / standalone) x every
pre-registered condition (gate.SPEC), on point-in-time S&P 500 members, with two hypothesis tiers.

Tier 1 = the live long setups (crash_rebound, breakout, squeeze) and standalone long entries: the question the
         study started with ("improve the live strategy"); Bonferroni k = tier-1 in-sample survivors.
Tier 2 = everything added later (live short setups, standalone shorts, 30 online long and 10 online short
         strategies); its own Bonferroni k. A tier-2 pass is as strict as tier 1 on its own family of tests.
Pass rule per test (same as gate.py): IS |t| >= 2 with both groups >= 30 and the excess delta of the same sign;
OOS same sign with t >= z(1 - 0.05 / k) and excess t >= 1; pooled t >= 2.5.
Percentile features are ranked against ALL eligible members on the same day (the grid).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.backtest.edge_hunt.gate import CRASH_ONLY, IS, OOS, SPEC, cluster_t

LIVE_LONG = ("crash_rebound", "breakout", "squeeze")
LIVE_SHORT = ("breakdown", "rally_fail", "ema21_loss")
EXIT_OF = {"crash_rebound": "crash", "breakout": "bo", "squeeze": "sq"}
STORY_NEEDS_SHORT_OK = True


def build_grid(d: str) -> pd.DataFrame:
    g = pd.read_parquet(os.path.join(d, "grid.parquet"))
    s = pd.read_parquet(os.path.join(d, "grid_src.parquet"))
    g = g.merge(s, on=["symbol", "date"], how="left")
    g["tgt_upside"] = g["tgt_mean"] / g["close"] - 1
    g["serial"] = (g["crash_cnt"] >= 1).astype("float32").where(g["crash_cnt"].notna())

    def b(cond, *need):
        known = pd.concat(need, axis=1).notna().all(axis=1)
        return cond.astype("float32").where(known)

    g["conf2"] = b(g["conf_n5"] >= 2, g["conf_n5"])
    g["conf3"] = b(g["conf_n5"] >= 3, g["conf_n5"])
    g["conf1d2"] = b(g["conf_n1"] >= 2, g["conf_n1"])
    g["multifam"] = b(g["n_fam5"] >= 2, g["n_fam5"])
    g["ens_pos"] = b(g["ens5"] > 0, g["ens5"])
    is_m = (g["date"] >= IS[0]) & (g["date"] <= IS[1])
    g["ens_hi"] = b(g["ens5"] >= g.loc[is_m & (g["ens5"] > 0), "ens5"].quantile(0.75), g["ens5"])
    g["catalyst_bo"] = b((g["earn_in3"] >= 0.5) & (g["last_surprise"] > 0), g["earn_in3"], g["last_surprise"])
    g["defend"] = b((g["up5"] >= 0.5) | (g["traise5"] >= 0.5), g["up5"], g["traise5"])
    feats = sorted({f for _, f, _, ref in SPEC if ref == "date" and f != "quality"})
    pct = {}
    for f in feats:
        pct[f] = g.groupby("date")[f].rank(pct=True).astype("float32")
    q = pd.concat([pct["gpa"], pct["roa"], 1 - pct["accruals"], 1 - pct["lev"]], axis=1)
    g["quality"] = q.mean(axis=1).where(q.notna().sum(axis=1) >= 3)
    pct["quality"] = g.groupby("date")["quality"].rank(pct=True).astype("float32")
    g["pess"] = b((pct["svr_abn"] >= 2 / 3) & ((g["down5"] >= 0.5) | (g["tcut5"] >= 0.5)), pct["svr_abn"], g["down5"], g["tcut5"])
    g["retail_dip"] = b((pct["offex_abn"] >= 2 / 3) & (g["ret5"] < 0), pct["offex_abn"], g["ret5"])
    g["smart_conf"] = b((g["conf_n5"] >= 2) & (pct["quality"] >= 2 / 3), g["conf_n5"], pct["quality"])
    for f, p in pct.items():
        g[f"pct_{f}"] = p
    return g


def conditions(g: pd.DataFrame, crash_rows: pd.DataFrame) -> list[str]:
    cols = []
    is_m = (g["date"] >= IS[0]) & (g["date"] <= IS[1])
    daily = g.drop_duplicates("date").set_index("date")
    for fam, f, kind, ref in SPEC:
        name = f"cond__{f}__{kind}"
        if ref == "date":
            p = g.get(f"pct_{f}")
            if p is None:
                continue
            v = (p >= 2 / 3) if kind == "top" else (p <= 1 / 3)
            g[name] = v.astype("float32").where(p.notna())
        elif ref == "signal":
            base = crash_rows.loc[(crash_rows["date"] >= IS[0]) & (crash_rows["date"] <= IS[1]), f].dropna()
            lo, hi = base.quantile(1 / 3), base.quantile(2 / 3)
            v = g[f]
            g[name] = ((v >= hi) if kind == "top" else (v <= lo)).astype("float32").where(v.notna())
        elif ref == "daily":
            s = daily[f]
            s = s[(s.index >= ("2010-01-01")) & (s.index <= ("2015-12-31" if f == "pc10" else IS[1]))].dropna()
            lo, hi = s.quantile(1 / 3), s.quantile(2 / 3)
            v = g[f]
            g[name] = ((v >= hi) if kind == "top" else (v <= lo)).astype("float32").where(v.notna())
        else:
            if f not in g:
                continue
            v = g[f]
            g[name] = (v.astype("float32") >= 0.5).astype("float32").where(v.notna())
        cols.append(name)
    return cols


def build_rows(d: str, sig_path: str) -> pd.DataFrame:
    s = pd.read_parquet(sig_path)
    s = s[s["member"] & (s["date"] >= "2009-06-01")]
    rnd = s[s["setup"] == "random"]
    dm = {k: rnd.groupby("date")[f"r_{k}"].mean() for k in ("crash", "bo", "sq", "short")}
    parts = []
    for st in LIVE_LONG:
        x = s[s["setup"] == st]
        parts.append(pd.DataFrame({"strategy": st, "side": 1, "tier": 1, "symbol": x["symbol"], "date": x["date"],
                                   "y": x["r"], "exc": x["r"] - x["date"].map(dm[EXIT_OF[st]])}))
    for st in LIVE_SHORT:
        x = s[s["setup"] == st]
        parts.append(pd.DataFrame({"strategy": st, "side": -1, "tier": 2, "symbol": x["symbol"], "date": x["date"],
                                   "y": x["r"], "exc": x["r"] - x["date"].map(dm["short"])}))
    parts.append(pd.DataFrame({"strategy": "standalone_long", "side": 1, "tier": 1, "symbol": rnd["symbol"], "date": rnd["date"],
                               "y": rnd["r_crash"], "exc": rnd["r_crash"] - rnd["date"].map(dm["crash"])}))
    parts.append(pd.DataFrame({"strategy": "standalone_short", "side": -1, "tier": 2, "symbol": rnd["symbol"], "date": rnd["date"],
                               "y": rnd["r_short"], "exc": rnd["r_short"] - rnd["date"].map(dm["short"])}))
    r = pd.concat(parts, ignore_index=True)
    fx = [f"fx{k}" for k in (1, 3, 5, 10, 20)]
    fw = pd.read_parquet(os.path.join(d, "sig_fwd.parquet")).drop_duplicates(["symbol", "date", "setup"])
    fw["strategy"] = fw["setup"].replace({"random": "standalone_long"})
    fs = fw[fw["strategy"] == "standalone_long"].assign(strategy="standalone_short")
    fw = pd.concat([fw, fs])[["symbol", "date", "strategy"] + fx]
    r = r.merge(fw, on=["symbol", "date", "strategy"], how="left")
    for c in fx:                       # forward excess in the trade's direction
        r[c] = r[c] * r["side"]
    z = pd.read_parquet(os.path.join(d, "zoo_rows.parquet"))
    z["tier"] = 2
    z = z[z["date"] >= "2009-06-01"]
    return pd.concat([r, z], ignore_index=True).dropna(subset=["y", "exc"])


def run_tests(R: pd.DataFrame, conds: list[str]) -> pd.DataFrame:
    res = []
    R = R.copy()
    R["month"] = R["date"].dt.to_period("M").astype(str)
    per = {"is": (R["date"] >= IS[0]) & (R["date"] <= IS[1]), "oos": (R["date"] >= OOS[0]) & (R["date"] <= OOS[1]),
           "all": (R["date"] >= IS[0]) & (R["date"] <= OOS[1])}
    for (strategy, side, tier), x in R.groupby(["strategy", "side", "tier"]):
        for col in conds:
            f = col.split("__")[1]
            if f in CRASH_ONLY and strategy != "crash_rebound":
                continue
            c_all = x[col]
            ok = c_all.notna()
            row = {"strategy": strategy, "side": side, "tier": tier, "feature": f, "kind": col.split("__")[2]}
            for pname, pm in per.items():
                if f == "pc10":
                    pm = {"is": (x["date"] >= "2010-01-01") & (x["date"] <= "2015-12-31"),
                          "oos": (x["date"] >= "2016-01-01") & (x["date"] <= "2019-10-31"),
                          "all": (x["date"] >= "2010-01-01") & (x["date"] <= "2019-10-31")}[pname]
                else:
                    pm = pm.loc[x.index]
                z = x[ok & pm]
                c = z[col].to_numpy(dtype=float)
                d_, t_, n1, n0 = cluster_t(z["y"].to_numpy(dtype=float), c, z["month"].to_numpy())
                de, te, _, _ = cluster_t(z["exc"].to_numpy(dtype=float), c, z["month"].to_numpy())
                row.update({f"{pname}_d": d_, f"{pname}_t": t_, f"{pname}_n1": n1, f"{pname}_n0": n0, f"{pname}_de": de, f"{pname}_te": te})
                if pname != "is":
                    sel = z[c == 1]
                    row[f"{pname}_r_c"] = float(sel["y"].mean()) if len(sel) else math.nan
                    row[f"{pname}_edge_c"] = float(sel["exc"].mean()) if len(sel) else math.nan
                if pname == "all":
                    ll = z[z["live_like"] >= 0.5]
                    dl, tl, _, _ = cluster_t(ll["y"].to_numpy(dtype=float), ll[col].to_numpy(dtype=float), ll["month"].to_numpy())
                    row.update({"ll_d": dl, "ll_t": tl})
            res.append(row)
    out = pd.DataFrame(res)
    out["is_pass"] = ((out["is_n1"] >= 30) & (out["is_n0"] >= 30) & (out["is_t"].abs() >= 2.0)
                      & (np.sign(out["is_de"]) == np.sign(out["is_d"])))
    out["z_needed"] = np.nan
    for tier in (1, 2):
        m = out["tier"] == tier
        k = int((out["is_pass"] & m).sum())
        out.loc[m, "z_needed"] = norm.ppf(1 - 0.05 / max(k, 1))
    sign = np.sign(out["is_d"])
    out["oos_pass"] = out["is_pass"] & (sign * out["oos_t"] >= out["z_needed"]) & (sign * out["oos_te"] >= 1.0)
    out["pass"] = out["oos_pass"] & (sign * out["all_t"] >= 2.5)
    return out


def diagnostics(R: pd.DataFrame, g_daily: pd.DataFrame) -> pd.DataFrame:
    """Per strategy: R, edge vs same-day random, t, win rate, payoff, forward excess decay, regime split."""
    rows = []
    R = R.merge(g_daily, left_on="date", right_index=True, how="left")
    for (strategy, side, tier), x in R.groupby(["strategy", "side", "tier"]):
        row = {"strategy": strategy, "side": side, "tier": tier}
        for pname, (a, b) in (("is", IS), ("oos", OOS)):
            z = x[(x["date"] >= a) & (x["date"] <= b)]
            if len(z) < 10:
                continue
            g = z.groupby(z["date"].dt.to_period("M"))["exc"]
            s = z["exc"] - z["exc"].mean()
            sums = s.groupby(z["date"].dt.to_period("M")).sum()
            se = math.sqrt((sums ** 2).sum()) / len(z)
            wins, losses = z.loc[z["y"] > 0, "y"], z.loc[z["y"] <= 0, "y"]
            yrs = z.groupby(z["date"].dt.year)["exc"].mean()
            row.update({f"{pname}_n": len(z), f"{pname}_r": z["y"].mean(), f"{pname}_edge": z["exc"].mean(),
                        f"{pname}_t": z["exc"].mean() / se if se > 0 else math.nan, f"{pname}_win": (z["y"] > 0).mean(),
                        f"{pname}_payoff": wins.mean() / -losses.mean() if len(losses) and losses.mean() < 0 else math.nan,
                        f"{pname}_posyears": (yrs > 0).mean()})
            for k in (1, 3, 5, 10, 20):
                if f"fx{k}" in z:
                    row[f"{pname}_fx{k}"] = z[f"fx{k}"].mean() * 100
            for lab, m in (("vix_hi", z["vix_t"] == 2), ("vix_lo", z["vix_t"] == 0), ("bull", z["spy_bull"] == 1),
                           ("bear", z["spy_bull"] == 0)):
                row[f"{pname}_edge_{lab}"] = z.loc[m, "exc"].mean() if m.sum() >= 20 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)

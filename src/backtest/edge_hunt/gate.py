"""Pre-registered conditions and the pass rule (written before any condition result was looked at).

For each setup S (crash_rebound, breakout, squeeze, plus 'standalone' = random entries with the live crash exit)
and condition C, on point-in-time S&P 500 members:

    delta = mean R of S-trades with C  -  mean R of S-trades without C
    t     = OLS of R on [1, C] with standard errors clustered by calendar month (CR1)

computed in sample (2010-01..2017-12) and out of sample (2018-01..2026-10), on raw R and on excess R
(R minus the mean R of same-day random entries with the same exit, which removes market-timing luck).

Pass rule:
  1. in sample: both groups >= 30 trades, |t raw| >= 2.0, excess delta has the same sign
  2. out of sample: same sign, t raw >= z(1 - 0.05 / k) with k = number of in-sample survivors (Bonferroni,
     one-sided in the pre-chosen direction), excess delta same sign with t >= 1.0
  3. pooled 2010-2026 t >= 2.5
Reported, not gated: the same delta in the live-like subset (beta >= 1.3, 60-day vol >= 35%) and in the terminal's
own (survivor) stock list, and the edge of the filtered setup over same-day random entries.
Percentile features are ranked against the same-day random entries (point-in-time cross-section); crash-shape
features against the in-sample crash signals; market features against their own in-sample daily distribution.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

IS = ("2010-01-01", "2017-12-31")
OOS = ("2018-01-01", "2026-10-31")
EXIT_OF = {"crash_rebound": "crash", "breakout": "bo", "squeeze": "sq", "standalone": "crash"}

# family, feature, kind, reference
SPEC: list[tuple[str, str, str, str]] = []
for f in ("svr1", "svr5", "svr_abn", "offex", "offex_abn"):
    SPEC += [("A short-sale flow (FINRA)", f, k, "date") for k in ("top", "bottom")]
for f in ("gpa", "roa", "accruals", "asset_growth", "issuance", "lev", "bm", "ep", "sales_growth", "cash_assets",
          "rd_int", "mcap", "quality"):
    SPEC += [("B fundamentals (SEC XBRL)", f, k, "date") for k in ("top", "bottom")]
SPEC += [("B fundamentals (SEC XBRL)", "prof", "true", "-")]
for f in ("earn_in5", "earn_in3", "earn_ahead20"):
    SPEC += [("C earnings events", f, "true", "-")]
for f in ("last_surprise", "days_since_earn"):
    SPEC += [("C earnings events", f, k, "date") for k in ("top", "bottom")]
for f in ("down5", "up5", "tcut5", "traise5"):
    SPEC += [("F analyst actions", f, "true", "-")]
SPEC += [("F analyst actions", "tgt_upside", k, "date") for k in ("top", "bottom")]
for f in ("vvix", "skew", "vix9_ratio", "pc10"):
    SPEC += [("D options / sentiment (market)", f, k, "daily") for k in ("top", "bottom")]
for f in ("sys_share", "on_share5", "speed"):
    SPEC += [("E self-made: crash shape", f, k, "signal") for k in ("top", "bottom")]
for f in ("ivol60", "max20", "mom12_1", "seas"):
    SPEC += [("E self-made: stock character", f, k, "date") for k in ("top", "bottom")]
SPEC += [("E self-made: stock character", "serial", "true", "-")]
# added 2026-11-03 18:10 after Neo asked to combine the online swing strategies (before any result was seen)
for f in ("conf2", "conf3", "conf1d2", "multifam", "fam_mr5", "fam_bo5", "fam_smc5", "fam_vp5", "ens_pos", "ens_hi"):
    SPEC += [("G multi-strategy confluence (42 online styles)", f, "true", "-")]
for f in ("pess", "retail_dip", "smart_conf", "catalyst_bo", "defend"):
    SPEC += [("H story combinations", f, "true", "-")]

CRASH_ONLY = {"sys_share", "on_share5", "speed"}
SETUPS = ("crash_rebound", "breakout", "squeeze", "standalone")


def _b(cond: pd.Series, *need: pd.Series) -> pd.Series:
    known = pd.concat(need, axis=1).notna().all(axis=1) if need else pd.Series(True, index=cond.index)
    return cond.astype(float).where(known)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "tgt_mean" in df:
        df["tgt_upside"] = df["tgt_mean"] / df["close"] - 1
    df["serial"] = (df["crash_cnt"] >= 1).astype(float).where(df["crash_cnt"].notna())
    if "conf_n5" in df:
        df["conf2"] = _b(df["conf_n5"] >= 2, df["conf_n5"])
        df["conf3"] = _b(df["conf_n5"] >= 3, df["conf_n5"])
        df["conf1d2"] = _b(df["conf_n1"] >= 2, df["conf_n1"])
        df["multifam"] = _b(df["n_fam5"] >= 2, df["n_fam5"])
        df["ens_pos"] = _b(df["ens5"] > 0, df["ens5"])
        ref = df[(df["setup"] == "random") & df["member"] & (df["date"] >= IS[0]) & (df["date"] <= IS[1]) & (df["ens5"] > 0)]["ens5"]
        df["ens_hi"] = _b(df["ens5"] >= ref.quantile(0.75), df["ens5"])
    if "earn_in3" in df and "last_surprise" in df:
        df["catalyst_bo"] = _b((df["earn_in3"] >= 0.5) & (df["last_surprise"] > 0), df["earn_in3"], df["last_surprise"])
    if "up5" in df:
        df["defend"] = _b((df["up5"] >= 0.5) | (df["traise5"] >= 0.5), df["up5"], df["traise5"])
    return df


def add_stories(df: pd.DataFrame, pct: dict[str, pd.Series]) -> None:
    if "svr_abn" in pct and "down5" in df:
        df["pess"] = _b((pct["svr_abn"] >= 2 / 3) & ((df["down5"] >= 0.5) | (df["tcut5"] >= 0.5)), pct["svr_abn"], df["down5"], df["tcut5"])
    if "offex_abn" in pct:
        df["retail_dip"] = _b((pct["offex_abn"] >= 2 / 3) & (df["ret5"] < 0), pct["offex_abn"], df["ret5"])
    if "quality" in pct and "conf_n5" in df:
        df["smart_conf"] = _b((df["conf_n5"] >= 2) & (pct["quality"] >= 2 / 3), df["conf_n5"], pct["quality"])


def date_pct(df: pd.DataFrame, col: str) -> pd.Series:
    """Percentile of each row's value among the same-day random entries (point-in-time members)."""
    ref = df[(df["setup"] == "random") & df["member"] & df[col].notna()]
    out = pd.Series(np.nan, index=df.index)
    refg = {d: np.sort(g.to_numpy(dtype=float)) for d, g in ref.groupby("date")[col]}
    for d, g in df[df[col].notna()].groupby("date")[col]:
        r = refg.get(d)
        if r is None or len(r) < 20:
            continue
        out.loc[g.index] = np.searchsorted(r, g.to_numpy(dtype=float), side="right") / len(r)
    return out


def cluster_t(y: np.ndarray, c: np.ndarray, groups: np.ndarray) -> tuple[float, float, int, int]:
    """delta (mean y | c=1 minus mean y | c=0) and its month-clustered t."""
    n1, n0 = int(c.sum()), int((1 - c).sum())
    if n1 < 2 or n0 < 2:
        return math.nan, math.nan, n1, n0
    X = np.column_stack([np.ones_like(y), c])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    s = pd.DataFrame({"g": groups, "e0": e, "e1": e * c}).groupby("g")[["e0", "e1"]].sum().to_numpy()
    meat = s.T @ s
    G = len(s)
    V = XtX_inv @ meat @ XtX_inv * (G / max(G - 1, 1)) * ((len(y) - 1) / max(len(y) - 2, 1))
    se = math.sqrt(V[1, 1]) if V[1, 1] > 0 else math.nan
    return float(b[1]), float(b[1] / se) if se and se > 0 else math.nan, n1, n0


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Adds same-day random means (per exit) and the condition masks cond__<feature>__<kind>."""
    df = add_derived(df)
    rnd = df[(df["setup"] == "random") & df["member"]]
    for ex in ("crash", "bo", "sq"):
        df["rd_" + ex] = df["date"].map(rnd.groupby("date")["r_" + ex].mean())
    feats_date = sorted({f for _, f, _, ref in SPEC if ref == "date" and f != "quality" and f in df})
    pct = {f: date_pct(df, f) for f in feats_date}
    if all(k in pct for k in ("gpa", "roa", "accruals", "lev")):
        q = pd.concat([pct["gpa"], pct["roa"], 1 - pct["accruals"], 1 - pct["lev"]], axis=1)
        pct["quality"] = q.mean(axis=1).where(q.notna().sum(axis=1) >= 3)
        # composite is itself re-expressed as a same-day percentile
        df["quality"] = pct["quality"]
        pct["quality"] = date_pct(df, "quality")
    add_stories(df, pct)
    is_mask = (df["date"] >= IS[0]) & (df["date"] <= IS[1])
    for fam, f, kind, ref in SPEC:
        if f not in df and f not in pct:
            continue
        name = f"cond__{f}__{kind}"
        if ref == "date":
            p = pct.get(f)
            if p is None:
                continue
            df[name] = np.where(p.isna(), np.nan, (p >= 2 / 3) if kind == "top" else (p <= 1 / 3)).astype(float)
        elif ref == "signal":
            base = df.loc[is_mask & (df["setup"] == "crash_rebound") & df["member"], f].dropna()
            lo, hi = base.quantile(1 / 3), base.quantile(2 / 3)
            v = df[f]
            df[name] = np.where(v.isna(), np.nan, (v >= hi) if kind == "top" else (v <= lo)).astype(float)
        elif ref == "daily":
            daily = df.drop_duplicates("date").set_index("date")[f]
            base = daily[(daily.index >= IS[0]) & (daily.index <= IS[1])].dropna()
            if f == "pc10":   # equity put/call ends 2019-10: its own split, in sample 2010-2015
                base = daily[(daily.index >= "2010-01-01") & (daily.index <= "2015-12-31")].dropna()
            lo, hi = base.quantile(1 / 3), base.quantile(2 / 3)
            v = df[f]
            df[name] = np.where(v.isna(), np.nan, (v >= hi) if kind == "top" else (v <= lo)).astype(float)
        else:
            v = df[f]
            df[name] = np.where(v.isna(), np.nan, v.astype(float) >= 0.5).astype(float)
    return df


def _rows(df: pd.DataFrame, setup: str, pool: str) -> pd.DataFrame:
    m = {"P": df["member"], "live_like": df["member"] & df["live_like"], "B": df["live_list"]}[pool]
    if setup == "standalone":
        x = df[(df["setup"] == "random") & m].copy()
        x["y"] = x["r_crash"]
    else:
        x = df[(df["setup"] == setup) & m].copy()
        x["y"] = x["r"]
    x["exc"] = x["y"] - x["rd_" + EXIT_OF[setup]]
    return x


def test_all(df: pd.DataFrame, extra_spec: list | None = None) -> pd.DataFrame:
    res = []
    spec = SPEC + (extra_spec or [])
    pools = {p: {s: _rows(df, s, p) for s in SETUPS} for p in ("P", "live_like", "B")}
    for setup in SETUPS:
        for fam, f, kind, ref in spec:
            if setup != "crash_rebound" and f in CRASH_ONLY:
                continue
            if setup == "standalone" and f in ("earn_in5", "earn_in3") and False:
                continue
            col = f"cond__{f}__{kind}"
            if col not in df:
                continue
            row: dict[str, Any] = {"family": fam, "feature": f, "kind": kind, "setup": setup}
            for pool in ("P", "live_like", "B"):
                x = pools[pool][setup]
                x = x[x[col].notna() & x["y"].notna() & x["exc"].notna()]
                for per, (a, b) in (("is", IS), ("oos", OOS), ("all", (IS[0], OOS[1]))):
                    if f == "pc10" and per != "all":
                        a, b = (("2010-01-01", "2015-12-31") if per == "is" else ("2016-01-01", "2019-10-31"))
                    if f == "pc10" and per == "all":
                        a, b = "2010-01-01", "2019-10-31"
                    z = x[(x["date"] >= a) & (x["date"] <= b)]
                    if pool != "P" and per != "all":
                        continue
                    g = z["date"].dt.to_period("M").astype(str).to_numpy()
                    c = z[col].to_numpy(dtype=float)
                    d, t, n1, n0 = cluster_t(z["y"].to_numpy(dtype=float), c, g)
                    de, te, _, _ = cluster_t(z["exc"].to_numpy(dtype=float), c, g)
                    pre = f"{pool}_{per}" if pool != "P" else per
                    row.update({f"{pre}_d": d, f"{pre}_t": t, f"{pre}_n1": n1, f"{pre}_n0": n0, f"{pre}_de": de, f"{pre}_te": te})
                    if pool == "P":
                        sel = z[c == 1]
                        row[f"{pre}_edge_c"] = float(sel["exc"].mean()) if len(sel) else math.nan
                        row[f"{pre}_r_c"] = float(sel["y"].mean()) if len(sel) else math.nan
            res.append(row)
    out = pd.DataFrame(res)
    s1 = (out["is_n1"] >= 30) & (out["is_n0"] >= 30) & (out["is_t"].abs() >= 2.0) & (np.sign(out["is_de"]) == np.sign(out["is_d"]))
    out["is_pass"] = s1
    k = int(s1.sum())
    z = norm.ppf(1 - 0.05 / max(k, 1))
    sign = np.sign(out["is_d"])
    out["z_needed"] = z
    out["oos_pass"] = s1 & (sign * out["oos_t"] >= z) & (sign * out["oos_te"] >= 1.0)
    out["pass"] = out["oos_pass"] & (sign * out["all_t"] >= 2.5)
    return out

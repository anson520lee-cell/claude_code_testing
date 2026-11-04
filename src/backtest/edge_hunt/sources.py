"""New information sources -> point-in-time features per (symbol, signal date).

Every value uses only what was public by the signal day's close:
  FINRA Reg SHO daily short-sale volume (off-exchange prints), published the same evening
  SEC XBRL company facts, usable from the day AFTER the 'filed' date
  SEC 8-K item 2.02 (results of operations = earnings release) filing dates
  Yahoo earnings dates with EPS surprise (the reported EPS is known on the earnings date)
  Yahoo analyst rating changes / price targets (timestamped)
  Cboe: equity put/call ratio (2006-11 .. 2019-10), VVIX, SKEW, VIX9D / VIX (Yahoo)
Upcoming earnings inside the holding window use the ACTUAL next earnings date (companies announce it weeks
ahead and the live terminal reads the scheduled date, so this is the usual approximation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------------- FINRA
def finra_features(shvol: pd.DataFrame, prices: dict[str, pd.DataFrame], keys: pd.DataFrame) -> pd.DataFrame:
    """svr1 / svr5 (short volume / off-exchange volume), svr_abn (5-day minus own 60-day baseline, in own sd),
    offex (off-exchange volume / consolidated Yahoo volume, 5-day) and offex_abn (5-day / 60-day)."""
    out = []
    shvol = shvol.rename(columns={"Date": "date", "Symbol": "symbol"})
    for sym, g in shvol.groupby("symbol"):
        k = keys[keys["symbol"] == sym]
        if k.empty or sym not in prices:
            continue
        g = g.set_index("date").sort_index()
        vol = prices[sym]["volume"].reindex(g.index)
        sv, tv = g["ShortVolume"].astype(float), g["TotalVolume"].astype(float)
        svr1 = sv / tv.where(tv > 0)
        svr5 = sv.rolling(5, min_periods=4).sum() / tv.rolling(5, min_periods=4).sum().where(lambda x: x > 0)
        base = svr1.shift(5).rolling(60, min_periods=40)
        svr_abn = (svr5 - base.mean()) / base.std()
        off = tv / vol.where(vol > 0)
        off5 = tv.rolling(5, min_periods=4).sum() / vol.rolling(5, min_periods=4).sum().where(lambda x: x > 0)
        off60 = off.shift(5).rolling(60, min_periods=40).mean()
        f = pd.DataFrame({"svr1": svr1, "svr5": svr5, "svr_abn": svr_abn, "offex": off5, "offex_abn": off5 / off60})
        # align to signal dates: the last FINRA day <= signal date (missing files -> previous day, max 3 days)
        d = pd.DatetimeIndex(k["date"].unique()).sort_values()
        f = f.reindex(f.index.union(d)).ffill(limit=3).reindex(d)
        f["symbol"] = sym
        out.append(f.rename_axis("date").reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------------------------------------- SEC XBRL
FLOW = {"ni": ["NetIncomeLoss"], "rev": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
        "gp": ["GrossProfit"], "cogs": ["CostOfRevenue"], "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
        "rd": ["ResearchAndDevelopmentExpense"], "opinc": ["OperatingIncomeLoss"]}
STOCK = {"assets": ["Assets"], "liab": ["Liabilities"], "equity": ["StockholdersEquity"],
         "cash": ["CashAndCashEquivalentsAtCarryingValue"],
         "shares": ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"]}


def _asof_table(x: pd.DataFrame) -> pd.DataFrame:
    """x: one concept for one company with columns end, val, filed (first-reported value per end).
    Returns rows sorted by filed with the LATEST-period value known at each filing date and the value of the
    period about one year earlier (end 330-400 days before), both as first reported."""
    x = x.sort_values(["end", "filed"]).drop_duplicates(["end"], keep="first").sort_values("filed").reset_index(drop=True)
    if x.empty:
        return x
    best_end = x["end"].cummax()
    is_best = x["end"].eq(best_end)
    idx = pd.Series(np.where(is_best, np.arange(len(x)), np.nan)).ffill().astype(int).to_numpy()
    out = pd.DataFrame({"filed": x["filed"].to_numpy(), "end": x["end"].to_numpy()[idx], "val": x["val"].to_numpy()[idx]})
    ends, vals = x["end"].to_numpy(), x["val"].to_numpy()
    prev = []
    for e in out["end"].to_numpy():
        lo, hi = e - np.timedelta64(400, "D"), e - np.timedelta64(330, "D")
        m = (ends >= lo) & (ends <= hi)
        prev.append(vals[m][np.argmax(ends[m])] if m.any() else np.nan)
        # (older periods are virtually always filed before newer ones)
    out["prev"] = prev
    prev_end = []
    for e in out["end"].to_numpy():
        lo, hi = e - np.timedelta64(400, "D"), e - np.timedelta64(330, "D")
        m = (ends >= lo) & (ends <= hi)
        prev_end.append(ends[m][np.argmax(ends[m])] if m.any() else np.datetime64("NaT"))
    out["prev_end"] = prev_end
    return out.drop_duplicates("filed", keep="last")


def sec_features(facts: pd.DataFrame, splits: pd.DataFrame, prices: dict[str, pd.DataFrame], keys: pd.DataFrame) -> pd.DataFrame:
    facts = facts.copy()
    facts["filed"] = pd.to_datetime(facts["filed"], errors="coerce")
    facts["val"] = pd.to_numeric(facts["val"], errors="coerce")
    facts["end"] = pd.to_datetime(facts["end"], errors="coerce")
    facts["start"] = pd.to_datetime(facts["start"], errors="coerce")
    facts = facts.dropna(subset=["filed", "val", "end"])
    dur = (facts["end"] - facts["start"]).dt.days
    facts["annual"] = (dur >= 340) & (dur <= 380)
    facts["instant"] = facts["start"].isna()
    out = []
    for sym, k in keys.groupby("symbol"):
        g = facts[facts["symbol"] == sym]
        if g.empty or sym not in prices:
            continue
        dates = pd.DatetimeIndex(sorted(k["date"].unique()))
        res = pd.DataFrame({"date": dates, "avail": dates - pd.Timedelta(days=1)})
        for name, concepts in list(FLOW.items()) + list(STOCK.items()):
            flow = name in FLOW
            unit = "shares" if name == "shares" else "USD"
            sel = g[g["concept"].isin(concepts) & (g["unit"] == unit) & (g["annual"] if flow else g["instant"])]
            # one concept per company: the alias with most rows
            if sel.empty:
                res[name] = np.nan; res[name + "_prev"] = np.nan; res[name + "_end"] = pd.NaT; res[name + "_prev_end"] = pd.NaT
                continue
            best = sel["concept"].value_counts().index[0] if name != "shares" else None
            if best is not None:
                sel = sel[sel["concept"] == best]
            tab = _asof_table(sel[["end", "val", "filed"]])
            m = pd.merge_asof(res[["avail"]].sort_values("avail"), tab.rename(columns={"filed": "avail"}), on="avail", direction="backward")
            stale = (m["avail"] - m["end"]).dt.days > (550 if flow else 400)
            res[name] = m["val"].where(~stale).to_numpy()
            res[name + "_prev"] = m["prev"].where(~stale).to_numpy()
            res[name + "_end"] = m["end"].to_numpy()
            res[name + "_prev_end"] = m["prev_end"].to_numpy()
        sp = splits[splits["symbol"] == sym]
        spd = pd.to_datetime(sp["date"])
        if getattr(spd.dt, "tz", None) is not None:
            spd = spd.dt.tz_localize(None)
        sp_d, sp_r = spd.to_numpy(), sp["ratio"].to_numpy(dtype=float)

        def fac(a, b):  # product of split ratios with a < date <= b
            if pd.isna(a) or pd.isna(b) or len(sp_d) == 0:
                return 1.0
            m = (sp_d > np.datetime64(a)) & (sp_d <= np.datetime64(b))
            return float(np.prod(sp_r[m])) if m.any() else 1.0

        close = prices[sym]["close"].reindex(dates, method="ffill").to_numpy()
        far = np.datetime64("2100-01-01")
        raw_px = close * np.array([fac(d, far) for d in dates])
        sh = res["shares"].to_numpy()
        sh_day = np.array([s * fac(e, d) if np.isfinite(s) else np.nan for s, e, d in zip(sh, res["shares_end"], dates)])
        res["mcap"] = sh_day * raw_px
        res["issuance"] = [s / (p * fac(pe, e)) - 1 if np.isfinite(s) and np.isfinite(p) and p > 0 else np.nan
                           for s, p, e, pe in zip(sh, res["shares_prev"], res["shares_end"], res["shares_prev_end"])]
        res["symbol"] = sym
        out.append(res)
    df = pd.concat(out, ignore_index=True)
    a = df["assets"].where(df["assets"] > 0)
    gp = df["gp"].fillna(df["rev"] - df["cogs"])
    m = df["mcap"].where(df["mcap"] > 0)
    liab = df["liab"].fillna(df["assets"] - df["equity"])
    return pd.DataFrame({
        "symbol": df["symbol"], "date": df["date"],
        "prof": (df["ni"] > 0).astype(float).where(df["ni"].notna()),
        "gpa": gp / a, "roa": df["ni"] / a, "accruals": (df["ni"] - df["cfo"]) / a,
        "asset_growth": df["assets"] / df["assets_prev"].where(df["assets_prev"] > 0) - 1,
        "issuance": pd.Series(df["issuance"]).where(lambda x: (x > -0.3) & (x < 0.4)),
        "lev": liab / a, "bm": df["equity"] / m, "ep": df["ni"] / m,
        "sales_growth": df["rev"] / df["rev_prev"].where(df["rev_prev"] > 0) - 1,
        "cash_assets": df["cash"] / a, "rd_int": df["rd"] / df["rev"].where(df["rev"] > 0),
        "mcap": m,
    })


# ---------------------------------------------------------------------------------------------- events
def _sessions(cal: pd.DatetimeIndex, d: pd.Series) -> np.ndarray:
    """Index of the first trading session on or after each date."""
    return np.searchsorted(cal.values, d.values.astype("datetime64[ns]"), side="left")


def event_features(keys: pd.DataFrame, cal: pd.DatetimeIndex, filings: pd.DataFrame, earn: pd.DataFrame,
                   ratings: pd.DataFrame) -> pd.DataFrame:
    """Earnings (SEC 8-K 2.02 + Yahoo dates/surprise) and analyst actions, computed on every session of a stock
    (vectorised) and then read at the signal dates.

    earn_in5 / earn_in3   an earnings release in the last 5 / 3 sessions (incl. today; after-close = next session)
    earn_ahead20          the next release falls within the next 20 sessions (actual date, see module note)
    days_since_earn       sessions since the last release (< 120)
    last_surprise         EPS surprise % of the last release (< 100 sessions old)
    down5 / up5           an analyst downgrade / upgrade in the last 5 sessions
    tcut5 / traise5       a price-target cut / raise in the last 5 sessions
    tgt_mean              mean of the price targets published in the last 63 sessions (>= 3 targets)
    Analyst fields need >= 250 sessions of rating history for the stock (coverage must exist to be 'absent')."""
    n = len(cal)
    calv = cal.values
    e8 = filings[filings["form"].str.startswith("8-K") & filings["items"].fillna("").str.contains("2.02")].copy()
    e8["dt"] = pd.to_datetime(e8["accepted"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    e8["dt"] = e8["dt"].fillna(pd.to_datetime(e8["filing_date"]) + pd.Timedelta(hours=17))
    ey = earn.copy()
    if len(ey):
        ey["dt"] = pd.to_datetime(ey["Earnings Date"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        ey["surprise"] = pd.to_numeric(ey.get("Surprise(%)"), errors="coerce")
        ey["reported"] = pd.to_numeric(ey.get("Reported EPS"), errors="coerce")
    rt = ratings.copy()
    if len(rt):
        rt["dt"] = pd.to_datetime(rt["GradeDate"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)

    def sess(dt: pd.Series) -> np.ndarray:
        d = dt.dt.normalize() + pd.to_timedelta((dt.dt.hour >= 16).astype(int), unit="D")
        return np.searchsorted(calv, d.values.astype("datetime64[ns]"), side="left")

    def recent(idx: np.ndarray, w: int) -> np.ndarray:
        cnt = np.bincount(idx[(idx >= 0) & (idx < n)], minlength=n).astype(float)
        cs = np.cumsum(cnt)
        lag = np.r_[np.zeros(w + 1), cs[:-(w + 1)]] if n > w + 1 else np.zeros(n)
        return (cs - lag[:n]) > 0             # any event in [t - w, t]

    out = []
    grid = np.arange(n)
    for sym, g in keys.groupby("symbol"):
        ti = np.searchsorted(calv, g["date"].values.astype("datetime64[ns]"))
        a = e8[e8["symbol"] == sym]
        b = ey[ey["symbol"] == sym] if len(ey) else ey
        es = np.unique(np.r_[sess(a["dt"]) if len(a) else [], sess(b["dt"]) if len(b) else []]).astype(int)
        row = pd.DataFrame({"symbol": sym, "date": g["date"].to_numpy()})
        if len(es):
            pos = np.searchsorted(es, ti, side="right")          # releases <= t are es[:pos]
            last = np.where(pos > 0, es[np.maximum(pos - 1, 0)], -10 ** 6)
            nxt = np.where(pos < len(es), es[np.minimum(pos, len(es) - 1)], 10 ** 7)
            row["earn_in5"] = (ti - last <= 5).astype(float)
            row["earn_in3"] = (ti - last <= 3).astype(float)
            row["earn_ahead20"] = ((nxt - ti >= 1) & (nxt - ti <= 20)).astype(float)
            row["days_since_earn"] = np.where(ti - last < 120, ti - last, np.nan).astype(float)
            # data must exist around the date to call it 'no release'
            covered = (ti >= es.min() - 5) & (ti <= es.max() + 70)
            for c in ("earn_in5", "earn_in3", "earn_ahead20"):
                row.loc[~covered, c] = np.nan
        if len(b):
            bs = b.dropna(subset=["reported"])
            if len(bs):
                si = sess(bs["dt"])
                order = np.argsort(si)
                si, sv = si[order], bs["surprise"].to_numpy()[order]
                p = np.searchsorted(si, ti, side="right") - 1
                ok = (p >= 0) & (ti - si[np.maximum(p, 0)] < 100)
                row["last_surprise"] = np.where(ok, sv[np.maximum(p, 0)], np.nan)
        r = rt[rt["symbol"] == sym] if len(rt) else rt
        if len(r):
            si = sess(r["dt"])
            act = r["Action"].fillna("").str.lower().to_numpy()
            cur = pd.to_numeric(r.get("currentPriceTarget"), errors="coerce").to_numpy()
            pri = pd.to_numeric(r.get("priorPriceTarget"), errors="coerce").to_numpy()
            have = ti >= si.min() + 250
            for name, m in (("down5", act == "down"), ("up5", act == "up"),
                            ("tcut5", (cur < pri) & (pri > 0)), ("traise5", (cur > pri) & (pri > 0))):
                v = recent(si[m], 5)[np.minimum(ti, n - 1)].astype(float)
                row[name] = np.where(have, v, np.nan)
            ok = np.isfinite(cur) & (cur > 0) & (si < n)
            s_sum = np.bincount(si[ok], weights=cur[ok], minlength=n)[:n]
            s_cnt = np.bincount(si[ok], minlength=n)[:n].astype(float)
            cs, cc = np.cumsum(s_sum), np.cumsum(s_cnt)
            w = 63
            ls = np.r_[np.zeros(w + 1), cs[:-(w + 1)]][:n]
            lc = np.r_[np.zeros(w + 1), cc[:-(w + 1)]][:n]
            tot, cnt = (cs - ls)[np.minimum(ti, n - 1)], (cc - lc)[np.minimum(ti, n - 1)]
            row["tgt_mean"] = np.where(have & (cnt >= 3), tot / np.maximum(cnt, 1), np.nan)
        out.append(row)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------------------- market
def market_features(prices: dict[str, pd.DataFrame], pc_csv: str | None) -> pd.DataFrame:
    def c(sym):
        return prices[sym]["close"] if sym in prices else None
    vix, vvix, skew, vix9 = c("^VIX"), c("^VVIX"), c("^SKEW"), c("^VIX9D")
    m = pd.DataFrame({"vvix": vvix, "skew": skew})
    if vix is not None and vix9 is not None:
        m["vix9_ratio"] = vix9 / vix
    if pc_csv:
        pc = pd.read_csv(pc_csv, skiprows=2)
        pc.columns = [x.strip() for x in pc.columns]
        pc["DATE"] = pd.to_datetime(pc["DATE"].str.strip(), errors="coerce")
        pc = pc.dropna(subset=["DATE"]).set_index("DATE")["P/C Ratio"].astype(float)
        m["pc10"] = pc.rolling(10, min_periods=8).mean().reindex(m.index)
    return m

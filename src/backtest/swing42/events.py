"""Strategies that need their own evaluator: same-day stop entries (ORB, Oops), events (earnings, insider
buying, index changes), index timing (seasonality, OPEX, intermarket), pairs, sector rotation, ETF proxies."""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from . import core
from .core import COST_LONG, COST_SHORT, BORROW, IS_END, START, exit_hold, exit_level, fresh, shift


def _sel():
    from .run import select_cells
    return select_cells


def _cells_from(p, grid_sigs, exits, get_exit):
    cells = []
    for prm, sig in grid_sigs:
        for ek in exits:
            out, unit = get_exit(ek)
            cells.append({"params": prm, "exit": ek, "sig": sig, "out": out, "unit": unit, "bench": None})
    return cells


def _rows(dates: pd.DatetimeIndex, ts) -> np.ndarray:
    """Row of the first session on or after each timestamp."""
    return np.searchsorted(dates.to_numpy(), pd.DatetimeIndex(ts).to_numpy(), side="left")


def _stats(x: pd.Series, base: float = 0.0) -> dict:
    """Mean of x - base with t (x is already one value per independent period, e.g. a month)."""
    out = {}
    for name, m in (("IS", (x.index >= START) & (x.index <= IS_END)), ("OOS", x.index > IS_END), ("ALL", x.index >= START)):
        v = x[m].dropna() - base
        if len(v) < 3:
            out[name] = {"n": int(len(v))}
            continue
        yy = v.groupby(v.index.year).mean()
        out[name] = {"n": int(len(v)), "mean": float(v.mean() + base), "edge": float(v.mean()),
                     "t": float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), "win": float((v > 0).mean()),
                     "pos_years": float((yy > 0).mean())}
    return out


# ====================================================================== 6 ORB (Crabel, daily stop-entry)
def s06_orb(A, X, ctxdf, get_exit, args):
    p = A
    rng = p.H - p.L
    prev = shift(rng, 1)
    nr7 = prev <= core.roll(prev, 7, "min")
    cells = []
    bench1 = p.C / p.O - 1 - COST_LONG
    cff = pd.DataFrame(p.C).ffill().to_numpy()
    for k in (0.2, 0.4, 0.6):
        level = p.O + k * prev
        for filt in ("all", "nr7"):
            for n in (1, 5):
                out = exit_level(p, level, 1, n)
                sig = np.isfinite(out) & (nr7 if filt == "nr7" else True)
                bench = bench1 if n == 1 else (shift(cff, -(n - 1)) / p.O - 1 - COST_LONG)
                cells.append({"params": {"k": k, "filter": filt}, "exit": f"day{n}", "sig": sig, "out": out, "unit": None,
                              "bench": bench})
    return _sel()(p, ctxdf, cells, sid=6, zh="開市區間突破 ORB", side=1, filter_lag=1,
                  proxy="冇歷史分鐘數據：用 Crabel 日線版 (開市價 + k×昨日波幅 做買入觸發價)，對比『開市買入』")


# ====================================================================== 29 Oops / gap-fill reversal
def s29_oops(A, X, ctxdf, get_exit, args):
    p = A
    a = core.atr(p)
    cff = pd.DataFrame(p.C).ffill().to_numpy()
    cells = []
    for g in (0.0, 0.5):
        for n in (1, 3):
            bench = (shift(cff, -(n - 1)) if n > 1 else cff) / p.O - 1 - COST_LONG
            gap = p.O < shift(p.L, 1) - g * shift(a, 1)
            level = np.where(gap, shift(p.L, 1), np.nan)
            out = exit_level(p, level, 1, n)
            cells.append({"params": {"gap_atr": g, "side": "long"}, "exit": f"day{n}", "sig": np.isfinite(out), "out": out,
                          "unit": None, "bench": bench})
    return _sel()(p, ctxdf, cells, sid=29, zh="缺口封閉反轉 Oops!", side=1, filter_lag=1,
                  proxy="開市跳空低過昨日低位，升返昨日低位先買 (日線可測)")


# ====================================================================== earnings (30b PEAD, 37 pre-earnings run-up)
def load_earnings(earn_dir: str, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        f = os.path.join(earn_dir, f"{t}.pkl")
        if not os.path.exists(f):
            continue
        e = pd.read_pickle(f)
        if e is None or len(e) == 0:
            continue
        e = e.reset_index()
        e.columns = ["ts"] + list(e.columns[1:])
        ts = pd.to_datetime(e["ts"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("America/New_York")
        hour = ts.dt.hour + ts.dt.minute / 60
        rows.append(pd.DataFrame({"sym": t, "date": ts.dt.tz_localize(None).dt.normalize(), "hour": hour,
                                  "surprise": pd.to_numeric(e.get("Surprise(%)"), errors="coerce"),
                                  "reported": pd.to_numeric(e.get("Reported EPS"), errors="coerce")}))
    d = pd.concat(rows, ignore_index=True)
    d = d.dropna(subset=["reported"]).drop_duplicates(["sym", "date"])
    return d


def _earn_rows(A, ev):
    col = {t: j for j, t in enumerate(A.tickers)}
    j = ev["sym"].map(col).to_numpy()
    r = _rows(A.dates, ev["date"])
    amc = (ev["hour"].to_numpy() >= 12) | (ev["hour"].to_numpy() == 0)
    # reaction day: same session for pre-market reports, next session for after-close (or unknown time)
    react = np.where(amc, r + 1, r)
    # last close BEFORE the report: the report day itself for after-close, the previous session for pre-market
    exact = A.dates.to_numpy()[np.clip(r, 0, A.D - 1)] == ev["date"].to_numpy()
    last_pre = np.where(amc, r, r - 1)
    ok = (r < A.D) & exact
    return j, react, last_pre, ok


def s30b_pead(A, X, ctxdf, get_exit, args):
    ev = load_earnings(args.earn, A.tickers)
    j, react, _, ok = _earn_rows(A, ev)
    sur = ev["surprise"].to_numpy()
    ret = A.C / shift(A.C, 1) - 1
    gap = A.O / shift(A.C, 1) - 1
    grid = []
    for smin in (0.0, 5.0, 15.0):
        for rmin in (0.0, 0.03, 0.06):
            sig = np.zeros((A.D, A.S), dtype=bool)
            m = ok & (react < A.D - 1) & (sur > smin)
            rr, jj = react[m], j[m]
            keep = ret[rr, jj] > rmin
            sig[rr[keep], jj[keep]] = True
            grid.append(({"surprise_min_%": smin, "reaction_min": rmin}, sig))
    cells = _cells_from(A, grid, ["hold20", "hold60", "live"], get_exit)
    r = _sel()(A, ctxdf, cells, sid="30b", zh="財報跳空漂移 PEAD (真實財報日+EPS surprise)", side=1,
               proxy=f"Yahoo 財報日及 EPS surprise，{len(ev)} 次財報；財報當日反應收市後買")
    r["n_events"] = int(len(ev))
    return r


def s37_pre_earnings(A, X, ctxdf, get_exit, args):
    ev = load_earnings(args.earn, A.tickers)
    j, _, last_pre, ok = _earn_rows(A, ev)
    grid_cells = []
    for k in (3, 5, 10):
        sig = np.zeros((A.D, A.S), dtype=bool)
        rows = last_pre - k
        m = ok & (rows > 0)
        sig[rows[m], j[m]] = True
        for prior in ("any", "beat_last"):
            s = sig
            if prior == "beat_last":
                evs = ev.assign(j=j, rows=rows, ok=m).sort_values(["sym", "date"])
                prev_beat = evs.groupby("sym")["surprise"].shift(1) > 0
                s = np.zeros_like(sig)
                mm = evs["ok"] & prev_beat
                s[evs.loc[mm, "rows"].to_numpy(), evs.loc[mm, "j"].to_numpy()] = True
            out, unit = get_exit(f"hold{k}")
            grid_cells.append({"params": {"days_before": k, "prior": prior}, "exit": f"hold{k}", "sig": s, "out": out,
                               "unit": None, "bench": None})
    return _sel()(A, ctxdf, grid_cells, sid=37, zh="財報前預期拉升 Pre-earnings run-up", side=1,
                  proxy="財報前 k 日開市買，財報前最後一個收市賣 (假設財報日提前已公佈)")


# ====================================================================== 34 insider cluster buying (SEC Form 4)
def s34_insider(A, X, ctxdf, get_exit, args):
    from src.backtest import insider
    files = sorted(glob.glob(os.path.join(args.sec, "*_form345.zip")))
    cache = os.path.join(os.path.dirname(args.cache) or ".", "insider_buys.pkl")
    if os.path.exists(cache):
        buys = pd.read_pickle(cache)
    else:
        buys = pd.concat([insider.parse_quarter(open(f, "rb").read()) for f in files], ignore_index=True).drop_duplicates()
        buys.to_pickle(cache)
    evs = insider.insider_events(buys)
    col = {t: j for j, t in enumerate(A.tickers)}
    grid = []
    for style in ("ins_cluster3", "ins_cluster2_250k", "ins_ceo_100k"):
        e = evs[(evs["style"] == style) & evs["sym"].isin(col)]
        r = _rows(A.dates, e["filing_date"])
        jj = e["sym"].map(col).to_numpy()
        m = r < A.D
        sig = np.zeros((A.D, A.S), dtype=bool)
        sig[r[m], jj[m]] = True
        grid.append(({"style": style}, sig))
    cells = _cells_from(A, grid, ["hold20", "hold60", "live"], get_exit)
    out = _sel()(A, ctxdf, cells, sid=34, zh="內部人集群買入", side=1,
                 proxy=f"SEC Form 4 (2006-2026)，只計 S&P 500 成分股 (細價股效應通常更強但我哋冇佢哋數據)；{len(files)} 個季度檔")
    out["n_events"] = int(len(evs))
    return out


# ====================================================================== 35 index inclusion / deletion
def s35_index(A, X, ctxdf, get_exit, args):
    d = pd.read_csv(args.hist, parse_dates=["date"]).sort_values("date")
    d = d[d["date"] >= "1999-06-01"]
    prev = None
    adds, dels = [], []
    for _, row in d.iterrows():
        cur = {t.replace(".", "-") for t in row["tickers"].split(",")}
        if prev is not None:
            adds += [(row["date"], t) for t in cur - prev]
            dels += [(row["date"], t) for t in prev - cur]
        prev = cur
    col = {t: j for j, t in enumerate(A.tickers)}
    has = np.isfinite(A.C) & np.isfinite(A.O) & (A.C >= 5)
    grid_cells = []
    for kind, lst in (("add", adds), ("del", dels)):
        e = pd.DataFrame(lst, columns=["date", "sym"])
        e = e[e["sym"].isin(col)]
        r = _rows(A.dates, e["date"])
        jj = e["sym"].map(col).to_numpy()
        for phase, off, ek in (("pre", -6, "hold5"), ("post", -1, "hold20"), ("post", -1, "hold60")):
            rr = r + off
            m = (rr > 0) & (rr < A.D)
            sig = np.zeros((A.D, A.S), dtype=bool)
            sig[rr[m], jj[m]] = True
            out, unit = get_exit(ek)
            grid_cells.append({"params": {"kind": kind, "phase": phase}, "exit": ek, "sig": sig, "out": out, "unit": None,
                               "bench": None})
    out = _sel()(A, ctxdf, grid_cells, sid=35, zh="指數納入/剔除", side=1, base_mask=has,
                 proxy="用 fja05680 生效日；冇公佈日 → 『pre』= 生效前 5 日入 (假設已公佈，約等於公佈後買)；被剔除股好多已退市冇數據")
    out["n_adds"], out["n_dels"] = len(adds), len(dels)
    return out


# ====================================================================== 39 sympathy / laggard catch-up
def s39_sympathy(A, X, ctxdf, get_exit, args):
    p = A
    ret = p.C / shift(p.C, 1) - 1
    month_end = np.r_[p.dates.month[1:] != p.dates.month[:-1], True]
    peers = {}
    for r in np.where(month_end)[0]:
        if r < 260:
            continue
        w = ret[r - 251:r + 1]
        okc = p.elig[r] & (np.isfinite(w).sum(0) >= 240)
        ix = np.where(okc)[0]
        if len(ix) < 50:
            continue
        z = np.nan_to_num(w[:, ix])
        z = (z - z.mean(0)) / (z.std(0) + 1e-12)
        c = z.T @ z / len(z)
        np.fill_diagonal(c, -1)
        top = np.argsort(-c, axis=1)[:, :5]
        peers[r] = (ix, ix[top])
    pe_rows = np.array(sorted(peers))
    vr = p.V / shift(core.sma(p.V, 50), 1)
    gap = p.O / shift(p.C, 1) - 1
    lead = p.elig & (gap >= 0.05) & (vr >= 3) & (p.C > p.O)
    li, lj = np.where(lead)
    grid = []
    for rmax in (0.0, 0.01, 9.9):
        for npeer in (3, 5):
            sig = np.zeros((p.D, p.S), dtype=bool)
            for d, s in zip(li, lj):
                k = np.searchsorted(pe_rows, d, side="left") - 1
                if k < 0:
                    continue
                ix, top = peers[pe_rows[k]]
                w = np.where(ix == s)[0]
                if len(w) == 0:
                    continue
                for q in top[w[0], :npeer]:
                    if q != s and ret[d, q] < rmax and not lead[d, q]:
                        sig[d, q] = True
            grid.append(({"peer_ret_max": rmax, "n_peers": npeer}, sig))
    cells = _cells_from(p, grid, ["hold5", "hold10", "live"], get_exit)
    return _sel()(p, ctxdf, cells, sid=39, zh="同業傳染 / 補漲", side=1,
                  proxy="冇行業分類歷史：同業 = 過去 252 日相關度最高嘅 3-5 隻；龍頭事件 = 跳空≥5%+3倍量")


# ====================================================================== index panel helpers
def _index_panel(X, names):
    q = X.sub(names)
    return q


def _idx_ctx(X):
    return X.df("C")


def s41_seasonality(A, X, ctxdf, get_exit, args):
    from .run import make_exits
    q = _index_panel(X, ["SPY", "QQQ", "IWM", "DIA"])
    ge = make_exits(q)
    dts = q.dates
    mon = dts.year * 12 + dts.month
    last = np.r_[mon[1:] != mon[:-1], True]
    lrow = np.where(last)[0]
    cells = []
    for start_off in (-1, -3):  # enter at the open of the last (or 3rd-last) session of the month
        for hold in (4, 6):
            sig = np.zeros((q.D, q.S), dtype=bool)
            rr = lrow + start_off  # signal one session before the entry session
            rr = rr[(rr > 0) & (rr < q.D)]
            sig[rr, :] = True
            out, unit = ge(f"hold{hold}")
            cells.append({"params": {"entry": f"月尾倒數第{-start_off}日開市", "hold": hold, "off": start_off},
                          "exit": f"hold{hold}", "sig": sig, "out": out, "unit": None, "bench": None})
    res = _sel()(q, ctxdf.reindex(q.dates), cells, sid=41, zh="月初月尾效應 (Turn-of-month) + 季節性", side=1,
                 proxy="SPY/QQQ/IWM/DIA；另計 Halloween (11-4月 vs 5-10月) 同總統週期")
    # stocks too (universe A), best index cell's timing
    best = res["chosen"]["params"]
    off = best["off"]
    sig = np.zeros((A.D, A.S), dtype=bool)
    rr = lrow + off
    rr = rr[(rr > 0) & (rr < A.D)]
    sig[rr, :] = True
    out, _ = get_exit(f"hold{best['hold']}")
    res["stocks"] = {k: v for k, v in core.evaluate(A, sig, out).items()}
    # Halloween & presidential cycle on SPY (monthly / yearly returns)
    spy = X.df("C")["SPY"].dropna()
    m = spy.resample("ME").last().pct_change().dropna()
    win = m[m.index.month.isin([11, 12, 1, 2, 3, 4])]
    summ = m[~m.index.month.isin([11, 12, 1, 2, 3, 4])]
    res["halloween"] = {"winter": _stats(win), "summer": _stats(summ)}
    y = spy.resample("YE").last().pct_change().dropna()
    res["presidential"] = {str(k): float(v) for k, v in y.groupby(y.index.year % 4).mean().items()}
    res["presidential_note"] = "year%4: 0=選舉年 1=第1年 2=中期 3=選前年；只有 ~6 個週期，統計上冇意義"
    return res


def s32_opex(A, X, ctxdf, get_exit, args):
    from .run import make_exits
    q = _index_panel(X, ["SPY", "QQQ", "IWM"])
    ge = make_exits(q)
    dts = q.dates
    cells = []
    fr = []
    for (y, mth), g in pd.Series(dts, index=dts).groupby([dts.year, dts.month]):
        third = pd.Timestamp(year=y, month=mth, day=15)
        third = third + pd.Timedelta(days=(4 - third.weekday()) % 7)  # third Friday
        r = np.searchsorted(dts.to_numpy(), np.datetime64(third), side="right") - 1  # opex session (Thu if holiday)
        fr.append(r)
    fr = np.array(fr)
    for kind, off, hold in (("opex_week", -5, 5), ("week_after", 0, 5)):
        sig = np.zeros((q.D, q.S), dtype=bool)
        rr = fr + off
        rr = rr[(rr > 0) & (rr < q.D - 1)]
        sig[rr, :] = True
        for side, ek in ((1, f"hold{hold}"), (-1, f"hold{hold}_s")):
            out, unit = ge(ek)
            cells.append({"params": {"window": kind, "side": "long" if side > 0 else "short"}, "exit": ek, "sig": sig,
                          "out": out, "unit": None, "bench": None})
    return _sel()(q, ctxdf.reindex(q.dates), cells, sid=32, zh="伽馬擠壓 / 期權到期週 (代理)", side=1,
                  proxy="冇 GEX/0DTE 數據：用每月期權到期週 (OPEX) 同之後一週嘅指數表現做代理")


def s40_intermarket(A, X, ctxdf, get_exit, args):
    from .run import make_exits
    q = _index_panel(X, ["SPY"])
    ge = make_exits(q)
    c = X.df("C")
    spy, hyg, vix, tlt = c["SPY"], c["HYG"], c["^VIX"], c["TLT"]
    cg = (c.get("HG=F") / c.get("GC=F")) if "HG=F" in c and "GC=F" in c else None
    variants = {
        "credit_bull": (spy.pct_change(10) < -0.02) & (hyg.pct_change(10) > 0),
        "vix_nonconfirm": (spy <= spy.rolling(10).min()) & (vix < vix.rolling(10).max()),
        "bonds_up_stocks_down": (spy.pct_change(10) < -0.02) & (tlt.pct_change(10) < 0),
    }
    if cg is not None:
        variants["copper_gold_up"] = (spy.pct_change(20) < 0) & (cg.pct_change(20) > 0.03)
    cells = []
    for name, cond in variants.items():
        sig = fresh(cond.fillna(False).to_numpy()[:, None], 5)
        for ek in ("hold5", "hold10", "hold20"):
            out, unit = ge(ek)
            cells.append({"params": {"variant": name}, "exit": ek, "sig": sig, "out": out, "unit": None, "bench": None})
    return _sel()(q, ctxdf.reindex(q.dates), cells, sid=40, zh="跨資產背離 (SPY 擇時)", side=1,
                  proxy="HYG 信用債 (2007+)、VIX、TLT、銅/金比；只買 SPY")


def s38b_sector_rotation(A, X, ctxdf, get_exit, args):
    secs = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
    c = X.df("C")[secs].dropna()
    me = c.resample("ME").last()
    mret = me.pct_change()
    ew = mret.mean(1)
    best = None
    grid = []
    for lb in (3, 6, 12):
        mom = me.pct_change(lb)
        for top in (1, 3):
            pick = mom.rank(axis=1, ascending=False) <= top
            held = pick.shift(1)
            port = (mret * held).sum(1) / top
            turn = (held.astype(float).diff().abs().sum(1) / (2 * top)).fillna(1)
            port = port - turn * 0.001
            exc = (port - ew).dropna()
            exc = exc[exc.index >= "2000-01-01"]
            s = _stats(exc)
            grid.append({"params": {"lookback_m": lb, "top": top}, **s})
    best = max(grid, key=lambda g: g["IS"].get("t", -99))
    out = {"id": "38b", "zh": "板塊輪動 (9 個 SPDR 板塊動量)", "chosen": {"params": best["params"], "exit": "每月換倉",
           "unit": "% / 月 (對比9板塊等權)"}, "IS": best["IS"], "OOS": best["OOS"], "grid": grid,
           "proxy": "月底揀過去 3/6/12 個月最強 1/3 個板塊，持有一個月，對比 9 板塊等權"}
    from .run import verdict
    out["verdict"] = verdict({"IS": best["IS"], "OOS": best["OOS"]}) if best["OOS"].get("n", 0) >= 20 else "樣本不足"
    out["grid_share_oos_pos"] = float(np.mean([g["OOS"].get("edge", -1) > 0 for g in grid]))
    return out


def s24_pairs(A, X, ctxdf, get_exit, args):
    p = A
    cff = pd.DataFrame(p.C).ffill().to_numpy()
    form, trade = 252, 126
    trades = []
    for f0 in range(260, p.D - 1, trade):
        f1 = f0 + form
        if f1 + 2 >= p.D:
            break
        t1 = min(f1 + trade, p.D - 1)
        okc = p.elig[f1] & np.all(np.isfinite(p.C[f0:f1 + 1]), axis=0)
        ix = np.where(okc)[0]
        P = p.C[f0:f1 + 1, ix] / p.C[f0, ix]
        sq = (P ** 2).sum(0)
        ssd = sq[:, None] + sq[None, :] - 2 * P.T @ P
        iu = np.triu_indices(len(ix), 1)
        order = np.argsort(ssd[iu])[:20]
        for o in order:
            a, b = ix[iu[0][o]], ix[iu[1][o]]
            sd = np.std(P[:, iu[0][o]] - P[:, iu[1][o]])
            pa = cff[f1:t1 + 1, a] / cff[f1, a]
            pb = cff[f1:t1 + 1, b] / cff[f1, b]
            spread = pa - pb
            pos = 0
            k = 1
            while k < len(spread) - 1:
                if pos == 0 and abs(spread[k]) > 2 * sd:
                    pos = -np.sign(spread[k])  # +1: long a short b
                    ea, eb, ek = pa[k + 1], pb[k + 1], k + 1  # enter next day (Gatev 1-day wait)
                    k += 1
                    continue
                if pos != 0 and (np.sign(spread[k]) == pos or k == len(spread) - 2):
                    xa, xb = pa[k + 1], pb[k + 1]
                    ra, rb = xa / ea - 1, xb / eb - 1
                    held = k + 1 - ek
                    r = 0.5 * (pos * ra - pos * rb) - (COST_LONG + COST_SHORT) / 2 - 0.5 * BORROW * held / 252
                    trades.append((p.dates[f1 + ek], r, held))
                    pos = 0
                k += 1
    t = pd.DataFrame(trades, columns=["date", "ret", "held"]).set_index("date")["ret"]
    # cluster by month: average trade return per month
    mt = t.groupby(t.index.to_period("M")).mean()
    mt.index = mt.index.to_timestamp()
    s = _stats(mt)
    out = {"id": 24, "zh": "配對交易 (Gatev 距離法)", "chosen": {"params": {"formation": form, "trade": trade, "pairs": 20,
           "open": "2σ"}, "exit": "價差回歸或期末", "unit": "% / 每筆 (市場中性，對比 0)"},
           "IS": {**s["IS"], "n_trades": int(((t.index >= START) & (t.index <= IS_END)).sum()), "mean_trade": float(t[t.index <= IS_END].mean())},
           "OOS": {**s["OOS"], "n_trades": int((t.index > IS_END).sum()), "mean_trade": float(t[t.index > IS_END].mean())},
           "proxy": "S&P 500 成分股，12 個月配對、6 個月交易，前 20 對；兩腳各 50%，計雙邊成本+借貨"}
    from .run import verdict
    out["verdict"] = verdict({"IS": out["IS"], "OOS": out["OOS"]})
    return out


def s36_etf_proxy(A, X, ctxdf, get_exit, args):
    c = X.df("C")
    res = {}
    spy = c["SPY"]
    for t, zh in (("MNA", "併購套利 ETF (IQ Merger Arb)"), ("CSD", "分拆上市 ETF (Invesco S&P Spin-off)")):
        if t not in c:
            continue
        x = c[t].dropna()
        if len(x) < 250:
            continue
        r = x.pct_change().dropna()
        s = spy.reindex(r.index).pct_change().dropna()
        r = r.reindex(s.index)
        yrs = len(r) / 252
        beta = float(np.cov(r, s)[0, 1] / np.var(s))
        res[t] = {"zh": zh, "start": str(x.index[0].date()), "end": str(x.index[-1].date()),
                  "cagr": float((1 + r).prod() ** (1 / yrs) - 1), "spy_cagr": float((1 + s).prod() ** (1 / yrs) - 1),
                  "vol": float(r.std() * np.sqrt(252)), "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
                  "beta": beta, "alpha_ann": float((r.mean() - beta * s.mean()) * 252)}
    return {"id": 36, "zh": "分拆上市與收購套利", "verdict": "無法直接測試", "etf": res,
            "proxy": "冇併購/分拆交易清單：用專門做呢類策略嘅 ETF 實際表現做旁證"}


SPECIAL = {6: s06_orb, 29: s29_oops, "30b": s30b_pead, 37: s37_pre_earnings, 34: s34_insider, 35: s35_index,
           39: s39_sympathy, 41: s41_seasonality, 32: s32_opex, 40: s40_intermarket, "38b": s38b_sector_rotation,
           24: s24_pairs, 36: s36_etf_proxy}

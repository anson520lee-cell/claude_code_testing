"""Runs the 42-strategy study and writes reports/swing42/results.json.

    python -m src.backtest.swing42.run --px <dir of Yahoo pickles> --hist <fja05680 components csv> \
        --earn <dir of Yahoo earnings pickles> --sec <dir of SEC form345 zips> --out reports/swing42

Selection protocol (fixed before looking at results):
  1. For each strategy, every (parameter set x exit) on the grid is evaluated in-sample (2000-2014) and
     out-of-sample (2015-2026-10). The cell with the best IN-SAMPLE t-stat of edge (n >= 30) is chosen.
  2. Optional context filter (SPY trend, VIX level, VIX term structure, breadth): chosen in-sample too, only if
     it raises the in-sample t and keeps n >= 30.
  3. Verdict on the chosen cell's OUT-OF-SAMPLE numbers:
       有效 (valid):  IS edge > 0, OOS edge > 0, OOS t >= 2, OOS positive in >= 60% of years;
                      "強" if OOS t >= 3 (about a Bonferroni bar for 42 strategies).
       邊緣 (marginal): IS and OOS edge > 0 with OOS t >= 1, or OOS t >= 2 with IS <= 0.
       無效 (no edge): otherwise.
     The whole grid is reported too (share of cells with OOS edge > 0) so a lucky cell is visible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback

import numpy as np
import pandas as pd

from . import core, data, events
from . import strategies as st
from .core import evaluate, exit_hold, exit_level, exit_path, fresh, shift


# ====================================================================== exits
def make_exits(p: core.Panel):
    cache = {}
    held_cache = {}

    def get(key):
        if key in cache:
            return cache[key]
        a = core.atr(p)
        entry = shift(p.O, -1)
        unit = None
        if key == "live":
            out, hd = exit_path(p, "live", 1, 20, 1.5, 3.0)
            held_cache[key] = hd.astype(np.float32)
            unit = 1.5 * a / entry
        elif key == "live_s":
            out, hd = exit_path(p, "live", -1, 20, 1.5, 3.0)
            held_cache[key] = hd.astype(np.float32)
            unit = 1.5 * a / entry
        elif key == "trend":
            out, hd = exit_path(p, "trend", 1, 60)
            held_cache[key] = hd.astype(np.float32)
        elif key == "mr":
            out, hd = exit_path(p, "mr", 1, 10)
            held_cache[key] = hd.astype(np.float32)
        elif key == "mr_s":
            out, hd = exit_path(p, "mr", -1, 10)
            held_cache[key] = hd.astype(np.float32)
        elif key == "mr20":
            out, hd = exit_path(p, "mr", 1, 15, ma_n=20)
            held_cache[key] = hd.astype(np.float32)
        elif key.startswith("hold"):
            n = int(key[4:].split("_")[0])
            out = exit_hold(p, n, -1 if key.endswith("_s") else 1)
            held_cache[key] = np.full(out.shape, n, dtype=np.float32)
        else:
            raise KeyError(key)
        cache[key] = (out.astype(np.float32), None if unit is None else unit.astype(np.float32))
        return cache[key]

    def held(key):
        get(key)
        return held_cache[key]
    get.held = held
    return get


EXIT_ZH = {"live": "實盤出場 1.5ATR止損+3ATR移動止損, 最多20日 (R)", "live_s": "做空實盤出場 (R)",
           "trend": "8%止損, 收市跌穿EMA21出, 最多60日", "mr": "收市高過SMA5出, 最多10日", "mr_s": "收市低過SMA5回補, 最多10日",
           "mr20": "收市高過SMA20出, 最多15日"}


def exit_zh(k):
    if k in EXIT_ZH:
        return EXIT_ZH[k]
    if k.startswith("hold"):
        n = k[4:].split("_")[0]
        return f"持有{n}日{'(做空)' if k.endswith('_s') else ''}"
    if k.startswith("day"):
        return "即日(收市平倉)" if k == "day1" else f"觸發後持有{k[3:]}日"
    return k


# ====================================================================== strategy catalogue (grid + exits)
G = lambda **kw: [dict(zip(kw, v)) for v in __import__("itertools").product(*kw.values())]

CATALOGUE = [
    (0, "參考：暴跌抄底 (實盤現用)", st.s00_crash, G(kind=["atr3", "drop5_20", "drop3_15", "sigma3"]), ["hold5", "mr"], 1, 10, "唔喺 42 個入面，用同一套方法做對照"),
    # id, zh, fn, grid, textbook exits, side, cooldown, proxy note
    (1, "波動率收縮突破 VCP", st.s01_vcp, G(tight=[0.4, 0.55], pivot=[20, 40], vm=[1.0, 1.4]), ["trend", "hold20"], 1, 10, ""),
    (2, "高而緊旗形 HTF", st.s02_htf, G(gain=[0.6, 0.9], depth=[0.2, 0.3]), ["trend", "hold20"], 1, 10, "S&P 500 大型股好少 HTF，樣本細"),
    (3, "口袋樞紐 Pocket Pivot", st.s03_pocket_pivot, G(ext=[0.03, 0.06], trend=["sma50", "tt"]), ["trend", "hold10"], 1, 10, ""),
    (4, "平底/達瓦斯箱體突破", st.s04_flat_base, G(n=[25, 40], depth=[0.12, 0.18], vm=[1.0, 1.5]), ["trend", "hold20"], 1, 10, ""),
    (5, "杯柄 / 欺騙底", st.s05_cup_handle, G(variant=["handle", "cheat", "lowcheat"], max_depth=[0.35, 0.5]), ["trend", "hold20"], 1, 10, "用日線幾何近似杯柄"),
    (7, "三週緊密 3WT", st.s07_3wt, G(tol=[0.01, 0.015, 0.02]), ["trend", "hold20"], 1, 10, ""),
    (8, "突破後首次回踩 / Holy Grail", st.s08_holy_grail, G(adx=[25, 30, 35]), ["hold5", "hold10"], 1, 10, ""),
    (9, "低位雙底 / 頭肩底", st.s09_double_bottom, G(pattern=["db", "ihs"], tol=[0.03, 0.05]), ["hold20", "trend"], 1, 10, "用確認咗嘅擺動低點(5日)"),
    (10, "假跌破反轉 U&R / Spring", st.s10_undercut_rally, G(n=[20, 50, 100], trend=[True, False]), ["hold5", "hold10", "mr"], 1, 10, ""),
    (11, "訂單塊回踩 OB", st.s11_order_block, G(m=[1.5, 2.0, 2.5]), ["hold5", "hold10"], 1, 10, "OB=大陽突破前最後一根陰燭"),
    (12, "公允價值缺口 FVG 回補", st.s12_fvg, G(min_gap=[0.25, 0.5, 1.0], trend=[False, True]), ["hold5", "hold10"], 1, 10, ""),
    (13, "BOS / CHoCH", st.s13_bos_choch, G(kind=["bos", "choch"], k=[3, 5]), ["hold5", "hold10", "hold20"], 1, 10, ""),
    (14, "破壞塊 / 緩解塊", st.s14_breaker, G(kind=["breaker", "mitigation"], k=[3, 5]), ["hold5", "hold10"], 1, 10, "以被突破嘅擺動高點區域做 breaker 區"),
    (15, "最優交易進場 OTE", st.s15_ote, G(k=[3, 5], band=["62-79", "50-62"]), ["hold5", "hold10"], 1, 10, ""),
    (16, "威科夫吸籌(Spring/SOS)/派發", st.s16_wyckoff, G(mode=[0, 1], width=[0.2, 0.3], n_range=[40, 60]), ["hold10", "hold20"], 1, 10, "日線近似；派發做空另列"),
    (17, "流動性池掃蕩 Sweep", st.s17_sweep, G(side=[1], k=[3, 5], tol_atr=[0.3, 0.6]), ["hold5", "mr"], 1, 10, "等低點(兩個擺動低點相差<0.3-0.6ATR)=流動性池"),
    (18, "錨定 VWAP 回踩", st.s18_avwap_bounce, G(anchor=["low52", "event", "ytd", "high52"]), ["hold5", "hold10"], 1, 10, "event 錨=最近一次跳空≥4%+2倍量(財報代理)"),
    (19, "多錨定 VWAP 共振 Pinch", st.s19_avwap_pinch, G(width=[0.5, 1.0, 2.0]), ["hold10", "hold20"], 1, 10, ""),
    (20, "籌碼 POC 上移", st.s20_poc_shift, G(n=[60], m=[0.5, 1.0, 2.0]), ["hold10", "hold20"], 1, 20, "60日成交量分佈, 24格"),
    (21, "低量節點 LVN 快速穿越", st.s21_lvn, G(n=[60], thin_max=[0.2, 0.35]), ["hold5", "hold10"], 1, 10, ""),
    (22, "價值區 VAH/VAL 回歸/突破", st.s22_value_area, G(kind=["reject", "breakout"], n=[20, 60]), ["hold5", "hold10"], 1, 10, "日線版 80% 法則"),
    (23, "天量K線支撐/阻力互換 HVB", st.s23_hvb_flip, G(n=[60, 120, 250]), ["hold5", "hold10"], 1, 10, ""),
    (25, "TTM Squeeze", st.s25_squeeze, G(n=[3, 6], kc=[1.5, 2.0], side=[1]), ["hold10", "hold20"], 1, 10, ""),
    (26, "Connors RSI(2)", st.s26_rsi2, G(x=[2, 5, 10], trend=[True, False]), ["mr", "hold5"], 1, 5, ""),
    (27, "Z-Score 均值回歸", st.s27_zscore, G(z=[2.0, 2.5, 3.0], trend=[True, False]), ["mr20", "mr", "hold5"], 1, 5, ""),
    (28, "拋物線做空 / 首根陰線", st.s28_parabolic_short, G(n=[5, 10], gain=[0.25, 0.4], first_red=[True, False]), ["hold5_s", "mr_s", "hold10_s"], -1, 10, "S&P 500 好少暴漲股"),
    (30, "催化劑跳空 EP (價格版)", st.s30_episodic_pivot, G(gap=[0.05, 0.08, 0.12], vm=[2.0, 3.0]), ["hold20", "hold60", "trend"], 1, 20, "冇新聞數據：跳空+放量做代理"),
    (31, "軋空 (代理)", st.s31_squeeze_proxy, G(dd=[0.4, 0.6], gap=[0.08, 0.12]), ["hold5", "hold20"], 1, 20, "冇歷史淡倉數據(FINRA 被擋)：用跌深+爆量大陽做代理"),
    (33, "異常期權大單 (代理)", st.s33_uoa_proxy, G(vm=[2.5, 4.0], move=[0.01, 0.02]), ["hold5", "hold20"], 1, 10, "冇期權/暗池數據：用『爆量但價格唔郁』做弱代理"),
    (38, "相對強度領頭羊 RS", st.s38_rs_leaders, G(top=[0.8, 0.9, 0.95], near_high=[0.0, 0.85]), ["hold20", "hold60"], 1, 1, "每月底揀 RS 前 5-20%"),
    (42, "年底稅損拋售反彈", st.s42_tax_loss, G(bottom=[0.05, 0.1, 0.2], entry_day=[10, 20]), ["hold20", "hold30"], 1, 1, ""),
]


def side_of(entry_side, prm):
    return prm.get("side", entry_side)


def score_cell(res):
    r = res.get("IS", {})
    return r.get("t", -99) if r.get("n", 0) >= 30 and np.isfinite(r.get("t", np.nan)) else -99


def verdict(sel, dm=False):
    i, o = sel["IS"], sel["OOS"]
    if o.get("n", 0) < 20:
        return "樣本不足"
    ek, tk = ("edge_dm", "t_dm") if dm else ("edge", "t")
    if dm and o.get(tk) is None:
        return "-"
    ie, oe, ot = i.get(ek, np.nan), o.get(ek, np.nan), o.get(tk, np.nan)
    if ie > 0 and oe > 0 and ot >= 2 and o.get("pos_years_dm" if dm else "pos_years", 0) >= 0.6:
        return "有效(強)" if ot >= 3 else "有效"
    if (ie > 0 and oe > 0 and ot >= 1) or (oe > 0 and ot >= 2):
        return "邊緣"
    return "無效"


def run_grid(p, ctxdf, sid, zh, fn, grid, exits, side, cooldown, proxy, get_exit, sigs=None, base_mask=None, bench=None):
    live = "live" if side > 0 else "live_s"
    ex_all = list(dict.fromkeys(exits + [live]))
    cells = []
    for prm in grid:
        sig = sigs[str(prm)] if sigs is not None else fn(p, None, **prm)
        sig = fresh(np.asarray(sig, dtype=bool), cooldown) if cooldown > 1 else np.asarray(sig, dtype=bool)
        for ek in ex_all:
            out, unit = get_exit(ek)
            cells.append({"params": prm, "exit": ek, "sig": sig, "out": out, "unit": unit, "bench": None})
    return select_cells(p, ctxdf, cells, sid=sid, zh=zh, proxy=proxy, side=side, base_mask=base_mask)


def select_cells(p, ctxdf, cells, sid, zh, proxy, side, base_mask=None, filter_lag=0):
    """Evaluate every cell, pick the best by in-sample t, try the context filters (chosen in-sample)."""
    t0 = time.time()
    for c in cells:
        c["res"] = evaluate(p, c["sig"], c["out"], base_mask=base_mask, unit=c["unit"], bench=c["bench"])
    best = max(cells, key=lambda c: score_cell(c["res"]))
    filt_res = {}
    best_f = None
    for fname, f in data.FILTERS.items():
        fm = f(ctxdf).fillna(False)
        if filter_lag:
            fm = fm.shift(filter_lag).fillna(False)
        m = fm.to_numpy(dtype=bool)[:, None]
        r = evaluate(p, best["sig"] & m, best["out"], base_mask=base_mask, unit=best["unit"], bench=best["bench"])
        filt_res[fname] = r
        if score_cell(r) > score_cell(best["res"]) and (best_f is None or score_cell(r) > score_cell(filt_res[best_f])):
            best_f = fname
    oos_edges = [c["res"]["OOS"].get("edge", np.nan) for c in cells if c["res"]["OOS"].get("n", 0) >= 20]
    sel = best["res"]
    out = {
        "id": sid, "zh": zh, "proxy": proxy, "side": side,
        "chosen": {"params": best["params"], "exit": best["exit"], "exit_zh": exit_zh(best["exit"]),
                   "unit": "R" if str(best["exit"]).startswith("live") else "%"},
        "IS": sel["IS"], "OOS": sel["OOS"], "ALL": sel["ALL"],
        "verdict": verdict(sel),
        "verdict_dm": verdict(sel, dm=True) if p.S > 4 else "-",
        "grid_share_oos_pos": float(np.mean(np.array(oos_edges) > 0)) if oos_edges else None,
        "grid_median_oos_edge": float(np.nanmedian(oos_edges)) if oos_edges else None,
        "grid": [{"params": c["params"], "exit": c["exit"], "IS": _brief(c["res"]["IS"]), "OOS": _brief(c["res"]["OOS"])}
                 for c in cells],
        "filters": {k: {"IS": _brief(v["IS"]), "OOS": _brief(v["OOS"])} for k, v in filt_res.items()},
        "best_filter": best_f,
        "secs": round(time.time() - t0, 1),
    }
    if best_f:
        fr = filt_res[best_f]
        out["filtered"] = {"filter": best_f, "IS": fr["IS"], "OOS": fr["OOS"], "verdict": verdict(fr)}
    lc = [c for c in cells if str(c["exit"]).startswith("live")]
    if lc:
        lb = max(lc, key=lambda c: score_cell(c["res"]))
        out["live_exit"] = {"params": lb["params"], "IS": _brief(lb["res"]["IS"]), "OOS": _brief(lb["res"]["OOS"])}
    for c in cells:
        c.pop("sig", None); c.pop("out", None)
    return out


def _brief(r):
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()
            if k in ("n", "mean", "win", "edge", "t", "edge_dm", "t_dm", "pos_years", "pos_years_dm", "base", "trades_per_year")}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", required=True)
    ap.add_argument("--hist", required=True)
    ap.add_argument("--earn", default="")
    ap.add_argument("--sec", default="")
    ap.add_argument("--cache", default="panel.pkl")
    ap.add_argument("--out", default="reports/swing42")
    ap.add_argument("--only", default="")
    ap.add_argument("--universe", default="A", help="A = point-in-time S&P 500; B = terminal focus universe (today's list)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = data.build(a.px, a.hist, a.cache)
    A, X = d["A"], d["X"]
    if a.universe == "B":
        import yaml
        ub = set(yaml.safe_load(open("config/universe_auto.default.yaml"))["symbols"])
        inb = np.array([t in ub for t in A.tickers])
        A = core.Panel(A.dates, A.tickers, A.O, A.H, A.L, A.C, A.V,
                       inb[None, :] & np.isfinite(A.O) & (np.nan_to_num(A.C) >= 5))
        print("universe B stocks with data:", int(inb.sum()), flush=True)
    ctxdf = data.context(X, A)
    resfile = os.path.join(a.out, "results.json")
    results = json.load(open(resfile)) if os.path.exists(resfile) else {}
    only = {x for x in a.only.split(",") if x}
    get_exit = make_exits(A)

    def save():
        with open(resfile, "w") as f:
            json.dump(clean(results), f, ensure_ascii=False, indent=1)

    for (sid, zh, fn, grid, exits, side, cd, proxy) in CATALOGUE:
        if only and str(sid) not in only:
            continue
        print(f"[{sid}] {zh} ...", flush=True)
        try:
            r = run_grid(A, ctxdf, sid, zh, fn, grid, exits, side, cd, proxy, get_exit)
            results[str(sid)] = r
            print(f"   -> {r['verdict']}  IS {r['IS'].get('edge')} t{r['IS'].get('t')}  OOS {r['OOS'].get('edge')} "
                  f"t{r['OOS'].get('t')} n{r['OOS'].get('n')}  ({r['secs']}s)", flush=True)
        except Exception:
            traceback.print_exc()
        save()
    # special evaluators (events, index timing, pairs, ETF proxies)
    for sid, fnx in events.SPECIAL.items():
        if only and str(sid) not in only:
            continue
        print(f"[{sid}] special ...", flush=True)
        try:
            r = fnx(A=A, X=X, ctxdf=ctxdf, get_exit=get_exit, args=a)
            results[str(sid)] = r
            print(f"   -> {r.get('verdict')}", flush=True)
        except Exception:
            traceback.print_exc()
        save()
    results["_meta"] = {"coverage": d["coverage"], "n_stocks": A.S, "dates": [str(A.dates[0].date()), str(A.dates[-1].date())]}
    save()


if __name__ == "__main__":
    main()

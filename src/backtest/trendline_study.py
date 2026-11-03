"""Driver of the trendline / EMA200 retest study (see src/backtest/trendline.py for the definitions)."""

from __future__ import annotations

from multiprocessing import Pool
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.backtest import trendline as T

Progress = Callable[[str], None]
BASES = ("tl_d", "tl_w", "ma200", "ma250")
FAMILIES = BASES + tuple(b + "_lim" for b in BASES)
FAMILY_LABEL = {"tl_d": "Daily trendline", "tl_w": "Weekly trendline", "ma200": "EMA200 retest", "ma250": "EMA250 retest"}
BASE_STEP = 3
W_ALL = (0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0)       # ATR stop widths simulated on the baselines (structure stops interpolate)
LIM_B = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5)               # limit distance below the previous close, in ATR (control orders)
MIN_CELL = 30
METS = ("eq", "r", "stopped", "win")


def is_lim(fam: str) -> bool:
    return fam.endswith("_lim")


# ====================================================================== data
def prepare(feats: dict[str, pd.DataFrame], last_date: str) -> dict[str, dict[str, Any]]:
    """Per-stock numpy arrays (partial last day removed)."""
    out = {}
    for sym, f in feats.items():
        f = f[f.index <= pd.Timestamp(last_date)]
        if len(f) < 400:
            continue
        a = {k: f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "atr", "ema200", "ema250")}
        a.update(T.stock_features(f))
        a["index"] = f.index
        out[sym] = a
    return out


def _signals_one(args):
    sym, a = args
    f = pd.DataFrame({k: a[k] for k in ("open", "high", "low", "close", "atr", "ema200", "ema250")}, index=a["index"])
    return sym, T.collect_signals(sym, f)


def collect(arrs: dict[str, dict[str, Any]], procs: int = 4, progress: Progress | None = None) -> dict[str, pd.DataFrame]:
    with Pool(procs) as pool:
        res = pool.map(_signals_one, list(arrs.items()), chunksize=4)
    return {k: pd.concat([o[k] for _, o in res], ignore_index=True) for k in FAMILIES}


def finish_signals(df: pd.DataFrame, arrs: dict[str, dict[str, Any]], syms: list[str], fam: str) -> pd.DataFrame:
    """Attach date, ATR, features and the structure stop level. Next-open families use the signal-day values;
    limit families use the previous session (the order is placed the evening before)."""
    sid = {s: i for i, s in enumerate(syms)}
    df = df.copy()
    df["sid"] = df["sym"].map(sid).astype(int)
    lim = is_lim(fam)
    base = fam[:-4] if lim else fam
    cols: dict[str, list] = {k: [] for k in ("date", "atr", "close", "low", "open", "dd10", "ext200", "leg30", "ma")}
    for s, d in zip(df["sym"].to_numpy(), df["d"].to_numpy()):
        a = arrs[s]
        j = d - 1 if lim else d
        cols["date"].append(a["index"][d])
        cols["atr"].append(a["atr"][j])
        cols["close"].append(a["close"][j])
        cols["low"].append(a["low"][d])
        cols["open"].append(a["open"][d])
        cols["dd10"].append(a["dd10"][j])
        cols["ext200"].append(a["ext200"][j])
        cols["leg30"].append(bool(a["leg30"][j]))
        cols["ma"].append(a["e200"][d] if base == "ma200" else a["e250"][d] if base == "ma250" else np.nan)
    for k, v in cols.items():
        df[k] = v
    level = df["level"] if base.startswith("tl") else df["ma"]
    if lim:
        df["entry_px"] = np.minimum(df["open"], df["lim_px"])
        df["struct"] = level - 0.5 * df["atr"]
        dist = ((df["close"] - df["lim_px"]) / df["atr"]).clip(lower=0).to_numpy()
        df["bucket"] = np.abs(dist[:, None] - np.array(LIM_B)[None, :]).argmin(1)
    else:
        df["struct"] = np.minimum(df["low"], level) - 0.5 * df["atr"]
    df["month"] = T.month_id(df["date"])
    df["ddb"] = T.bin_dd(df["dd10"].to_numpy())
    df["up"] = (df["ext200"] > 0).astype(int)
    df["legup"] = ((df["ext200"] > 0) & df["leg30"]).astype(int)
    df["half"] = T.half_id(df["date"])
    return df


def build_panel(arrs: dict[str, dict[str, Any]], syms: list[str], slip: float) -> pd.DataFrame:
    """Every stock-day: forward returns for every horizon (entry = next open), drawdown bucket, trend side."""
    parts = []
    for si, s in enumerate(syms):
        a = arrs[s]
        n = len(a["close"])
        d = np.arange(n - 1)
        e = a["open"][d + 1] * (1 + slip)
        df = {"sid": np.full(len(d), si, dtype=np.int16), "d": d.astype(np.int32), "date": a["index"][d]}
        for h in T.HZS:
            r = np.full(len(d), np.nan, dtype=np.float32)
            ok = d + h <= n - 1
            r[ok] = (a["close"][d[ok] + h] * (1 - slip) / e[ok] - 1).astype(np.float32)
            df[f"r{h}"] = r
        df["dd10"] = a["dd10"][d].astype(np.float32)
        df["ext200"] = a["ext200"][d].astype(np.float32)
        df["leg30"] = a["leg30"][d]
        parts.append(pd.DataFrame(df))
    p = pd.concat(parts, ignore_index=True)
    p = p[np.isfinite(p["dd10"]) & np.isfinite(p["ext200"])].reset_index(drop=True)
    p["month"] = T.month_id(p["date"])
    p["ddb"] = T.bin_dd(p["dd10"].to_numpy())
    p["up"] = (p["ext200"] > 0).astype(int)
    p["legup"] = ((p["ext200"] > 0) & p["leg30"]).astype(int)
    p["half"] = T.half_id(p["date"])
    return p


# ====================================================================== baselines for the forward returns
class FwdBaseline:
    def __init__(self, panel: pd.DataFrame):
        self.day = {}
        self.dip = {}
        key = [panel["month"], panel["ddb"], panel["up"]]
        keyl = [panel["month"], panel["ddb"], panel["legup"]]
        self.leg = {}
        for h in T.HZS:
            col = f"r{h}"
            self.day[h] = panel.groupby("date")[col].mean()
            self.dip[h] = panel.groupby(key)[col].agg(["mean", "count"])
            self.leg[h] = panel.groupby(keyl)[col].agg(["mean", "count"])

    def day_mean(self, dates: pd.Series, h: int) -> np.ndarray:
        return self.day[h].reindex(pd.DatetimeIndex(dates)).to_numpy()

    def dip_mean(self, sig: pd.DataFrame, h: int, kind: str = "dip") -> np.ndarray:
        third = sig["up"] if kind == "dip" else sig["legup"]
        idx = pd.MultiIndex.from_arrays([sig["month"].to_numpy(), sig["ddb"].to_numpy(), third.to_numpy()])
        g = (self.dip if kind == "dip" else self.leg)[h].reindex(idx)
        return np.where(g["count"].to_numpy() >= MIN_CELL, g["mean"].to_numpy(), np.nan)


def fwd_returns(sig: pd.DataFrame, arrs, syms, slip: float, h: int) -> np.ndarray:
    out = np.full(len(sig), np.nan)
    for si in np.unique(sig["sid"]):
        sel = np.flatnonzero(sig["sid"].to_numpy() == si)
        a = arrs[syms[si]]
        n = len(a["close"])
        d = sig["d"].to_numpy()[sel]
        ok = d + h <= n - 1
        e = a["open"][np.minimum(d + 1, n - 1)] * (1 + slip)
        r = np.full(len(d), np.nan)
        r[ok] = a["close"][d[ok] + h] * (1 - slip) / e[ok] - 1
        out[sel] = r
    return out


# ====================================================================== variants
def variants() -> list[dict[str, Any]]:
    v = []
    for base in ("tl_d", "tl_w"):
        for lim in (False, True):
            fam = base + ("_lim" if lim else "")
            for nt in ("3", "4", "5+", "all"):
                for sl in (("all",) if lim else ("all", "flat", "rising", "steep")):
                    for up in ("any", "up"):
                        v.append({"fam": fam, "name": f"{fam}|touch={nt}|slope={sl}|trend={up}", "nt": nt, "slope": sl, "up": up})
    for base in ("ma200", "ma250"):
        for lim in (False, True):
            fam = base + ("_lim" if lim else "")
            for L in ((30, 50) if lim else (30, 50, 80)):
                for no in ("1", "2", "3+"):
                    for rs in (("any",) if lim else ("any", "rising")):
                        v.append({"fam": fam, "name": f"{fam}|leg>={L}%|retest={no}|ma={rs}", "L": L, "no": no, "rising": rs})
    return v


def variant_mask(var: dict[str, Any], sig: pd.DataFrame) -> np.ndarray:
    m = np.ones(len(sig), dtype=bool)
    if var["fam"].startswith("tl"):
        nt = sig["nt"].to_numpy()
        if var["nt"] == "3":
            m &= nt == 3
        elif var["nt"] == "4":
            m &= nt == 4
        elif var["nt"] == "5+":
            m &= nt >= 5
        if var["slope"] != "all":
            lo, hi = T.SLOPE_BUCKETS[var["slope"]]
            ann = sig["ann"].to_numpy()
            m &= (ann >= lo) & (ann < hi)
        if var["up"] == "up":
            m &= sig["ext200"].to_numpy() > 0
    else:
        no = sig[f"no{var['L']}"].to_numpy()
        m &= (no == 1) if var["no"] == "1" else (no == 2) if var["no"] == "2" else (no >= 3)
        if var["rising"] == "rising":
            m &= sig["rising"].to_numpy(dtype=bool)
    return m


# ====================================================================== exits and baselines
def exit_configs() -> list[tuple]:
    return [(s, t, hz) for s in T.STOPS for t in T.TRAILS for hz in T.EXIT_HZ]


def cfg_name(c: tuple) -> str:
    s, t, hz = c
    return f"stop={'struct' if s == 'struct' else f'{s:g}ATR'}|trail={'none' if t is None else f'{t:g}ATR'}|hold={hz}d"


def cfg_from_name(name: str) -> tuple:
    parts = dict(p.split("=") for p in name.split("|"))
    s = parts["stop"]
    s = "struct" if s == "struct" else float(s.replace("ATR", ""))
    t = None if parts["trail"] == "none" else float(parts["trail"].replace("ATR", ""))
    return (s, t, int(parts["hold"].replace("d", "")))


def _outcomes(sim: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ok = np.isfinite(sim["eq"])
    return {"eq": sim["eq"].astype(float), "r": sim["r"].astype(float),
            "stopped": np.where(ok, sim["stopped"].astype(float), np.nan), "win": np.where(ok, (sim["ret"] > 0).astype(float), np.nan)}


class Baselines:
    """Random-entry outcomes through the very same exits, as lookups aligned to signal rows.
    open mode: entry at the next open of a stock-day (every 3rd session of every stock); day = same entry day +-1,
               dip = same month x drawdown bucket x side of the 200-day average.
    lim mode: buy limit `b` ATR under the previous close, filled when the next day trades there (fill at
              min(open, limit)); control = same month x side of the 200-day average x the signal's b."""

    def __init__(self, arrs, syms, panel: pd.DataFrame, slip: float, log: Progress):
        bs = panel[(panel["d"] % BASE_STEP == 0)].reset_index(drop=True)
        bs["atr"] = np.array([arrs[syms[s]]["atr"][d] for s, d in zip(bs["sid"].to_numpy(), bs["d"].to_numpy())])
        bs["pclose"] = np.array([arrs[syms[s]]["close"][d] for s, d in zip(bs["sid"].to_numpy(), bs["d"].to_numpy())])
        self.bs = bs = bs[np.isfinite(bs["atr"]) & (bs["atr"] > 0)].reset_index(drop=True)
        self.P = P = T.gather_paths(arrs, bs["sid"].to_numpy(), bs["d"].to_numpy(), syms, offset=1)
        self.slip = slip
        self.days = pd.DatetimeIndex(np.sort(panel["date"].unique()))
        atr = bs["atr"].to_numpy()
        log(f"baseline entries {len(bs):,}")
        key = [bs["month"].to_numpy(), bs["ddb"].to_numpy(), bs["up"].to_numpy()]
        keyl = [bs["month"].to_numpy(), bs["ddb"].to_numpy(), bs["legup"].to_numpy()]
        self.open: dict[tuple, dict[str, Any]] = {}
        for w in W_ALL:
            for t in T.TRAILS:
                for hz in T.EXIT_HZ:
                    out = _outcomes(T.simulate(P, slip, w * atr, hz, trail=t, atr=atr))
                    ent = {}
                    for met, x in out.items():
                        ok = np.isfinite(x)
                        ent[f"day_{met}"] = T.day_baseline(bs["date"].to_numpy()[ok], x[ok], self.days)
                        ent[f"dip_{met}"] = pd.Series(x, index=pd.MultiIndex.from_arrays(key)).groupby(level=[0, 1, 2]).agg(["mean", "count"])
                        ent[f"leg_{met}"] = pd.Series(x, index=pd.MultiIndex.from_arrays(keyl)).groupby(level=[0, 1, 2]).agg(["mean", "count"])
                    self.open[(w, t, hz)] = ent
        log("baseline (next open) sims done")
        self.lim: dict[tuple, dict[str, Any]] = {}
        self.lim_bounce: dict[tuple, dict[str, float]] = {}
        O0 = P["O"][:, 0].astype(float)
        L0 = P["L"][:, 0].astype(float)
        for bi, b in enumerate(LIM_B):
            limit = bs["pclose"].to_numpy() - b * atr
            sel = np.flatnonzero(np.isfinite(O0) & (L0 <= limit))
            sub = {k: v[sel] for k, v in P.items()}
            px = np.minimum(O0[sel], limit[sel])
            a_sel = atr[sel]
            mon, up = bs["month"].to_numpy()[sel], bs["up"].to_numpy()[sel]
            half, legup = bs["half"].to_numpy()[sel], bs["legup"].to_numpy()[sel]
            for w in W_ALL:
                for t in T.TRAILS:
                    for hz in T.EXIT_HZ:
                        out = _outcomes(T.simulate(sub, slip, w * a_sel, hz, trail=t, atr=a_sel, entry_px=px))
                        ent = {}
                        for met, x in out.items():
                            ent[met] = pd.Series(x, index=pd.MultiIndex.from_arrays([mon, up])).groupby(level=[0, 1]).agg(["mean", "count"])
                            ent[met + "_leg"] = pd.Series(x, index=pd.MultiIndex.from_arrays([half, legup])).groupby(level=[0, 1]).agg(["mean", "count"])
                        self.lim[(bi, w, t, hz)] = ent
            for kk in T.BOUNCE_K:
                sm = T.simulate(sub, slip, kk * a_sel, 10, atr=a_sel, tgtd=kk * a_sel, entry_px=px)
                ok = np.isfinite(sm["eq"])
                self.lim_bounce[(bi, kk)] = {"tgt": float(sm["target"][ok].mean()), "stop": float(sm["stopped"][ok].mean()), "n": int(ok.sum())}
        log("baseline (limit) sims done")
        self.open_bounce = {}
        for kk in T.BOUNCE_K:
            sm = T.simulate(P, slip, kk * atr, 10, atr=atr, tgtd=kk * atr)
            ok = np.isfinite(sm["eq"])
            self.open_bounce[kk] = {"tgt": float(sm["target"][ok].mean()), "stop": float(sm["stopped"][ok].mean())}

    def get(self, sg: pd.DataFrame, lim: bool, w: float | np.ndarray, t, hz: int, met: str, kind: str) -> np.ndarray:
        """Baseline mean for every signal row at ATR stop width w (a grid width, scalar or per row)."""
        n = len(sg)
        w = np.broadcast_to(np.asarray(w, dtype=float), (n,))
        out = np.full(n, np.nan)
        for wv in np.unique(w):
            sel = np.flatnonzero(w == wv)
            if not lim:
                ent = self.open[(float(wv), t, hz)]
                if kind == "day":
                    out[sel] = ent[f"day_{met}"].reindex(pd.DatetimeIndex(sg["date"].to_numpy()[sel])).to_numpy()
                else:
                    third = sg["up"] if kind == "dip" else sg["legup"]
                    gi = pd.MultiIndex.from_arrays([sg["month"].to_numpy()[sel], sg["ddb"].to_numpy()[sel], third.to_numpy()[sel]])
                    g = ent[f"{kind}_{met}"].reindex(gi)
                    out[sel] = np.where(g["count"].to_numpy() >= MIN_CELL, g["mean"].to_numpy(), np.nan)
            else:
                bks = sg["bucket"].to_numpy()[sel]
                for bi in np.unique(bks):
                    s2 = sel[bks == bi]
                    if kind == "leg":
                        ent = self.lim[(int(bi), float(wv), t, hz)][met + "_leg"]
                        gi = pd.MultiIndex.from_arrays([sg["half"].to_numpy()[s2], sg["legup"].to_numpy()[s2]])
                    else:
                        ent = self.lim[(int(bi), float(wv), t, hz)][met]
                        gi = pd.MultiIndex.from_arrays([sg["month"].to_numpy()[s2], sg["up"].to_numpy()[s2]])
                    g = ent.reindex(gi)
                    out[s2] = np.where(g["count"].to_numpy() >= MIN_CELL, g["mean"].to_numpy(), np.nan)
        return out

    def get_interp(self, sg, lim, w: np.ndarray, t, hz, met, kind) -> np.ndarray:
        """Same for arbitrary widths: log-linear interpolation between the neighbouring grid widths."""
        g = np.array(W_ALL)
        w = np.clip(w, g[0], g[-1])
        hi = np.clip(np.searchsorted(g, w), 1, len(g) - 1)
        lo = hi - 1
        frac = (np.log(w) - np.log(g[lo])) / (np.log(g[hi]) - np.log(g[lo]))
        b_lo = self.get(sg, lim, g[lo], t, hz, met, kind)
        b_hi = self.get(sg, lim, g[hi], t, hz, met, kind)
        return b_lo + frac * (b_hi - b_lo)


# ====================================================================== the study
def run(arrs, syms, fam_sigs: dict[str, pd.DataFrame], slip: float, progress: Progress | None = None, n_perm: int = 2000) -> dict[str, Any]:
    log = progress or (lambda m: None)
    panel = build_panel(arrs, syms, slip)
    log(f"panel {len(panel):,} stock-days")
    fb = FwdBaseline(panel)
    vlist = variants()
    res: dict[str, Any] = {"n_variants": len(vlist), "n_configs": len(exit_configs()), "periods": T.PERIODS}
    sigs = {k: finish_signals(v, arrs, syms, k) for k, v in fam_sigs.items()}
    res["signal_counts"] = {k: int(len(v)) for k, v in sigs.items()}
    for k, v in sigs.items():
        log(f"{k}: {len(v):,} signals")

    # ---------------- stage 1: forward returns of the next-open families (buy the open after the touch, hold h sessions)
    fwd_rows = []
    for k, sg in sigs.items():
        if is_lim(k):
            continue
        for h in T.HZS:
            sg[f"r{h}"] = fwd_returns(sg, arrs, syms, slip, h)
            sg[f"bd{h}"] = fb.day_mean(sg["date"], h)
            sg[f"bp{h}"] = fb.dip_mean(sg, h)
            sg[f"bl{h}"] = fb.dip_mean(sg, h, "leg")
    pm = {k: T.period_masks(sg["date"]) for k, sg in sigs.items()}
    for var in vlist:
        if is_lim(var["fam"]):
            continue
        sg = sigs[var["fam"]]
        vm = variant_mask(var, sg)
        row = {"variant": var["name"], "fam": var["fam"], "n_all": int(vm.sum())}
        for h in T.HZS:
            r = sg[f"r{h}"].to_numpy()
            ed = r - sg[f"bd{h}"].to_numpy()
            ep = r - sg[f"bp{h}"].to_numpy()
            el = r - sg[f"bl{h}"].to_numpy()
            for per in ("all", "gfc", "hold", "is", "oos"):
                m = vm & (pm[var["fam"]][per] if per != "all" else True)
                cl = sg["month"].to_numpy()[m]
                mu_d, t_d, n = T.cluster_stat(ed[m], cl)
                mu_p, t_p, _ = T.cluster_stat(ep[m], cl)
                mu_l, t_l, _ = T.cluster_stat(el[m], cl)
                rr = r[m]
                row[f"{per}|h{h}"] = {"n": n, "ret": float(np.nanmean(rr)) if n else None, "win": float(np.nanmean(rr[np.isfinite(rr)] > 0)) if n else None,
                                      "ex_day": mu_d, "t_day": t_d, "ex_dip": mu_p, "t_dip": t_p, "ex_leg": mu_l, "t_leg": t_l}
        fwd_rows.append(row)
    res["forward"] = fwd_rows
    res["forward_base"] = {f"h{h}": {"ret": float(panel[f"r{h}"].mean()), "win": float((panel[f"r{h}"].dropna() > 0).mean())} for h in T.HZS}

    # ---------------- baselines through the exits
    log("baselines...")
    bl = Baselines(arrs, syms, panel, slip, log)
    cfgs = exit_configs()
    cand = []
    store: dict[tuple, dict[str, Any]] = {}
    res["bounce"] = {}
    for fam in FAMILIES:
        lim = is_lim(fam)
        sg = sigs[fam]
        sg = sg[np.isfinite(sg["atr"]) & (sg["atr"] > 0) & np.isfinite(sg["dd10"])].reset_index(drop=True)
        sigs[fam] = sg
        pm_f = T.period_masks(sg["date"])
        P = T.gather_paths(arrs, sg["sid"].to_numpy(), sg["d"].to_numpy(), syms, offset=0 if lim else 1)
        atr = sg["atr"].to_numpy()
        entry_px = sg["entry_px"].to_numpy() if lim else P["O"][:, 0].astype(float)
        struct_d = np.clip(entry_px * (1 + slip) - sg["struct"].to_numpy(), 0.75 * atr, 8.0 * atr)
        w_struct = struct_d / atr
        vmasks = {v["name"]: variant_mask(v, sg) for v in vlist if v["fam"] == fam}
        # bounce test: +k ATR before -k ATR within 10 sessions of the fill
        res["bounce"][fam] = {name: {} for name in vmasks}
        for kk in T.BOUNCE_K:
            sm = T.simulate(P, slip, kk * atr, 10, atr=atr, tgtd=kk * atr, entry_px=entry_px if lim else None)
            ok0 = np.isfinite(sm["eq"])
            if lim:
                bb = np.full(len(sg), np.nan)
                bs_ = np.full(len(sg), np.nan)
                for bi in np.unique(sg["bucket"]):
                    sel = sg["bucket"].to_numpy() == bi
                    bb[sel] = bl.lim_bounce[(int(bi), kk)]["tgt"]
                    bs_[sel] = bl.lim_bounce[(int(bi), kk)]["stop"]
            else:
                bb = np.full(len(sg), bl.open_bounce[kk]["tgt"])
                bs_ = np.full(len(sg), bl.open_bounce[kk]["stop"])
            for name, vm in vmasks.items():
                ok = ok0 & vm
                res["bounce"][fam][name][f"k{kk:g}"] = {
                    "n": int(ok.sum()), "tgt": float(sm["target"][ok].mean()) if ok.sum() else None,
                    "stop": float(sm["stopped"][ok].mean()) if ok.sum() else None,
                    "base_tgt": float(np.nanmean(bb[ok])) if ok.sum() else None, "base_stop": float(np.nanmean(bs_[ok])) if ok.sum() else None}
        for c in cfgs:
            s, t, hz = c
            sd = struct_d if s == "struct" else s * atr
            sim = T.simulate(P, slip, sd, hz, trail=t, atr=atr, entry_px=entry_px if lim else None)
            outc = _outcomes(sim)
            ex = {}
            for met in METS:
                if s == "struct":
                    bd = bl.get_interp(sg, lim, w_struct, t, hz, met, "day")
                    bp = bl.get_interp(sg, lim, w_struct, t, hz, met, "dip")
                    bg = bl.get_interp(sg, lim, w_struct, t, hz, met, "leg")
                else:
                    bd = bl.get(sg, lim, float(s), t, hz, met, "day")
                    bp = bl.get(sg, lim, float(s), t, hz, met, "dip")
                    bg = bl.get(sg, lim, float(s), t, hz, met, "leg")
                ex[met] = (outc[met], bd, bp, bg)
            for name, vm in vmasks.items():
                rec = {"variant": name, "fam": fam, "cfg": cfg_name(c)}
                for per in ("all", "gfc", "hold", "is", "oos"):
                    m = vm & (pm_f[per] if per != "all" else True)
                    cl = sg["month"].to_numpy()[m]
                    x, bd, bp, bg = ex["eq"]
                    mu_d, t_d, n = T.cluster_stat((x - bd)[m], cl)
                    mu_p, t_p, _ = T.cluster_stat((x - bp)[m], cl)
                    mu_g, t_g, _ = T.cluster_stat((x - bg)[m], cl)
                    xr, bdr, bpr, bgr = ex["r"]
                    mr_d, tr_d, _ = T.cluster_stat((xr - bdr)[m], cl)
                    mr_p, tr_p, _ = T.cluster_stat((xr - bpr)[m], cl)
                    mr_g, tr_g, _ = T.cluster_stat((xr - bgr)[m], cl)
                    xs, bds, _, _ = ex["stopped"]
                    xw, bdw, _, _ = ex["win"]
                    rec[per] = {"n": n, "eq": float(np.nanmean(x[m]) * 100) if n else None, "ex_eq": mu_d * 100, "t": t_d,
                                "ex_eq_dip": mu_p * 100, "t_dip": t_p, "ex_eq_leg": mu_g * 100, "t_leg": t_g, "r": float(np.nanmean(xr[m])) if n else None,
                                "ex_r": mr_d, "t_r": tr_d, "ex_r_dip": mr_p, "t_r_dip": tr_p, "ex_r_leg": mr_g, "t_r_leg": tr_g,
                                "stop_rate": float(np.nanmean(xs[m])) if n else None, "base_stop_rate": float(np.nanmean(bds[m])) if n else None,
                                "win": float(np.nanmean(xw[m])) if n else None, "base_win": float(np.nanmean(bdw[m])) if n else None}
                cand.append(rec)
            store[(fam, c)] = {"d": ex["eq"][0] - ex["eq"][1], "dp": ex["eq"][0] - ex["eq"][2], "dl": ex["eq"][0] - ex["eq"][3], "month": sg["month"].to_numpy(),
                               "masks": vmasks, "pm": pm_f}
        log(f"{fam}: {len(sg):,} signals done")
    res["candidates"] = cand
    res["gate"] = gate(cand, store, n_perm, log)
    return res


# ====================================================================== pre-registered gate
def gate(cand: list[dict[str, Any]], store, n_perm: int, log: Progress) -> dict[str, Any]:
    from scipy.stats import norm
    out: dict[str, Any] = {}
    dkey = {"day": "d", "dip": "dp", "leg": "dl"}
    for tag, tk, ek in (("day", "t", "ex_eq"), ("dip", "t_dip", "ex_eq_dip"), ("leg", "t_leg", "ex_eq_leg")):
        elig = []
        for r in cand:
            i = r["is"]
            if i.get("n", 0) >= 50 and i.get(ek) is not None and i[ek] > 0 and i[tk] == i[tk] and i[tk] >= 1.65:
                elig.append(r)
        n_el = len(elig)
        z_b = float(norm.isf(0.05 / max(n_el, 1)))
        need = max(2.5, z_b)
        conf = []
        for r in elig:
            key = (r["fam"], cfg_from_name(r["cfg"]))
            st = store[key]
            vm = st["masks"][r["variant"]]
            d = st[dkey[tag]]
            cm = vm & (st["pm"]["hold"] | st["pm"]["oos"])
            mu, t, n = T.cluster_stat(d[cm], st["month"][cm])
            h, o = r["hold"], r["oos"]
            conf.append({"variant": r["variant"], "cfg": r["cfg"], "is_t": r["is"][tk], "is_n": r["is"]["n"], "is_ex": r["is"][ek], "hold_ex": h.get(ek), "hold_t": h.get(tk),
                         "oos_ex": o.get(ek), "oos_t": o.get(tk), "oos_n": o.get("n"), "pooled_ex": mu * 100, "pooled_t": t,
                         "pass": bool(h.get(ek) is not None and o.get(ek) is not None and h[ek] > 0 and o[ek] > 0 and t == t and t >= need),
                         "_cm": cm, "_key": key})
        perm_p, real = None, None
        if conf:
            months = np.unique(np.concatenate([store[c["_key"]]["month"][c["_cm"]] for c in conf]))
            mi = {m: i for i, m in enumerate(months)}
            A = np.zeros((len(conf), len(months)))
            for ci, c in enumerate(conf):
                st = store[c["_key"]]
                x = st[dkey[tag]][c["_cm"]]
                mo = st["month"][c["_cm"]]
                ok = np.isfinite(x)
                for m_, v_ in pd.Series(x[ok]).groupby(mo[ok]).sum().items():
                    A[ci, mi[m_]] = v_
            rng = np.random.default_rng(11)
            sd = np.sqrt((A ** 2).sum(1))
            sd[sd == 0] = np.nan
            real = float(np.nanmax(A.sum(1) / sd))
            mx = np.empty(n_perm)
            for b in range(n_perm):
                sgn = rng.choice([-1.0, 1.0], size=len(months))
                mx[b] = np.nanmax((A @ sgn) / sd)
            perm_p = float((mx >= real).mean())
        for c in conf:
            c.pop("_cm"), c.pop("_key")
        conf.sort(key=lambda c: -(c["pooled_t"] if c["pooled_t"] == c["pooled_t"] else -9))
        out[tag] = {"n_is_candidates": n_el, "bonferroni_z": z_b, "need_t": need, "n_pass": int(sum(c["pass"] for c in conf)),
                    "max_t_perm_p": perm_p, "max_t_real": real, "top": conf[:40]}
        log(f"gate[{tag}]: IS candidates {n_el}, need t>={need:.2f}, pass {out[tag]['n_pass']}, perm p {perm_p}")
    return out


# ====================================================================== compact copy (shipped in config/)
HEADLINE = ("tl_d|touch=3|slope=all|trend=any", "tl_d|touch=3|slope=all|trend=up", "tl_w|touch=3|slope=all|trend=any", "tl_w|touch=3|slope=all|trend=up",
            "tl_d_lim|touch=3|slope=all|trend=any", "tl_d_lim|touch=3|slope=all|trend=up", "tl_w_lim|touch=3|slope=all|trend=any", "tl_w_lim|touch=3|slope=all|trend=up",
            "ma200|leg>=30%|retest=1|ma=any", "ma200|leg>=30%|retest=2|ma=any", "ma200|leg>=30%|retest=3+|ma=any", "ma250|leg>=30%|retest=1|ma=any",
            "ma200|leg>=50%|retest=1|ma=any", "ma200_lim|leg>=30%|retest=1|ma=any", "ma200_lim|leg>=30%|retest=2|ma=any", "ma200_lim|leg>=30%|retest=3+|ma=any",
            "ma250_lim|leg>=30%|retest=1|ma=any", "ma250_lim|leg>=30%|retest=2|ma=any", "ma200_lim|leg>=50%|retest=1|ma=any")


def _round(x: Any, nd: int = 4) -> Any:
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, nd) for v in x]
    return x


COMPACT_KEYS = ("n", "eq", "r", "ex_r", "t_r", "ex_r_dip", "t_r_dip", "ex_r_leg", "t_r_leg", "stop_rate", "base_stop_rate", "win", "base_win")


def compact(res: dict[str, Any]) -> dict[str, Any]:
    """Headline variants x the 40-session exits without a trail, plus the gate: small enough to ship as
    config/trendline_study.default.json."""
    head = set(HEADLINE)
    out = {k: res[k] for k in ("n_variants", "n_configs", "periods", "signal_counts", "forward_base", "generated_at", "session", "n_stocks", "survivorship_note") if k in res}
    out["forward"] = [{k: ({kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in r.items() if not k.startswith(("gfc|h", "all|h5", "all|h60", "hold|h5", "is|h5", "oos|h5", "hold|h60", "is|h60", "oos|h60"))}
                      for r in res["forward"] if r["variant"] in head]
    out["bounce"] = {fam: {v: x for v, x in d.items() if v in head} for fam, d in res["bounce"].items()}
    cands = []
    for c in res["candidates"]:
        if c["variant"] not in head or "hold=40d" not in c["cfg"] or "trail=none" not in c["cfg"]:
            continue
        cands.append({"variant": c["variant"], "fam": c["fam"], "cfg": c["cfg"],
                      **{per: {k: c[per][k] for k in COMPACT_KEYS if k in c[per]} for per in ("all", "hold", "is", "oos")}})
    out["candidates"] = cands
    out["gate"] = {tag: {**{k: v for k, v in g.items() if k != "top"}, "top": g["top"][:15]} for tag, g in res["gate"].items()}
    return _round(out)

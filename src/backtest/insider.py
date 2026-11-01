"""Insider buying study (research only): do open-market purchases by a company's own officers and directors
predict the next weeks, tested the same honest way as the strategy discovery study?

Source: SEC "Insider Transactions Data Sets" (every Form 3/4/5 since 2006, one zip per quarter):
https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
Only Form 4 / 4/A, non-derivative transactions with code P (open-market or private purchase) and
acquired-disposed flag A, by an officer or director (10% owners and "other" are left out), worth >= $5,000.

The signal day is the FILING date (the day the public could know), entry at the next session's open. Three
styles, fixed before looking at any result:
    ins_cluster3       >= 3 different officers/directors filed purchases within 30 days
    ins_cluster2_250k  >= 2 different officers/directors filed purchases worth >= $250k in total within 30 days
    ins_ceo_100k       the CEO / CFO / President filed a purchase worth >= $100k
Each style re-arms 60 days after its last signal in the same stock.

Stocks: every company with such a filing whose bars are on Yahoo (so delisted companies are missing:
survivorship bias, results look better than reality), price >= $5 and 20-day median dollar volume >= $5M on
the signal day. Random entries: every 10th session of the same stocks, same liquidity rule. Exits and the pass
rule are the discovery study's (src/backtest/discovery.py): exit chosen on the first 60% of signal dates,
judged on the rest against random entries with the same exit, t counted by signal date, pass = edge >= 0.05R
in AND out of sample and out-of-sample t >= 2.5. The terminal's own 248-stock universe is reported separately.
"""

from __future__ import annotations

import io
import math
import zipfile
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.backtest import discovery as dc

DATASET_PAGE = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
STYLE_NAMES = {
    "ins_cluster3": "Insider cluster: >= 3 officers/directors buy within 30 days",
    "ins_cluster2_250k": "Insider cluster: >= 2 buy, >= $250k in total within 30 days",
    "ins_ceo_100k": "CEO / CFO / President open-market buy >= $100k",
}
MIN_VALUE = 5_000
WINDOW_DAYS = 30
REARM_DAYS = 60
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 5e6


def parse_quarter(content: bytes) -> pd.DataFrame:
    """Purchases (code P, acquired) on Form 4 / 4/A from one quarterly zip."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        def rd(name: str, cols: list[str]) -> pd.DataFrame:
            return pd.read_csv(zf.open(name), sep="\t", usecols=cols, dtype=str, quoting=3, on_bad_lines="skip")
        sub = rd("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERTRADINGSYMBOL"])
        tr = rd("NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                                       "TRANS_ACQUIRED_DISP_CD"])
        own = rd("REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
    tr = tr[(tr["TRANS_CODE"] == "P") & (tr["TRANS_ACQUIRED_DISP_CD"] == "A")]
    sub = sub[sub["DOCUMENT_TYPE"].isin(["4", "4/A"])]
    own = own.groupby("ACCESSION_NUMBER").agg(owner=("RPTOWNERCIK", "first"), rel=("RPTOWNER_RELATIONSHIP", "first"),
                                              title=("RPTOWNER_TITLE", "first")).reset_index()
    m = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
    shares = pd.to_numeric(m["TRANS_SHARES"], errors="coerce")
    price = pd.to_numeric(m["TRANS_PRICEPERSHARE"], errors="coerce")
    return pd.DataFrame({"sym": m["ISSUERTRADINGSYMBOL"].fillna("").str.strip().str.upper(),
                         "filing_date": pd.to_datetime(m["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
                         "owner": m["owner"], "rel": m["rel"].fillna(""), "title": m["title"].fillna(""),
                         "value": shares * price, "accession": m["ACCESSION_NUMBER"]})


def insider_events(buys: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, style, signal filing date)."""
    d = buys[(buys["value"] >= MIN_VALUE) & buys["rel"].str.contains("Director|Officer")
             & buys["sym"].str.fullmatch(r"[A-Z]{1,5}")].copy()
    if d.empty:
        return pd.DataFrame(columns=["sym", "style", "filing_date", "n_buyers", "value_30d"])
    t = d["title"].str.upper()
    d["ceo"] = t.str.contains(r"\bCEO\b|CHIEF EXECUTIVE|\bCFO\b|CHIEF FINANCIAL|PRESIDENT")
    f = d.groupby(["sym", "owner", "filing_date"]).agg(value=("value", "sum"), ceo=("ceo", "max")).reset_index()
    out = []
    win, rearm = pd.Timedelta(days=WINDOW_DAYS), pd.Timedelta(days=REARM_DAYS)
    for sym, g in f.groupby("sym"):
        g = g.sort_values("filing_date")
        last: dict[str, pd.Timestamp] = {}
        for day in g["filing_date"].unique():
            w = g[(g["filing_date"] > day - win) & (g["filing_date"] <= day)]
            nb, tot = w["owner"].nunique(), float(w["value"].sum())
            today = g[g["filing_date"] == day]
            for name, ok in (("ins_cluster3", nb >= 3), ("ins_cluster2_250k", nb >= 2 and tot >= 250_000),
                             ("ins_ceo_100k", bool((today["ceo"] & (today["value"] >= 100_000)).any()))):
                if ok and (name not in last or day - last[name] > rearm):
                    out.append((sym, name, pd.Timestamp(day), nb, tot))
                    last[name] = pd.Timestamp(day)
    return pd.DataFrame(out, columns=["sym", "style", "filing_date", "n_buyers", "value_30d"])


def collect(events: pd.DataFrame, prices: dict[str, pd.DataFrame], spy: pd.Series, regime: pd.Series,
            universe: set[str], base_step: int = 10) -> dict[str, Any]:
    from src.scanner.features import build_features
    H = dc.H
    rows, fwd = [], []
    by_sym = {s: g for s, g in events.groupby("sym")}
    for k, (sym, df) in enumerate(sorted(prices.items())):
        if sym not in by_sym or df is None or len(df) < 300:
            continue
        try:
            f = build_features(df, spy, None)
        except Exception:  # noqa: BLE001 - a malformed history only drops that stock
            continue
        o, h, lo, c, a = (f[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close", "atr"))
        sw = f["swing_low_last"].to_numpy(dtype=float)
        dv = (f["close"] * f["volume"]).rolling(20).median().to_numpy(dtype=float)
        n = len(f)
        liquid = (c >= MIN_PRICE) & (dv >= MIN_DOLLAR_VOLUME)

        def add(name: str, t: int, extra: dict) -> None:
            if t < 260 or t + H >= n or not liquid[t] or not (np.isfinite(a[t]) and a[t] > 0):
                return
            d = f.index[t]
            rows.append({"symbol": sym, "setup": name, "date": d, "ref": c[t], "atr": a[t], "structural": c[t] - 1.5 * a[t],
                         "swing": sw[t], "conditional": False, "long": True, "regime": regime.get(d, "n/a"),
                         "in_universe": sym in universe, **extra})
            fwd.append(np.stack([o[t + 1: t + 1 + H], h[t + 1: t + 1 + H], lo[t + 1: t + 1 + H], c[t + 1: t + 1 + H]]))
        for e in by_sym[sym].itertuples():
            # last session on / before the filing date; entry at the next open
            add(e.style, int(f.index.searchsorted(e.filing_date, side="right")) - 1,
                {"n_buyers": e.n_buyers, "value_30d": e.value_30d})
        for t in range(260 + k % base_step, n - H, base_step):
            add("baseline", t, {})
    sig = pd.DataFrame(rows)
    for col in ("score", "tier", "planned_rr", "chip_stop", "chip_target", "resistance"):
        sig[col] = np.nan
    arr = np.array(fwd) if fwd else np.empty((0, 4, H))
    return {"signals": sig, "O": arr[:, 0], "H": arr[:, 1], "L": arr[:, 2], "C": arr[:, 3]}


def run(data: dict[str, Any], slip: float, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda m: None)
    dc.STYLE_NAMES.update(STYLE_NAMES)
    sig = data["signals"]
    styles = list(STYLE_NAMES)
    cutoff = sig.loc[sig["setup"] != "baseline", "date"].quantile(0.6)
    res: dict[str, Any] = {"available": True, "method": __doc__, "source": DATASET_PAGE, "style_names": STYLE_NAMES,
                           "cutoff": pd.Timestamp(cutoff).strftime("%Y-%m-%d"), "n_stocks": int(sig["symbol"].nunique()),
                           "first": sig["date"].min().strftime("%Y-%m-%d"), "last": sig["date"].max().strftime("%Y-%m-%d"),
                           "n_signals": {s: int((sig["setup"] == s).sum()) for s in styles}}
    say("All liquid stocks with insider filings:")
    res["styles"] = dc.evaluate_styles(data, cutoff, slip, styles, say)
    say("The terminal's own universe only:")
    m = sig["in_universe"].to_numpy(dtype=bool)
    res["universe"] = dc.evaluate_styles(dc._subset(data, m), cutoff, slip, styles, say, min_is=30) if m.any() else {}
    res["robustness"] = robustness(data, cutoff, slip)
    res["verdict"] = verdict(res)
    return res


ROBUST_EXITS = [("atr1.5", "trail3", 20), ("atr1.0", "trail3", 20), ("atr2.0", "r2", 20)]


def robustness(data: dict[str, Any], cutoff: pd.Timestamp, slip: float) -> list[dict[str, Any]]:
    """Each style with three fixed exits (no exit chosen on the data): is the edge there whichever exit is used?"""
    from src.backtest.research import simulate
    sig = data["signals"]
    out = []
    for sm, tm, hz in ROBUST_EXITS:
        br = simulate(dc._subset(data, (sig["setup"] == "baseline").to_numpy()), sm, tm, hz, slip)
        bi, bo = br[br["date"] <= cutoff]["r"].mean(), br[br["date"] > cutoff]["r"].mean()
        for name in STYLE_NAMES:
            m = (sig["setup"] == name).to_numpy()
            if m.sum() < 30:
                continue
            tr = simulate(dc._subset(data, m), sm, tm, hz, slip)
            i, o = tr[tr["date"] <= cutoff], tr[tr["date"] > cutoff]
            ci, co = dc.clustered(i), dc.clustered(o)
            out.append({"style": name, "stop": sm, "target": tm, "horizon": hz, "is_r": ci.get("avg_r"), "oos_r": co.get("avg_r"),
                        "edge_is": ci.get("avg_r", np.nan) - bi, "edge_oos": co.get("avg_r", np.nan) - bo,
                        "t_is": ci.get("t_clustered"), "t_oos": co.get("t_clustered"), "n_is": ci.get("n", 0), "n_oos": co.get("n", 0),
                        "per_year": {str(y): float(g["r"].mean()) for y, g in tr.groupby(tr["date"].dt.year)},
                        "random_per_year": {str(y): float(g["r"].mean()) for y, g in br.groupby(br["date"].dt.year)}})
    return out


def period_check(data: dict[str, Any], slip: float, style: str = "ins_ceo_100k", split: str = "2020-01-01",
                 main: tuple[str, str, int] = ("atr1.0", "trail3", 20)) -> dict[str, Any]:
    """Independent check on another period: the style's edge over random entries before and after ``split`` with the
    three fixed exits. Used to validate the 2020-2026 study on 2010-2019 filings, which played no part in designing
    it. ``main`` = the exit the 2020-2026 study chose in sample. Passes (rule fixed before the check) if, with that
    exit, the earlier period beats random entries by >= 0.05R with t >= 2.5, the later period by >= 0.05R, and every
    fixed exit is positive in both periods."""
    from src.backtest.research import simulate
    sig = data["signals"]
    cut = pd.Timestamp(split)
    rows = []
    for sm, tm, hz in ROBUST_EXITS:
        for label, m in (("before", (sig["date"] < cut).to_numpy()), ("after", (sig["date"] >= cut).to_numpy())):
            st = m & (sig["setup"] == style).to_numpy()
            if st.sum() < 30:
                continue
            tr = simulate(dc._subset(data, st), sm, tm, hz, slip)
            br = simulate(dc._subset(data, m & (sig["setup"] == "baseline").to_numpy()), sm, tm, hz, slip)
            c = dc.clustered(tr)
            rows.append({"period": label, "stop": sm, "target": tm, "horizon": hz, "n": c.get("n", 0), "avg_r": c.get("avg_r"),
                         "random_r": float(br["r"].mean()), "edge": c.get("avg_r", np.nan) - float(br["r"].mean()),
                         "t": c.get("t_clustered"), "first": tr["date"].min().strftime("%Y-%m-%d") if len(tr) else None,
                         "last": tr["date"].max().strftime("%Y-%m-%d") if len(tr) else None})
    edges = [r["edge"] for r in rows]
    mr = {r["period"]: r for r in rows if (r["stop"], r["target"], r["horizon"]) == tuple(main)}
    passes = bool(len(mr) == 2 and mr["before"]["edge"] >= 0.05 and (mr["before"]["t"] or 0) >= 2.5
                  and mr["after"]["edge"] >= 0.05 and edges and min(edges) > 0)
    return {"style": style, "split": split, "rows": rows, "passes": passes, "main_exit": list(main),
            "edge_min": min(edges, default=None)}


def _f(x: Any, d: int = 2) -> str:
    return f"{x:+.{d}f}" if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"


def verdict(res: dict[str, Any]) -> list[str]:
    st = {k: v for k, v in res["styles"].items() if not v.get("skipped")}
    passed = [v for v in st.values() if v["passes"]]
    out = [f"{len(st)} insider styles tested on {res['n_stocks']} liquid stocks ({res['first']} to {res['last']}); "
           f"{len(passed)} passed the strict test (edge >= 0.05R in AND out of sample, out-of-sample t >= 2.5 by date)."]
    for v in sorted(st.values(), key=lambda x: -(x["edge_oos"] if math.isfinite(x["edge_oos"]) else -9)):
        near = v["edge_is"] >= 0.05 and v["edge_oos"] >= 0.05 and (v["oos"].get("t_clustered") or 0) >= 2
        out.append(f"{'PASS' if v['passes'] else 'near pass' if near else 'fail'} · {v['name']}: out of sample "
                   f"{_f(v['oos'].get('avg_r'))}R over {v['oos'].get('n', 0)} trades, edge {_f(v['edge_oos'])}R "
                   f"(in sample {_f(v['edge_is'])}R), t {_f(v['oos'].get('t_clustered'), 1)}.")
    uni = {k: v for k, v in (res.get("universe") or {}).items() if not v.get("skipped")}
    if uni:
        out.append("Terminal universe only: " + "; ".join(f"{v['name']} edge {_f(v['edge_oos'])}R out of sample over "
                                                          f"{v['oos'].get('n', 0)} trades (t {_f(v['oos'].get('t_clustered'), 1)})"
                                                          for v in uni.values()) + " - too few to judge.")
    pc = res.get("period_check")
    if pc and pc.get("rows"):
        bf = [r for r in pc["rows"] if r["period"] == "before"]
        af = [r for r in pc["rows"] if r["period"] == "after"]
        out.append(f"Independent check, {STYLE_NAMES.get(pc['style'], pc['style'])}: filings {bf[0]['first'][:4] if bf else '?'}-2019 "
                   f"(never used to design the study) edge {', '.join(_f(r['edge']) + 'R' for r in bf)} "
                   f"(t {', '.join(_f(r['t'], 1) for r in bf)}); 2020 onwards {', '.join(_f(r['edge']) + 'R' for r in af)} "
                   f"(t {', '.join(_f(r['t'], 1) for r in af)}) with the three fixed exits - "
                   f"{'PASSES in both periods: used live as a score bonus.' if pc['passes'] else 'does not pass in both periods.'}")
    out.append("Survivorship: companies delisted since are missing (no Yahoo history), which flatters every style.")
    return out

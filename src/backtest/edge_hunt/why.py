"""python -m src.backtest.edge_hunt.why --out V2/out [--rep REP/out]

Per strategy: why it does (not) work, from diagnostics.csv / tests.csv of run2 (and the replication's levels.csv).
Writes why.csv with a failure class, the horizon profile, the regime split and the best condition from our data.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

NAMES = {
    "crash_rebound": "暴跌反彈 (live)", "breakout": "突破 (live)", "squeeze": "收縮突破 squeeze (live)",
    "breakdown": "跌破 (live 空)", "rally_fail": "反彈失敗 (live 空)", "ema21_loss": "失守 EMA21 (live 空)",
    "standalone_long": "隨機做多 (基準)", "standalone_short": "隨機做空 (基準)",
    "L01": "VCP 波動收縮突破", "L02": "高而緊旗形 HTF", "L03": "口袋樞紐 Pocket Pivot", "L04": "達瓦斯箱體突破",
    "L05": "杯柄 / 欺騙底", "L07": "三週緊密 3WT", "L08": "Holy Grail 首次回踩", "L09": "雙底 / 頭肩底",
    "L10": "假跌破反轉 U&R", "L11": "訂單塊 OB", "L12": "公允價值缺口 FVG", "L13": "BOS / CHoCH",
    "L14": "破壞塊 Breaker", "L15": "OTE 回撤進場", "L16": "威科夫 Spring", "L17": "流動性掃蕩 Sweep",
    "L18": "錨定 VWAP 回踩", "L19": "多錨 VWAP 共振 Pinch", "L20": "POC 上移", "L21": "低量節點 LVN",
    "L22": "價值區 VAH/VAL", "L23": "天量 K 線 HVB", "L25": "TTM Squeeze", "L26": "Connors RSI(2)",
    "L27": "Z-Score 均值回歸", "L30": "催化劑跳空 EP", "L31": "軋空 (代理)", "L33": "異常大單 (代理)",
    "L38": "相對強度領頭羊 RS", "L42": "年底稅損反彈",
    "s28": "拋物線做空", "x01": "RSI(2)≥95 做空 (200 日線下)", "x02": "Z-Score≥2 做空 (200 日線下)",
    "x03": "TTM Squeeze 向下", "x04": "掃高點流動性做空", "x05": "跳空高開回落", "x06": "假突破做空",
    "x07": "死貓反彈做空", "x08": "量增創 52 週新低", "x09": "箱體跌破",
}
FEAT = {
    "svr5": "5 日沽空量比", "svr1": "1 日沽空量比", "svr_abn": "沽空量異常", "offex": "場外成交佔比",
    "offex_abn": "場外佔比異常", "mom12_1": "12-1 月動量", "ivol60": "特異波動", "ens_hi": "多策略加權共振高",
    "mcap": "市值", "bm": "賬面市值比", "accruals": "應計項目", "lev": "槓桿", "quality": "質量綜合",
    "skew": "SKEW 指數", "vvix": "VVIX", "pc10": "認沽認購比", "tgt_upside": "目標價空間", "last_surprise": "上次 EPS 驚喜",
    "max20": "20 日最大單日升幅", "multifam": "多家族共振", "smart_conf": "共振 + 高質量", "fam_vp5": "成交量分佈家族共振",
    "traise5": "近 5 日上調目標價", "cash_assets": "現金/資產",
}


def classify(r: pd.Series) -> str:
    it, ot = r.get("is_t", np.nan), r.get("oos_t", np.nan)
    if pd.isna(it):
        it = 0.0
    if r["n_total"] < 300:
        return "樣本太少"
    if it >= 2 and ot >= 2:
        return "兩段都贏隨機" + ("，但絕對虧錢" if r["oos_r"] < 0 else "")
    if (it >= 1.5 and ot >= 1.5) or (it >= 2 and ot >= 1):
        return "兩段同方向，但不夠顯著" + ("，而且絕對虧錢" if r["oos_r"] < 0 else "")
    if it >= 1.5 and ot < 1:
        return "樣本內有、樣本外消失"
    if it < 1.5 and ot >= 1.5:
        return "只在近年出現（未證實）"
    if it <= -1.5 and ot <= -1:
        return "持續輸給隨機入場"
    if (it <= -1.5 and ot >= 1) or (it >= 1.5 and ot <= -1):
        return "方向反轉（不穩定）"
    return "沒有資訊（等於隨機）"


def horizon(r: pd.Series) -> str:
    fx = [r.get(f"oos_fx{k}", np.nan) for k in (1, 3, 5, 10, 20)]
    if any(pd.isna(v) for v in fx):
        return ""
    best = int(np.argmax(fx))
    return f"樣本外超額 1/5/20 日 {fx[0]:+.2f}% / {fx[2]:+.2f}% / {fx[4]:+.2f}%，峰值在 {(1, 3, 5, 10, 20)[best]} 日"


def regime(r: pd.Series) -> str:
    hi, lo = r.get("oos_edge_vix_hi", np.nan), r.get("oos_edge_vix_lo", np.nan)
    bu, be = r.get("oos_edge_bull", np.nan), r.get("oos_edge_bear", np.nan)
    parts = []
    if pd.notna(hi) and pd.notna(lo):
        parts.append(f"高 VIX {hi:+.2f}R / 低 VIX {lo:+.2f}R")
    if pd.notna(bu) and pd.notna(be):
        parts.append(f"牛市 {bu:+.2f}R / 熊市 {be:+.2f}R")
    return "；".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rep", default="")
    a = ap.parse_args()
    d = pd.read_csv(os.path.join(a.out, "diagnostics.csv"))
    t = pd.read_csv(os.path.join(a.out, "tests.csv"))
    s = np.sign(t["is_d"])
    t["stab"] = np.minimum(s * t["is_t"], s * t["oos_t"]).where(t["is_pass"])
    best = t[t["is_pass"]].sort_values("stab", ascending=False).groupby("strategy").head(1).set_index("strategy")
    d["n_total"] = d["is_n"].fillna(0) + d["oos_n"].fillna(0)
    rows = []
    for _, r in d.iterrows():
        b = best.loc[r["strategy"]] if r["strategy"] in best.index else None
        cond = ""
        if b is not None:
            f = FEAT.get(b["feature"], b["feature"])
            kind = {"top": "最高三分一", "bottom": "最低三分一", "true": "成立"}[b["kind"]]
            verdict = "通過" if bool(b["pass"]) else ("樣本外同方向但不夠顯著" if b["stab"] >= 1 else "樣本外不成立")
            cond = (f"{f} {kind}: 樣本內 {b['is_d']:+.3f}R (t {b['is_t']:.1f})，樣本外 {b['oos_d']:+.3f}R (t {b['oos_t']:.1f})"
                    f"，過濾後樣本外 {b['oos_r_c']:+.3f}R → {verdict}")
        rows.append({"strategy": r["strategy"], "name": NAMES.get(r["strategy"], r["strategy"]), "side": "多" if r["side"] > 0 else "空",
                     "n": int(r["n_total"]), "is_r": r["is_r"], "oos_r": r["oos_r"], "is_edge": r["is_edge"], "is_t": r["is_t"],
                     "oos_edge": r["oos_edge"], "oos_t": r["oos_t"], "oos_win": r["oos_win"], "oos_payoff": r["oos_payoff"],
                     "class": classify(r), "horizon": horizon(r), "regime": regime(r), "best_condition": cond})
    w = pd.DataFrame(rows)
    if a.rep and os.path.exists(os.path.join(a.rep, "levels.csv")):
        lv = pd.read_csv(os.path.join(a.rep, "levels.csv")).rename(columns={"n": "rep_n", "r": "rep_r", "edge": "rep_edge", "t": "rep_t"})
        w = w.merge(lv.drop(columns=["side"]), on="strategy", how="left")
    w.to_csv(os.path.join(a.out, "why.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 80):
        print(w.drop(columns=["horizon", "regime", "best_condition"]).round(3).to_string())


if __name__ == "__main__":
    main()

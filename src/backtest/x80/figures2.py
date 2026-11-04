"""python -m src.backtest.x80.figures2 OUT_DIR FIG_DIR : figures for the second round (crypto factors, short vol, combinations)."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from src.backtest.x80.common import load
from src.backtest.x80.figures import (
    GRAY,
    INK,
    INK2,
    PAL,
    RED,
    ZH,
)

ZH2 = {**ZH, "crypto funding L/S": "幣圈資金費率多空（新）", "BTC trend (28-rule avg)": "BTC 趨勢 28 規則平均（新）",
       "ETH trend (28-rule avg)": "ETH 趨勢 28 規則平均（新）", "multi-coin trend": "多幣種趨勢（新）",
       "BTC short vol (DVOL)": "賣 BTC 期權波動率（新）", "S&P momentum top20": "S&P 集中動量 20 隻（新）"}
PCT = matplotlib.ticker.PercentFormatter(1.0, decimals=0)


def fig_sleeves(out: str, fig: str) -> None:
    w = pd.read_parquet(f"{out}/meta_sleeves_weekly.parquet")
    with open(f"{out}/meta_summary.json") as fh:
        km = json.load(fh)["kelly_math"]["2022-26"]
    need, need30 = km["sharpe_needed_80_at_best_lev"], km["sharpe_needed_80_at_30vol"]
    lneed = (need - (km["funded"] * km["spread"] - km["unfunded"] * km["tbill"]) / km["vol"]) / km["vol"]   # best leverage at that Sharpe
    rows = []
    for c in w:
        x18, x22 = w[c][w.index >= "2018-01-01"].dropna(), w[c][w.index >= "2022-01-01"].dropna()
        rows.append((ZH2.get(c, c), x18.mean() / x18.std() * np.sqrt(52), x22.mean() / x22.std() * np.sqrt(52)))
    d = pd.DataFrame(rows, columns=["name", "s18", "s22"]).sort_values("s22")
    f, ax = plt.subplots(figsize=(8.2, 6.6))
    y = np.arange(len(d))
    ax.barh(y + 0.2, d["s18"], 0.38, color=PAL[0], label="2018-26")
    ax.barh(y - 0.2, d["s22"], 0.38, color=PAL[1], label="2022-26")
    ax.axvline(need, color=RED, lw=1, ls="--", label=f"{need:.2f}：80%/年的最低要求（扣借錢成本，還要約 {lneed:.0f} 倍槓桿）")
    ax.axvline(need30, color=RED, lw=1, ls=":", label=f"{need30:.1f}：只用 30% 波動做到 80%/年需要的水平")
    ax.set_yticks(y, d["name"], fontsize=8.5)
    ax.set_xlabel("Sharpe（週，扣 T-bill）")
    ax.set_title("每個方法單獨的 Sharpe：2022-26 全部低於 80% 需要的水平")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.1), ncol=2, fontsize=8)
    ax.text(1.33, list(d["name"]).index(ZH2["crypto funding L/S"]) + 0.2, "← 含選參數期 2020-21", fontsize=7.5, color=INK2, va="center")
    f.tight_layout()
    f.savefig(f"{fig}/fig_r2_sleeves.png", dpi=170)
    plt.close(f)


def fig_frontier(out: str, fig: str) -> None:
    t = pd.read_csv(f"{out}/meta.csv")
    h = pd.read_csv(f"{out}/meta_hindsight.csv")
    with open(f"{out}/meta_summary.json") as fh:
        ms = json.load(fh)
    pick, win = (ms["allocator_pick"]["method"], int(ms["allocator_pick"]["look"])), ms["vol_window_pick"]
    zh = {"inverse vol": "反波動", "inverse vol, positive only": "反波動（只要過去賺錢的）", "shrunk mean-variance": "收縮均值方差",
          "Sharpe-weighted": "Sharpe 加權", "equal risk": "等風險貢獻"}
    lines = [(m, 104, win) for m in ("equal risk", "inverse vol", "Sharpe-weighted", "shrunk mean-variance", "inverse vol, positive only")]
    lines = sorted(lines, key=lambda z: z[:2] != pick) + [(pick[0], pick[1], "(26w)")]
    f, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
    for ax, lab in zip(axes, ("2018-26", "2022-26")):
        x = t[(t["period"] == lab) & t["lev"].str.startswith("vol")]
        for i, (m, look, w) in enumerate(lines):
            g = x[(x["method"] == m) & (x["look"] == look) & x["lev"].str.endswith(w)].copy()
            g["tv"] = g["lev"].str.extract(r"vol (\d+)%").astype(float)
            g = g.sort_values("tv")
            main = (m, look) == pick and w == win
            wz = "全部歷史估計波動" if w == "(all)" else w.strip("()").replace("w", " 週") + "估計波動"
            ax.plot(-g["max_dd"], g["cagr"], "o-" if w == win else "o--", ms=3, lw=2.4 if main else 1.1,
                    color=PAL[5] if main else PAL[i], label=f"{zh[m]} {look} 週，{wz}" + ("（2010-17 選出）" if main else ""))
        hh = h[h["period"] == lab]
        ax.plot(-hh["max_dd"], hh["cagr"], "s--", ms=3, lw=1.3, color=GRAY, label="事後最佳權重（作弊上限）")
        ax.axhline(0.8, color=RED, lw=1)
        ax.text(0.02, 0.83, "年化 80%", color=RED, fontsize=8.5)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.1, 1.6)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"-{v * 100:.0f}%" if v > 0 else "0%"))
        ax.yaxis.set_major_formatter(PCT)
        ax.set_xlabel("最大回撤")
        ax.set_title(f"{lab}：每條線是波動率目標 10% → 100%")
    axes[0].set_ylabel("年化")
    f.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    f.tight_layout(rect=(0, 0.13, 1, 1))
    f.savefig(f"{fig}/fig_r2_frontier.png", dpi=170)
    plt.close(f)


def fig_curve(out: str, fig: str) -> None:
    s = pd.read_parquet(f"{out}/meta_series.parquet")
    with open(f"{out}/meta_summary.json") as fh:
        ms = json.load(fh)
    m, look, win = ms["allocator_pick"]["method"], int(ms["allocator_pick"]["look"]), ms["vol_window_pick"]
    px = load()
    f, ax = plt.subplots(figsize=(8.6, 4.2))
    for i, tv in enumerate((30, 60, 80)):
        x = s[f"{m} {look}w @vol {tv}% {win}"].dropna()
        x = x[x.index >= "2018-01-01"]
        ax.plot((1 + x).cumprod(), color=PAL[i], lw=1.5, label=f"組合 波動率 {tv}%")
    for j, (sym, lab) in enumerate((("SPY", "SPY"), ("BTC-USD", "BTC"))):
        p = px[sym]["adj_close"]
        p = p[p.index >= "2018-01-01"]
        ax.plot(p / p.iloc[0], color=[INK2, GRAY][j], lw=1, ls="--", label=lab)
    t = pd.date_range("2018-01-01", "2026-11-01", freq="D")
    yrs = (t - t[0]).days / 365
    ax.plot(t, 1.8 ** yrs, color=RED, lw=1, label="每年 80%（目標）")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    zh = {"inverse vol": "反波動", "equal risk": "等風險貢獻", "Sharpe-weighted": "Sharpe 加權"}.get(m, m)
    ax.set_title(f"2010-17 選出的組合規則（{zh} {look} 週，全部歷史估計波動）對比 80%/年")
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{fig}/fig_r2_curve.png", dpi=170)
    plt.close(f)


def fig_funding(out: str, fig: str) -> None:
    s = pd.read_parquet(f"{out}/crypto_factors_series.parquet")
    c = "funding L/S (chosen design)"
    price = "funding 3d, fifths, weekly, price part only (no funding)"
    f, ax = plt.subplots(figsize=(8.6, 4.0))
    for i, (col, lab) in enumerate(((c, "做多低資金費率 / 做空高資金費率（1 倍）"), (price, "只看價格部分（不計資金費率）"),
                                    (f"{c} @vol 60%", "同一策略，波動率目標 60%"))):
        x = s[col].dropna()
        ax.plot((1 + x).cumprod(), color=PAL[i], lw=1.4, label=lab)
    ax.axvline(pd.Timestamp("2022-01-01"), color=GRAY, lw=1, ls=":")
    ax.text(pd.Timestamp("2022-02-15"), 1.3, "← 2020-21 選參數　2022 起是樣本外 →", color=INK2, fontsize=8)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_title("幣圈資金費率多空：2022 年後賺的全是資金費率，價格部分是虧的")
    ax.legend(fontsize=8, loc="upper left")
    f.tight_layout()
    f.savefig(f"{fig}/fig_r2_funding.png", dpi=170)
    plt.close(f)


def fig_vrp(out: str, fig: str) -> None:
    s = pd.read_parquet(f"{out}/crypto_vrp_series.parquet")
    dv = pd.read_csv("/home/user/research_data/x80/raw/deribit/dvol.csv")
    dv = dv[dv["currency"] == "BTC"]
    dv.index = pd.to_datetime(dv["time_ms"], unit="ms")
    sp = pd.read_parquet("/home/user/research_data/x80/raw/binance/spot.parquet", columns=["date", "symbol", "close"])
    btc = sp[sp["symbol"] == "BTCUSDT"].set_index("date")["close"].sort_index()
    rv_next = np.log(btc).diff().rolling(30).std().shift(-30) * np.sqrt(365) * 100      # what the next 30 days delivered
    prem = (dv["close"] - rv_next.reindex(dv.index)).rolling(90, min_periods=60).mean()
    f, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    x = s["BTC short vol, iv>rv10, size 0.5, haircut 2"].dropna()
    a1.plot((1 + x).cumprod(), color=PAL[0], lw=1.4, label="賣 BTC 30 日波動率（2021-22 選出的版本）")
    a1.axvline(pd.Timestamp("2023-01-01"), color=GRAY, lw=1, ls=":")
    a1.set_title("賣比特幣期權波動率：2023 年後溢價變小，年化只剩約 6%（和國債差不多）")
    a1.legend(fontsize=8)
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    a2.plot(prem, color=PAL[1], lw=1.3, label="隱含波動率 - 之後 30 日實際波動率（90 日平均，百分點）")
    a2.axhline(2, color=RED, lw=1, ls="--")
    a2.text(pd.Timestamp("2023-02-01"), -6.5, "紅虛線：買賣價差 + 對沖成本約 2 點", color=RED, fontsize=8)
    a2.axhline(0, color=INK, lw=0.6)
    a2.legend(fontsize=8)
    f.tight_layout()
    f.savefig(f"{fig}/fig_r2_vrp.png", dpi=170)
    plt.close(f)


def main() -> None:
    out, fig = sys.argv[1:3]
    os.makedirs(fig, exist_ok=True)
    for fn in (fig_sleeves, fig_frontier, fig_curve, fig_funding, fig_vrp):
        fn(out, fig)
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()

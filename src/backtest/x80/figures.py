"""python -m src.backtest.x80.figures OUT_DIR FIG_DIR : the report figures (PNG, light, print)."""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest.x80.common import cash_daily, load  # noqa: E402
from src.backtest.x80.frontier import spread_of  # noqa: E402

INK, INK2, GRID, GRAY, RED = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984", "#c0392b"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#d4a20b", "#d03f7a", "#3aa6c4", "#6b8e23", "#8a6d3b", "#555555", "#b5651d"]
plt.rcParams.update({"font.family": ["WenQuanYi Zen Hei"], "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False, "axes.unicode_minus": False})
ZH = {"BTC trend": "BTC 趨勢", "ETH trend": "ETH 趨勢", "crypto XS momentum": "幣圈橫截面動量", "BTC funding carry": "BTC 資金費率套利",
      "short VIX + term filter": "做空 VIX + 期限結構", "QQQ vol target": "QQQ 波動率目標", "multi-asset trend": "多資產趨勢",
      "S&P 500 long/short": "S&P 500 市場中性", "PutWrite": "賣認沽 (PUT 指數)", "live swing (crash+squeeze)": "現行波段（暴跌反彈+squeeze）",
      "funding rotation": "資金費率輪動", "BTC vs alts": "做多 BTC / 做空山寨幣"}


SKIP = {"BTC funding carry", "funding rotation"}     # shown on their own (fig_x80_carry): not leverable, see frontier.EXCLUDE


def lever_curve(ex: pd.Series, cash: pd.Series, levs, spread: float = 0.01) -> list[tuple[float, float, float]]:
    out = []
    for L in levs:
        r = (L * ex + cash - max(L - 1, 0) * spread / 52).clip(lower=-1)
        eq = (1 + r).cumprod()
        yrs = len(r) / 52
        cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
        out.append((L, cagr, (eq / eq.cummax() - 1).min()))
    return out


def fig_lever(out: str, fig: str, cash_w: pd.Series) -> None:
    w = pd.read_parquet(os.path.join(out, "sleeves_weekly.parquet"))
    levs = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8]
    fig_, axes = plt.subplots(1, 2, figsize=(7.8, 5.0), sharey=True)
    for ax, (lab, a) in zip(axes, (("2018-26", "2018-01-01"), ("2022-26", "2022-01-01"))):
        for i, c in enumerate([c for c in w.columns if c not in SKIP]):
            x = w[c][w.index >= a].dropna()
            if len(x) < 52:
                continue
            cv = lever_curve(x, cash_w.reindex(x.index).fillna(0), levs, spread_of(c))
            ax.plot([-d * 100 for _, _, d in cv], [g * 100 for _, g, _ in cv], marker="o", ms=2.6, lw=1.3, color=PAL[i % len(PAL)],
                    label=ZH.get(c, c))
        ax.axhline(80, color=RED, lw=1.2, ls="--")
        ax.text(2, 83, "年化 80%", color=RED, fontsize=8.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(-40, 120)
        ax.set_title(lab)
        ax.set_xlabel("最大回撤 %（線上的點 = 0.5x → 8x 槓桿）")
    axes[0].set_ylabel("年化 %")
    h, lab_ = axes[0].get_legend_handles_labels()
    fig_.legend(h, lab_, loc="lower center", ncol=4, fontsize=7.6, bbox_to_anchor=(0.5, 0.0))
    fig_.suptitle("每個方法加槓桿：年化最多到哪裡、要承受多大回撤", fontsize=11, fontweight="bold")
    fig_.tight_layout(rect=(0, 0.12, 1, 1))
    fig_.savefig(os.path.join(fig, "fig_x80_lever.png"), dpi=200)
    plt.close(fig_)


def fig_decay(out: str, fig: str) -> None:
    w = pd.read_parquet(os.path.join(out, "sleeves_weekly.parquet"))
    rows = []
    for c in w.columns:
        if c in SKIP:
            continue
        x = w[c].dropna()
        early = x[x.index < "2018-01-01"]
        lab_e = "2017 前"
        if len(early) < 104:
            early, lab_e = x[x.index < "2022-01-01"], "2021 前"
        late = x[x.index >= "2022-01-01"]
        sr = lambda y: y.mean() / y.std() * np.sqrt(52) if len(y) > 52 else np.nan
        rows.append((ZH.get(c, c), sr(early), sr(late), lab_e, early.index[0].year if len(early) else None))
    rows.sort(key=lambda r: -np.nan_to_num(r[1]))
    fig_, ax = plt.subplots(figsize=(7.4, 4.6))
    y = np.arange(len(rows))[::-1]
    for yi, (n, e, l, lab_e, y0) in zip(y, rows):
        ax.plot([e, l], [yi, yi], color=GRAY, lw=1.4, zorder=2)
        ax.scatter([e], [yi], color="#2a78d6", s=40, zorder=3)
        ax.scatter([l], [yi], color="#eb6834", s=40, zorder=3)
        ax.text(2.45, yi, f"{y0}-{'17' if lab_e == '2017 前' else '21'}: {e:.2f} → 2022-26: {l:.2f}", va="center", fontsize=7.8, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.axvline(1.07, color=RED, lw=1.1, ls="--")
    ax.text(1.1, y[0] + 0.55, "80% 需要 ≥ 1.07", color=RED, fontsize=8.5, va="center")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-1.2, 4.3)
    ax.set_ylim(y[-1] - 0.6, y[0] + 0.9)
    ax.set_xlabel("Sharpe（藍 = 早段，橙 = 2022-26）")
    ax.set_title("所有方法的 Sharpe：早段高，近年大多跌到 0.5 左右")
    ax.grid(axis="y", visible=False)
    fig_.tight_layout()
    fig_.savefig(os.path.join(fig, "fig_x80_decay.png"), dpi=200)
    plt.close(fig_)


def fig_frontier(out: str, fig: str) -> None:
    t = pd.read_csv(os.path.join(out, "frontier.csv"))
    fig_, ax = plt.subplots(figsize=(7.2, 4.4))
    sty = {("walk-forward inverse vol", "2018-26"): ("#2a78d6", "-", "逐步向前（誠實）2018-26"),
           ("walk-forward inverse vol", "2022-26"): ("#2a78d6", ":", "逐步向前（誠實）2022-26"),
           ("hindsight max-Sharpe", "2018-26"): ("#eb6834", "-", "事後最佳權重（作弊上限）2018-26"),
           ("hindsight max-Sharpe", "2022-26"): ("#eb6834", ":", "事後最佳權重（作弊上限）2022-26")}
    for (mix, per), (col, ls, lab) in sty.items():
        x = t[(t["mix"] == mix) & (t["period"] == per)].sort_values("target_vol")
        ax.plot(-x["max_dd"] * 100, x["cagr"] * 100, color=col, ls=ls, marker="o", ms=3.5, lw=1.6, label=lab)
        for _, r in x.iterrows():
            if r["target_vol"] in (0.2, 0.6, 1.0) and per == "2018-26":
                ax.annotate(f"{int(r['target_vol'] * 100)}% 波動", (-r["max_dd"] * 100, r["cagr"] * 100), xytext=(4, -10),
                            textcoords="offset points", fontsize=7.5, color=col)
    ax.axhline(80, color=RED, lw=1.2, ls="--")
    ax.text(2, 83, "年化 80%", color=RED, fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("最大回撤 %")
    ax.set_ylabel("年化 %")
    ax.set_title("把所有方法組合再加槓桿：誠實版本到不了 80%")
    ax.legend(loc="upper left", fontsize=8)
    fig_.tight_layout()
    fig_.savefig(os.path.join(fig, "fig_x80_frontier.png"), dpi=200)
    plt.close(fig_)


def fig_crypto(out: str, fig: str, px: dict) -> None:
    w = pd.read_parquet(os.path.join(out, "sleeves_weekly.parquet"))
    cash_w = (1 + cash_daily(px, pd.date_range("2014-01-01", "2026-11-04"), 365)).resample("W-FRI").prod() - 1
    btc = px["BTC-USD"]["close"]
    b = btc.resample("W-FRI").last().pct_change()
    fig_, ax = plt.subplots(figsize=(7.4, 4.0))
    start = "2018-01-01"
    ser = {"BTC 買入持有": b, "BTC 趨勢（前段選的規則）": w["BTC trend"] + cash_w.reindex(w.index).fillna(0),
           "ETH 趨勢（前段選的規則）": w["ETH trend"] + cash_w.reindex(w.index).fillna(0),
           "幣圈橫截面動量（含已下架幣）": w["crypto XS momentum"] + cash_w.reindex(w.index).fillna(0)}
    for i, (lab, s) in enumerate(ser.items()):
        s = s[s.index >= start].dropna()
        eq = (1 + s).cumprod()
        ax.plot(eq.index, eq / eq.iloc[0], lw=1.5, color=[GRAY, "#2a78d6", "#1baf7a", "#eb6834"][i], label=lab)
    ax.axvspan(pd.Timestamp("2020-10-01"), pd.Timestamp("2021-11-30"), color="#f3e3c3", alpha=0.6, lw=0)
    ax.text(pd.Timestamp("2020-10-15"), 0.13, "2020-21 牛市", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_title("加密貨幣：大部分利潤集中在 2020-21 一段，之後明顯變差")
    ax.set_ylabel("淨值（2018 = 1，對數）")
    ax.legend(loc="upper left", fontsize=8)
    fig_.tight_layout()
    fig_.savefig(os.path.join(fig, "fig_x80_crypto.png"), dpi=200)
    plt.close(fig_)


def fig_vrp(fig: str, px: dict) -> None:
    from src.backtest.x80 import vrp
    s, cash = vrp.short_vol_series(px)
    ts = (px["^VIX"]["close"] / px["^VIX3M"]["close"]).reindex(s.index)
    fig_, ax = plt.subplots(figsize=(7.4, 3.8))
    for lab, ex, col in (("一直做空（1 倍）", pd.Series(1.0, index=s.index), GRAY), ("只在 VIX < VIX3M 時做空（1 倍）", (ts < 1.0).astype(float), "#2a78d6"),
                         ("同上，2 倍", (ts < 1.0).astype(float) * 2, "#eb6834")):
        r = vrp.account(s, cash, ex, 1)
        eq = (1 + r).cumprod()
        ax.plot(eq.index, eq.clip(lower=1e-3), lw=1.4, color=col, label=lab)
    ax.axvline(pd.Timestamp("2018-01-01"), color=INK2, lw=0.8, ls=":")
    ax.text(pd.Timestamp("2018-02-15"), 0.95, "2018 起樣本外", fontsize=8, color=INK2, transform=ax.get_xaxis_transform(), va="top")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_title("做空 VIX 期貨：2011-17 年化 57-65%，2018 年後最好的版本只有約 10%")
    ax.set_ylabel("淨值（對數）")
    ax.legend(loc="upper left", fontsize=8)
    fig_.tight_layout()
    fig_.savefig(os.path.join(fig, "fig_x80_vrp.png"), dpi=200)
    plt.close(fig_)


def fig_carry(out: str, fig: str) -> None:
    p = os.path.join(out, "crypto_carry_series.parquet")
    if not os.path.exists(p):
        return
    c = pd.read_parquet(p)
    y = c.groupby(c.index.year).sum() * 100
    fig_, ax = plt.subplots(figsize=(7.0, 3.4))
    xs = np.arange(len(y))
    names = {"BTC always on": "BTC 一直持有", "ETH always on": "ETH 一直持有", "top-5 funding rotation": "費率最高 5 隻（每日）",
             "top-5 funding, weekly": "費率最高 5 隻（每週）"}
    for i, col in enumerate(y.columns):
        bars = ax.bar(xs + (i - 1.5) * 0.2, y[col], width=0.2, color=["#2a78d6", "#1baf7a", "#eb6834", "#8e5bd0"][i], label=names.get(col, col), zorder=3)
        for r, v in zip(bars, y[col]):
            ax.text(r.get_x() + r.get_width() / 2, v + (0.6 if v >= 0 else -0.6), f"{v:.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7, color=INK)
    ax.set_xticks(xs, [str(i) for i in y.index])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("每年 % （1 倍，扣成本，未加槓桿）")
    ax.set_title("資金費率套利：每年收益逐步變薄")
    ax.legend(fontsize=8, loc="upper right")
    fig_.tight_layout()
    fig_.savefig(os.path.join(fig, "fig_x80_carry.png"), dpi=200)
    plt.close(fig_)


def main() -> None:
    out, fig = sys.argv[1:3]
    os.makedirs(fig, exist_ok=True)
    px = load()
    w = pd.read_parquet(os.path.join(out, "sleeves_weekly.parquet"))
    cash_w = ((1 + cash_daily(px, pd.date_range("2004-01-01", "2026-11-04"), 365)).resample("W-FRI").prod() - 1).reindex(w.index).fillna(0)
    fig_lever(out, fig, cash_w)
    fig_decay(out, fig)
    fig_frontier(out, fig)
    fig_crypto(out, fig, px)
    fig_vrp(fig, px)
    fig_carry(out, fig)


if __name__ == "__main__":
    main()

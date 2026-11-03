"""python -m src.backtest.edge_hunt.figures --v2 V2 --rep REP --out DIR : the report figures (PNG, light, print)."""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": ["WenQuanYi Zen Hei"], "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False, "axes.unicode_minus": False})


def fig_strategies(v2: str, rep: str, out: str) -> None:
    w = pd.read_csv(os.path.join(v2, "out", "why.csv"))
    w = w[w["n"] >= 300].dropna(subset=["oos_t", "rep_t"])
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for side, col, lab in (("多", BLUE, "做多策略"), ("空", ORANGE, "做空策略")):
        x = w[w["side"] == side]
        ax.scatter(x["oos_t"], x["rep_t"], s=46, color=col, edgecolor="white", linewidth=1.2, label=lab, zorder=3)
    for v in (-2, 2):
        ax.axvline(v, color=GRAY, lw=0.8, ls="--")
        ax.axhline(v, color=GRAY, lw=0.8, ls="--")
    lab = {"breakout": "突破(live)", "squeeze": "squeeze(live)", "crash_rebound": "暴跌反彈(live)", "L26": "RSI(2)",
           "L19": "VWAP Pinch", "x02": "Z-Score 做空", "L01": "VCP", "L25": "TTM", "x07": "死貓反彈空", "L22": "VAH/VAL"}
    for _, r in w[w["strategy"].isin(lab)].iterrows():
        ax.annotate(lab[r["strategy"]], (r["oos_t"], r["rep_t"]), xytext=(5, 4), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("S&P 500 當年成分股，2018-26 樣本外：贏隨機入場的 t 值")
    ax.set_ylabel("S&P 400/600（從未用過），2010-26：t 值")
    ax.set_title("48 個策略：沒有一個在兩個獨立股票池都 > 2")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_strategies.png"), dpi=200)
    plt.close(fig)


def fig_is_oos(v2: str, out: str) -> None:
    t = pd.read_csv(os.path.join(v2, "out", "tests.csv"))
    t = t[t["is_pass"]]
    s = np.sign(t["is_d"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for tier, col, lab in ((2, ORANGE, "Tier 2（做空 / 網上策略）"), (1, BLUE, "Tier 1（現行 setup / 隨機做多）")):
        x = t[t["tier"] == tier]
        ax.scatter((s * t["is_t"])[x.index], (s * t["oos_t"])[x.index], s=22, color=col, edgecolor="white", linewidth=0.8,
                   alpha=0.9, label=f"{lab} {len(x)} 個", zorder=3)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axhline(3.56, color=GRAY, lw=0.8, ls="--")
    ax.text(ax.get_xlim()[1], 3.6, "樣本外門檻 3.56 (Tier 2)", ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.axhline(2.82, color=GRAY, lw=0.8, ls=":")
    ax.text(ax.get_xlim()[1], 2.86, "2.82 (Tier 1)", ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xlabel("樣本內 2010-17 t 值（同方向為正）")
    ax.set_ylabel("樣本外 2018-26 t 值")
    ax.set_title("樣本內過關的 292 個組合，到樣本外大多歸零")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_is_oos.png"), dpi=200)
    plt.close(fig)


def fig_quintiles(v2: str, rep: str, out: str) -> None:
    a = pd.read_csv(os.path.join(v2, "out", "flow_quintiles_sp.csv"))
    b = pd.read_csv(os.path.join(rep, "out", "desc.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for ax, d, title in ((axes[0], a, "S&P 500 當年成分股 2010-26"), (axes[1], b, "S&P 400/600 獨立驗證 2010-26")):
        x = d[(d["strategy"] == "standalone_long") & (d["feature"] == "svr5")].sort_values("q")
        bars = ax.bar(x["q"], x["exc"], color=[BLUE if q == 1 else GRAY for q in x["q"]], width=0.62, zorder=3)
        for r, v in zip(bars, x["exc"]):
            ax.text(r.get_x() + r.get_width() / 2, v + (0.002 if v >= 0 else -0.002), f"{v:+.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylim(min(x["exc"].min(), 0) * 1.6 - 0.002, max(x["exc"].max(), 0) * 1.25 + 0.002)
        ax.set_xticks([1, 2, 3, 4, 5], ["1\n最低", "2", "3", "4", "5\n最高"])
        ax.set_title(title)
        ax.set_xlabel("5 日沽空量比（當日五分位）")
    axes[0].set_ylabel("隨機做多：超額 R（對同日平均）")
    fig.suptitle("沽空量比最低的兩成股票，做多結果最好（兩個股票池一致）", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_quintiles.png"), dpi=200)
    plt.close(fig)


def fig_replication(rep: str, out: str) -> None:
    m = pd.read_csv(os.path.join(rep, "out", "main.csv"))
    a = pd.read_csv(os.path.join(rep, "out", "addendum.csv"))
    e = pd.read_csv(os.path.join(rep, "out", "edge.csv"))
    names = {"H1": "H1 隨機做多 × 沽空量比低", "H2": "H2 隨機做多 × 場外佔比低", "H3": "H3 隨機做多 × 籌碼綜合低",
             "H4": "H4 突破 × 多策略共振高", "H5": "H5 U&R × 場外佔比低", "H6": "H6 暴跌反彈 × 沽空量比低",
             "H7": "H7 跌破做空 × 12-1 動量低", "H8": "H8 隨機做空 × 場外佔比高", "A1": "A1 所有做空 × 特異波動高",
             "A2": "A2 Z-Score 做空本身", "A3": "A3 RSI(2) 本身", "A4": "A4 VWAP Pinch 本身"}
    rows = [(r["id"], r["t"], r["z_needed"], bool(r["pass"])) for _, r in pd.concat([m, a]).iterrows()]
    rows += [(r["id"], r["t"], r["z_needed"], bool(r["pass"])) for _, r in e.iterrows()]
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[BLUE if r[3] else GRAY for r in rows], height=0.62, zorder=3)
    for i, r in enumerate(rows):
        ax.text(max(r[1], 0) + 0.06, i, f"{r[1]:.2f}" + ("  通過" if r[3] else ""), va="center", fontsize=8.5, color=INK)
        ax.plot([r[2], r[2]], [i - 0.38, i + 0.38], color=ORANGE, lw=1.6, zorder=4)
    ax.set_yticks(y, [names[r[0]] for r in rows])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("t 值（預先登記方向）；橙線 = 門檻（H：2.50，A：2.24）")
    ax.set_title("獨立驗證：12 個預先登記假設，2 個通過")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_replication.png"), dpi=200)
    plt.close(fig)


def fig_account(v2: str, rep: str, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    for ax, d, title in ((axes[0], v2, "S&P 500 當年成分股"), (axes[1], rep, "S&P 400/600 獨立驗證")):
        c = pd.read_parquet(os.path.join(d, "acct", "account_curves.parquet"))
        c2 = pd.read_parquet(os.path.join(d, "acct", "account_setups_curves.parquet"))
        series = (("SPY", c["SPY"], GRAY, "SPY 買入持有"), ("live", c["live_like_only"], ORANGE, "現行三個 setup"),
                  ("filt", c["live_like_filter"], BLUE, "現行 + 只做沽空量比低"),
                  ("cs", c2["crash_squeeze_filter"], AQUA, "去掉突破 + 只做沽空量比低"))
        for _, s, col, lab in series:
            s = s.dropna()
            ax.plot(s.index, s / s.iloc[0], color=col, lw=1.6, label=lab)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(4))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    axes[0].set_ylabel("賬戶淨值（起點 = 1，對數）")
    axes[0].legend(loc="upper left", fontsize=8.2)
    fig.suptitle("賬戶模擬 2010-26（10 格、每筆風險 1%、成本已扣）", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_account.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--rep", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_strategies(a.v2, a.rep, a.out)
    fig_is_oos(a.v2, a.out)
    fig_quintiles(a.v2, a.rep, a.out)
    fig_replication(a.rep, a.out)
    fig_account(a.v2, a.rep, a.out)


if __name__ == "__main__":
    main()

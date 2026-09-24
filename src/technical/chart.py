"""technical_chart.png: a clean 4-panel swing chart of roughly the last 6 months,
laid out like the trader's own chart setup.

Panels (shared x-axis, one slot per trading day so weekends leave no gaps):
    1. Price: daily candles, EMA9 / EMA21 / EMA50 (distinct colours), EMA200 / EMA250
       (thin grey context lines), BOLL (20, 1.8) upper / middle / lower, nearest
       support / resistance
    2. RSI6 (primary) and RSI14 with 70 / 50 / 30 reference lines
    3. MACD 12/26/9: DIF, DEA (signal) and histogram, with the zero line
    4. Daily volume (coloured by up / down day) with MAVOL20
Indicators are calculated on the full history, so values at the left edge are
already warmed up. No events are annotated here (see price_chart.png for those).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

from src.visualization.charts import (  # noqa: E402
    AXIS,
    DOWN,
    INK,
    INK_SECONDARY,
    MUTED,
    NEUTRAL,
    PRICE_LINE,
    SURFACE,
    UP,
    _compact_number,
    _legend_above,
    _new_figure,
    _save,
    _titles,
)

EMA9_COLOR = "#2a78d6"    # blue   - immediate momentum
EMA21_COLOR = "#eb6834"   # orange - primary swing trend
EMA50_COLOR = "#4a3aa7"   # violet - medium-term structure
EMA200_COLOR = "#898781"  # grey   - long-term context
EMA250_COLOR = "#b3b1a8"  # light grey - ~1-year context
BAND_FILL = "#ecebe6"
BAND_MIDDLE = "#a8a69d"
CONTEXT_STRETCH = 1.35    # EMA200 / EMA250 may widen the price axis by at most 35%, otherwise they are clipped


def _month_ticks(dates: pd.DatetimeIndex) -> tuple[list[int], list[str]]:
    """One tick at the first session of each month; the year is shown on the first tick and in January."""
    positions, labels = [], []
    for i, date in enumerate(dates):
        if i == 0 or date.month != dates[i - 1].month:
            if i > 0 or date.day <= 7:
                positions.append(i)
                labels.append(date.strftime("%b %Y") if (not labels or date.month == 1) else date.strftime("%b"))
    return positions, labels


def _candles(ax: plt.Axes, x: np.ndarray, view: pd.DataFrame) -> None:
    o, h, lo, c = (view[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    ax.vlines(x, lo, h, color=PRICE_LINE, linewidth=0.8, zorder=3)
    bottom = np.minimum(o, c)
    height = np.maximum(np.abs(c - o), np.nanmedian(c) * 0.0008)  # dojis stay visible
    up = c >= o
    ax.bar(x[up], height[up], bottom=bottom[up], width=0.62, facecolor=SURFACE, edgecolor=PRICE_LINE,
           linewidth=0.8, zorder=4)
    ax.bar(x[~up], height[~up], bottom=bottom[~up], width=0.62, facecolor=PRICE_LINE, edgecolor=PRICE_LINE,
           linewidth=0.8, zorder=4)


def _price_limits(view: pd.DataFrame, levels: list[float]) -> tuple[float, float]:
    """Y-range from the candles, BOLL, EMA9/21/50 and the drawn levels; EMA200 / EMA250 are included
    only if that widens the range by at most 35% (they are context, not the focus)."""
    core = pd.concat([view[c] for c in ("low", "high", "close", "bb_upper", "bb_lower", "ema9", "ema21", "ema50")]
                     + [pd.Series(levels, dtype=float)])
    low, high = float(np.nanmin(core)), float(np.nanmax(core))
    span = max(high - low, abs(high) * 0.01, 1e-9)
    for column in ("ema200", "ema250"):
        values = view[column].dropna()
        if values.empty:
            continue
        wider_low, wider_high = min(low, float(values.min())), max(high, float(values.max()))
        if wider_high - wider_low <= CONTEXT_STRETCH * span:
            low, high = wider_low, wider_high
    pad = 0.04 * (high - low)
    return low - pad, high + pad


def technical_chart(result: dict[str, Any], path: Path, days: int = 126) -> Path:
    """Draw and save the swing technical chart; returns the path."""
    frame: pd.DataFrame = result["frame"]
    view = frame.tail(days)
    n = len(view)
    x = np.arange(n)
    last = view.iloc[-1]
    lookback = result["levels"]["lookback"]
    fig, (price_ax, rsi_ax, macd_ax, volume_ax) = _new_figure(nrows=4, height=12.5, height_ratios=[3.4, 1.1, 1.1, 1.1],
                                                             sharex=True)

    # ---- 1. price panel
    price_ax.fill_between(x, view["bb_lower"], view["bb_upper"], color=BAND_FILL, linewidth=0, zorder=1,
                          label="BOLL (20, 1.8)")
    for column in ("bb_upper", "bb_lower"):
        price_ax.plot(x, view[column], color=AXIS, linewidth=0.9, zorder=1)
    price_ax.plot(x, view["bb_mid"], color=BAND_MIDDLE, linewidth=0.9, linestyle=(0, (6, 2, 1.5, 2)), zorder=1,
                  label="BOLL middle (MA20)")
    if view["open"].notna().mean() > 0.9:
        _candles(price_ax, x, view)
    else:
        price_ax.plot(x, view["close"], color=PRICE_LINE, linewidth=1.6, zorder=4, label="Close")
    price_ax.plot(x, view["ema9"], color=EMA9_COLOR, linewidth=1.6, zorder=6, label="EMA9")
    price_ax.plot(x, view["ema21"], color=EMA21_COLOR, linewidth=1.9, zorder=6, label="EMA21")
    price_ax.plot(x, view["ema50"], color=EMA50_COLOR, linewidth=1.7, zorder=5, label="EMA50")
    price_ax.plot(x, view["ema200"], color=EMA200_COLOR, linewidth=1.1, linestyle=(0, (6, 3)), zorder=2,
                  label="EMA200")
    price_ax.plot(x, view["ema250"], color=EMA250_COLOR, linewidth=1.1, linestyle=(0, (1.5, 2)), zorder=2,
                  label="EMA250")

    # nearest support / resistance: dashed lines across the structure window, labelled on the right
    start = max(0, n - lookback)
    label_x = n + 0.6
    drawn: list[float] = []
    for name, levels in (("R", result["levels"]["resistance"]), ("S", result["levels"]["support"])):
        for i, level in enumerate(levels, start=1):
            price_ax.hlines(level.price, start, n - 1 + 0.5, color=INK_SECONDARY, linewidth=1.0, linestyles="--",
                            zorder=3)
            price_ax.text(label_x, level.price, f"{name}{i} {level.price:,.2f}", va="center", ha="left",
                          fontsize=8.5, color=INK_SECONDARY)
            drawn.append(level.price)
    bottom, top = _price_limits(view, drawn)
    price_ax.set_ylim(bottom, top)
    # EMA200 / EMA250 outside the visible range: show their value at the edge instead of squashing the chart
    offset = 0.0
    for column, name in (("ema200", "EMA200"), ("ema250", "EMA250")):
        value = last.get(column)
        if pd.notna(value) and not bottom <= value <= top:
            above = value > top
            y = (top - (0.03 + offset) * (top - bottom)) if above else (bottom + (0.03 + offset) * (top - bottom))
            price_ax.text(label_x, y, f"{name} {value:,.2f} {'↑' if above else '↓'}", va="center", ha="left",
                          fontsize=8, color=MUTED)
            offset += 0.05
    price_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.2f}"))
    price_ax.set_ylabel("Price", color=INK_SECONDARY, fontsize=9)
    _legend_above(price_ax, ncol=8)

    # ---- 2. RSI panel: RSI6 primary
    for level in (70, 50, 30):
        rsi_ax.axhline(level, color=MUTED if level != 50 else AXIS, linewidth=0.9, linestyle="--" if level != 50 else "-")
    rsi_ax.plot(x, view["rsi6"], color=INK, linewidth=1.7, zorder=3, label="RSI6")  # primary: listed first, drawn on top
    rsi_ax.plot(x, view["rsi14"], color=EMA50_COLOR, linewidth=1.1, zorder=2, label="RSI14")
    rsi_ax.set_ylim(0, 100)
    rsi_ax.yaxis.set_major_locator(FixedLocator([30, 50, 70]))
    rsi_ax.set_ylabel("RSI 6 / 14", color=INK_SECONDARY, fontsize=9)
    _legend_above(rsi_ax, ncol=2)

    # ---- 3. MACD 12/26/9 panel
    hist = np.nan_to_num(view["macd_hist"].to_numpy(dtype=float))
    positive = hist >= 0
    macd_ax.bar(x[positive], hist[positive], width=0.7, color=UP, alpha=0.55, linewidth=0, label="Histogram > 0")
    macd_ax.bar(x[~positive], hist[~positive], width=0.7, color=DOWN, alpha=0.55, linewidth=0, label="Histogram < 0")
    macd_ax.axhline(0, color=AXIS, linewidth=0.9)
    macd_ax.plot(x, view["macd"], color=INK, linewidth=1.4, label="DIF (12, 26)")
    macd_ax.plot(x, view["macd_signal"], color=EMA21_COLOR, linewidth=1.3, label="DEA / signal (9)")
    macd_ax.set_ylabel("MACD 12/26/9", color=INK_SECONDARY, fontsize=9)
    _legend_above(macd_ax, ncol=4)

    # ---- 4. volume panel with MAVOL20
    volume = view["volume"].fillna(0).to_numpy(dtype=float)
    direction = view["return_1d"].to_numpy(dtype=float)
    up_day, down_day = direction > 0, direction < 0
    volume_ax.bar(x[up_day], volume[up_day], width=0.7, color=UP, alpha=0.55, linewidth=0, label="Volume (up day)")
    volume_ax.bar(x[down_day], volume[down_day], width=0.7, color=DOWN, alpha=0.55, linewidth=0,
                  label="Volume (down day)")
    other = ~(up_day | down_day)
    if other.any():
        volume_ax.bar(x[other], volume[other], width=0.7, color=NEUTRAL, linewidth=0)
    volume_ax.plot(x, view["mavol20"], color=INK, linewidth=1.3, label="MAVOL20")
    volume_ax.set_ylim(bottom=0)
    volume_ax.yaxis.set_major_formatter(FuncFormatter(_compact_number))
    volume_ax.set_ylabel("Volume", color=INK_SECONDARY, fontsize=9)
    _legend_above(volume_ax, ncol=3)

    positions, labels = _month_ticks(view.index)
    volume_ax.xaxis.set_major_locator(FixedLocator(positions))
    volume_ax.set_xticklabels(labels)
    for ax in (price_ax, rsi_ax, macd_ax, volume_ax):
        ax.set_xlim(-1, n + 7)

    currency = "$" if (result.get("currency") or "").upper() == "USD" else ""
    _titles(
        fig, f"{result['ticker']} - swing technical view (last {n} sessions)",
        f"{view.index[0]:%Y-%m-%d} to {view.index[-1]:%Y-%m-%d} · last close {currency}{last['close']:,.2f} · "
        f"daily candles, dividend-adjusted · support (S) / resistance (R) from the last {lookback} sessions · "
        "descriptive, not a signal",
    )
    return _save(fig, path)

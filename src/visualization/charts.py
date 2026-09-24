"""Create the report charts with matplotlib (saved as PNG files).

Colour rules used in every chart (checked for colour-blind safety):
* up move   -> blue  (#2a78d6) with an upward triangle marker
* down move -> red   (#e34948) with a downward triangle marker
  (shape carries the direction too, so colour is never the only cue)
* volume-only event (no abnormal price move) -> hollow grey circle
* 20-day MA -> yellow (#eda100), 50-day MA -> violet (#4a3aa7)
* price / main line -> near-black; everything "normal" -> light grey
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only; no window needed

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, PercentFormatter  # noqa: E402

logger = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
PRICE_LINE = "#2e2d2a"
UP = "#2a78d6"
DOWN = "#e34948"
MA_COLORS = {20: "#eda100", 50: "#4a3aa7"}
BENCHMARK_LINE = "#4a3aa7"
NEUTRAL = "#c3c2b7"
REFERENCE = "#eb6834"

DPI = 150
# Space reserved at the top of every figure for the title and subtitle (inches).
HEADER_INCHES = 0.8


def _style_axes(ax: plt.Axes) -> None:
    """Quiet chart chrome: hairline solid grid, only left/bottom axis lines."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, color=AXIS)


def _date_axis(ax: plt.Axes) -> None:
    """Readable date ticks. Call after plotting (uses the x-range of the data)."""
    start, end = ax.get_xlim()
    if end - start < 10:  # only a few days: one tick per day (never clock times for daily data)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        return
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _new_figure(nrows: int = 1, height: float = 6.0, height_ratios: list[float] | None = None,
                sharex: bool = False):
    fig, axes = plt.subplots(
        nrows, 1, figsize=(12, height), facecolor=SURFACE, sharex=sharex,
        gridspec_kw={"height_ratios": height_ratios} if height_ratios else None,
    )
    axes_list = list(np.atleast_1d(axes))
    for ax in axes_list:
        _style_axes(ax)
    return fig, axes_list


def _titles(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Title and subtitle at a fixed distance (in inches) from the top, whatever the figure height."""
    height = fig.get_figheight()
    # Plain figure text (not fig.suptitle) so tight_layout does not reserve the space twice.
    fig.text(0.01, 1 - 0.12 / height, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.45 / height, subtitle, ha="left", va="top", fontsize=9.5, color=INK_SECONDARY)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout(rect=(0, 0, 1, 1 - HEADER_INCHES / fig.get_figheight()))
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved chart %s", path)
    return path


def _legend(ax: plt.Axes, **kwargs) -> None:
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, **kwargs)


def _legend_above(ax: plt.Axes, ncol: int) -> None:
    """Legend in one row just above the plot area, so it can never hide data."""
    _legend(ax, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=ncol, borderaxespad=0.2,
            handletextpad=0.4, columnspacing=1.2)


def _compact_number(value: float, _position=None) -> str:
    """1_250_000 -> '1.25M'."""
    for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= divisor:
            return f"{value / divisor:.3g}{suffix}"
    return f"{value:.0f}"


# (label, face colour, edge colour, marker) for each kind of event marker
EVENT_STYLES = {
    "up": ("Up move", UP, SURFACE, "^"),
    "down": ("Down move", DOWN, SURFACE, "v"),
    "volume_only": ("Volume only", SURFACE, INK_SECONDARY, "o"),
}


def _event_subsets(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split events into price-rule up moves, price-rule down moves and volume-only events."""
    if events.empty:
        return {key: events for key in EVENT_STYLES}
    price = events["is_price_event"].fillna(False).astype(bool)
    return {
        "up": events[price & (events["direction"] == "up")],
        "down": events[price & (events["direction"] == "down")],
        "volume_only": events[~price],
    }


def _plot_event_markers(ax: plt.Axes, events: pd.DataFrame, x_column: str, y_column: str,
                        size: float = 7.5) -> None:
    """Up moves as blue ▲, down moves as red ▼, volume-only events as hollow ○."""
    for kind, subset in _event_subsets(events).items():
        subset = subset.dropna(subset=[x_column, y_column])
        if subset.empty:
            continue
        label, face, edge, marker = EVENT_STYLES[kind]
        ax.plot(
            subset[x_column], subset[y_column], linestyle="none", marker=marker, markersize=size,
            markerfacecolor=face, markeredgecolor=edge, markeredgewidth=1.2 if kind == "volume_only" else 1.0,
            label=f"{label} ({len(subset)})", zorder=5,
        )


# ---------------------------------------------------------------- the charts
def price_chart(df: pd.DataFrame, events: pd.DataFrame, ticker: str, path: Path) -> Path:
    """Close price with 20/50-day moving averages and abnormal-event markers."""
    fig, (ax,) = _new_figure(height=6.2)
    ax.plot(df.index, df["close"], color=PRICE_LINE, linewidth=1.6, label="Close")
    for n, color in MA_COLORS.items():
        column = f"ma_{n}"
        if column in df and df[column].notna().any():
            ax.plot(df.index, df[column], color=color, linewidth=1.4, label=f"{n}-day moving average")
    _plot_event_markers(ax, events, "event_date", "close")
    _date_axis(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.2f}"))
    ax.set_ylabel("Price (close)", color=INK_SECONDARY, fontsize=9)
    _legend_above(ax, ncol=3)
    _titles(
        fig, f"{ticker} - daily close, moving averages and abnormal events",
        f"{df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} · markers show days flagged by the event rules "
        "(cause not implied)",
    )
    return _save(fig, path)


def returns_distribution_chart(df: pd.DataFrame, ticker: str, path: Path) -> Path:
    """Histogram of daily returns with mean, median, ±1 std and a normal curve for comparison."""
    returns = df["daily_return"].dropna() * 100  # in percent
    mean, median, std = returns.mean(), returns.median(), returns.std(ddof=1)
    fig, (ax,) = _new_figure(height=6.0)
    bins = min(100, max(20, int(np.sqrt(len(returns)) * 2)))
    counts, edges, _ = ax.hist(returns, bins=bins, color=UP, edgecolor=SURFACE, linewidth=0.6,
                               label="Daily returns")
    def show(value: float, spec: str, suffix: str = "%") -> str:
        return "n/a" if pd.isna(value) else f"{value:{spec}}{suffix}"

    has_spread = pd.notna(std) and std > 0  # needs at least 2 different returns
    if has_spread:
        width = edges[1] - edges[0]
        x = np.linspace(edges[0], edges[-1], 400)
        normal = len(returns) * width * np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        ax.plot(x, normal, color=REFERENCE, linewidth=2, label="Normal curve with same mean & std")
    ax.axvline(mean, color=INK, linewidth=1.3, label=f"Mean {show(mean, '+.2f')}")
    ax.axvline(median, color=INK, linewidth=1.3, linestyle=":", label=f"Median {show(median, '+.2f')}")
    if has_spread:
        for sign in (-1, 1):
            ax.axvline(mean + sign * std, color=MUTED, linewidth=1.1, linestyle="--",
                       label=f"Mean ± 1 std ({std:.2f}%)" if sign == 1 else None)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=None))
    ax.set_xlabel("Daily return", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Number of days", color=INK_SECONDARY, fontsize=9)
    _legend_above(ax, ncol=3)
    stats_text = (
        f"Days: {len(returns):,}\nMean: {show(mean, '+.3f')}\nMedian: {show(median, '+.3f')}\n"
        f"Std: {show(std, '.3f')}\nSkewness: {show(returns.skew(), '.2f', '')}\n"
        f"Excess kurtosis: {show(returns.kurt(), '.2f', '')}\n"
        f"Min: {show(returns.min(), '+.2f')}  Max: {show(returns.max(), '+.2f')}"
    )
    # Outside the plot area (right-hand side), so it can never cover bars of a skewed distribution.
    ax.text(1.015, 1.0, stats_text, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            color=INK_SECONDARY, family="monospace")
    _titles(
        fig, f"{ticker} - distribution of daily returns",
        f"{df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} · excess kurtosis > 0 means more extreme days "
        "than a normal distribution",
    )
    return _save(fig, path)


def volume_chart(df: pd.DataFrame, events: pd.DataFrame, ticker: str, path: Path,
                 volume_window: int) -> Path:
    """Daily volume with the previous-N-day average and abnormal-volume days highlighted."""
    fig, (ax,) = _new_figure(height=5.6)
    volume = df["volume"].fillna(0)
    ax.vlines(df.index, 0, volume, color=NEUTRAL, linewidth=0.9, label="Volume")
    if not events.empty:
        volume_events = events[events["is_volume_event"].fillna(False).astype(bool)]
        for direction, color, label in (("up", UP, "High volume, price up"),
                                         ("down", DOWN, "High volume, price down"),
                                         ("flat", MUTED, "High volume, price flat")):
            subset = volume_events[volume_events["direction"] == direction]
            if not subset.empty:
                ax.vlines(subset["event_date"], 0, subset["volume"], color=color, linewidth=1.6,
                          label=f"{label} ({len(subset)})")
    ax.plot(df.index, df["avg_volume"], color=INK, linewidth=1.3,
            label=f"Average of previous {volume_window} days")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(_compact_number))
    ax.set_ylabel("Shares traded", color=INK_SECONDARY, fontsize=9)
    _date_axis(ax)
    _legend_above(ax, ncol=3)
    _titles(
        fig, f"{ticker} - daily trading volume",
        "Coloured lines: days flagged by a volume rule (see summary.md for thresholds)",
    )
    return _save(fig, path)


def event_chart(df: pd.DataFrame, events: pd.DataFrame, ticker: str, path: Path,
                zscore_threshold: float | None, volume_ratio_threshold: float | None) -> Path:
    """Two panels: return Z-score over time, and Z-score vs volume ratio for every day."""
    fig, (top, bottom) = _new_figure(nrows=2, height=10.0, height_ratios=[1, 1.15])

    # Panel 1: Z-score timeline
    z = df["return_zscore"]
    top.vlines(df.index, 0, z.fillna(0), color=NEUTRAL, linewidth=0.8)
    top.axhline(0, color=AXIS, linewidth=0.8)
    limit = max(float(z.abs().max()) if z.notna().any() else 1.0, zscore_threshold or 0) * 1.35
    top.set_ylim(-limit, limit)  # symmetric, with room for the date labels
    if zscore_threshold:
        for level in (zscore_threshold, -zscore_threshold):
            top.axhline(level, color=MUTED, linewidth=1, linestyle="--")
        top.text(df.index[0], zscore_threshold, f"±{zscore_threshold:g}σ threshold", va="bottom",
                 fontsize=8.5, color=INK_SECONDARY)
    if not events.empty:
        _plot_event_markers(top, events, "event_date", "return_zscore")
        scored = events.dropna(subset=["return_zscore"])
        largest = scored.loc[scored["return_zscore"].abs().sort_values(ascending=False).index[:3]]
        for _, event in largest.iterrows():
            above = event["return_zscore"] > 0
            top.annotate(
                f"{event['event_date']:%Y-%m-%d}  {event['daily_return'] * 100:+.1f}%",
                (event["event_date"], event["return_zscore"]), xytext=(0, 9 if above else -9),
                textcoords="offset points", ha="center", va="bottom" if above else "top",
                fontsize=8, color=INK_SECONDARY,
            )
        _legend_above(top, ncol=3)
    top.set_ylabel("Return Z-score (σ)", color=INK_SECONDARY, fontsize=9)
    _date_axis(top)
    top.set_title("Return Z-score of every day (grey), flagged events marked", loc="left", fontsize=10.5,
                  color=INK, pad=8)

    # Panel 2: Z-score vs volume ratio
    both = df[["return_zscore", "volume_ratio"]].dropna()
    both = both[both["volume_ratio"] > 0]
    bottom.scatter(both["return_zscore"], both["volume_ratio"], s=9, color=NEUTRAL, alpha=0.7,
                   linewidths=0, label="Other days")
    if not events.empty:
        positive = events[events["volume_ratio"] > 0]
        _plot_event_markers(bottom, positive, "return_zscore", "volume_ratio", size=8)
    bottom.set_yscale("log")
    low = both["volume_ratio"].min() if not both.empty else 1.0
    high = max(both["volume_ratio"].max() if not both.empty else 1.0, volume_ratio_threshold or 0)
    ticks = [2.0 ** k for k in range(-4, 11) if low / 2 <= 2.0 ** k <= high * 2]
    bottom.yaxis.set_major_locator(FixedLocator(ticks))
    bottom.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    bottom.yaxis.set_minor_formatter(NullFormatter())
    if zscore_threshold:
        for level in (zscore_threshold, -zscore_threshold):
            bottom.axvline(level, color=MUTED, linewidth=1, linestyle="--")
    if volume_ratio_threshold:
        bottom.axhline(volume_ratio_threshold, color=MUTED, linewidth=1, linestyle="--")
        bottom.text(bottom.get_xlim()[0], volume_ratio_threshold, f" volume ratio {volume_ratio_threshold:g}×",
                    va="bottom", fontsize=8.5, color=INK_SECONDARY)
    bottom.set_xlabel("Return Z-score (σ)", color=INK_SECONDARY, fontsize=9)
    bottom.set_ylabel("Volume ratio (log scale)", color=INK_SECONDARY, fontsize=9)
    _legend_above(bottom, ncol=4)
    bottom.set_title("How unusual was each day? Price move vs. volume", loc="left", fontsize=10.5,
                     color=INK, pad=8)

    _titles(fig, f"{ticker} - abnormal event detection",
            "Every trading day with enough history is scored; events lie beyond a dashed threshold line")
    return _save(fig, path)


def benchmark_chart(df: pd.DataFrame, ticker: str, benchmark: str, path: Path) -> Path:
    """Cumulative return of stock vs benchmark, and cumulative relative performance."""
    data = df[["cum_return", "benchmark_cum_return", "relative_performance"]].dropna()
    fig, (top, bottom) = _new_figure(nrows=2, height=8.0, height_ratios=[1.3, 1], sharex=True)
    top.plot(data.index, data["cum_return"] * 100, color=PRICE_LINE, linewidth=1.8, label=ticker)
    top.plot(data.index, data["benchmark_cum_return"] * 100, color=BENCHMARK_LINE, linewidth=1.8, label=benchmark)
    top.axhline(0, color=AXIS, linewidth=0.8)
    top.yaxis.set_major_formatter(PercentFormatter(decimals=None))
    top.set_ylabel("Cumulative return", color=INK_SECONDARY, fontsize=9)
    _legend_above(top, ncol=2)
    top.set_title(f"Cumulative return since {data.index[0]:%Y-%m-%d}", loc="left", fontsize=10.5, color=INK)

    relative = data["relative_performance"] * 100
    bottom.plot(data.index, relative, color=PRICE_LINE, linewidth=1.5)
    bottom.fill_between(data.index, 0, relative, where=relative >= 0, color=UP, alpha=0.12, linewidth=0)
    bottom.fill_between(data.index, 0, relative, where=relative < 0, color=DOWN, alpha=0.12, linewidth=0)
    bottom.axhline(0, color=AXIS, linewidth=0.8)
    bottom.yaxis.set_major_formatter(PercentFormatter(decimals=None))
    bottom.set_ylabel(f"{ticker} vs {benchmark}", color=INK_SECONDARY, fontsize=9)
    bottom.set_title(f"Relative performance = (1 + {ticker} return) / (1 + {benchmark} return) - 1",
                     loc="left", fontsize=10.5, color=INK)
    _date_axis(bottom)
    _titles(fig, f"{ticker} vs {benchmark} - benchmark comparison",
            "Both series start at 0% on the first common trading day; returns include dividends when "
            "adjusted prices are available")
    return _save(fig, path)

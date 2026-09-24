"""Write technical_analysis.md and the compact "Swing Technical Snapshot" for summary.md.

Uses the trader's chart settings: EMA 9/21/50/200/250, RSI 6/14, MACD 12/26/9,
MAVOL20 and BOLL(20, 1.8). The wording follows the indicator priority for a
2-7 day swing: price structure and EMA9 / EMA21 first, EMA200 / EMA250 only as
context.

The text is generated from the deterministic classifications with fixed
sentence templates, so the same data always gives the same wording. Sections
respect the length limits of the report design (e.g. at most 4 sentences for
the trend section, at most 150 words for the swing summary). The words Buy,
Sell and Hold are never used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.technical import classification as cls
from src.technical.classification import COMPONENTS, MAX_SCORE, last_cross_age
from src.technical.indicators import (
    ATR_PERIOD,
    BOLL_STD,
    BOLL_WINDOW,
    EMA_SPANS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MAVOL_WINDOW,
    RSI_PERIODS,
)
from src.technical.structure import COMPRESSION_ATR, NEAR_ATR, NEAR_EXTREME_ATR

EVENT_RISK_BANNER = "EVENT RISK WITHIN TYPICAL HOLDING WINDOW"
BAND_POSITION_TEXT = {
    cls.ABOVE_UPPER: "above the upper band", cls.NEAR_UPPER: "near the upper band",
    cls.UPPER_HALF: "between the middle and upper band", cls.NEAR_MIDDLE: "near the middle band",
    cls.LOWER_HALF: "between the middle and lower band", cls.NEAR_LOWER: "near the lower band",
    cls.BELOW_LOWER: "below the lower band",
}
BAND_CONTEXT_SHORT = {  # compact wording for the 150-word swing summary
    cls.UPPER_WALK: "upper-band walk", cls.UPPER_PUSH: "close above the upper band with trend and volume",
    cls.UPPER_EXTENSION: "unconfirmed extension above the upper band", cls.LOWER_WALK: "lower-band walk",
    cls.LOWER_PUSH: "close below the lower band with trend and volume",
    cls.LOWER_EXTENSION: "unconfirmed drop below the lower band",
}
COMPONENT_NAMES = {"structure": "structure", "trend": "EMA9/21", "volume": "volume", "rsi": "RSI",
                   "macd": "MACD", "boll": "BOLL", "relative_strength": "RS", "ema50": "EMA50"}


# ------------------------------------------------------------------ formatting
def _ok(value: Any) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _no_negative_zero(value: float, decimals: int) -> float:
    """Values that round to zero are shown as +0.0, never -0.0."""
    return 0.0 if abs(value) < 0.5 * 10 ** -decimals else value


def pct(value: Any, decimals: int = 1) -> str:
    return f"{_no_negative_zero(float(value) * 100, decimals):+.{decimals}f}%" if _ok(value) else "n/a"


def pct_plain(value: Any, decimals: int = 0) -> str:
    """Unsigned percentage, e.g. volatility levels."""
    return f"{float(value) * 100:.{decimals}f}%" if _ok(value) else "n/a"


def num(value: Any, decimals: int = 1) -> str:
    return f"{float(value):.{decimals}f}" if _ok(value) else "n/a"


def signed(value: Any, decimals: int = 1) -> str:
    return f"{_no_negative_zero(float(value), decimals):+.{decimals}f}" if _ok(value) else "n/a"


def macd_decimals(atr_value: Any) -> int:
    """MACD values scale with the price level: 2 decimals for ATR >= 1, 3 for ATR >= 0.1, and so on."""
    if not _ok(atr_value) or float(atr_value) <= 0:
        return 3
    return int(max(2, 2 - np.floor(np.log10(float(atr_value)))))


def sig(value: Any, decimals: int = 2) -> str:
    """Signed value with a fixed number of decimals (see macd_decimals)."""
    return signed(value, decimals)


def trend_word(value: Any, dead_zone: float = 0.05) -> str:
    """rising / falling / flat for normalised slopes (flat inside +/- dead_zone)."""
    if not _ok(value) or abs(float(value)) < dead_zone:
        return "flat"
    return "rising" if float(value) > 0 else "falling"


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def atrs(value: Any) -> str:
    """Distance in ATRs, 2 decimals so values either side of a 0.5 ATR threshold stay distinguishable."""
    return f"{signed(value, 2)} ATR" if _ok(value) else "n/a"


def ratio(value: Any) -> str:
    return f"{float(value):.2f}×" if _ok(value) else "n/a"


def ordinal(value: Any) -> str:
    if not _ok(value):
        return "n/a"
    n = int(round(float(value)))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def volume_text(value: Any) -> str:
    if not _ok(value):
        return "n/a"
    value = float(value)
    for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= divisor:
            return f"{value / divisor:.2f}{suffix}"
    return f"{value:.0f}"


class Money:
    """Formats prices with the trading currency ($ for USD, otherwise the plain number)."""

    def __init__(self, currency: str | None) -> None:
        self.symbol = "$" if (currency or "").upper() == "USD" else ""  # unknown currency: no symbol

    def __call__(self, value: Any) -> str:
        return f"{self.symbol}{float(value):,.2f}" if _ok(value) else "n/a"


def table(headers: list[str], rows: list[list[Any]], align: str | None = None) -> str:
    """Markdown table; ``align`` is one character per column: l (left) or r (right)."""
    align = align or "l" * len(headers)
    rule = "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"
    body = ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows]
    return "\n".join(["| " + " | ".join(headers) + " |", rule, *body])


# ------------------------------------------------------------------ shared phrases
def _vs(row: pd.Series, column: str) -> float:
    """Close relative to a reference: C / reference - 1."""
    return row["close"] / row[column] - 1 if _ok(row.get(column)) and row[column] else float("nan")


def _gap_atr(row: pd.Series, column: str) -> float:
    return (row["close"] - row[column]) / row["atr"] if _ok(row.get(column)) and _ok(row.get("atr")) else float("nan")


def _change(series: pd.Series, days: int) -> float:
    """Change of the last value vs ``days`` sessions earlier (NaN if not enough data)."""
    return float(series.iloc[-1] - series.iloc[-1 - days]) if len(series) > days else float("nan")


def _direction_word(value: Any, up: str = "up", down: str = "down", flat: str = "flat") -> str:
    if not _ok(value) or float(value) == 0:
        return flat
    return up if float(value) > 0 else down


def _side(value: Any, reference: Any) -> str:
    if not (_ok(value) and _ok(reference)):
        return "n/a vs"
    return "above" if value > reference else "below" if value < reference else "at"


def _ago(age: int) -> str:
    return "today" if age == 0 else f"{plural(age, 'session')} ago"


def _day_text(row: pd.Series) -> str:
    direction = row.get("day_direction")
    if direction == "up":
        return f"an up day ({pct(row['return_1d'])})"
    if direction == "down":
        return f"a down day ({pct(row['return_1d'])})"
    return "an unchanged day" if direction == "flat" else "a day with no previous close"


def _day_short(row: pd.Series) -> str:
    direction = row.get("day_direction")
    if direction in ("up", "down"):
        return f"{direction} day {pct(row['return_1d'])}"
    return "unchanged day" if direction == "flat" else "no previous close"


def _volume_verdict(row: pd.Series) -> str:
    """Volume vs MAVOL20 read together with the day's price direction."""
    level, direction = row["volume_level"], row.get("day_direction")
    if level in cls.HEAVY_VOLUME:
        if direction == "up":
            return "heavy volume on an up day, so buying pressure is confirmed by volume"
        if direction == "down":
            return "heavy volume on a down day, which is selling pressure, not confirmation of strength"
        return "heavy volume on an unchanged close"
    if level == cls.WEAK_PARTICIPATION:
        move = {"up": "the up move", "down": "the down move"}.get(direction, "the move")
        return f"light volume, so {move} has little participation"
    if level == cls.VOLUME_NORMAL:
        return "volume close to its 20-session average, so no strong confirmation either way"
    return "not available"


def _ema_events(r: dict[str, Any], short: bool = False) -> list[str]:
    """EMA9 / EMA21 events of the last few sessions (crosses, reclaims, losses, compression)."""
    row, frame = r["latest"], r["frame"]
    events: list[str] = []
    age = last_cross_age(frame["ema9"], frame["ema21"])
    if age is not None:
        events.append(f"EMA9 crossed {'above' if row['ema9'] > row['ema21'] else 'below'} EMA21 {_ago(age)}")
    for name in ("EMA21", "EMA9"):
        if row.get(f"Price reclaiming {name}"):
            events.append(f"price reclaimed {name} today")
        if row.get(f"Price losing {name}"):
            events.append(f"price lost {name} today")
    if row.get("EMA9 / EMA21 compression"):
        gap = num(abs(row["ema9"] - row["ema21"]) / row["atr"], 2)
        events.append(f"EMA9/EMA21 compressed ({gap} ATR)" if short else
                      f"EMA9 and EMA21 are compressed ({gap} ATR apart, threshold {COMPRESSION_ATR:g})")
    return events


def _rsi_details(r: dict[str, Any]) -> list[str]:
    row, frame = r["latest"], r["frame"]
    details: list[str] = []
    age = last_cross_age(frame["rsi6"], frame["rsi14"])
    if age is not None:
        details.append(f"RSI6 crossed {'above' if row['rsi6'] > row['rsi14'] else 'below'} RSI14 {_ago(age)}")
    elif _ok(row["rsi6"]) and _ok(row["rsi14"]):
        details.append(f"RSI6 {_side(row['rsi6'], row['rsi14'])} RSI14")
    if row.get("rsi_recovering_oversold"):
        details.append("RSI6 recovering from oversold")
    if row.get("rsi_losing_extended"):
        details.append("RSI6 losing momentum from an extended level")
    return details


def _macd_details(r: dict[str, Any]) -> list[str]:
    row, frame = r["latest"], r["frame"]
    details: list[str] = []
    age = last_cross_age(frame["macd"], frame["macd_signal"])
    if age is not None:
        details.append(f"{'bullish' if row['macd_hist'] > 0 else 'bearish'} crossover {_ago(age)}")
    if row.get("macd_hist_trend"):
        details.append(f"histogram {row['macd_hist_trend']}")
    if row.get("macd_zero_context"):
        details.append(f"MACD {row['macd_zero_context']}")
    return details


def _event_sentence(risk: dict[str, Any]) -> str:
    if risk["status"] == "unknown":
        return "Earnings date unknown: the earnings calendar was not available, so event risk cannot be assessed."
    if risk["status"] == "none":
        return "The provider's earnings calendar lists no upcoming earnings date."
    when = f"{risk['release_time']:%Y-%m-%d %H:%M} New York time"
    sessions = risk["sessions_until"]
    if risk["within_window"]:
        return (f"{EVENT_RISK_BANNER}: earnings listed for {when}; the first session that can react is "
                f"{risk['reaction_date']:%Y-%m-%d}, {plural(sessions, 'trading day')} ahead.")
    return (f"No known earnings within the next {plural(risk['window_days'], 'trading day')} "
            f"(next listed: {when}, {plural(sessions, 'trading day')} ahead).")


def _event_short(risk: dict[str, Any]) -> str:
    if risk["status"] == "unknown":
        return "Earnings date unknown (no calendar), so event risk cannot be assessed."
    if risk["status"] == "none":
        return "No upcoming earnings date is listed."
    if risk["within_window"]:
        return (f"{EVENT_RISK_BANNER}: earnings {risk['release_time']:%Y-%m-%d %H:%M} New York time, first "
                f"reaction {risk['reaction_date']:%Y-%m-%d} (in {plural(risk['sessions_until'], 'trading day')}).")
    return (f"No known earnings within {plural(risk['window_days'], 'trading day')} (next: "
            f"{risk['release_time']:%Y-%m-%d}, in {plural(risk['sessions_until'], 'trading day')}).")


def _no_level_text(side: str, lookback: int, second: bool = False) -> str:
    if second:
        return f"none (no second distinct level in the last {lookback} sessions)"
    where = "at or above" if side == "resistance" else "at or below"
    return f"none (price is {where} every level of the last {lookback} sessions)"


def _level_text(levels: list, side: str, lookback: int, money: "Money", short: bool = False) -> str:
    if not levels:
        where = "above" if side == "resistance" else "below"
        return f"none {where} the price (last {lookback} sessions)" if short else _no_level_text(side, lookback)
    level = levels[0]
    return f"{money(level.price)} ({pct(level.distance_pct)}, {atrs(level.distance_atr)})"


def _points(value: Any) -> str:
    return "0" if _ok(value) and float(value) == 0 else signed(value, 0)


def _score_breakdown(row: pd.Series) -> str:
    parts = [f"{COMPONENT_NAMES[c]} {_points(row.get(f'points_{c}'))}" for c in COMPONENTS]
    return f"score {signed(row.get('swing_score'), 0)} of ±{MAX_SCORE} (" + ", ".join(parts) + ")"


# ------------------------------------------------------------------ technical_analysis.md
def _snapshot(r: dict[str, Any], money: Money) -> str:
    row, frame, bench = r["latest"], r["frame"], r["benchmark"]
    d = macd_decimals(row["atr"])
    lookback = r["levels"]["lookback"]

    def ema_row(span: int, role: str) -> list[str]:
        col = f"ema{span}"
        text = f"Price {pct(_vs(row, col))} ({atrs(_gap_atr(row, col))})"
        if f"{col}_slope" in row:
            text += (f"; slope {_direction_word(row[f'{col}_slope'], 'rising', 'falling')} "
                     f"({pct(row[f'{col}_slope'], 2)} over 3D)")
        return [f"EMA{span}", money(row[col]), f"{text} - {role}"]

    macd_extra = _macd_details(r)
    context = row["band_context"]
    resistance, support = r["levels"]["resistance"], r["levels"]["support"]

    def level_row(name: str, levels: list, side: str) -> list[str]:
        if not levels:
            return [name, "none", _no_level_text(side, lookback)]
        level = levels[0]
        return [name, money(level.price), f"{pct(level.distance_pct)}, {atrs(level.distance_atr)} ({level.label()})"]

    risk = r["event_risk"]
    if risk["within_window"]:
        risk_reading = f"**Within {plural(risk['window_days'], 'trading day')}**"
    else:
        risk_reading = {"unknown": "Unknown", "none": "None listed"}.get(risk["status"], "Not within window")
    rows = [
        ["Price", money(row["close"]), f"Swing trend (EMA9 / EMA21): {row['trend']}; structure: {r['structure']}"],
        ema_row(9, "immediate momentum"),
        ema_row(21, "primary swing trend"),
        ema_row(50, "secondary confirmation"),
        ema_row(200, "long-term context only"),
        ema_row(250, "~1-year context only"),
        ["RSI6", num(row["rsi6"]),
         f"{row['rsi_zone']}; {signed(_change(frame['rsi6'], 1))} / {signed(_change(frame['rsi6'], 3))} / "
         f"{signed(_change(frame['rsi6'], 5))} pts over 1D / 3D / 5D; " + "; ".join(_rsi_details(r))],
        ["RSI14", num(row["rsi14"]), f"Secondary; {signed(_change(frame['rsi14'], 5))} pts over 5D"],
        ["MACD 12/26/9", row["macd_state"],
         f"DIF {sig(row['macd'], d)} / DEA {sig(row['macd_signal'], d)} / histogram {sig(row['macd_hist'], d)} "
         f"({sig(frame['macd_hist'].diff(1).iloc[-1], d)} 1D, {sig(frame['macd_hist'].diff(3).iloc[-1], d)} 3D)"
         + ("; " + ", ".join(macd_extra) if macd_extra else "")],
        ["Volume", f"{ratio(row['volume_ratio'])} MAVOL20",
         f"{row['volume_level']} on {_day_text(row)}; {volume_text(row['volume'])} vs MAVOL20 "
         f"{volume_text(row['mavol20'])}"],
        ["BOLL (20, 1.8)", row["bb_position"],
         f"%B {num(row['bb_pct_b'], 2)}; bands {money(row['bb_lower'])} / {money(row['bb_mid'])} / "
         f"{money(row['bb_upper'])}; width: {row['band_state'].lower()} ({pct(row['bb_width_change'])} over 5D)"
         + (f"; {context}" if context not in (cls.NO_BAND_CONTEXT, cls.NA) else "")],
        ["ATR(14)", f"{money(row['atr'])} ({num(row['atr_pct'], 2)}%)",
         f"{row['volatility']} volatility ({ordinal(row['atr_pct_percentile'])} percentile of the past 252 sessions)"
         if _ok(row["atr_pct_percentile"]) else "Volatility percentile needs 126+ previous sessions"],
        ["Relative Strength 5D", f"{pct(row['rel_5d'], 2)} vs {bench}" if bench else "n/a",
         f"{row['relative_strength']} (2D {pct(row['rel_2d'])}, 7D {pct(row['rel_7d'])}, 20D {pct(row['rel_20d'])})"
         if bench else "No benchmark data"],
        level_row("Nearest Resistance", resistance, "resistance"),
        level_row("Nearest Support", support, "support"),
        ["Event risk", risk_reading, _event_short(risk)],
        ["Swing Condition", f"**{r['condition']}**", _score_breakdown(row) if r["condition"] != cls.NA
         else "not available (insufficient history)"],
    ]
    return "## Snapshot\n\n" + table(["Metric", "Reading", "Interpretation"], rows, "lrl")


def _trend_section(r: dict[str, Any]) -> str:
    row = r["latest"]
    events = _ema_events(r)
    sentences = [
        f"The swing trend (EMA9 / EMA21) is **{row['trend']}**: the close is {pct(_vs(row, 'ema9'))} vs EMA9 and "
        f"{pct(_vs(row, 'ema21'))} vs EMA21, and EMA9 is {_side(row['ema9'], row['ema21'])} EMA21 "
        f"({pct(row['ema9'] / row['ema21'] - 1 if _ok(row['ema21']) else float('nan'))}).",
        f"Over the last 3 sessions EMA9 changed {pct(row['ema9_slope'], 2)} and EMA21 {pct(row['ema21_slope'], 2)}"
        + (f"; {', '.join(events)}" if events else "") + ".",
        f"EMA50 (secondary confirmation): the close is {pct(_vs(row, 'ema50'))} vs EMA50, which is "
        f"{_direction_word(row['ema50_slope'], 'rising', 'falling')} ({pct(row['ema50_slope'], 2)} over 3 sessions).",
    ]
    if _ok(row["ema250"]):
        sentences.append(f"Context only: the close is {pct(_vs(row, 'ema200'))} vs EMA200 and "
                         f"{pct(_vs(row, 'ema250'))} vs EMA250; ADX(14) is {num(row['adx'])} ({row['adx_state']}).")
    else:
        sentences.append(f"Context only: EMA200 / EMA250 need 200 / 250 sessions of data (close vs EMA200: "
                         f"{pct(_vs(row, 'ema200'))}); ADX(14) is {num(row['adx'])} ({row['adx_state']}).")
    return "## 1. Short-Term Trend\n\n" + " ".join(sentences)


def _momentum_section(r: dict[str, Any]) -> str:
    row, frame = r["latest"], r["frame"]
    rsi6 = frame["rsi6"]
    d = macd_decimals(row["atr"])
    hist = frame["macd_hist"]
    macd_extra = _macd_details(r)
    sentences = [
        f"RSI6 is {num(row['rsi6'])} ({row['rsi_zone']}), {signed(_change(rsi6, 1))} points over 1 day, "
        f"{signed(_change(rsi6, 3))} over 3 and {signed(_change(rsi6, 5))} over 5; RSI14 is {num(row['rsi14'])} "
        f"({signed(_change(frame['rsi14'], 5))} over 5 days); " + "; ".join(_rsi_details(r)) + ".",
        f"MACD 12/26/9 is **{row['macd_state']}**: DIF {sig(row['macd'], d)} vs DEA {sig(row['macd_signal'], d)}, "
        f"histogram {sig(row['macd_hist'], d)} ({sig(hist.diff(1).iloc[-1], d)} over 1 day, "
        f"{sig(hist.diff(3).iloc[-1], d)} over 3 days)" + (f"; {', '.join(macd_extra)}" if macd_extra else "") + ".",
        f"Returns: 2D {pct(row['return_2d'])}, 3D {pct(row['return_3d'])}, 5D {pct(row['return_5d'])}, "
        f"7D {pct(row['return_7d'])} (10D {pct(row['return_10d'])}, 20D {pct(row['return_20d'])}); "
        f"short-term momentum is **{row['momentum']}**.",
    ]
    volume = f"volume {ratio(row['volume_ratio'])} MAVOL20, {_day_short(row)}"
    if _ok(row["rsi6"]) and row["rsi6"] >= 70:
        if row["trend"] in cls.UP_TRENDS:
            sentences.append(f"RSI6 is high inside a rising swing trend (structure: {r['structure']}; {volume}): "
                             "strong momentum that can persist, not a reversal signal by itself.")
        else:
            sentences.append(f"RSI6 is high without a confirmed swing uptrend ({volume}): a sharp short-term move "
                             "rather than an established trend.")
    elif _ok(row["rsi6"]) and row["rsi6"] < 30:
        if row["trend"] in cls.DOWN_TRENDS:
            sentences.append(f"RSI6 is oversold inside a falling swing trend (structure: {r['structure']}; "
                             f"{volume}): weak momentum that can persist, not a rebound signal by itself.")
        else:
            sentences.append(f"RSI6 is oversold without a confirmed swing downtrend ({volume}): a sharp short-term "
                             "drop rather than an established trend.")
    return "## 2. Momentum\n\n" + " ".join(sentences)


def _volume_section(r: dict[str, Any]) -> str:
    row = r["latest"]
    move = _direction_word(row["return_5d"])
    obv_against = (_ok(row["obv_slope10"]) and _ok(row["move_5d_atr"]) and move != "flat"
                   and np.sign(row["obv_slope10"]) == -np.sign(row["return_5d"])
                   and abs(row["obv_slope10"]) >= 0.1 and abs(row["move_5d_atr"]) >= 1)
    sentences = [
        f"Today's volume is {volume_text(row['volume'])}, {ratio(row['volume_ratio'])} MAVOL20 "
        f"({volume_text(row['mavol20'])}, the 20-session average including today), on {_day_text(row)}.",
        f"Participation: **{row['volume_level']}** - {_volume_verdict(row)}.",
        f"Secondary: the 5-day average is {ratio(row['volume_5d_vs_mavol20'])} MAVOL20, and OBV is "
        f"{trend_word(row['obv_slope10'])} over 10 sessions ({signed(row['obv_slope10'], 2)}× the average daily "
        f"volume per session)" + (f", against the 5-day {move} move (divergence)" if obv_against else "") + ".",
    ]
    return "## 3. Volume & Participation\n\n" + " ".join(sentences)


def _volatility_section(r: dict[str, Any], money: Money) -> str:
    row = r["latest"]
    if _ok(row["atr_pct_percentile"]):
        first = (f"ATR(14) is {money(row['atr'])} ({num(row['atr_pct'], 2)}% of price), the "
                 f"{ordinal(row['atr_pct_percentile'])} percentile of the previous 252 sessions: "
                 f"**{row['volatility']}** volatility.")
    else:
        first = (f"ATR(14) is {money(row['atr'])} ({num(row['atr_pct'], 2)}% of price); its percentile needs at "
                 "least 126 previous sessions.")
    band = {cls.EXPANSION_AFTER_SQUEEZE: "expanding after a volatility squeeze", cls.BAND_EXPANSION: "expanding",
            cls.VOLATILITY_SQUEEZE: "in a volatility squeeze (narrowest 20% of the past 126 sessions)",
            cls.BAND_CONTRACTION: "contracting", cls.BAND_STABLE: "stable"}.get(row["band_state"], "n/a")
    context = row["band_context"]
    sentences = [
        first,
        f"Realised volatility (annualised) is {pct_plain(row['rv20'])} over 20 days and "
        f"{pct_plain(row['rv60'])} over 60 days.",
        f"BOLL (20, 1.8): price is {BAND_POSITION_TEXT.get(row['bb_position'], 'n/a')} (%B {num(row['bb_pct_b'], 2)}; "
        f"lower {money(row['bb_lower'])}, middle {money(row['bb_mid'])}, upper {money(row['bb_upper'])}) and band "
        f"width is {band} ({pct(row['bb_width_change'])} over 5 sessions)"
        + (f"; {context[0].lower() + context[1:]}" if context not in (cls.NO_BAND_CONTEXT, cls.NA) else "") + ".",
        "The ranges below are volatility references around the current price, not price targets.",
    ]
    ranges = []
    for multiple in (1.0, 1.5, 2.0):
        if _ok(row["atr"]):
            ranges.append([f"±{multiple:g} ATR", money(row["close"] - multiple * row["atr"]),
                           money(row["close"] + multiple * row["atr"])])
    text = "## 4. Volatility & Swing Risk\n\n" + " ".join(sentences)
    if ranges:
        text += "\n\n" + table(["Range", "Low", "High"], ranges, "lrr")
    return text


def _relative_section(r: dict[str, Any]) -> str:
    row, bench = r["latest"], r["benchmark"]
    if not bench:
        return "## 5. Relative Strength\n\nNo benchmark data was available, so relative strength is not calculated."
    sentences = [
        f"Versus {bench}: 2D {pct(row['rel_2d'])}, 5D {pct(row['rel_5d'])}, 7D {pct(row['rel_7d'])} and "
        f"20D {pct(row['rel_20d'])} (stock return minus {bench} return).",
        f"Relative strength is **{row['relative_strength']}**: the 5D and 7D relative returns average "
        f"{signed(row['relative_strength_score'], 2)} standard deviations of the stock's typical relative moves.",
    ]
    short = row["relative_strength_score"]
    if _ok(short) and _ok(row["rel_20d"]) and abs(short) >= 0.5 and np.sign(short) != np.sign(row["rel_20d"]):
        sentences.append(f"The 20D relative return points the other way, so the recent "
                         f"{'out' if short > 0 else 'under'}performance is a short-term shift.")
    return "## 5. Relative Strength\n\n" + " ".join(sentences)


def _structure_section(r: dict[str, Any], money: Money) -> str:
    row, levels = r["latest"], r["levels"]
    lookback = levels["lookback"]

    def level_row(name: str, levels_on_side: list, number: int, side: str) -> list[str]:
        if len(levels_on_side) < number:
            return [name, _no_level_text(side, lookback, second=number == 2), "—", "—"]
        level = levels_on_side[number - 1]
        return [f"{name} ({level.label()})", money(level.price), pct(level.distance_pct), atrs(level.distance_atr)]

    resistance, support = levels["resistance"], levels["support"]
    rows = [
        level_row("Resistance 2", resistance, 2, "resistance"),
        level_row("Resistance 1", resistance, 1, "resistance"),
        ["Current", money(row["close"]), "—", "—"],
        level_row("Support 1", support, 1, "support"),
        level_row("Support 2", support, 2, "support"),
    ]
    text = "## 6. Price Structure\n\n" + table(["Level", "Price", "Distance", "ATR Distance"], rows, "lrrr")
    references = _reference_rows(row, money)
    if references:
        text += "\n\n" + table(["Reference", "Price", "Distance", "ATR Distance", "Note"], references, "lrrrl")
    states = r["states"]
    structure = (f"Detected structure: **{states[0]}**" + (f" (also: {', '.join(states[1:])})" if len(states) > 1
                                                          else "") + ".") if states else \
        "Detected structure: no distinct breakout, pullback, EMA or range state."
    text += "\n\n" + structure
    text += (f"\n\nDistance = reference / price - 1. Levels come from swing highs/lows and 20D/60D extremes of the "
             f"last {lookback} sessions; levels within 0.5 ATR of each other are merged. \"Near\" = within "
             f"{NEAR_ATR:g} ATR (EMA9, EMA21, EMA50, VWAP20) or {NEAR_EXTREME_ATR:g} ATR (20D high / low). VWAP20 "
             "is a secondary reference approximated from daily bars.")
    return text


def _reference_rows(row: pd.Series, money: Money) -> list[list[str]]:
    """20D / 60D extremes and the swing EMAs, as distances from the current price."""
    close, a = row["close"], row["atr"]
    has_atr = _ok(a) and a > 0
    rows: list[list[str]] = []

    def add(name: str, value: Any, note: str = "") -> None:
        if _ok(value):
            rows.append([name, money(value), pct(value / close - 1), atrs((value - close) / a) if has_atr else "n/a",
                         note or "—"])

    high_note = low_note = ""
    if _ok(row.get("prior_high20")) and close > row["prior_high20"]:
        high_note = f"**Close above the previous 20D high ({money(row['prior_high20'])})**"
    elif has_atr and _ok(row.get("high20")) and row["high20"] - close <= NEAR_EXTREME_ATR * a:
        high_note = "**Within 1 ATR**"
    if _ok(row.get("prior_low20")) and close < row["prior_low20"]:
        low_note = f"**Close below the previous 20D low ({money(row['prior_low20'])})**"
    elif has_atr and _ok(row.get("low20")) and close - row["low20"] <= NEAR_EXTREME_ATR * a:
        low_note = "**Within 1 ATR**"
    add("20D high", row.get("high20"), high_note)
    add("20D low", row.get("low20"), low_note)
    add("60D high", row.get("high60"))
    add("60D low", row.get("low60"))
    for column, name in (("ema9", "EMA9"), ("ema21", "EMA21"), ("ema50", "EMA50"), ("vwap20", "VWAP20")):
        value = row.get(column)
        near = has_atr and _ok(value) and abs(close - value) <= NEAR_ATR * a
        add(name, value, "**Near**" if near else "")
    return rows


def _history_section(r: dict[str, Any]) -> str:
    bench = r["benchmark"]
    text = ("## 7. 2–7 Day Historical Context\n\nWhat happened after past days in this stock's own history that "
            "matched today's conditions. N = days with a complete 7-day future; episodes = separate runs of "
            "consecutive days (overlapping windows make N overstate independent evidence). Descriptive only - "
            "no significance test, not a prediction.")
    for group in r["history"]:
        episodes = "" if group.get("baseline") else f" ({plural(group['episodes'], 'episode')})"
        header = f"\n\n**{group['label']}** — **N = {plural(group['n'], 'day')}{episodes}**"
        if group["insufficient"]:
            text += header + "\n\nToo few comparable days (fewer than 5) - no statistics shown."
            continue
        if group["small_sample"]:
            header += " - *small sample: treat as anecdotal*"
        rows = []
        for item in group["rows"]:
            excursion = item["mfe_avg"] is not None
            rows.append([
                f"{item['horizon']}D", pct(item["avg"]), pct(item["median"]),
                f"{item['pct_positive'] * 100:.0f}%" if _ok(item["pct_positive"]) else "n/a",
                pct(item["avg_vs_benchmark"]) if bench else "n/a",
                f"{pct(item['mfe_avg'])} / {pct(item['mfe_median'])}" if excursion else "—",
                f"{pct(item['mae_avg'])} / {pct(item['mae_median'])}" if excursion else "—",
            ])
        text += header + "\n\n" + table(
            ["Horizon", "Avg", "Median", "% positive", f"Avg vs {bench or 'benchmark'}", "MFE avg / median",
             "MAE avg / median"], rows, "lrrrrrr")
    text += ("\n\nMFE = highest High in the next h days / close - 1 (best price reached); MAE = lowest Low / close - 1 "
             "(worst price reached).")
    return text


def swing_summary(r: dict[str, Any], money: Money) -> str:
    """At most 150 words answering the standard swing questions, ending with the condition."""
    row, frame, bench = r["latest"], r["frame"], r["benchmark"]
    d = macd_decimals(row["atr"])
    lookback = r["levels"]["lookback"]
    events = _ema_events(r, short=True)
    ema = (f"EMA9/EMA21: {row['trend'].lower()} - price {_side(row['close'], row['ema9'])} EMA9 "
           f"({pct(_vs(row, 'ema9'))}) and {_side(row['close'], row['ema21'])} EMA21 ({pct(_vs(row, 'ema21'))}), "
           f"EMA9 {_side(row['ema9'], row['ema21'])} EMA21" + (f", {events[0]}" if events else "") + ".")
    rsi3 = _change(frame["rsi6"], 3)
    state = row["macd_state"]
    if state == cls.MACD_IMPROVING and _ok(rsi3) and rsi3 >= 0:
        pace = "building"
    elif state == cls.MACD_WEAKENING and _ok(rsi3) and rsi3 <= 0:
        pace = "fading"
    else:
        pace = "mixed"
    momentum = (f"Short-term momentum is {pace}: RSI6 {num(row['rsi6'], 0)} ({row['rsi_zone'].lower()}, "
                f"{signed(rsi3)} over 3 days) vs RSI14 {num(row['rsi14'], 0)}; MACD 12/26/9 {state.lower()} "
                f"(histogram {sig(frame['macd_hist'].diff(3).iloc[-1], d)} over 3 days).")
    level, direction = row["volume_level"], row.get("day_direction")
    if level in cls.HEAVY_VOLUME:
        verdict = "confirming the up move" if direction == "up" else "selling pressure, not confirmation"
    elif level == cls.WEAK_PARTICIPATION:
        verdict = "light participation"
    else:
        verdict = "no strong confirmation"
    volume = f"Volume {ratio(row['volume_ratio'])} MAVOL20 on {_day_text(row)}: {verdict}."
    context = row["band_context"]
    boll = (f"BOLL(20, 1.8): {BAND_POSITION_TEXT.get(row['bb_position'], 'n/a')} (%B {num(row['bb_pct_b'], 2)})"
            + (f", {BAND_CONTEXT_SHORT[context]}" if context in BAND_CONTEXT_SHORT else "") + ".")
    if row["volatility"] == cls.NA:
        volatility = f"Volatility regime not classified yet (ATR {num(row['atr_pct'], 2)}% of price)."
    else:
        unusual = "unusually high" if row["volatility"] in ("High", "Extreme") else row["volatility"].lower()
        volatility = (f"Volatility is {unusual} (ATR {num(row['atr_pct'], 2)}% of price, "
                      f"{ordinal(row['atr_pct_percentile'])} percentile).")
    resistance, support = r["levels"]["resistance"], r["levels"]["support"]
    levels = (f"Nearest support {_level_text(support, 'support', lookback, money, short=True)}; nearest "
              f"resistance {_level_text(resistance, 'resistance', lookback, money, short=True)}.")
    relative = (f"Vs {bench} over 5 / 7 days: {pct(row['rel_5d'], 2)} / {pct(row['rel_7d'], 2)} "
                f"({row['relative_strength'].lower()})." if bench else "Relative strength: no benchmark data.")
    sentences = [ema, momentum, volume, boll, volatility, levels, relative, _event_short(r["event_risk"])]
    condition = r["condition"] if r["condition"] != cls.NA else "not available (insufficient history)"
    return " ".join(sentences) + f"\n\n**Swing Technical Condition: {condition}**"


def build_technical_markdown(r: dict[str, Any], currency: str | None = "USD") -> str:
    """The full technical_analysis.md text."""
    money = Money(currency)
    bench = f" · benchmark {r['benchmark']}" if r["benchmark"] else ""
    head = (f"# Swing Technical Analysis — {r['ticker']}\n\n"
            f"Data through {r['as_of']:%Y-%m-%d} · daily bars, dividend-adjusted · typical holding period "
            f"2-7 trading days{bench}\n\n"
            "Indicators: EMA 9 / 21 / 50 / 200 / 250 · RSI 6 / 14 · MACD 12 / 26 / 9 · MAVOL20 · BOLL (20, 1.8) · "
            "ATR(14)\n\n"
            "> Descriptive technical state from deterministic rules (README.md, section 8). Not a price prediction, "
            "not a price target and not a trading recommendation.")
    if r["event_risk"]["within_window"]:
        head += f"\n\n> **{_event_sentence(r['event_risk'])}**"
    if r["notes"]:  # data-quality notes go at the top, so the report still ends with the condition line
        head += "\n\n> *Data notes:* " + " ".join(r["notes"])
    parts = [
        head,
        _snapshot(r, money),
        "---",
        _trend_section(r),
        _momentum_section(r),
        _volume_section(r),
        _volatility_section(r, money),
        _relative_section(r),
        _structure_section(r, money),
        _history_section(r),
        "## 8. Swing Summary\n\n" + swing_summary(r, money),
    ]
    return "\n\n".join(parts) + "\n"


# ------------------------------------------------------------------ summary.md section
def build_summary_snapshot(r: dict[str, Any], currency: str | None = "USD") -> str:
    """Compact "## Swing Technical Snapshot" for summary.md."""
    money = Money(currency)
    row, bench = r["latest"], r["benchmark"]
    lookback = r["levels"]["lookback"]
    risk = r["event_risk"]
    if risk["status"] == "unknown":
        event = "Unknown (earnings calendar not available)"
    elif risk["status"] == "none":
        event = "No upcoming earnings date listed"
    elif risk["within_window"]:
        event = (f"**{EVENT_RISK_BANNER}** (earnings {risk['release_time']:%Y-%m-%d}, first reaction in "
                 f"{plural(risk['sessions_until'], 'trading day')})")
    else:
        event = f"None within {plural(risk['window_days'], 'trading day')} (next: {risk['release_time']:%Y-%m-%d})"

    def ema(span: int) -> list[str]:
        return [f"EMA{span}", f"{money(row[f'ema{span}'])} (price {pct(_vs(row, f'ema{span}'))})"]

    rows = [
        ["Price", f"{money(row['close'])} (swing trend: {row['trend']})"],
        ema(9), ema(21), ema(50), ema(200), ema(250),
        ["RSI6", f"{num(row['rsi6'])} ({row['rsi_zone']})"],
        ["RSI14", num(row["rsi14"])],
        ["MACD 12/26/9", row["macd_state"]],
        ["Volume", f"{ratio(row['volume_ratio'])} MAVOL20 ({row['volume_level']}, {_day_short(row)})"],
        ["BOLL (20, 1.8)", f"{row['bb_position']} (%B {num(row['bb_pct_b'], 2)})"],
        ["ATR(14)", f"{num(row['atr_pct'], 2)}% of price ("
                    + (f"{row['volatility']} volatility)" if row["volatility"] != cls.NA else "regime needs 126+ sessions)")],
        ["Relative Strength 5D", f"{pct(row['rel_5d'], 2)} vs {bench} ({row['relative_strength']})" if bench else "n/a"],
        ["Nearest Resistance", _level_text(r["levels"]["resistance"], "resistance", lookback, money)],
        ["Nearest Support", _level_text(r["levels"]["support"], "support", lookback, money)],
        ["Structure", r["structure"]],
        ["Event risk", event],
        ["**Swing Condition**", f"**{r['condition']}**"],
    ]
    return ("## Swing Technical Snapshot\n\n*Descriptive 2-7 trading-day technical state from deterministic rules "
            "(EMA 9/21/50/200/250, RSI 6/14, MACD 12/26/9, MAVOL20, BOLL 20/1.8); not a recommendation.*\n\n"
            + table(["Measure", "Reading"], rows) + "\n\nFull report: [technical_analysis.md](technical_analysis.md)")


def technical_json(r: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable technical snapshot for analysis.json."""
    row = r["latest"]
    keys = ["close", "ema9", "ema21", "ema50", "ema200", "ema250", "ema9_slope", "ema21_slope", "ema50_slope",
            "rsi6", "rsi14", "macd", "macd_signal", "macd_hist", "return_1d", "return_2d", "return_3d", "return_5d",
            "return_7d", "return_10d", "return_20d", "volume", "mavol20", "volume_ratio", "avg_volume_5d",
            "volume_5d_vs_mavol20", "obv_slope10", "obv_slope20", "bb_upper", "bb_mid", "bb_lower", "bb_pct_b",
            "bb_width", "bb_width_change", "bb_width_percentile", "atr", "atr_pct", "atr_pct_percentile", "rv20",
            "rv60", "vwap20", "adx", "high20", "low20", "high60", "low60", "rel_2d", "rel_5d", "rel_7d", "rel_20d",
            "relative_strength_score", "swing_score", *[f"points_{c}" for c in COMPONENTS]]
    labels = ["trend", "ema50_state", "rsi_zone", "macd_state", "macd_hist_trend", "macd_zero_context", "momentum",
              "volume_level", "day_direction", "vwap_position", "bb_position", "band_state", "band_context",
              "volatility", "relative_strength", "adx_state", "swing_condition"]

    def level(lv) -> dict[str, Any]:
        return {"price": lv.price, "origin": lv.origin, "label": lv.label(), "distance_pct": lv.distance_pct,
                "distance_atr": lv.distance_atr,
                "merged": [{"origin": o, "price": p} for o, p in zip(lv.also, lv.also_prices)]}

    return {
        "as_of": r["as_of"], "benchmark": r["benchmark"],
        "parameters": {"ema": list(EMA_SPANS), "rsi": list(RSI_PERIODS), "macd": [MACD_FAST, MACD_SLOW, MACD_SIGNAL],
                       "mavol": MAVOL_WINDOW, "boll": {"period": BOLL_WINDOW, "std": BOLL_STD}, "atr": ATR_PERIOD},
        "values": {k: (float(row[k]) if _ok(row.get(k)) else None) for k in keys},
        "classifications": {k: (row.get(k) or None) for k in labels},
        "rsi_details": _rsi_details(r), "macd_details": _macd_details(r), "ema_events": _ema_events(r),
        "structure": r["structure"], "structure_states": r["states"], "proximity": r["proximity"],
        "resistance": [level(lv) for lv in r["levels"]["resistance"]],
        "support": [level(lv) for lv in r["levels"]["support"]],
        "event_risk": {k: v for k, v in r["event_risk"].items()},
        "historical_context": r["history"],
        "swing_condition": r["condition"],
        "notes": r["notes"],
    }

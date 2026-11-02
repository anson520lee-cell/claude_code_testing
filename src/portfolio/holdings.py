"""Holdings: the positions you recorded as bought, re-analysed at every data update (research only - no orders).

For every open journal trade (Portfolio / Holdings page, "Bought" button on a candidate) this module answers, from the
latest stored daily bars and the latest hourly bars of the intraday job:

1. Where the LIVE exit rule stands: the stop in force for the next session (initial stop raised by the trailing stop =
   highest high since entry - k x ATR at the signal), the target (squeeze), the time stop, and whether the rule has
   already been hit (then it says SELL - the rule, not a forecast).
2. The trade's state: day k of the holding period, open profit in R, best (MFE) and worst (MAE) R so far, give-back
   from the best, distance to the stop.
3. What is visible today - daily candle patterns (with volume), weekly trend, today's hourly patterns (e.g. a pin bar
   through yesterday's low that recovered, 打針收回), overbought / oversold readings, trade-state signals - with what
   each one did in the backtest (``holding_study``): the move over the next sessions across all stocks, and the paired
   result of selling at the next open the first time it appeared in trades of this setup. A signal is only called
   actionable if it passed the study's pre-registered gate; everything else is shown as information with its numbers.
4. What similar trades did from here (same setup, same day of the trade, same open-R bucket): average final R, the
   chance of ending below zero, and holding on versus selling now.

The pattern / rule definitions are the study's own functions applied to one row, so the page and the backtest can
never drift apart.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import holding_study as hs
from src.market import calendar
from src.technical import candles

DEFAULT_RULE = {s: dict(v) for s, v in hs.LIVE_EXIT.items() if s != "random"}
DEFAULT_RULE["other"] = {"stop_atr": 1.5, "trail": 3.0, "tgt_r": None}
HORIZON = hs.H


# ====================================================================== study results
def load_study(backtest_dir: Path, default_file: Path) -> dict[str, Any] | None:
    """Newest reports/backtest/<session>/holding_study_compact.json, else the shipped default."""
    try:
        runs = sorted((p for p in backtest_dir.iterdir() if (p / "holding_study_compact.json").exists()), reverse=True) \
            if backtest_dir.exists() else []
    except OSError:
        runs = []
    for f in [*(r / "holding_study_compact.json" for r in runs), default_file]:
        try:
            return json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


# ====================================================================== the plan stored with a journal trade
def trade_plan(trade: dict[str, Any]) -> dict[str, Any]:
    """Exit-rule parameters saved with the trade (plan_json) or the live defaults for its setup."""
    raw = trade.get("plan_json")
    plan: dict[str, Any] = {}
    if isinstance(raw, str) and raw.strip():
        try:
            plan = json.loads(raw)
        except ValueError:
            plan = {}
    elif isinstance(raw, dict):
        plan = dict(raw)
    setup = trade.get("setup") or plan.get("setup") or "other"
    base = DEFAULT_RULE.get(setup, DEFAULT_RULE["other"])
    out = {"setup": setup, "stop_atr": base["stop_atr"], "trail": base["trail"], "tgt_r": base["tgt_r"], "horizon": HORIZON}
    for k_src, k_dst in (("trail_atr", "trail"), ("horizon_days", "horizon"), ("atr", "atr"), ("ref", "ref"),
                         ("signal_date", "signal_date"), ("target", "target"), ("stop_atr", "stop_atr")):
        if plan.get(k_src) not in (None, ""):
            out[k_dst] = plan[k_src]
    if "trail_atr" in plan and plan["trail_atr"] in (None, ""):
        out["trail"] = None
    if setup == "squeeze" and plan.get("target") in (None, "") and trade.get("target"):
        out["target"] = trade["target"]
    return out


# ====================================================================== live exit state
def exit_state(f: pd.DataFrame, trade: dict[str, Any], plan: dict[str, Any], raw: pd.DataFrame | None = None) -> dict[str, Any]:
    """Replay the live exit rule over the bars since the entry date (daily bars, same rules as the backtest).

    ``raw`` = the stored (unadjusted) daily prices: the journal holds real fill prices, so the replay uses real highs and
    lows (the feature frame is dividend-adjusted, which only differs before an ex-dividend date)."""
    entry = float(trade["entry_price"])
    stop0 = float(trade["stop"])
    risk = entry - stop0
    if risk <= 0:
        return {"error": "the stop must be below the entry price for a long position"}
    try:
        start = pd.Timestamp(trade.get("entry_date") or str(trade.get("created_at", ""))[:10])
    except (TypeError, ValueError):
        start = f.index[-1]
    pos = np.flatnonzero(f.index >= start)
    if pos.size == 0:
        return {"days_held": 0, "status": "waiting", "note": "No completed daily bar since the entry date yet.",
                "stop_next": stop0, "entry": entry, "stop0": stop0, "risk": risk, "atr": _atr(f, plan, None)}
    e = int(pos[0])
    a = _atr(f, plan, e)
    trail = plan.get("trail")
    tgt = float(plan["target"]) if plan.get("target") not in (None, "") else (
        (float(plan.get("ref") or entry) + float(plan["tgt_r"]) * (float(plan.get("ref") or entry) - stop0)) if plan.get("tgt_r") else None)
    horizon = int(plan.get("horizon") or HORIZON)
    px = raw.reindex(f.index) if raw is not None and not raw.empty else f
    o, h, lo, c = (px[k].fillna(f[k]).to_numpy(float) for k in ("open", "high", "low", "close"))
    hi_prev = entry
    status, exit_date, exit_px = "open", None, None
    k = 0
    for k, i in enumerate(range(e, len(f)), start=1):
        st = stop0 if trail is None else max(stop0, hi_prev - float(trail) * a)
        if lo[i] <= st:
            status, exit_date, exit_px = "stop", f.index[i], min(o[i], st)
            break
        if tgt is not None and h[i] >= tgt:
            status, exit_date, exit_px = "target", f.index[i], max(o[i], tgt) if k > 1 else tgt
            break
        hi_prev = max(hi_prev, h[i])
        if k >= horizon:
            status, exit_date, exit_px = "time", f.index[i], c[i]
            break
    last = len(f) - 1
    seg = slice(e, last + 1)
    hi_run = float(np.max(h[seg]))
    lo_run = float(np.min(lo[seg]))
    stop_next = stop0 if trail is None else max(stop0, max(entry, hi_run) - float(trail) * a)
    days = last - e + 1
    pos_hi = int(np.argmax(h[seg]))
    ts_date = _session_after(f.index[e].date(), horizon - 1)
    out = {"entry": entry, "stop0": stop0, "risk": risk, "atr": a, "trail_atr": trail, "trail_distance": float(trail) * a if trail else None,
           "target": tgt, "horizon": horizon, "days_held": days, "entry_bar": f.index[e].strftime("%Y-%m-%d"),
           "last_bar": f.index[-1].strftime("%Y-%m-%d"), "last_close": float(c[-1]), "status": status,
           "exit_date": exit_date.strftime("%Y-%m-%d") if exit_date is not None else None,
           "exit_px": float(exit_px) if exit_px is not None else None,
           "stop_next": float(stop_next), "trail_on": bool(stop_next > stop0 + 1e-9),
           "open_r": float((c[-1] - entry) / risk), "mfe_r": float((hi_run - entry) / risk), "mae_r": float((lo_run - entry) / risk),
           "locked_r": float((stop_next - entry) / risk), "days_since_high": int(days - 1 - pos_hi),
           "dist_stop_pct": float(c[-1] / stop_next - 1), "dist_stop_atr": float((c[-1] - stop_next) / a) if a > 0 else None,
           "time_stop_date": ts_date.isoformat() if ts_date else None,
           "high_since_entry": hi_run,
           "raise_stop_above": (stop0 + float(trail) * a) if trail else None}
    out["giveback_r"] = out["mfe_r"] - out["open_r"]
    return out


def _atr(f: pd.DataFrame, plan: dict[str, Any], e: int | None) -> float:
    if plan.get("atr") not in (None, ""):
        try:
            v = float(plan["atr"])
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    i = (e - 1) if e is not None and e > 0 else len(f) - 1
    return float(f["atr"].iloc[i])


def _session_after(d: date, n: int) -> date | None:
    try:
        x = d
        for _ in range(max(0, n)):
            x = calendar.next_trading_day(x)
        return x
    except Exception:  # noqa: BLE001
        return None


# ====================================================================== today's signals (the study's own definitions)
def signal_row(f: pd.DataFrame, st: dict[str, Any], hourly: pd.DataFrame | None, today_session: str | None) -> pd.DataFrame:
    """One trade-day row with the columns the study's mask functions read (last completed daily bar + the latest
    session's hourly bars)."""
    o, h, lo, c, v, a = (f[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close", "volume", "atr"))
    X = hs.numeric_frame(f).iloc[[-1]].reset_index(drop=True)
    bars = candles.bar_features(o, h, lo, c, v, a)
    flags = candles.detect(o, h, lo, c, a)
    wk = candles.weekly_context(f).iloc[[-1]].reset_index(drop=True)
    row = {"k": [int(st.get("days_held") or 1)], "s_open_r": [st.get("open_r")], "s_mfe_r": [st.get("mfe_r")],
           "s_mae_r": [st.get("mae_r")], "s_days_since_high": [st.get("days_since_high")], "s_giveback_r": [st.get("giveback_r")],
           "s_dist_stop_atr": [st.get("dist_stop_atr")], "s_locked_r": [st.get("locked_r")]}
    for k_, arr in bars.items():
        row[f"b_{k_}"] = [float(arr[-1])]
    for k_ in candles.PATTERN_IDS:
        row[f"p_{k_}"] = [float(flags[k_][-1])]
    df = pd.concat([pd.DataFrame(row), X, wk], axis=1)
    if hourly is not None and not hourly.empty:
        hourly = hourly.sort_index()
        hf = candles.hourly_flags(hourly)
        days = np.asarray(hourly.index.date)
        last_day = days[-1]
        m = days == last_day
        vr = hf["vol_ratio"].to_numpy(float)
        complete = session_complete(hourly.index[-1])
        for k_ in candles.HOURLY_IDS:
            fl = hf[k_].to_numpy(bool) & m
            if not complete and k_ in ("last_hour_strong", "last_hour_weak"):
                fl = fl & False   # the latest bar is not the session's last hour yet
            df[f"hd_{k_}"] = float(fl.any())
            df[f"hd_{k_}@hv"] = float((fl & (vr >= candles.HV)).any())
            df[f"hd_{k_}@lv"] = float((fl & (vr <= candles.LV)).any())
        df.attrs["hourly_session"] = str(last_day)
        df.attrs["hourly_last_bar"] = hourly.index[-1].strftime("%H:%M")
        df.attrs["hourly_complete"] = complete
    return df


def session_complete(last_bar_start: pd.Timestamp) -> bool:
    """True when the bar starting at ``last_bar_start`` (New York) is the session's final 60-minute bar (Yahoo's
    last bar is the half hour before the close: 15:30, or 12:30 on early-close days)."""
    t = last_bar_start.tz_convert(calendar.NEW_YORK) if last_bar_start.tzinfo else last_bar_start
    close = calendar.close_time(t.date())
    return (t.hour * 60 + t.minute) >= close.hour * 60 + close.minute - 30


def todays_signals(row: pd.DataFrame) -> list[str]:
    masks = {}
    masks.update(hs.pattern_masks(row))
    masks.update(hs.classic_masks(row))
    masks.update(hs.state_masks(row))
    masks.update(hs.hourly_masks(row))
    out = []
    for name, m in masks.items():
        if bool(np.asarray(m)[0]):
            out.append(name)
    r = float(row["s_open_r"].iloc[0]) if pd.notna(row["s_open_r"].iloc[0]) else 0.0
    ctx = []
    for name in out:
        if name.startswith(("candle:", "mtf:", "hourly_eod:")):
            if r >= hs.PROFIT_R:
                ctx.append(name + "|in_profit")
            elif r <= hs.LOSS_R and not name.startswith("hourly_eod:"):
                ctx.append(name + "|in_loss")
    return out + ctx


def _label(name: str) -> str:
    kind, _, rest = name.partition(":")
    base, _, ctx = rest.partition("|")
    pid, _, vol = base.partition("@")
    info = candles.PATTERN_INFO.get(pid) or candles.HOURLY_INFO.get(pid)
    nm = info["name"] if info else pid.replace("_", " ")
    if kind == "hourly_eod":
        nm = "Hourly: " + nm
    if kind == "mtf":
        nm += f" ({ctx.replace('_', ' ')})"
        ctx = ""
    if vol == "hv":
        nm += ", volume >= 1.5x"
    elif vol == "lv":
        nm += ", volume <= 0.7x"
    if ctx and kind != "mtf":
        nm += f" [{ctx.replace('_', ' ')}]"
    return nm


def evidence(setup: str, names: list[str], study: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The backtest numbers behind each signal visible today."""
    if not study:
        return [{"rule": n, "name": _label(n), "verdict": "no study"} for n in names]
    s = (study.get("setups") or {}).get(setup) or {}
    studied = bool(s)
    look = s.get("lookup") or {}
    hlook = s.get("hourly_lookup") or {}
    validated = set((s.get("gate") or {}).get("validated") or []) | set((s.get("hourly_gate") or {}).get("validated") or [])
    up = study.get("universe_patterns") or {}
    uh = study.get("universe_hourly_patterns") or {}
    out = []
    for n in names:
        kind, _, rest = n.partition(":")
        base = rest.split("|")[0] if kind != "mtf" else rest
        pid = base.split("@")[0]
        info = candles.PATTERN_INFO.get(pid) or candles.HOURLY_INFO.get(pid) or {}
        item: dict[str, Any] = {"rule": n, "name": _label(n), "folklore": info.get("folklore"), "kind": kind}
        rec = look.get(n) or hlook.get(n)
        if rec:
            item["in_trade"] = rec
        if kind == "hourly_eod":   # same pattern, intraday mode (sell at the next hour's open)
            intr = hlook.get("hourly:" + rest)
            if intr:
                item["intraday"] = intr
        if (kind == "candle" and "@" not in base and "|" not in rest) or kind == "mtf":
            tight = look.get(n.replace("candle:", "tighten_to_low:").replace("mtf:", "tighten_to_low:mtf:"))
            if tight:
                item["tighten"] = tight
        if kind == "candle" and "|" not in rest and base in up:
            item["all_stocks"] = up[base]
        if kind == "mtf" and base in up:
            item["all_stocks"] = up[base]
        if kind == "hourly_eod" and "|" not in rest and base in uh:
            item["all_stocks_hourly"] = uh[base]
        if not studied:
            item["verdict"] = "setup not studied"
        elif n in validated:
            item["verdict"] = "validated"
        elif rec:
            pooled_t = rec.get("pt") if isinstance(rec, dict) else None
            pooled_d = rec.get("pd") if isinstance(rec, dict) else None
            if pooled_t is not None and pooled_d is not None and abs(pooled_t) >= 2:
                item["verdict"] = "selling helped (not validated)" if pooled_d > 0 else "selling hurt"
            else:
                item["verdict"] = "no reliable effect"
        else:
            item["verdict"] = "too rare to test"
        out.append(item)
    order = {"validated": 0, "selling helped (not validated)": 1, "selling hurt": 2, "no reliable effect": 3, "too rare to test": 4, "setup not studied": 5, "no study": 6}
    return sorted(out, key=lambda x: order.get(x["verdict"], 9))


def similar_trades(setup: str, st: dict[str, Any], study: dict[str, Any] | None) -> dict[str, Any] | None:
    """Day-of-trade x open-R bucket row of the study's day profile (nearest tracked day at or before today)."""
    if not study:
        return None
    rows = ((study.get("setups") or {}).get(setup) or {}).get("day_profile") or []
    k = int(st.get("days_held") or 0)
    r = st.get("open_r")
    if not rows or r is None or k <= 0:
        return None
    days = sorted({x["day"] for x in rows if x["day"] <= k}) or [min(x["day"] for x in rows)]
    d = days[-1]
    for x in rows:
        lo_s, hi_s = x["open_r"].split("..")
        if x["day"] == d and float(lo_s) < r <= float(hi_s):
            return {**x, "matched_day": d}
    return None


# ====================================================================== one holding
def analyze_holding(trade: dict[str, Any], f: pd.DataFrame, study: dict[str, Any] | None,
                    hourly: pd.DataFrame | None = None, quote: dict[str, Any] | None = None,
                    raw: pd.DataFrame | None = None) -> dict[str, Any]:
    plan = trade_plan(trade)
    st = exit_state(f, trade, plan, raw)
    shares = float(trade.get("shares") or 0)
    out: dict[str, Any] = {"id": trade.get("id"), "symbol": trade["symbol"], "setup": plan["setup"], "shares": shares,
                           "entry_price": float(trade["entry_price"]), "entry_date": trade.get("entry_date"),
                           "stop0": float(trade["stop"]), "plan": plan, "exit": st, "notes": trade.get("notes")}
    if "error" in st:
        out["error"] = st["error"]
        return out
    last = float(quote["last"]) if quote and quote.get("last") else st.get("last_close")
    out["last"] = last
    out["live"] = bool(quote and quote.get("last"))
    out["live_slot"] = quote.get("bar_time") if quote else None
    entry, risk = st["entry"], st["risk"]
    out["pnl"] = (last - entry) * shares if last else None
    out["pnl_pct"] = last / entry - 1 if last else None
    out["r_now"] = (last - entry) / risk if last else None
    if quote and quote.get("low") is not None and st.get("status") == "open" and quote["low"] <= st["stop_next"]:
        out["intraday_stop_touch"] = True
    if st.get("days_held", 0) <= 0:
        out["action"] = {"kind": "wait", "text": "Entered today: the first daily bar of the trade is not complete yet."}
        return out
    row = signal_row(f, st, hourly, None)
    names = todays_signals(row)
    out["signals"] = evidence(plan["setup"], names, study)
    out["hourly_session"] = row.attrs.get("hourly_session")
    out["hourly_last_bar"] = row.attrs.get("hourly_last_bar")
    out["hourly_complete"] = row.attrs.get("hourly_complete")
    r0 = row.iloc[0]
    out["today"] = {"date": st["last_bar"], "rsi6": _f(r0.get("f_rsi6")), "rsi14": _f(r0.get("f_rsi14")),
                    "bb_pct_b": _f(r0.get("f_bb_pct_b")), "extension_atr": _f(r0.get("f_extension_atr")),
                    "vol_ratio": _f(r0.get("b_vol_ratio")), "clv": _f(r0.get("b_clv")), "range_atr": _f(r0.get("b_range_atr")),
                    "weekly_up": _f(r0.get("w_up_trend")), "w_rsi14": _f(r0.get("w_rsi14")), "adx": _f(r0.get("f_adx")),
                    "macd_hist_atr": _f(r0.get("d_macd_hist")), "above_ema21": _f(r0.get("d_ema21"))}
    out["similar"] = similar_trades(plan["setup"], st, study)
    validated = [s for s in out["signals"] if s.get("verdict") == "validated"]
    if st["status"] != "open":
        why = {"stop": f"the stop {st['exit_px']:.2f} traded on {st['exit_date']}",
               "target": f"the target {st['exit_px']:.2f} traded on {st['exit_date']}",
               "time": f"the {st['horizon']}-session time stop was reached on {st['exit_date']}"}[st["status"]]
        out["action"] = {"kind": "sell", "text": f"The live exit rule says SELL: {why}. Record the exit when you sell."}
    elif out.get("intraday_stop_touch"):
        out["action"] = {"kind": "sell", "text": f"Today's price has touched the stop {st['stop_next']:.2f} (live quote): the rule sells there."}
    elif validated:
        out["action"] = {"kind": "sell", "text": "A validated exit signal fired today: " + "; ".join(v["name"] for v in validated)
                         + ". The backtest says sell at the next open."}
    else:
        out["action"] = {"kind": "hold", "text": f"Hold: no signal visible today has a proven edge over the live exit. Next stop "
                         f"{st['stop_next']:.2f} ({st['dist_stop_pct'] * 100:+.1f}% from the last close)"
                         + (f", time stop {st['time_stop_date']}" if st.get("time_stop_date") else "") + "."}
    return out


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def hourly_frames(payload: dict[str, Any] | None) -> dict[str, pd.DataFrame]:
    """data/intraday/hourly.json -> {symbol: 60-minute bars (New York index)}."""
    out: dict[str, pd.DataFrame] = {}
    for sym, rows in ((payload or {}).get("bars") or {}).items():
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        idx = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(calendar.NEW_YORK)
        out[sym] = df.drop(columns="ts").set_index(pd.DatetimeIndex(idx)).astype(float)
    return out

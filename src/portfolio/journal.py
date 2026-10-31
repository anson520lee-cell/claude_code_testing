"""Trade journal and price alerts (stored in the terminal database; the user records trades by hand).

Nothing here talks to a broker - the journal is a notebook for reviewing your own decisions, and
alerts are checked against the stored daily bars (after every daily run and when the dashboard loads).

Journal review: R multiple = direction x (exit - entry) / |entry - stop| (the INITIAL stop), P&L =
direction x (exit - entry) x shares. Open trades are marked to the latest stored close, with a flag
when that close is beyond the stop or the target.

Alert kinds (checked on every daily bar after the alert was created, first hit wins):
    above        high >= level            below        low <= level
    close_above  close >= level           close_below  close <= level
"""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Any

import numpy as np

from src.daily.store import utc_now

ALERT_KINDS = {"above": "盤中升穿", "below": "盤中跌穿", "close_above": "收市高於", "close_below": "收市低於"}
JOURNAL_FIELDS = ("symbol", "direction", "setup", "entry_date", "entry_price", "stop", "target", "shares",
                  "exit_date", "exit_price", "status", "notes")


def _rows(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
    cur = conn.execute(sql, args)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _clean(data: dict[str, Any]) -> dict[str, Any]:
    out = {k: data.get(k) for k in JOURNAL_FIELDS if k in data}
    if "symbol" in out and out["symbol"]:
        out["symbol"] = str(out["symbol"]).upper().strip()
    if "direction" in out and out["direction"] not in ("long", "short"):
        raise ValueError("方向必須是 long 或 short")
    for k in ("entry_price", "stop", "target", "shares", "exit_price"):
        if k in out and out[k] not in (None, ""):
            out[k] = float(out[k])
        elif k in out:
            out[k] = None
    if out.get("exit_price") is not None and "status" not in data:
        out["status"] = "closed"
    return out


def add_trade(conn: sqlite3.Connection, data: dict[str, Any]) -> int:
    d = _clean(data)
    for k in ("symbol", "direction", "entry_price", "stop"):
        if d.get(k) in (None, ""):
            raise ValueError(f"缺少欄位：{k}")
    if d["entry_price"] == d["stop"]:
        raise ValueError("止損不能等於進場價")
    d.setdefault("status", "closed" if d.get("exit_price") is not None else "open")
    cols = ["created_at", *d]
    with conn:
        cur = conn.execute(f"INSERT INTO trade_journal ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                           (utc_now(), *d.values()))
    return int(cur.lastrowid)


def update_trade(conn: sqlite3.Connection, trade_id: int, data: dict[str, Any]) -> None:
    d = _clean(data)
    if not d:
        return
    with conn:
        conn.execute(f"UPDATE trade_journal SET {', '.join(f'{k} = ?' for k in d)} WHERE id = ?", (*d.values(), trade_id))


def delete_trade(conn: sqlite3.Connection, trade_id: int) -> None:
    with conn:
        conn.execute("DELETE FROM trade_journal WHERE id = ?", (trade_id,))


def _last_close(conn: sqlite3.Connection, symbol: str) -> tuple[str, float] | None:
    row = conn.execute("SELECT date, close FROM price_data WHERE ticker = ? ORDER BY date DESC LIMIT 1", (symbol,)).fetchone()
    return (row[0], float(row[1])) if row else None


def journal_review(conn: sqlite3.Connection) -> dict[str, Any]:
    trades = _rows(conn, "SELECT * FROM trade_journal ORDER BY COALESCE(entry_date, created_at) DESC, id DESC")
    closed_r = []
    for t in trades:
        s = 1 if t["direction"] == "long" else -1
        risk = abs(t["entry_price"] - t["stop"])
        if t["status"] == "closed" and t["exit_price"] is not None:
            t["r"] = s * (t["exit_price"] - t["entry_price"]) / risk if risk else None
            t["pnl"] = s * (t["exit_price"] - t["entry_price"]) * (t["shares"] or 0)
            if t["r"] is not None:
                closed_r.append((t["r"], t["setup"] or "其他", t["pnl"]))
        else:
            last = _last_close(conn, t["symbol"])
            if last:
                t["mark_date"], t["mark"] = last
                t["r_open"] = s * (last[1] - t["entry_price"]) / risk if risk else None
                t["pnl_open"] = s * (last[1] - t["entry_price"]) * (t["shares"] or 0)
                t["flag"] = ("收市已穿止損" if s * (last[1] - t["stop"]) <= 0 else
                             "收市已到目標" if t["target"] and s * (last[1] - t["target"]) >= 0 else None)
    stats: dict[str, Any] = {"n_closed": len(closed_r), "n_open": sum(t["status"] != "closed" for t in trades)}
    if closed_r:
        r = np.array([x[0] for x in closed_r])
        wins, losses = r[r > 0], r[r <= 0]
        stats.update({"win_rate": float((r > 0).mean()), "avg_r": float(r.mean()), "total_r": float(r.sum()),
                      "avg_win_r": float(wins.mean()) if wins.size else None, "avg_loss_r": float(losses.mean()) if losses.size else None,
                      "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
                      "total_pnl": float(sum(x[2] for x in closed_r)),
                      "by_setup": [{"setup": k, "n": int(len(v)), "avg_r": float(np.mean(v)), "win_rate": float(np.mean(np.array(v) > 0))}
                                   for k, v in _group(closed_r).items()]})
    return {"trades": trades, "stats": stats}


def _group(items: list[tuple]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r, k, _ in items:
        out.setdefault(k, []).append(r)
    return out


# ---------------------------------------------------------------- alerts
def add_alert(conn: sqlite3.Connection, symbol: str, kind: str, level: float, note: str | None = None) -> int:
    if kind not in ALERT_KINDS:
        raise ValueError(f"提醒類型必須是 {', '.join(ALERT_KINDS)}")
    if not symbol or not np.isfinite(float(level)) or float(level) <= 0:
        raise ValueError("請輸入代號和正數價位")
    with conn:
        cur = conn.execute("INSERT INTO price_alerts (created_at, symbol, kind, level, note) VALUES (?, ?, ?, ?, ?)",
                           (utc_now(), symbol.upper().strip(), kind, float(level), note))
    return int(cur.lastrowid)


def delete_alert(conn: sqlite3.Connection, alert_id: int) -> None:
    with conn:
        conn.execute("DELETE FROM price_alerts WHERE id = ?", (alert_id,))


def rearm_alert(conn: sqlite3.Connection, alert_id: int) -> None:
    with conn:
        conn.execute("UPDATE price_alerts SET active = 1, triggered_session = NULL, triggered_price = NULL, created_at = ? WHERE id = ?",
                     (utc_now(), alert_id))


def _hit(kind: str, level: float, bar: tuple) -> float | None:
    _, o, h, lo, c = bar
    if kind == "above" and h is not None and h >= level:
        return max(level, o) if o is not None else level
    if kind == "below" and lo is not None and lo <= level:
        return min(level, o) if o is not None else level
    if kind == "close_above" and c >= level:
        return c
    if kind == "close_below" and c <= level:
        return c
    return None


def check_alerts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Trigger active alerts against stored bars dated after the alert's creation. Returns newly triggered."""
    fired = []
    for a in _rows(conn, "SELECT * FROM price_alerts WHERE active = 1"):
        since = a["created_at"][:10]
        bars = conn.execute("SELECT date, open, high, low, close FROM price_data WHERE ticker = ? AND date > ? ORDER BY date",
                            (a["symbol"], since)).fetchall()
        for bar in bars:
            px = _hit(a["kind"], a["level"], bar)
            if px is not None:
                with conn:
                    conn.execute("UPDATE price_alerts SET active = 0, triggered_session = ?, triggered_price = ? WHERE id = ?",
                                 (bar[0][:10], px, a["id"]))
                fired.append({**a, "active": 0, "triggered_session": bar[0][:10], "triggered_price": px})
                break
    return fired


def list_alerts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    out = []
    for a in _rows(conn, "SELECT * FROM price_alerts ORDER BY active DESC, COALESCE(triggered_session, '') DESC, id DESC"):
        last = _last_close(conn, a["symbol"])
        a["kind_name"] = ALERT_KINDS.get(a["kind"], a["kind"])
        if last:
            a["last_date"], a["last"] = last
            a["distance"] = a["level"] / last[1] - 1
        out.append(a)
    return out


def recent_triggers(conn: sqlite3.Connection, since: date) -> list[dict[str, Any]]:
    return _rows(conn, "SELECT * FROM price_alerts WHERE triggered_session >= ? ORDER BY triggered_session DESC", (since.isoformat(),))

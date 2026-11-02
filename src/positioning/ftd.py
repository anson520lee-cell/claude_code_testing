"""SEC fails-to-deliver (FTD) data.

Source: https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data - one zip per half month
(``cnsfailsYYYYMMa.zip`` = 1st-15th, ``...b.zip`` = 16th-end), pipe-delimited:
    SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE
Published with a lag of roughly two to four weeks, so the newest figures are always a few weeks old.
The quantity is the AGGREGATE net balance of shares that failed to be delivered on that settlement date
(not new fails that day). Needs SEC_USER_AGENT (SEC fair-access policy); files are cached forever.

What is reported per stock (last ~2 published files, about a month):
    latest / max / average fails, max fails as a share of average daily volume and of the stock's
    float-proxy (not available -> volume only), sessions with fails >= 0.5% of ADV, the trend
    (second half vs first half), and the value of the largest fail (shares x price).
Persistent large FTDs can mean hard-to-borrow shares or settlement stress (sometimes before a squeeze),
but they are noisy and frequently benign (ETF creation / redemption timing). Context only.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date
from typing import Any

import pandas as pd

BASE = "https://www.sec.gov/files/data/fails-deliver-data/cnsfails{ym}{half}.zip"


def candidate_files(today: date, months: int = 3) -> list[tuple[str, str]]:
    """(url, cache name) newest first, for the last ``months`` months."""
    out = []
    y, m = today.year, today.month
    for _ in range(months):
        ym = f"{y}{m:02d}"
        for half in ("b", "a"):
            out.append((BASE.format(ym=ym, half=half), f"cnsfails{ym}{half}.zip"))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return out


def parse_ftd(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = z.namelist()[0]
        text = z.read(name).decode("latin-1")
    frame = pd.read_csv(io.StringIO(text), sep="|", dtype=str, on_bad_lines="skip")
    frame.columns = [c.strip().upper() for c in frame.columns]
    frame = frame.rename(columns={"SETTLEMENT DATE": "date", "SYMBOL": "symbol", "QUANTITY (FAILS)": "fails", "PRICE": "price"})
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="coerce")
    frame["fails"] = pd.to_numeric(frame["fails"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    return frame.dropna(subset=["date", "symbol", "fails"])[["date", "symbol", "fails", "price"]]


def load_ftd(sec: Any, today: date, files: int = 2) -> pd.DataFrame:
    """The newest ``files`` published half-month files (via SecClient.get_file)."""
    frames = []
    for url, name in candidate_files(today):
        content = sec.get_file(url, name)
        if content:
            try:
                frames.append(parse_ftd(content))
            except (zipfile.BadZipFile, ValueError, KeyError):
                continue
        if len(frames) >= files:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "symbol", "fails", "price"])


def analyze_ftd(ftd: pd.DataFrame, symbol: str, volume: pd.Series) -> dict[str, Any]:
    if ftd is None or ftd.empty:
        return {"available": False, "reason": "No SEC FTD data (needs SEC_USER_AGENT, or not yet published)."}
    rows = ftd[ftd["symbol"] == symbol].sort_values("date")
    period = f"{ftd['date'].min():%Y-%m-%d} to {ftd['date'].max():%Y-%m-%d}"
    if rows.empty:
        return {"available": True, "period": period, "n": 0, "note": f"No fails to deliver {period} (or below the SEC listing threshold)."}
    adv = float(volume.loc[:rows["date"].max()].tail(20).mean()) if len(volume) else None
    mx = rows.loc[rows["fails"].idxmax()]
    half = len(rows) // 2
    trend = (rows["fails"].iloc[half:].mean() / rows["fails"].iloc[:half].mean() - 1) if half and rows["fails"].iloc[:half].mean() > 0 else None
    big = int((rows["fails"] >= 0.005 * adv).sum()) if adv else None
    out = {"available": True, "period": period, "n": int(len(rows)), "latest_date": f"{rows['date'].iloc[-1]:%Y-%m-%d}",
           "latest": float(rows["fails"].iloc[-1]), "max": float(mx["fails"]), "max_date": f"{mx['date']:%Y-%m-%d}",
           "avg": float(rows["fails"].mean()), "max_value": float(mx["fails"] * mx["price"]) if pd.notna(mx["price"]) else None,
           "max_vs_adv": float(mx["fails"] / adv) if adv else None, "sessions_over_half_pct_adv": big, "trend": trend,
           "series": [{"date": f"{d:%Y-%m-%d}", "fails": float(v)} for d, v in zip(rows["date"], rows["fails"])]}
    level = out["max_vs_adv"] or 0
    out["note"] = (f"{period}: fails on {len(rows)} settlement days, peak {out['max']:,.0f} shares ({out['max_date']})"
                   + (f", {level * 100:.1f}% of 20-day average volume. " if adv else ". ")
                   + ("High relative to volume - possible borrow stress, worth watching." if level >= 0.1
                      else "Small relative to volume - normal." if level < 0.02 else "Moderate.")
                   + (f" Second half {'up' if trend > 0 else 'down'} {abs(trend) * 100:.0f}% vs the first." if trend is not None and abs(trend) >= 0.25 else ""))
    return out

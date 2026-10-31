"""Build the high-beta / high-volatility stock universe automatically (up to ``max_size`` names).

Steps (nothing is added by hand):
1. Yahoo Finance equity screener: US-listed, market cap >= min_market_cap, price >= min_price,
   3-month average volume >= min_avg_shares (a coarse first pass).
2. Coarse dollar-volume pass: price x 3-month average volume >= min_avg_dollar_volume.
3. One year of daily bars for every survivor; the SAME rules as the daily focus filter are
   computed from those bars: 120-session beta vs SPY, 60-session annualised volatility and
   20-session average dollar volume.
4. Survivors are ranked by 20-day dollar volume (most liquid first) and capped at max_size.
5. Sector / industry from Yahoo company profiles -> the sector ETF used for relative strength
   (semiconductors -> SMH, biotech -> XBI, aerospace & defence -> ITA, regional banks -> KRE).

The result is written to config/universe_auto.yaml and merged with the hand-picked core list in
config/terminal.yaml (core first). The daily focus filter still runs on every name every day.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

AUTO_PATH = Path(__file__).resolve().parents[2] / "config" / "universe_auto.yaml"
SECTOR_ETF = {
    "Technology": "XLK", "Communication Services": "XLC", "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Energy": "XLE", "Financial Services": "XLF", "Healthcare": "XLV", "Industrials": "XLI",
    "Basic Materials": "XLB", "Real Estate": "XLRE", "Utilities": "XLU",
}


def sector_etf(sector: str | None, industry: str | None) -> str:
    ind = (industry or "").lower()
    if "semiconductor" in ind:
        return "SMH"
    if "biotechnology" in ind:
        return "XBI"
    if "aerospace" in ind or "defense" in ind:
        return "ITA"
    if "banks - regional" in ind:
        return "KRE"
    return SECTOR_ETF.get(sector or "", "XLK")


def screen_candidates(cfg: dict[str, Any], screener: Callable | None = None) -> list[dict[str, Any]]:
    """Paginate the Yahoo equity screener. Returns quote dicts (symbol, marketCap, price, avg volume)."""
    if screener is None:
        import yfinance as yf
        from yfinance import EquityQuery as Q

        query = Q("and", [Q("eq", ["region", "us"]), Q("gte", ["intradaymarketcap", cfg["min_market_cap"]]),
                          Q("gte", ["avgdailyvol3m", cfg.get("min_avg_shares", 500_000)]),
                          Q("gte", ["intradayprice", cfg.get("min_price", 5)])])

        def screener(offset: int) -> dict:
            return yf.screen(query, sortField="intradaymarketcap", sortAsc=False, size=250, offset=offset)
    out, offset = [], 0
    while True:
        page = screener(offset)
        quotes = page.get("quotes", []) or []
        out.extend(quotes)
        offset += len(quotes)
        if not quotes or offset >= int(page.get("total", 0) or 0) or offset >= 5000:
            break
    seen, unique = set(), []
    for q in out:
        sym = str(q.get("symbol") or "")
        if sym and sym not in seen and q.get("quoteType", "EQUITY") == "EQUITY" and "." not in sym:
            seen.add(sym)
            unique.append(q)
    return unique


def measure(frames: dict[str, pd.DataFrame], spy: pd.Series) -> dict[str, dict[str, float]]:
    """beta120 vs SPY, hv60 (annualised) and 20-day dollar volume from daily bars - same as the daily filter."""
    spy_ret = spy.pct_change()
    out = {}
    for sym, f in frames.items():
        close = f["adj_close"].where(f["adj_close"].notna(), f["close"]).astype(float)
        ret = close.pct_change()
        joined = pd.concat([ret, spy_ret.reindex(ret.index)], axis=1).dropna().tail(120)
        beta = float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / joined.iloc[:, 1].var()) if len(joined) >= 100 else float("nan")
        hv60 = float(ret.tail(60).std(ddof=1) * math.sqrt(252)) if ret.notna().sum() >= 60 else float("nan")
        dv20 = float((f["close"] * f["volume"]).tail(20).mean())
        out[sym] = {"beta": beta, "hv60": hv60, "dollar_volume20": dv20, "close": float(f["close"].iloc[-1]),
                    "sessions": int(len(f))}
    return out


def build_universe(settings: dict[str, Any], downloader: Any = None, screener: Callable | None = None,
                   profile: Callable[[str], dict] | None = None, progress: Callable[[str], None] | None = None,
                   path: Path = AUTO_PATH) -> dict[str, Any]:
    say = progress or (lambda m: logger.info(m))
    uf, liq = settings["universe_filter"], settings["liquidity"]
    cfg = {**(settings.get("universe_builder") or {}), "min_market_cap": uf["min_market_cap"],
           "min_price": liq["min_price"]}
    max_size = int(cfg.get("max_size", 500))
    min_dv = float(liq["min_avg_dollar_volume"])

    say("篩選器：取得全美股市值及成交量…")
    quotes = screen_candidates(cfg, screener)
    coarse = [q for q in quotes if (q.get("regularMarketPrice") or 0) * (q.get("averageDailyVolume3Month") or 0) >= min_dv * 0.35]
    caps = {q["symbol"]: q.get("marketCap") for q in coarse}
    say(f"市值 ≥ ${uf['min_market_cap'] / 1e9:.0f}B 的美股 {len(quotes)} 隻，成交額初步合格 {len(coarse)} 隻；下載一年日線計算 Beta 及波動率…")

    if downloader is None:
        from src.data.market_data import YahooBatchDownloader
        downloader = YahooBatchDownloader()
    symbols = [q["symbol"] for q in coarse]
    frames = downloader.download(["SPY", *symbols], period="1y")
    spy = frames.pop("SPY")["adj_close"]
    stats = measure(frames, spy)
    passed = []
    for sym, st in stats.items():
        ok = (np.isfinite(st["beta"]) and st["beta"] >= uf["min_beta"] and np.isfinite(st["hv60"])
              and st["hv60"] >= uf["min_hv60"] and st["dollar_volume20"] >= min_dv and st["sessions"] >= int(settings["data"].get("min_history_days", 260)))
        if ok:
            passed.append(sym)
    passed.sort(key=lambda s: -stats[s]["dollar_volume20"])
    grid = {}
    for b in (1.3, 1.2, 1.1, 1.0):
        for hv in (0.35, 0.30, 0.25):
            for dv in (min_dv, min_dv / 2):
                n = sum(1 for st in stats.values() if np.isfinite(st["beta"]) and st["beta"] >= b and np.isfinite(st["hv60"])
                        and st["hv60"] >= hv and st["dollar_volume20"] >= dv and st["sessions"] >= 150)
                grid[f"beta>={b} hv60>={int(hv * 100)}% dv>={int(dv / 1e6)}M"] = n
    core = list(settings.get("core_universe") or settings["universe"])
    room = max(0, max_size - len(core))
    chosen = [s for s in passed if s not in core][:room]
    say(f"通過 Beta ≥ {uf['min_beta']}、波動率 ≥ {uf['min_hv60'] * 100:.0f}%、成交額 ≥ ${min_dv / 1e6:.0f}M 的有 {len(passed)} 隻；"
        f"加入 {len(chosen)} 隻（核心名單 {len(core)} 隻，上限 {max_size}）。正在讀取行業分類…")

    if profile is None:
        import yfinance as yf

        def profile(sym: str) -> dict:
            try:
                info = yf.Ticker(sym).get_info()
                return {"sector": info.get("sector"), "industry": info.get("industry"), "name": info.get("longName")}
            except Exception:  # noqa: BLE001
                return {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        profiles = dict(zip(chosen, pool.map(profile, chosen)))
    symbols_out = {s: sector_etf(profiles[s].get("sector"), profiles[s].get("industry")) for s in chosen}
    doc = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_for": date.today().isoformat(),
        "rules": {"min_market_cap": uf["min_market_cap"], "min_beta": uf["min_beta"], "min_hv60": uf["min_hv60"],
                  "min_avg_dollar_volume": min_dv, "max_size": max_size, "ranked_by": "20-day dollar volume"},
        "screened": len(quotes), "coarse_liquidity": len(coarse), "passed": len(passed), "threshold_grid": grid,
        "symbols": symbols_out,
        "details": {s: {"name": profiles[s].get("name"), "sector": profiles[s].get("sector"),
                        "industry": profiles[s].get("industry"), "market_cap": caps.get(s),
                        **{k: round(v, 4) for k, v in stats[s].items() if k in ("beta", "hv60")},
                        "dollar_volume20": round(stats[s]["dollar_volume20"])} for s in chosen},
    }
    path.write_text("# Generated by `python daily_run.py --rebuild-universe` - do not edit by hand.\n"
                    + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    say(f"已寫入 {path.name}：{len(chosen)} 隻新增股票。")
    return doc


def load_auto(path: Path = AUTO_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or None
    except yaml.YAMLError:
        return None


def is_stale(doc: dict[str, Any] | None, today: date, refresh_days: int) -> bool:
    if not doc or not doc.get("generated_for"):
        return True
    return (today - date.fromisoformat(str(doc["generated_for"]))).days > refresh_days

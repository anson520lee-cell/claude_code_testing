"""Options-chain analysis from Yahoo Finance (delayed, end-of-day snapshot).

What the data IS:   per contract - last price, bid/ask, session volume, open interest, Yahoo's implied vol.
What it is NOT:     trade-level prints. Opening vs closing transactions, buyer vs seller, and
                    hedging vs speculation are NOT observable. Open interest is the OCC figure as of
                    the PREVIOUS session (it updates overnight), so today's volume cannot be in it yet.

Outputs (each labelled observed / derived / inferred):
* call / put volume and open interest, put/call ratios                         (observed)
* ATM implied volatility (front expiry >= 3 calendar days, mean of the call and put
  IVs at the two strikes nearest spot, quotes with bid > 0 only)                 (observed, Yahoo IV)
* IV term structure: front vs next expiry ATM IV                                (derived)
* expected move to an expiry = ATM straddle mid price (call mid + put mid)       (derived)
  also as % of spot, and IV-implied 1-sigma move S x IV x sqrt(days / 365)
* IV / HV20 (implied vs realised volatility)                                     (derived)
* strike concentration: strikes with the largest open interest                   (observed)
* max pain for the front expiry (strike that minimises total option payout)      (derived)
* unusual activity: contracts with volume >= 500, volume > 2 x open interest
  and premium >= $250k -> "volume exceeds prior-day OI: at least part is likely
  opening activity" (inferred - direction and intent remain unknown)
* gamma exposure (GEX) estimate under the common ASSUMPTION that dealers are long
  calls and short puts (customers sell calls / buy puts). Black-Scholes gamma from
  Yahoo IV. Treat as a rough positioning sketch, not an observation.  (inferred)
* IV rank / percentile: only once >= 60 daily ATM IV snapshots have been stored in the
  database (this tool records one per run); otherwise "insufficient history (N)".
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_EXPIRIES = 6
MAX_DAYS = 60
MIN_IV_HISTORY = 60


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_gamma(spot: float, strike: np.ndarray, iv: np.ndarray, t_years: float, rate: float = 0.04) -> np.ndarray:
    t = max(t_years, 1 / 365)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(spot / strike) + (rate + 0.5 * iv ** 2) * t) / (iv * math.sqrt(t))
        return np.where(iv > 0, _norm_pdf(d1) / (spot * iv * math.sqrt(t)), 0.0)


def _clean_chain(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    for col in ("strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"):
        f[col] = pd.to_numeric(f.get(col), errors="coerce")
    f["volume"] = f["volume"].fillna(0)
    f["openInterest"] = f["openInterest"].fillna(0)
    f["mid"] = np.where((f["bid"] > 0) & (f["ask"] > 0), (f["bid"] + f["ask"]) / 2, np.nan)
    f["iv_ok"] = (f["impliedVolatility"] > 0.01) & (f["impliedVolatility"] < 5) & (f["bid"] > 0)
    return f


def fetch_chains(ticker_obj: Any, today: date, max_expiries: int = MAX_EXPIRIES) -> list[dict[str, Any]]:
    """[{expiry, days, calls, puts}] for expiries within MAX_DAYS (yfinance Ticker object)."""
    out = []
    for expiry in list(ticker_obj.options or [])[: max_expiries * 2]:
        days = (date.fromisoformat(expiry) - today).days
        if days < 0 or days > MAX_DAYS:
            continue
        chain = ticker_obj.option_chain(expiry)
        out.append({"expiry": expiry, "days": days, "calls": _clean_chain(chain.calls), "puts": _clean_chain(chain.puts)})
        if len(out) >= max_expiries:
            break
    return out


def _atm_iv(chain: dict[str, Any], spot: float) -> float | None:
    ivs = []
    for side in ("calls", "puts"):
        f = chain[side]
        f = f[f["iv_ok"]]
        if f.empty:
            continue
        nearest = f.iloc[(f["strike"] - spot).abs().argsort()[:2]]
        ivs.extend(nearest["impliedVolatility"].tolist())
    return float(np.mean(ivs)) if ivs else None


def _straddle(chain: dict[str, Any], spot: float) -> float | None:
    calls, puts = chain["calls"], chain["puts"]
    strikes = sorted(set(calls["strike"]) & set(puts["strike"]), key=lambda k: abs(k - spot))
    for k in strikes[:3]:
        c = calls.loc[calls["strike"] == k, "mid"].iloc[0]
        p = puts.loc[puts["strike"] == k, "mid"].iloc[0]
        if np.isfinite(c) and np.isfinite(p):
            return float(c + p)
    return None


def max_pain(chain: dict[str, Any]) -> float | None:
    calls, puts = chain["calls"], chain["puts"]
    strikes = np.array(sorted(set(calls["strike"]) | set(puts["strike"])), dtype=float)
    if strikes.size == 0 or (calls["openInterest"].sum() + puts["openInterest"].sum()) == 0:
        return None
    cs, co = calls["strike"].to_numpy(float), calls["openInterest"].to_numpy(float)
    ps, po = puts["strike"].to_numpy(float), puts["openInterest"].to_numpy(float)
    payout = [np.sum(np.clip(s - cs, 0, None) * co) + np.sum(np.clip(ps - s, 0, None) * po) for s in strikes]
    return float(strikes[int(np.argmin(payout))])


def analyze_chains(chains: list[dict[str, Any]], spot: float, hv20: float | None, horizon_days: int = 7,
                   iv_history: pd.Series | None = None, as_of: date | None = None) -> dict[str, Any]:
    if not chains:
        return {"available": False, "reason": "沒有上市期權（或無法下載期權鏈）。"}
    calls = pd.concat([c["calls"].assign(expiry=c["expiry"], days=c["days"]) for c in chains])
    puts = pd.concat([c["puts"].assign(expiry=c["expiry"], days=c["days"]) for c in chains])
    out: dict[str, Any] = {"available": True, "source": "Yahoo Finance 期權鏈（延遲，收市數據）",
                           "spot": spot, "expiries": [c["expiry"] for c in chains]}
    cv, pv = float(calls["volume"].sum()), float(puts["volume"].sum())
    coi, poi = float(calls["openInterest"].sum()), float(puts["openInterest"].sum())
    out.update({"call_volume": cv, "put_volume": pv, "call_oi": coi, "put_oi": poi,
                "pcr_volume": pv / cv if cv else None, "pcr_oi": poi / coi if coi else None,
                "volume_to_oi": (cv + pv) / (coi + poi) if (coi + poi) else None})
    front = next((c for c in chains if c["days"] >= 3), chains[0])
    nxt = next((c for c in chains if c["days"] > front["days"] + 5), None)
    out["atm_iv"] = _atm_iv(front, spot)
    out["atm_iv_expiry"] = front["expiry"]
    out["atm_iv_next"] = _atm_iv(nxt, spot) if nxt else None
    out["iv_term_slope"] = (out["atm_iv_next"] - out["atm_iv"]) if out["atm_iv"] and out["atm_iv_next"] else None
    out["iv_hv_ratio"] = out["atm_iv"] / hv20 if out["atm_iv"] and hv20 else None
    target = next((c for c in chains if c["days"] >= horizon_days * 7 / 5), chains[-1])
    straddle = _straddle(target, spot)
    out["expected_move_expiry"] = target["expiry"]
    out["expected_move"] = straddle
    out["expected_move_pct"] = straddle / spot if straddle else None
    iv_t = _atm_iv(target, spot)
    out["iv_move_1sd_pct"] = iv_t * math.sqrt(max(target["days"], 1) / 365) if iv_t else None
    # concentration
    def top(frame: pd.DataFrame, n: int = 5) -> list[dict]:
        g = frame.groupby("strike")["openInterest"].sum().sort_values(ascending=False).head(n)
        return [{"strike": float(k), "open_interest": float(v)} for k, v in g.items() if v > 0]
    out["top_call_oi_strikes"] = top(calls)
    out["top_put_oi_strikes"] = top(puts)
    out["oi_by_expiry"] = [{"expiry": c["expiry"], "days": c["days"],
                            "call_oi": float(c["calls"]["openInterest"].sum()),
                            "put_oi": float(c["puts"]["openInterest"].sum()),
                            "call_volume": float(c["calls"]["volume"].sum()),
                            "put_volume": float(c["puts"]["volume"].sum())} for c in chains]
    out["max_pain_front"] = max_pain(front)
    # unusual activity (inferred)
    unusual = []
    for side, frame in (("call", calls), ("put", puts)):
        price = frame["mid"].fillna(frame["lastPrice"])
        premium = frame["volume"] * price * 100
        mask = (frame["volume"] >= 500) & (frame["volume"] > 2 * frame["openInterest"]) & (premium >= 250_000)
        for _, r in frame[mask].assign(premium=premium[mask]).sort_values("premium", ascending=False).head(5).iterrows():
            unusual.append({"side": side, "expiry": r["expiry"], "strike": float(r["strike"]),
                            "volume": float(r["volume"]), "open_interest": float(r["openInterest"]),
                            "premium": float(r["premium"]), "iv": float(r["impliedVolatility"]),
                            "moneyness": float(r["strike"] / spot - 1)})
    out["unusual"] = sorted(unusual, key=lambda u: -u["premium"])[:8]
    # GEX estimate (inferred, assumption-laden)
    gex = 0.0
    for c in chains:
        t = max(c["days"], 1) / 365
        for side, sign in (("calls", 1), ("puts", -1)):
            f = c[side][c[side]["iv_ok"]]
            if f.empty:
                continue
            g = bs_gamma(spot, f["strike"].to_numpy(float), f["impliedVolatility"].to_numpy(float), t)
            gex += sign * float(np.sum(g * f["openInterest"].to_numpy(float) * 100 * spot * spot * 0.01))
    out["gex_estimate"] = gex
    out["gex_note"] = ("每 1% 價格變動的美元 Gamma，假設莊家持有認購期權好倉、認沽期權淡倉。"
                       "正數表示莊家對沖傾向壓抑波動，負數表示放大波動。這是推算，不是觀察數據。")
    # IV rank
    if iv_history is not None and len(iv_history.dropna()) >= MIN_IV_HISTORY and out["atm_iv"]:
        h = iv_history.dropna()
        lo, hi = float(h.min()), float(h.max())
        out["iv_rank"] = (out["atm_iv"] - lo) / (hi - lo) * 100 if hi > lo else None
        out["iv_percentile"] = float((h <= out["atm_iv"]).mean() * 100)
        out["iv_history_n"] = int(len(h))
    else:
        out["iv_rank"] = out["iv_percentile"] = None
        out["iv_history_n"] = int(len(iv_history.dropna())) if iv_history is not None else 0
        out["iv_rank_note"] = (f"已儲存的隱含波動率歷史不足（{out['iv_history_n']}/{MIN_IV_HISTORY} 個交易日），"
                               "累積足夠運行次數後會自動顯示 IV Rank。")
    out["interpretation"] = interpret(out)
    out["limitations"] = [
        "未平倉合約是上一交易日的數字，今日成交量尚未計入。",
        "成交量無法顯示交易是開倉還是平倉，也看不出是誰買誰賣。",
        "認購期權成交量高，可能是備兌沽出或對沖，不一定代表看好。",
        "流動性低的行權價，Yahoo 的隱含波動率可能不可靠；只採用有買入報價的合約。",
    ]
    return out


def interpret(o: dict[str, Any]) -> list[str]:
    notes = []
    pcr = o.get("pcr_volume")
    if pcr is not None:
        if pcr >= 1.2:
            notes.append(f"認沽／認購成交比 {pcr:.2f}：今日以認沽為主（可能是對沖，也可能是看淡投機，數據無法分辨）。")
        elif pcr <= 0.5:
            notes.append(f"認沽／認購成交比 {pcr:.2f}：今日以認購為主（可能是投機、備兌沽出或淡倉對沖）。")
    r = o.get("iv_hv_ratio")
    if r is not None:
        if r >= 1.4:
            notes.append(f"隱含波動率是實際波動率的 {r:.1f} 倍：市場預期波幅大於近期（通常有事件將至）。")
        elif r <= 0.8:
            notes.append(f"隱含波動率是實際波動率的 {r:.1f} 倍：相對近期走勢，期權偏便宜。")
    if o.get("iv_term_slope") is not None and o["iv_term_slope"] < -0.03:
        notes.append("隱含波動率期限結構倒掛（近月高於次月）：市場在為近期事件或壓力定價。")
    if o.get("unusual"):
        u = o["unusual"][0]
        side = "認購" if u["side"] == "call" else "認沽"
        notes.append(f"成交量遠超未平倉的最大一筆：{side} {u['strike']:g}（{u['expiry']} 到期），成交 {u['volume']:,.0f}，"
                     f"未平倉 {u['open_interest']:,.0f}，約 ${u['premium'] / 1e6:.1f}M 權利金。很可能包含開倉，但方向未知。")
    return notes


def fetch_and_analyze(symbol: str, spot: float, hv20: float | None, today: date, horizon_days: int,
                      iv_history: pd.Series | None = None, ticker_factory: Any = None) -> dict[str, Any]:
    try:
        if ticker_factory is None:
            import yfinance
            ticker_factory = yfinance.Ticker
        chains = fetch_chains(ticker_factory(symbol), today)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Options for %s unavailable: %s", symbol, exc)
        return {"available": False, "reason": f"無法下載期權數據（{type(exc).__name__}）。"}
    result = analyze_chains(chains, spot, hv20, horizon_days, iv_history, today)
    result["retrieved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return result

"""A broad, self-contained library of technical indicators on daily OHLCV (300+ series).

Neo (2026-11-02) asked for "every indicator, popular or obscure" to be tested together with the ATR trailing stop and
in combinations. The terminal's own features (``src.scanner.features`` etc.) cover the common ones; this module adds
the long tail of the usual charting-site list (moving-average families, oscillators, trend-strength, volatility,
volume, statistical and price-structure measures), each in several lengths.

Rules:
* Causal: the value at bar t uses bars <= t only (``tests`` check it by truncating the input). Nothing is shifted
  forward (Ichimoku spans are compared with today's price as computed today; DPO uses the past SMA).
* Scale-free: prices are expressed as % of close or in ATR(14) units, volumes as ratios, so values compare across
  stocks and years.
* No external dependency (numpy + pandas), so the Windows host needs nothing new.

``compute(frame)`` returns a float32 DataFrame indexed like ``frame`` with columns prefixed ``L_``; ``INFO`` maps each
column to (family, short description).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INFO: dict[str, tuple[str, str]] = {}


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def _rma(x: pd.Series, n: int) -> pd.Series:   # Wilder smoothing
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def _wma(x: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return x.rolling(n, min_periods=n).apply(lambda v: float(np.dot(v, w) / w.sum()), raw=True)


def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up, dn = _rma(d.clip(lower=0), n), _rma((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _std(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).std()


def _slope(x: pd.Series, n: int) -> pd.Series:
    """Least-squares slope over the last n bars."""
    t = np.arange(n, dtype=float)
    t -= t.mean()
    den = (t ** 2).sum()
    return x.rolling(n, min_periods=n).apply(lambda v: float(np.dot(v - v.mean(), t) / den), raw=True)


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    o = frame["open"].astype(float)
    h = frame["high"].astype(float)
    lo = frame["low"].astype(float)
    c = frame["close"].astype(float)
    v = frame["volume"].astype(float).replace(0, np.nan)
    out: dict[str, pd.Series] = {}

    def put(name: str, s: pd.Series, fam: str, desc: str) -> None:
        out["L_" + name] = s
        INFO["L_" + name] = (fam, desc)

    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr = _rma(tr, 14)
    rng = (h - lo).replace(0, np.nan)
    ret = c.pct_change()
    lr = np.log(c).diff()

    # ------------------------------------------------------------------ moving-average families: close vs MA (%)
    for n in (5, 10, 20, 50, 100, 150, 200):
        put(f"sma{n}_dist", c / _sma(c, n) - 1, "ma", f"close vs SMA {n} (%)")
        put(f"ema{n}_dist", c / _ema(c, n) - 1, "ma", f"close vs EMA {n} (%)")
    for n in (10, 20, 50):
        put(f"wma{n}_dist", c / _wma(c, n) - 1, "ma", f"close vs WMA {n}")
    for n in (9, 21, 55):
        hma = _wma(2 * _wma(c, n // 2) - _wma(c, n), int(np.sqrt(n)))
        put(f"hma{n}_dist", c / hma - 1, "ma", f"close vs Hull MA {n}")
        put(f"hma{n}_slope", hma.pct_change(3), "ma", f"Hull MA {n} 3-day change")
    for n in (10, 20, 50):
        e1 = _ema(c, n)
        e2 = _ema(e1, n)
        put(f"dema{n}_dist", c / (2 * e1 - e2) - 1, "ma", f"close vs DEMA {n}")
        put(f"tema{n}_dist", c / (3 * e1 - 3 * e2 + _ema(e2, n)) - 1, "ma", f"close vs TEMA {n}")
    for n in (10, 30):   # Kaufman adaptive MA
        change = (c - c.shift(n)).abs()
        vol = c.diff().abs().rolling(n).sum()
        er = (change / vol.replace(0, np.nan)).fillna(0)
        sc = (er * (2 / 3 - 2 / 31) + 2 / 31) ** 2
        kama = np.full(len(c), np.nan)
        cv, scv = c.to_numpy(), sc.to_numpy()
        for i in range(n, len(c)):
            kama[i] = cv[i] if np.isnan(kama[i - 1]) else kama[i - 1] + scv[i] * (cv[i] - kama[i - 1])
        put(f"kama{n}_dist", c / pd.Series(kama, c.index) - 1, "ma", f"close vs KAMA {n}")
    for n in (10, 20):
        put(f"zlema{n}_dist", c / _ema(c + (c - c.shift((n - 1) // 2)), n) - 1, "ma", f"close vs zero-lag EMA {n}")
        put(f"vwma{n}_dist", c / ((c * v).rolling(n).sum() / v.rolling(n).sum()) - 1, "ma", f"close vs volume-weighted MA {n}")
    for a_, b_ in ((5, 20), (10, 50), (20, 50), (50, 200), (20, 100)):
        put(f"ema{a_}_{b_}_spread", (_ema(c, a_) - _ema(c, b_)) / atr, "ma", f"EMA {a_} - EMA {b_} (ATR)")
        put(f"sma{a_}_{b_}_spread", (_sma(c, a_) - _sma(c, b_)) / atr, "ma", f"SMA {a_} - SMA {b_} (ATR)")
    for n in (10, 20, 50, 200):
        put(f"ema{n}_slope5", _ema(c, n).pct_change(5), "ma", f"EMA {n} 5-day change")
    put("ma_ribbon_order", sum((_ema(c, a_) > _ema(c, b_)).astype(float) for a_, b_ in ((5, 10), (10, 20), (20, 50), (50, 100), (100, 200))),
        "ma", "how many of 5 EMA pairs are in bullish order (0-5)")

    # ------------------------------------------------------------------ oscillators
    for n in (2, 3, 4, 5, 6, 9, 14, 21, 28):
        put(f"rsi{n}", _rsi(c, n), "osc", f"RSI {n}")
    put("rsi14_slope3", _rsi(c, 14).diff(3), "osc", "RSI 14 3-day change")
    for n, d in ((5, 3), (9, 3), (14, 3), (21, 5)):
        ll, hh = lo.rolling(n).min(), h.rolling(n).max()
        k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
        put(f"stoch{n}_k", k, "osc", f"Stochastic %K {n}")
        put(f"stoch{n}_d", _sma(k, d), "osc", f"Stochastic %D {n},{d}")
        put(f"stoch{n}_kd", k - _sma(k, d), "osc", f"Stochastic %K - %D {n}")
    for n in (14, 5):
        r = _rsi(c, n)
        srsi = (r - r.rolling(14).min()) / (r.rolling(14).max() - r.rolling(14).min()).replace(0, np.nan)
        put(f"stochrsi{n}", 100 * _sma(srsi, 3), "osc", f"Stochastic RSI {n}")
    for n in (10, 14, 28):
        put(f"willr{n}", -100 * (h.rolling(n).max() - c) / (h.rolling(n).max() - lo.rolling(n).min()).replace(0, np.nan), "osc", f"Williams %R {n}")
    tp = (h + lo + c) / 3
    for n in (10, 20, 40):
        md = tp.rolling(n).apply(lambda x: float(np.mean(np.abs(x - x.mean()))), raw=True)
        put(f"cci{n}", (tp - _sma(tp, n)) / (0.015 * md.replace(0, np.nan)), "osc", f"CCI {n}")
    for n in (1, 2, 3, 5, 10, 20, 40, 60, 120, 250):
        put(f"roc{n}", c / c.shift(n) - 1, "momentum", f"rate of change {n} days")
    for f_, s_, sg in ((12, 26, 9), (5, 35, 5), (8, 17, 9), (19, 39, 9)):
        m = _ema(c, f_) - _ema(c, s_)
        sig = _ema(m, sg)
        tag = f"{f_}_{s_}"
        put(f"macd{tag}", m / atr, "osc", f"MACD {f_},{s_} line (ATR)")
        put(f"macd{tag}_hist", (m - sig) / atr, "osc", f"MACD {f_},{s_},{sg} histogram (ATR)")
        put(f"macd{tag}_hist_chg", (m - sig).diff() / atr, "osc", f"MACD {f_},{s_} histogram 1-day change (ATR)")
    ppo = 100 * (_ema(c, 12) / _ema(c, 26) - 1)
    put("ppo", ppo, "osc", "PPO 12,26")
    put("ppo_hist", ppo - _ema(ppo, 9), "osc", "PPO histogram")
    for n in (9, 15, 30):
        trix = 100 * _ema(_ema(_ema(c, n), n), n).pct_change()
        put(f"trix{n}", trix, "osc", f"TRIX {n}")
    put("trix15_sig", 100 * _ema(_ema(_ema(c, 15), 15), 15).pct_change() - _sma(100 * _ema(_ema(_ema(c, 15), 15), 15).pct_change(), 9), "osc", "TRIX 15 - signal")
    for r_, s_ in ((25, 13), (13, 7)):
        m = c.diff()
        tsi = 100 * _ema(_ema(m, r_), s_) / _ema(_ema(m.abs(), r_), s_)
        put(f"tsi{r_}_{s_}", tsi, "osc", f"True strength index {r_},{s_}")
    bp = c - pd.concat([lo, pc], axis=1).min(axis=1)
    put("ultosc", 100 * (4 * bp.rolling(7).sum() / tr.rolling(7).sum() + 2 * bp.rolling(14).sum() / tr.rolling(14).sum()
                         + bp.rolling(28).sum() / tr.rolling(28).sum()) / 7, "osc", "Ultimate oscillator 7,14,28")
    mid = (h + lo) / 2
    ao = _sma(mid, 5) - _sma(mid, 34)
    put("ao", ao / atr, "osc", "Awesome oscillator (ATR)")
    put("ao_chg", ao.diff() / atr, "osc", "Awesome oscillator change")
    put("ac", (ao - _sma(ao, 5)) / atr, "osc", "Accelerator oscillator")
    rcma = [_sma(c / c.shift(n) - 1, s) for n, s in ((10, 10), (15, 10), (20, 10), (30, 15))]
    kst = 100 * (rcma[0] + 2 * rcma[1] + 3 * rcma[2] + 4 * rcma[3])
    put("kst", kst, "momentum", "Know Sure Thing")
    put("kst_sig", kst - _sma(kst, 9), "momentum", "KST - signal")
    put("coppock", _wma(100 * (c / c.shift(14) - 1) + 100 * (c / c.shift(11) - 1), 10), "momentum", "Coppock curve (daily lengths)")
    for n in (20, 50):
        put(f"dpo{n}", (c - _sma(c, n).shift(n // 2 + 1)) / atr, "osc", f"Detrended price oscillator {n} (ATR, causal)")
    for n in (9, 14, 20):
        d = c.diff()
        su, sd = d.clip(lower=0).rolling(n).sum(), (-d).clip(lower=0).rolling(n).sum()
        put(f"cmo{n}", 100 * (su - sd) / (su + sd).replace(0, np.nan), "osc", f"Chande momentum oscillator {n}")
    num = _sma((c - o) + 2 * (c.shift(1) - o.shift(1)) + 2 * (c.shift(2) - o.shift(2)) + (c.shift(3) - o.shift(3)), 10)
    den = _sma((h - lo) + 2 * (h.shift(1) - lo.shift(1)) + 2 * (h.shift(2) - lo.shift(2)) + (h.shift(3) - lo.shift(3)), 10)
    put("rvgi", num / den.replace(0, np.nan), "osc", "Relative vigor index 10")
    e13 = _ema(c, 13)
    put("elder_bull", (h - e13) / atr, "osc", "Elder bull power (ATR)")
    put("elder_bear", (lo - e13) / atr, "osc", "Elder bear power (ATR)")
    for n in (9, 20):
        hh, ll = mid.rolling(n).max(), mid.rolling(n).min()
        x = (2 * (mid - ll) / (hh - ll).replace(0, np.nan) - 1).clip(-0.999, 0.999)
        xs = x.ewm(alpha=0.33, adjust=False).mean().clip(-0.999, 0.999)
        put(f"fisher{n}", 0.5 * np.log((1 + xs) / (1 - xs)), "osc", f"Fisher transform {n}")
    # Connors RSI: RSI3 + RSI2 of the up/down streak + percent rank of the 1-day change over 100 days
    sgn = np.sign(c.diff().fillna(0)).to_numpy()
    streak = np.zeros(len(c))
    for i in range(1, len(c)):
        streak[i] = (streak[i - 1] + 1 if streak[i - 1] > 0 else 1) if sgn[i] > 0 else ((streak[i - 1] - 1 if streak[i - 1] < 0 else -1) if sgn[i] < 0 else 0)
    streak_s = pd.Series(streak, c.index)
    prank = ret.rolling(100).apply(lambda x: float((x[:-1] < x[-1]).mean() * 100), raw=True)
    put("connors_rsi", (_rsi(c, 3) + _rsi(streak_s, 2) + prank) / 3, "osc", "Connors RSI")
    put("streak", streak_s, "structure", "consecutive up (+) / down (-) closes")
    gam = 0.7
    cv_ = c.to_numpy()
    L0 = L1 = L2 = L3 = cv_[0]
    lag = np.full(len(c), np.nan)
    for i in range(len(c)):
        p0, p1, p2 = L0, L1, L2
        L0 = (1 - gam) * cv_[i] + gam * p0
        L1 = -gam * L0 + p0 + gam * p1
        L2 = -gam * L1 + p1 + gam * p2
        L3 = -gam * L2 + p2 + gam * L3
        cu = max(L0 - L1, 0) + max(L1 - L2, 0) + max(L2 - L3, 0)
        cd = max(L1 - L0, 0) + max(L2 - L1, 0) + max(L3 - L2, 0)
        lag[i] = cu / (cu + cd) if cu + cd > 0 else np.nan
    lag[:20] = np.nan
    put("laguerre_rsi", pd.Series(lag, c.index), "osc", "Laguerre RSI (gamma 0.7)")
    m = _ema(c, 23) - _ema(c, 50)
    st1 = 100 * (m - m.rolling(10).min()) / (m.rolling(10).max() - m.rolling(10).min()).replace(0, np.nan)
    st1 = st1.ewm(alpha=0.5, adjust=False).mean()
    st2 = 100 * (st1 - st1.rolling(10).min()) / (st1.rolling(10).max() - st1.rolling(10).min()).replace(0, np.nan)
    put("schaff", st2.ewm(alpha=0.5, adjust=False).mean(), "osc", "Schaff trend cycle")
    for n in (8, 14):
        put(f"qstick{n}", _sma(c - o, n) / atr, "osc", f"QStick {n} (ATR)")
    put("bop14", _sma((c - o) / rng, 14), "osc", "Balance of power 14")
    put("psy12", 100 * (c.diff() > 0).astype(float).rolling(12).mean(), "osc", "Psychological line 12")
    put("pgo14", (c - _sma(c, 14)) / _ema(tr, 14), "osc", "Pretty good oscillator 14")

    # ------------------------------------------------------------------ trend strength
    up, dn = h.diff(), -lo.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), c.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), c.index)
    for n in (7, 14, 28):
        a_ = _rma(tr, n)
        pdi, ndi = 100 * _rma(pdm, n) / a_, 100 * _rma(ndm, n) / a_
        dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
        adx = _rma(dx, n)
        put(f"adx{n}", adx, "trend", f"ADX {n}")
        put(f"di_diff{n}", pdi - ndi, "trend", f"+DI - -DI {n}")
        if n == 14:
            put("pdi14", pdi, "trend", "+DI 14")
            put("ndi14", ndi, "trend", "-DI 14")
            put("adx14_slope", adx.diff(3), "trend", "ADX 14 3-day change")
    for n in (14, 25, 50):
        au = 100 * h.rolling(n + 1).apply(lambda x: float(np.argmax(x)), raw=True) / n
        ad = 100 * lo.rolling(n + 1).apply(lambda x: float(np.argmin(x)), raw=True) / n
        put(f"aroon_up{n}", au, "trend", f"Aroon up {n}")
        put(f"aroon_dn{n}", ad, "trend", f"Aroon down {n}")
        put(f"aroon_osc{n}", au - ad, "trend", f"Aroon oscillator {n}")
    for n in (14, 21):
        vmp, vmm = (h - lo.shift(1)).abs().rolling(n).sum(), (lo - h.shift(1)).abs().rolling(n).sum()
        trs = tr.rolling(n).sum()
        put(f"vortex{n}", (vmp - vmm) / trs, "trend", f"Vortex VI+ - VI- {n}")
    for n in (14, 28):
        put(f"chop{n}", 100 * np.log10(tr.rolling(n).sum() / (h.rolling(n).max() - lo.rolling(n).min()).replace(0, np.nan)) / np.log10(n), "trend", f"Choppiness index {n}")
    for n in (10, 20):
        put(f"er{n}", (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum().replace(0, np.nan), "trend", f"Kaufman efficiency ratio {n}")
    for n in (10, 20, 50):
        sl = _slope(c, n)
        put(f"linreg{n}_slope", sl / atr, "trend", f"linear regression slope {n} (ATR per day)")
        r2 = c.rolling(n).corr(pd.Series(np.arange(len(c), dtype=float), c.index)) ** 2
        put(f"linreg{n}_r2", r2, "trend", f"linear regression R squared {n}")
        fit_end = _sma(c, n) + sl * (n - 1) / 2
        put(f"linreg{n}_resid", (c - fit_end) / atr, "trend", f"close vs regression line {n} (ATR)")
    for n, mult in ((10, 3.0), (10, 2.0), (20, 3.0)):
        a_ = _rma(tr, n)
        hl2 = (h + lo) / 2
        ub, lb = (hl2 + mult * a_).to_numpy(), (hl2 - mult * a_).to_numpy()
        cc = c.to_numpy()
        fu, fl = ub.copy(), lb.copy()
        dirn = np.ones(len(c))
        for i in range(1, len(c)):
            if np.isnan(ub[i]):
                continue
            fu[i] = ub[i] if (np.isnan(fu[i - 1]) or ub[i] < fu[i - 1] or cc[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = lb[i] if (np.isnan(fl[i - 1]) or lb[i] > fl[i - 1] or cc[i - 1] < fl[i - 1]) else fl[i - 1]
            dirn[i] = 1 if cc[i] > fu[i - 1] else (-1 if cc[i] < fl[i - 1] else dirn[i - 1])
        line = np.where(dirn > 0, fl, fu)
        tag = f"{n}_{int(mult)}"
        put(f"supertrend{tag}_dist", (c - pd.Series(line, c.index)) / atr, "trend", f"close vs Supertrend {n},{mult} (ATR)")
        put(f"supertrend{tag}_dir", pd.Series(dirn, c.index).where(atr.notna()), "trend", f"Supertrend {n},{mult} direction")
    for step, mx in ((0.02, 0.2), (0.01, 0.1)):
        hh_, ll_ = h.to_numpy(), lo.to_numpy()
        sar = np.full(len(c), np.nan)
        bull, af, ep = True, step, hh_[0]
        sar[0] = ll_[0]
        for i in range(1, len(c)):
            s = sar[i - 1] + af * (ep - sar[i - 1])
            if bull:
                s = min(s, ll_[i - 1], ll_[i - 2] if i > 1 else ll_[i - 1])
                if ll_[i] < s:
                    bull, s, ep, af = False, ep, ll_[i], step
                elif hh_[i] > ep:
                    ep, af = hh_[i], min(af + step, mx)
            else:
                s = max(s, hh_[i - 1], hh_[i - 2] if i > 1 else hh_[i - 1])
                if hh_[i] > s:
                    bull, s, ep, af = True, ep, hh_[i], step
                elif ll_[i] < ep:
                    ep, af = ll_[i], min(af + step, mx)
            sar[i] = s
        sar[:20] = np.nan
        tag = "" if step == 0.02 else "_slow"
        put(f"psar{tag}_dist", (c - pd.Series(sar, c.index)) / atr, "trend", f"close vs parabolic SAR {step},{mx} (ATR)")
    tenkan = (h.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (h.rolling(52).max() + lo.rolling(52).min()) / 2
    put("ichi_tenkan_dist", (c - tenkan) / atr, "trend", "close vs Ichimoku conversion line (ATR)")
    put("ichi_kijun_dist", (c - kijun) / atr, "trend", "close vs Ichimoku base line (ATR)")
    put("ichi_tk_spread", (tenkan - kijun) / atr, "trend", "Ichimoku conversion - base (ATR)")
    put("ichi_cloud_dist", (c - pd.concat([span_a, span_b], axis=1).max(axis=1)) / atr, "trend", "close vs top of today's Ichimoku cloud (ATR)")
    put("ichi_cloud_thick", (span_a - span_b) / atr, "trend", "Ichimoku span A - span B (ATR)")
    put("ichi_chikou", c / c.shift(26) - 1, "trend", "close vs close 26 days ago (lagging span)")
    e9 = _ema(h - lo, 9)
    put("mass_index", (e9 / _ema(e9, 9)).rolling(25).sum(), "volatility", "Mass index 25")

    # ------------------------------------------------------------------ volatility
    for n in (5, 10, 14, 20, 50):
        put(f"atr{n}_pct", _rma(tr, n) / c, "volatility", f"ATR {n} (% of price)")
    put("atr5_50_ratio", _rma(tr, 5) / _rma(tr, 50), "volatility", "ATR 5 / ATR 50")
    put("atr14_pctrank", (atr / c).rolling(250).rank(pct=True), "volatility", "ATR% percentile over 250 days")
    for n, k in ((10, 2.0), (20, 2.0), (20, 1.0), (50, 2.0), (20, 3.0)):
        mb, sd = _sma(c, n), _std(c, n)
        tag = f"{n}_{int(k)}"
        put(f"bb{tag}_pctb", (c - (mb - k * sd)) / (2 * k * sd).replace(0, np.nan), "volatility", f"Bollinger %b {n},{k}")
    bw = 4 * _std(c, 20) / _sma(c, 20)
    put("bb20_width", bw, "volatility", "Bollinger width 20")
    put("bb20_width_pctrank", bw.rolling(120).rank(pct=True), "volatility", "Bollinger width percentile over 120 days")
    for n, k in ((20, 2.0), (20, 1.5), (10, 1.0)):
        mid_ = _ema(c, n)
        put(f"kelt{n}_{k}_pos", (c - mid_) / (k * _rma(tr, n)), "volatility", f"Keltner position {n},{k} (+1 = upper band)")
    sq = (_std(c, 20) * 2) < (1.5 * _rma(tr, 20))
    put("ttm_squeeze_on", sq.astype(float).where(atr.notna()), "volatility", "TTM squeeze on (Bollinger inside Keltner)")
    put("ttm_squeeze_days", sq.astype(int).groupby((~sq).cumsum()).cumsum().astype(float), "volatility", "days the TTM squeeze has been on")
    for n in (10, 20, 55, 100):
        hh, ll = h.rolling(n).max(), lo.rolling(n).min()
        put(f"donch{n}_pos", (c - ll) / (hh - ll).replace(0, np.nan), "structure", f"Donchian position {n} (0 = low, 1 = high)")
    for n in (5, 10, 20, 60, 120):
        put(f"hv{n}", _std(lr, n) * np.sqrt(252), "volatility", f"historical volatility {n} days")
    put("hv10_60_ratio", _std(lr, 10) / _std(lr, 60), "volatility", "volatility 10 / 60 days")
    put("hv20_120_ratio", _std(lr, 20) / _std(lr, 120), "volatility", "volatility 20 / 120 days")
    for n in (14, 28):
        dd = 100 * (c / c.rolling(n).max() - 1)
        put(f"ulcer{n}", np.sqrt((dd ** 2).rolling(n).mean()), "volatility", f"Ulcer index {n}")
    put("chaikin_vol", _ema(h - lo, 10).pct_change(10), "volatility", "Chaikin volatility 10")
    put("parkinson20", np.sqrt((np.log(h / lo) ** 2).rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252), "volatility", "Parkinson volatility 20")
    gk = 0.5 * np.log(h / lo) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    put("garman_klass20", np.sqrt(gk.rolling(20).mean() * 252), "volatility", "Garman-Klass volatility 20")
    put("range_atr", (h - lo) / atr, "volatility", "today's range (ATR)")
    put("range_atr5", _sma(h - lo, 5) / atr, "volatility", "5-day average range (ATR)")
    put("gap_atr", (o - pc) / atr, "structure", "today's gap (ATR)")
    put("body_atr", (c - o) / atr, "structure", "today's body (ATR, + = up)")
    put("clv", (2 * c - h - lo) / rng, "structure", "close location value (-1 low .. +1 high)")
    put("clv5", _sma((2 * c - h - lo) / rng, 5), "structure", "5-day average close location")
    put("upper_wick_atr", (h - pd.concat([o, c], axis=1).max(axis=1)) / atr, "structure", "upper wick (ATR)")
    put("lower_wick_atr", (pd.concat([o, c], axis=1).min(axis=1) - lo) / atr, "structure", "lower wick (ATR)")
    put("nr7", ((h - lo) <= (h - lo).rolling(7).min()).astype(float), "volatility", "narrowest range of 7 days")
    put("wr7", ((h - lo) >= (h - lo).rolling(7).max()).astype(float), "volatility", "widest range of 7 days")
    put("inside_day", ((h <= h.shift(1)) & (lo >= lo.shift(1))).astype(float), "structure", "inside day")
    put("outside_day", ((h > h.shift(1)) & (lo < lo.shift(1))).astype(float), "structure", "outside day")

    # ------------------------------------------------------------------ volume
    for n in (5, 10, 20, 50):
        put(f"vol_ratio{n}", v / v.shift(1).rolling(n).mean(), "volume", f"volume / average of previous {n} days")
    put("vol5_50", _sma(v, 5) / _sma(v, 50), "volume", "5-day / 50-day average volume")
    put("vol_z20", (np.log(v) - np.log(v).shift(1).rolling(20).mean()) / np.log(v).shift(1).rolling(20).std(), "volume", "log volume z-score 20")
    obv = (np.sign(c.diff()).fillna(0) * v.fillna(0)).cumsum()
    av20 = _sma(v, 20)
    for n in (5, 10, 20):
        put(f"obv_slope{n}", (obv - obv.shift(n)) / (av20 * n), "volume", f"OBV {n}-day change / average volume")
    put("obv_vs_ema", (obv - _ema(obv, 20)) / av20, "volume", "OBV vs its EMA 20 (average volumes)")
    mfm = ((c - lo) - (h - c)) / rng
    adl = (mfm.fillna(0) * v.fillna(0)).cumsum()
    put("adl_slope10", (adl - adl.shift(10)) / (av20 * 10), "volume", "accumulation/distribution 10-day change")
    put("chaikin_osc", (_ema(adl, 3) - _ema(adl, 10)) / av20, "volume", "Chaikin oscillator (average volumes)")
    for n in (10, 20, 50):
        put(f"cmf{n}", (mfm * v).rolling(n).sum() / v.rolling(n).sum(), "volume", f"Chaikin money flow {n}")
    rmf = tp * v
    for n in (7, 14, 28):
        pos = rmf.where(tp > tp.shift(1), 0).rolling(n).sum()
        neg = rmf.where(tp < tp.shift(1), 0).rolling(n).sum()
        put(f"mfi{n}", 100 - 100 / (1 + pos / neg.replace(0, np.nan)), "volume", f"Money flow index {n}")
    fi = c.diff() * v
    for n in (2, 13):
        put(f"force{n}", _ema(fi, n) / (atr * av20), "volume", f"Force index {n} (normalised)")
    emv = ((h + lo) / 2 - (h.shift(1) + lo.shift(1)) / 2) * (h - lo) / v
    put("eom14", _sma(emv, 14) / (atr ** 2 / av20), "volume", "Ease of movement 14 (normalised)")
    vpt = (ret.fillna(0) * v.fillna(0)).cumsum()
    put("vpt_slope10", (vpt - vpt.shift(10)) / av20, "volume", "volume-price trend 10-day change")
    nvi = (1 + ret.where(v < v.shift(1), 0).fillna(0)).cumprod()
    pvi = (1 + ret.where(v > v.shift(1), 0).fillna(0)).cumprod()
    put("nvi_vs_ema", nvi / _ema(nvi, 255) - 1, "volume", "negative volume index vs EMA 255")
    put("pvi_vs_ema", pvi / _ema(pvi, 255) - 1, "volume", "positive volume index vs EMA 255")
    for n in (5, 20, 50):
        put(f"vwap{n}_dist", (c - (tp * v).rolling(n).sum() / v.rolling(n).sum()) / atr, "volume", f"close vs {n}-day VWAP (ATR)")
    put("updown_vol10", v.where(c > pc, 0).rolling(10).sum() / v.rolling(10).sum(), "volume", "share of 10-day volume on up days")
    put("pvo", 100 * (_ema(v, 12) / _ema(v, 26) - 1), "volume", "Percentage volume oscillator")
    dm_ = np.where(((h + lo + c) / 3) > ((h + lo + c) / 3).shift(1), 1, -1)
    kvf = pd.Series(dm_, c.index) * v
    kvo = _ema(kvf, 34) - _ema(kvf, 55)
    put("klinger", kvo / av20, "volume", "Klinger oscillator (average volumes)")
    put("klinger_sig", (kvo - _ema(kvo, 13)) / av20, "volume", "Klinger - signal")
    put("vol_price_corr20", c.pct_change().rolling(20).corr(v.pct_change()), "volume", "20-day correlation of returns and volume changes")
    put("dollar_vol_rank", (c * v).rolling(250).rank(pct=True), "volume", "dollar volume percentile over 250 days")

    # ------------------------------------------------------------------ price structure / statistics
    for n in (5, 10, 20, 55, 120, 252):
        put(f"dist_high{n}", (c - h.rolling(n).max()) / atr, "structure", f"close below the {n}-day high (ATR)")
        put(f"dist_low{n}", (c - lo.rolling(n).min()) / atr, "structure", f"close above the {n}-day low (ATR)")
    for n in (20, 55):
        put(f"days_since_high{n}", h.rolling(n).apply(lambda x: float(len(x) - 1 - np.argmax(x)), raw=True), "structure", f"days since the {n}-day high")
        put(f"days_since_low{n}", lo.rolling(n).apply(lambda x: float(len(x) - 1 - np.argmin(x)), raw=True), "structure", f"days since the {n}-day low")
    for n in (10, 20, 50, 100):
        put(f"zscore{n}", (c - _sma(c, n)) / _std(c, n), "stat", f"z-score of close vs {n}-day mean")
    for n in (20, 60):
        put(f"skew{n}", ret.rolling(n).skew(), "stat", f"return skewness {n}")
        put(f"kurt{n}", ret.rolling(n).kurt(), "stat", f"return kurtosis {n}")
        put(f"autocorr{n}", ret.rolling(n).corr(ret.shift(1)), "stat", f"lag-1 autocorrelation of returns {n}")
        put(f"upday_share{n}", (ret > 0).astype(float).rolling(n).mean(), "stat", f"share of up days {n}")
    put("pivot_dist", (c - (h.shift(1) + lo.shift(1) + c.shift(1)) / 3) / atr, "structure", "close vs yesterday's classic pivot (ATR)")
    put("r1_dist", (c - (2 * (h.shift(1) + lo.shift(1) + c.shift(1)) / 3 - lo.shift(1))) / atr, "structure", "close vs pivot R1 (ATR)")
    put("s1_dist", (c - (2 * (h.shift(1) + lo.shift(1) + c.shift(1)) / 3 - h.shift(1))) / atr, "structure", "close vs pivot S1 (ATR)")
    put("hl_range20_pos", (c - lo.rolling(20).min()) / (h.rolling(20).max() - lo.rolling(20).min()).replace(0, np.nan), "structure", "position in 20-day range")
    put("round_num_dist", (c - (c / 10 ** np.floor(np.log10(c))).round() * 10 ** np.floor(np.log10(c))) / atr, "structure", "distance to the nearest round number (ATR)")
    put("hurst_proxy60", np.log(_std(c.diff(10), 60) / _std(c.diff(), 60)) / np.log(10), "stat", "variance-ratio Hurst proxy 60 (0.5 = random walk)")
    put("ret_vs_vol20", ret / _std(ret, 20).shift(1), "stat", "today's return in 20-day standard deviations")
    put("ret5_vs_vol", (c / c.shift(5) - 1) / (_std(ret, 60) * np.sqrt(5)), "stat", "5-day return in standard deviations")

    df = pd.DataFrame(out, index=frame.index)
    return df.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def columns() -> dict[str, tuple[str, str]]:
    """Column -> (family, description); computes once on a synthetic frame if needed."""
    if not INFO:
        idx = pd.bdate_range("2020-01-01", periods=320)
        rng = np.random.default_rng(0)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
        compute(pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6}, index=idx))
    return dict(INFO)

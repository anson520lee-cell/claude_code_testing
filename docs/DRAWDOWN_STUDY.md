# Drawdown-reduction study (2026-11-02, research only)

Question (Neo): the account simulation of the three live setups (10 slots, 1% risk per trade) has a maximum drawdown of about -65%. Which account-level risk control cuts it without giving up the return?

`python drawdown_study.py --prices prices.pkl [--cache data.pkl]` (about 8 minutes on 4 cores). Code: `src/backtest/drawdown.py`, tests `tests/test_drawdown.py`, compact results `config/drawdown_study.default.json`.

## Design
* Same trades as the account illustration of 7k (live exits, 20-session horizon, 10 bps slippage, daily mark-to-market, cash account, 10 slots, 1% risk). A second copy uses a 5 ATR trail for breakout. Only the account layer changes. Median over 16 (24 for the final table) random orders of same-day signals.
* Mechanisms (about 90 single settings): lower risk per trade, fewer slots, total open-risk (heat) cap, per-sector cap, correlation cap; regime filters (SPY vs 50/200d, SPY drawdown, VIX, breadth), each for all setups and for breakout + squeeze only; volatility targeting (SPY vol, VIX, own account vol); account drawdown circuit breakers and an equity-curve trend filter; per-setup treatment of breakout (risk x, slot cap, priority). Regime data are known at the previous close.
* Windows: in-sample 2020-01..2024-06 (the only window used for ranking), hold 2010-2019, oos 2024-07.., all 2010-2026, plus five sub-periods.
* Pass bar for a mechanism: max drawdown <= 80% of the base in IS, hold and oos AND more CAGR than plain risk scaling at the same drawdown in all three (the "frontier" from risk 0.3%..1.5%).

## Result
* Seed noise of one cell is large (drawdown sd 3-7 pp, oos CAGR sd 40 pp), so only big differences mean anything.
* The -65% comes from long stretches with no edge: 2011-2016 and 2021-2022 (breakout R sum negative in 8 of 17 years) and a 5-year underwater period. A risk control cannot create edge; it can only reduce exposure.
* Market-regime filters (SPY 200d / 50d, SPY drawdown, VIX, breadth), volatility targeting, correlation caps and sector caps cut the drawdown by 0-15% at most and lose money in oos. None passes. Account breakers and plain risk reduction cut the drawdown by 40-55% but sit on the frontier (they just scale the return down).
* Total open-risk cap measured against CURRENT equity is the only family that beats the frontier consistently: after losses the same dollar risk is a larger share of the account, so the cap tightens automatically in a drawdown. The terminal's own `portfolio.max_heat` is 4%, but the account simulation (10 x 1% = 10%) never applied it.
* Adaptive heat cap (8% normally, 4% once the account is 10-15% below its peak): the whole 54-cell grid is a plateau (drawdown -30% to -49%, hold CAGR positive in all), the SPY 200d switch adds nothing.

| 2010-26 / 2010-19 / 2020-24H1 / 2024H2-26 (CAGR, max DD) | all | hold | is | oos |
|---|---|---|---|---|
| base (10 slots, 1%) | 12.1%, -68% | -2.2%, -67% | 20.6%, -59% | 63.4%, -35% |
| risk 0.5% | 9.0%, -41% | 1.6%, -41% | 12.5%, -34% | 31.8%, -21% |
| heat cap 4% (terminal's max_heat) | 10.7%, -37% | 6.3%, -37% | 11.4%, -30% | 28.0%, -18% |
| adaptive heat 8% -> 4% (dd 15%) | 11.9%, -44% | 3.2%, -42% | 20.1%, -36% | 52.5%, -25% |
| adaptive heat 8% -> 4% (dd 10%) | 12.4%, -39% | 3.6%, -38% | 17.0%, -34% | 47.7%, -24% |
| SPY buy & hold | 14.0%, -34% | 13.3%, -19% | 13.9%, -34% | 16.6%, -19% |

Nothing here makes the strategy beat SPY (Calmar 0.33 at best against 0.41). The adaptive rule was picked on the in-sample window from 54 settings, so it is a candidate for forward confirmation, not a validated rule. No live rule is changed.

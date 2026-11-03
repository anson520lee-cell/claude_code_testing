# Replication pre-registration (written 2026-11-03 18:50 UTC, before any replication data were loaded)

Universe: current S&P 400 + S&P 600 members whose ticker never appears in the S&P 500 price set used for discovery
(886 tickers). A stock-day is eligible when: unadjusted close >= $5, >= 260 sessions of history, and the 20-session
average dollar volume (unadjusted close x volume) >= $5M. Period 2010-01-01 .. 2026-10-31 (all of it is out of sample:
none of these stocks was used for discovery). Same code, same exits, same costs, same confluence weights (S&P IS).
Percentile features are ranked against all eligible stocks of this universe on the same day.

Statistic: delta of mean R (live exit) between condition = 1 and condition = 0 within the strategy's rows,
month-clustered t (CR1). One-sided in the discovery direction. Pass: t >= 2.50 (Bonferroni 0.05/8, one-sided)
AND excess-over-same-day-random delta t >= 1 in the same direction.

H1 standalone long (random entries, crash exit)      svr5 bottom tercile            -> higher R
H2 standalone long                                    offex bottom tercile           -> higher R
H3 standalone long                                    flow composite bottom tercile  -> higher R
   (flow composite = mean of the same-day svr5 and offex percentiles, then ranked again)
H4 breakout (live)                                    ens5 >= S&P IS threshold (ens_hi) -> higher R
H5 L10 Undercut & Rally (swing42 params)              offex bottom tercile           -> higher R
H6 crash_rebound (live)                               svr5 bottom tercile            -> higher R
H7 breakdown short (live short)                       mom12_1 bottom tercile         -> higher short R
H8 standalone short (random entries, short mirror)    offex top tercile              -> higher short R
Descriptive only (not counted): the same tests split 2010-2017 / 2018-2026, svr5 top for shorts, quintiles.

## Addendum A (written 2026-11-03 18:55 UTC, still before any replication data were loaded). Own family, Bonferroni 0.05/4 one-sided -> t >= 2.24
A1 all short strategies pooled (live shorts + x01..x09 + s28)   ivol60 top tercile -> higher short R
A2 x02 Z-score short: mean excess over same-day random shorts > 0
A3 L26 RSI(2) long: mean excess over same-day random longs > 0
A4 L19 AVWAP pinch long: mean excess over same-day random longs > 0

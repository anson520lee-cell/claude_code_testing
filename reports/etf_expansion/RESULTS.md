# ETF expansion: completed historical screen

1728 of 1728 preregistered trials completed, each transactionally checkpointed.
All four selection subperiods at CAGR >=80%: **0**.
No strategy from this study qualifies for stable 80-90% annual returns.

The frozen pre-2025 winner is trial 332: universe us,
252-session positive momentum, top 1, inverse 60-session
volatility sizing, target vol 30%, exposure target capped at
2.0x, rebalance every 5 sessions.
Signals execute at the following close and first earn returns one session later.

| Period | CAGR | Daily maximum drawdown |
|---|---:|---:|
| 2008-2012 | 8.34% | -31.81% |
| 2013-2017 | 17.93% | -39.13% |
| 2018-2021 | 16.24% | -38.10% |
| 2022-2024 | 44.51% | -21.52% |
| 2008-2024 | 18.78% | -39.13% |
| Reused 2025-2026 | 34.22% | -27.98% |

The 2008-24 minimum rolling three-year CAGR is -5.27%.
Worst calendar-year return, including partial years, is -15.13%.
Peak marked-to-market exposure before the closing risk adjustment was
2.111x; a closing target cap does not prove intraday margin headroom.

One-way trading/slippage costs were 10 bp; the same frozen rule was then checked
at 20 bp (validation.json), without selecting or tuning again. Borrowing uses
the lagged T-bill proxy plus 1%, for actual calendar days. Individual taxes are excluded.

This is a separate expansion, not completion of the original 10026-configuration
30-sleeve study. The original inputs remain incomplete. Prior project research has
already observed 2025-26; it is reused historical validation, never a fresh holdout.
Fixed ETF fund selection has design/survivorship limitations, and neither tradable
intraday margin nor a future return guarantee is established.

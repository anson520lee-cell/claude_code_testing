# ETF allocator expansion: pre-registered protocol (2026-11-04)

This is a NEW study; it is not the original 30-sleeve / 10026-configuration search.
Dataset: research_data/frozen_20261104, exclusive end 2026-11-04. Fixed Yahoo symbols with real adjusted daily bars.
Do not replace or overwrite any old report or trial. No scaling-in rules or live changes.
All historical periods have already been observed in earlier project research; none is an untouched holdout.

Hypothesis: ETF rotation/reversal across different asset classes, inverse-volatility sizing and lagged
execution may produce a more persistent risk-adjusted return than a single equity/crypto sleeve.
Test this mechanism directly before considering aggressive leverage.

Fixed universes (fund selection itself may have survivorship/design bias):
- us: SPY QQQ IWM XLK XLF XLE
- global: SPY QQQ IWM EFA EEM TLT IEF GLD DBC VNQ
- defensive: SPY TLT IEF GLD EFA DBC

Grid (1728 trials, every trial retained):
- momentum: lookback 21/63/126/252 days, positive momentum only, SMA filter off/200,
  top 1/2/3, target vol off/30%/60%, exposure cap 1/2/3, rebalance every 5/21 sessions.
- reversal: lookback 2/5/10 days, negative-return assets only, SMA filter off/200,
  top 1/3, target vol off/30%, exposure cap 1/2/3, rebalance every 1/5 sessions.
- Selected names receive inverse 60-session volatility weights.
- Target vol uses the current chosen basket's covariance from the previous 60 observed returns.
- A signal at close t is executed at close t+1. It first earns returns on t+2.
- One-way transaction/slippage cost is 10 bp of traded notional; margin borrowing is prior observed
  13-week T-bill discount yield plus 1% annual, charged for actual calendar days. Free cash earns
  the observed T-bill proxy. Fund costs are embedded in traded ETF returns. Individual taxes are excluded.
- Daily leverage is monitored but intraday executable margin/liquidity is NOT established by this study.
  Candidates requiring borrowed exposure remain implementation-unverified. Account loss <= -100% is permanent ruin.
- Sources, config list and implementation are fingerprinted; SQLite saves after every trial.

Selection and reporting:
- Primary score: minimum CAGR of 2008-12, 2013-17, 2018-21 and 2022-24.
- Ties: higher aggregate 2008-24 Sharpe, then lower maximum daily marked-to-market drawdown.
- Freeze that one winner before inspecting 2025-26; that period is reused historical validation only.
- Report 2008-24, 2018-24, 2022-24 and 2025-26, worst year, rolling 3-year CAGR,
  daily drawdown from starting capital, peak exposure and ruin. Show funded (cap 1) separately.
- Count how many trials meet 80% CAGR in ALL four selection subperiods; do not substitute a bull-market result.
- A historical screen pass is only a candidate. Forward validation, cost stress, selection-bias checks and
  executable margin checks are required before any stable-return claim.

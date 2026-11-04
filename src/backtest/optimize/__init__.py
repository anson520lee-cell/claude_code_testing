"""Search for the best version of the x80 book without fooling ourselves (Neo, 2026-11-04).

Every choice is made on data before HOLDOUT (2025-01-01); the holdout is scored once, at the end. Modules:
  newsleeves  three sleeves the earlier rounds never had: Connors RSI(2) on point-in-time S&P 500 members,
              G10 FX carry, S&P 500 turn of the month
  pool        the candidate sleeves (x80's 16 + earlier variants + the new three), weekly and daily excess returns
  engine      a fast book: sleeve selection, weighting, covariance, caps, rebalancing bands, leverage bands, costs
  search      thousands of random configurations, each scored only on data before the holdout
  overfit     walk-forward re-optimisation, CSCV probability of backtest overfitting, deflated Sharpe
  final       the one holdout run
"""

HOLDOUT = "2025-01-01"

# Pre-registration (written 2026-11-04 11:19 UTC, while the search was running and before any search result was read)

Holdout: 2025-01-01 .. 2026-11-06. The search code cuts the data at 2025-01-01 before it builds anything.

Score of a configuration: Sharpe (excess of T-bills) of its weekly account returns levered to 80% target vol,
all costs included, 2010-01-01 .. 2024-12-31.

Candidates that go to the holdout (decided now, not after looking):
  A  max-Sharpe       the single configuration with the best score
  B  consensus        for every dimension, the option with the best median score across the random configurations,
                      put together into one configuration
  C  walk-forward     the rule "each January pick the configuration with the best score so far" (and its top-20 average
                      variant); its pick for 2025 is A by construction, so C's evidence is its 2013-24 track record
  X  x80 round 2      the current book (ERC, 104 weeks, 16 sleeves) as the benchmark

Which one is recommended is decided on the pre-holdout evidence (walk-forward record, CSCV probability of backtest
overfitting, deflated Sharpe, the same sign in 2010-17 and 2018-24). The holdout is then run once and reported as it is;
it can veto a candidate (a loss far outside its history) but it is not used to pick the best one.

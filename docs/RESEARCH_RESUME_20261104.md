# Research continuation: 2026-11-04

The user resumed the paused research and explicitly requested preservation of completed work. The goal remains
research into 80-90% annualized returns. **No newly validated strategy meets that goal. Nothing here changes live
trading rules, connects a broker, places orders, merges existing PRs, or deploys the terminal.**

## Authoritative starting point and preserved work

- Handoff revision 7, project `trading-terminal`, task `main`.
- Parent research commit: `2f015a09a95f7d81e568cd594114dbaa91aa3113` on `claude/strategy-optimize` (PR #15).
- All 307 tracked files in that snapshot were recovered. Previously completed notes and results were retained.
- `reports/optimize/PREREG.md` is unchanged. SHA-256:
  `628ea5e4c00ac378bd1b5b64cd3edbbbe915f937bc6b3a6e1a5d7909f26f65d5`.
- The approximately 4,000 trials lost before the original interruption are still lost; no result has been fabricated
  or described as recovered. The intended 10,026-configuration search has NOT been completed.
- Continuation branch: `codex/research-resume-20261104`. Preserve the original `STATUS.md` as the historical pause note;
  this document records the resumed state.

## Implemented and checked

1. **Restartable search:** `src/backtest/optimize/checkpoint.py` commits every completed configuration in SQLite.
   `search.py` skips committed trials on restart, preserves their stable IDs even when workers finish out of order,
   and fingerprints the pre-cutoff inputs, metadata, full configuration list and relevant implementation.
   Changed inputs/code require a new output directory. Workers default to at most four; use two on the shared host.
2. **Bounded batches:** the original positional CLI remains available, with `--workers` and `--max-new` added.
   An interrupted/incomplete search cannot enter `overfit.py` or `final.py`. `progress.json` is marked complete only
   after the final CSV/Parquet exports are published.
3. **Missing-return hindsight removed:** `run_book` no longer reallocates an allocated sleeve's weight after seeing
   that its current week's return is missing. Missing allocated returns fail explicitly and require documented
   closure/delisting treatment.
4. **Drawdown from initial capital:** an initial-period loss now counts in search, overfit and final-period drawdown
   instead of starting the running peak after that loss.
5. **Portable research paths:** `TRADING_RESEARCH_ROOT` selects research inputs. The existing Linux location is retained
   if present; otherwise the default is the checkout's `research_data/`. Core price, Cboe PUT, frontier and leverage
   study input paths use this resolver. Some auxiliary historical download scripts still contain old Linux paths.
6. **Resumable collection:** `scripts/research/collect.py` collects 50 Yahoo instruments plus eight Cboe option-strategy
   index histories. Files are hashed and completed downloads reused; both Yahoo and Cboe histories are cut at the
   explicit exclusive as-of date. It does not obtain historical executable option chains.
7. **Public-price recovery:** `scripts/research/recover_prices.py` reads only the terminal's `price_data` table, refuses
   an active WAL or a changing source, preserves a verified existing snapshot, and never reads account/alert tables.
   `research_data/` and the isolated research environment are excluded from Git.

Validation actually performed on Mac:

- 17 research checks passed: the 11 original optimization checks and six new regression checks, including an
  interrupted/restarted search that proves committed trials are not recomputed and an as-of-date leakage check.
- All 16 cases that failed in the previous Windows run passed in targeted Mac reruns, using UTF-8 and the correct
  project working directory. The previous Windows result was 362 passing / 16 failing, including CP950 decoding
  errors. This is NOT a claim that Windows has been fixed or revalidated. Its 362 passing cases were not rerun.
- The collection CLI help was checked; fresh provider downloads have not succeeded in this continuation.

## Recovered market data

The local terminal cache supplied **361,601 daily OHLCV rows across 350 symbols**, from **2021-11-01 through
2026-11-03**. A hash-verified Parquet snapshot, per-symbol inventory and manifest are under
`research_data/recovered_terminal_20261104/`; the price-only archive is
`research_data/archives/trading_terminal_market_prices_20261104.zip` and has been saved separately for recovery.

This is a present-universe cache with survivorship bias, NOT point-in-time index membership, and NOT a reconstruction
of the original 30-sleeve daily/weekly pool. Do not silently substitute it for that pool or call it untouched validation.

## Research blockers and newly verified issues

- The Windows command channel timed out; the separate data-fetch channel reported that its tunnel client had not
  connected for 300 seconds. Do not wait indefinitely or claim downloads completed. Bound a connectivity probe,
  preserve local state, and retry only missing data when the connection is available.
- Original research inputs were not uploaded with the repository. The full 30-sleeve pool still needs PIT equity,
  FINRA/SEC, Cboe, derivatives/crypto archives and rates/FX inputs described in `docs/X80.md`, `docs/LEVSCORE.md` and
  the preserved optimization `STATUS.md`. No synthetic return series have been substituted for them.
- **New deterministic cap violation:** the original sequential asset-cap and crypto-risk projections do not jointly
  enforce both constraints. With four initially equal weights, identity covariance, three crypto sleeves, a 25%
  asset cap and 35% crypto variance-share cap, the final non-crypto weight is **44.0339%**. The numerical evidence is
  in `reports/optimize/RESUME_AUDIT_20261104.json`. This remains OPEN and must be resolved and regression-tested before
  approving candidates that rely on these risk limits. Infeasible constraints must be explicit, never silently relaxed.
- Final evaluation currently reports weekly drawdowns and checks intraweek wipe-out. That does not establish daily
  maximum drawdown, executable margin headroom or option spread liquidity. Those remain qualification requirements.
- 2025-26 was already observed in earlier research. Retain the original selection protocol, but label its later
  evaluation as reused historical validation, not a clean untouched holdout. Freeze a candidate before new forward
  observations, and do not use the later evaluation to repeatedly select or retune a winner.
- The old result of approximately 88% annualized return at eight-times leverage with an 82% maximum drawdown is
  inherited research, not new evidence of stable 80-90% returns.

## Next bounded batch

1. Fix and test the verified joint-cap problem; audit post-allocation caps and cash/infeasibility behavior. Preserve
   baseline equivalence when constraints are inactive. Record any changed semantics before running the search.
2. Reuse the recovered price snapshot. Test connectivity once with a short timeout. If available, resume only missing
   provider downloads, keeping one frozen dataset per study. Install `requirements-research.txt` in an isolated env.
3. Rebuild the missing original sleeves according to the preserved reproduction instructions; record source hashes,
   coverage gaps and delistings. Fix auxiliary path assumptions as encountered, without changing the original protocol.
4. Once real inputs are complete and audited, resume the original search in bounded batches, for example:

   ```sh
   python -m src.backtest.optimize.search POOL_DIR CASH_WEEKLY OUT_DIR 10000 --workers 2 --max-new 250
   ```

   Repeat that exact study command to continue its checkpoints. Do not reduce the original trial list or overwrite a
   differently fingerprinted study. Then run overfit checks and the pre-registered evaluation once.
5. Pursue new option/positioning hypotheses only when suitable historical data exist. Register new hypotheses before
   results, count all trials, deduct financing/turnover/slippage, and report daily losses and liquidation scenarios.
   Report genuine improvements or changed blockers; do not repeatedly announce unchanged offline status.

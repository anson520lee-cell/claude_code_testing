# Progress

## Completed

* Inspected repository: it was empty (no commits, no existing code, no CSV event data to migrate).
* Environment: Python 3.11; yfinance 1.7, pandas 3.0, numpy 2.4, matplotlib 3.11, pytest.
* Configuration: `config/settings.yaml` + `src/config.py` (defaults, validation, CLI overrides).
* Data layer: `src/data/provider.py` (Yahoo Finance, retries, network-error vs invalid-ticker distinction),
  `src/data/prices.py` (standardising, cleaning, validation warnings, CSV import).
* Statistics, benchmark comparison, historical event outcomes (`src/analysis/`).
* Event detection and earnings-date evidence matching (`src/events/`).
* SQLite database (`src/database/`): 5 tables, duplicate-safe upserts, forward-return refresh,
  protection of manual research, research CSV import.
* Fundamentals (`src/fundamentals/`): reported snapshot + annual statements, calculated margins/growth,
  no invented values, fallback to stored copy.
* Charts (`src/visualization/charts.py`), colour palette checked for colour-blind safety,
  layout reviewed visually and fixed (legends above plots, fixed header spacing, readable log axis).
* Reports (`src/reporting/report.py`): summary.md, events.csv, fundamentals.csv, analysis.json, run.log.
* Pipeline (`src/pipeline.py`) and CLI (`analyze.py`), plus `event_stats.py` and `import_research.py`.
* 88 automated tests passing (also verified with pandas 2.2 / numpy 1.26).
* End-to-end run on 5 years of synthetic data (scratchpad only, not committed): all outputs generated,
  every event's return / Z-score / volume ratio / forward returns and the headline statistics
  re-computed independently with plain numpy - 0 mismatches.
* Real `python analyze.py AVAV` run: fails cleanly with a correct "network failure" message
  (exit code 3) because Yahoo Finance is blocked in this environment (see Known Issues).
* Bugs found and fixed during testing:
  - floating-point noise in a near-constant baseline produced meaningless Z-scores;
  - yfinance hid network errors and returned an empty table, which was misreported as an
    invalid ticker; yfinance is now told to raise, and each retry uses a fresh Ticker object;
  - fundamentals fallback only worked when *all* fundamentals were missing;
  - re-importing the same research CSV downgraded a manual verification status.
* README written.
* Final acceptance test (2026-09-24):
  - 100/100 automated tests pass (0 failed, 0 skipped) on pandas 3.0 / numpy 2.4 and on pandas 2.2 / numpy 1.26.
  - `python analyze.py AVAV`: blocked by the cloud network policy (proxy answers 403 for Yahoo hosts);
    the program reports it correctly as a network failure (exit code 3).
  - Full pipeline verified on the existing synthetic demo dataset and, through the real CLI, real
    YahooFinanceProvider and real yfinance 1.7 parsing code, against a simulated Yahoo server
    (prices, company info, statements, earnings calendar, 404 answers, network failures).
    Every event statistic, forward return and headline number re-computed independently: 0 mismatches.
  - Bugs found and fixed during acceptance (each with a regression test):
    - an unknown ticker could be reported as a network failure (Yahoo's HTTP 404 was retried);
    - `--period` was silently ignored for `--prices-csv` input;
    - report / database attributed fundamentals and earnings evidence to "Yahoo Finance" even when
      another source supplied them, and Data Limitations claimed Yahoo data for CSV input;
    - annual statement metrics with no data disappeared from the table instead of "Not available";
    - re-running on the same day left a stale benchmark chart; a failed re-run overwrote run.log;
    - chart labels: duplicate percent ticks for low-volatility stocks, clock times on very short
      histories, "nan" text, volume axis not labelled up to a high threshold; in-plot legends moved
      outside the data area.

* Swing technical analysis module for a 2-7 trading-day holding period (2026-09-24):
  - Built on the trader's own chart settings (changed mid-task at the user's request, replacing a first
    version based on EMA5/10/20, RSI7 and MACD 6/13/5): EMA 9 / 21 / 50 / 200 / 250, RSI 6 (primary) / 14,
    MACD 12 / 26 / 9 (DIF, DEA, histogram), MAVOL20, BOLL (20, 1.8), plus ATR(14), ATR% percentile,
    realised volatility, 2-20D returns, relative returns vs SPY, OBV / VWAP20 / ADX as secondary context.
  - Deterministic, documented rules (README section 8) with the requested priority: price structure,
    EMA9 / EMA21, volume vs MAVOL20 (always with the day's direction), RSI6 / RSI14, MACD momentum change,
    BOLL position / squeeze / band walk vs unconfirmed extension, ATR, support / resistance, relative
    strength, EMA50 / EMA200 / EMA250 context. 21 structure states (breakouts by volume, EMA9 / EMA21
    reclaim / loss / cross / pullback / compression, squeeze, range ...). The Swing Technical Condition is a
    priority-weighted points score (-12 ... +12; EMA200 / EMA250 add nothing) shown with its breakdown.
  - Kept: 2/3/5/7-day returns, forward 2/3/5/7-day returns (+1D / 20D), 3/5/7-day MFE / MAE, ATR ranges,
    support / resistance (merged within 0.5 ATR, polarity-aware), relative strength vs SPY, earnings event
    risk (banner within 7 trading days), historical 2-7 day context with N, episodes and small-sample flags.
  - Outputs: `technical_analysis.md` (fixed 8-section layout, sentence limits, <= 150-word summary answering
    the 11 swing questions), `technical_chart.png` (candles, EMA9/21/50 distinct, EMA200/250 subtle, BOLL
    20/1.8, S/R; RSI6/14; MACD DIF/DEA/histogram; volume by up/down day + MAVOL20), a compact "Swing Technical
    Snapshot" in `summary.md`, and a `technical` block (incl. the parameters) in `analysis.json`.
    A failure in this layer never stops the main report.
  - Event system extended: forward returns for 1/2/3/5/7/20 days (raw and vs benchmark) and 3/5/7-day
    MFE / MAE stored for every event; older databases are upgraded automatically (new columns added,
    no data lost); summary section 8 and `event_stats.py` show the 2-7 day horizons.
  - 217 automated tests pass (0 failed) on pandas 3.0 / numpy 2.4 and on pandas 2.2 / numpy 1.26, with no
    warnings in the technical module and a clean ruff / pyflakes run; the technical tests check the exact
    parameters, every rule at its thresholds and the report rules across 31 different market states.
  - Verified end to end through the real CLI against the simulated Yahoo server: all 13 outputs written;
    27 values (EMA9/21/50/200/250, RSI6/14, DIF/DEA/histogram, MAVOL20, BOLL 20/1.8, ATR, relative returns)
    re-computed independently from the stored prices - 0 mismatches; chart inspected (alignment, labels,
    legends, axes, S/R, BOLL, RSI, MACD, volume) and report wording reviewed; event-risk banner checked.
  - Issues found and fixed during review: MACD legend showed one histogram colour; BOLL middle line looked
    like EMA200; band walk rule too loose (tightened to %B >= 0.9); "-0.00", "1 session(s)", doubled phrases
    and nested parentheses in the text; ATR distances rounded so 0.47 and 0.51 ATR both read "0.5".
  - `python analyze.py AVAV` with live data: still blocked by the cloud network policy (exit code 3, see
    Known Issues).

## Currently Working On

* Nothing - waiting for a live run on a computer with internet access (see Remaining).

## Remaining

* Run `python analyze.py AVAV` locally with live Yahoo Finance data and check the report, including
  `technical_analysis.md` and `technical_chart.png` (the live code path is verified against simulated
  Yahoo responses, not yet against Yahoo itself).

## Known Issues

* The cloud environment used for development blocks Yahoo Finance
  (`query1/query2.finance.yahoo.com`, `fc.yahoo.com`, `guce.yahoo.com`) at the network proxy.
  To enable it: open the cloud environment settings (environment menu in the session title bar ->
  Edit -> Network access) and choose a broader access level or allow `*.yahoo.com`.
  On a normal computer with internet access this does not apply.
* The earnings calendar in yfinance is scraped from a Yahoo web page and may break if Yahoo changes
  the page; the analysis continues without it (noted in the report).
* When the internet is down, every download is retried 3 times before falling back to cached data,
  so such a run takes about 40 seconds; `--offline` skips the downloads immediately.
* Swing event-risk day counts use Monday-Friday and do not remove exchange holidays (can be one day high
  around a holiday). In `--offline` mode the earnings calendar is not downloaded, so upcoming earnings
  are shown as unknown.
* The swing classification thresholds are fixed rules of thumb documented in README section 8; they
  have not been tuned on real data, and the historical 2-7 day statistics are descriptive only.
* Indicator values can differ slightly from a charting platform: EMAs start at the first downloaded price
  (EMA200 / EMA250 need a few hundred sessions to settle; a note is shown below 500 sessions), BOLL uses the
  population standard deviation (a sample-std platform draws ~2.6% wider bands) and MAVOL20 includes today.

* Quantitative swing research terminal (2026-10-31):
  - Audit baseline: 217/217 tests passing; Yahoo, SEC EDGAR, FRED and FINRA reachable from this environment.
  - Added a universe-level daily workflow (`daily_run.py`) and a web dashboard (`serve.py`, FastAPI + vanilla JS +
    lightweight-charts) on top of the existing indicator / classification / structure code, which is reused unchanged.
  - New modules: NYSE calendar, batch market data with incremental refresh + corporate-action detection, data-quality
    gate, market regime (6 documented components, history), sector rotation, setup rules (5 long / 3 short), risk plans,
    universe-wide historical simulation, transparent composite score with penalties and supporting / contradicting
    evidence, point-in-time SEC XBRL fundamentals (Yahoo fallback with estimated publication dates), options analysis,
    positioning (short interest, 13F, Form 4, FINRA short volume and off-exchange share), catalysts (earnings, SEC
    filing classification, analyst actions, macro calendar), Markdown / HTML / JSON daily report.
  - Database: 6 new tables added non-destructively; existing databases are backed up before the first upgrade.
  - 254 tests pass (37 new). Real run for session 2026-10-30: 149 symbols, 0 excluded, regime Bearish (-26),
    40 deep dives, ~2 minutes; candidate indicators / plan / score re-computed independently from stored prices: 0 mismatches.
  - Known limitation in this environment: SEC EDGAR needs SEC_USER_AGENT (name + email); without it the run falls back
    to Yahoo statements and reports the missing SEC filing checks.
* SL / TP validation backtest (2026-10-31):
  - `backtest.py` + `src/backtest/research.py`: 5 years, 233 focus stocks, 54,086 signals; 8 stops x 8 targets x 3 time
    stops, long / short; in-sample (first 60% of dates) selection, out-of-sample judgement; random-entry baseline with
    the same exits; live-plan accuracy diagnostics; factor IC study and an in-sample-fitted score checked out-of-sample.
  - Research page: "How SL/TP are set" explainer, verdicts, stop x target heat map (side / horizon / period toggles),
    best combinations with baseline and setup edge, R:R calibration, MAE / MFE, score quintiles, factor IC, by setup /
    regime / quarter, re-run button (`/api/backtest`, `/api/backtest/run`). Every plan now states why its stop and
    targets sit where they do (`stop_basis` / `target_basis`) and shows the backtest's preferred alternative.
  - Findings: live plan +0.00R out-of-sample (no edge); targets too far; stops not too tight; best long exit
    (1.5 ATR + 3-ATR trail, 10 days) +0.17R OOS but random entries earn +0.14R with the same exit; short setups lose;
    score has no predictive power. The live plan is unchanged pending the user's decision.
  - 264 tests pass.
* Portfolio and context layer (2026-10-31):
  - Portfolio page: ATR/stop-based position sizing with regime budget and caps, direction-adjusted correlation matrix,
    total heat vs cap, effective bets, correlated clusters, trade journal (manual records, R review by setup), price alerts
    (checked after every run and on page load; macOS notification).
  - Market page: cross-asset table (DXY, WTI, gold, Bitcoin) with z-scores and QQQ correlation; universe advance / decline
    line and McClellan oscillator; industry-group RS ranking with 5-day rank change; each candidate shows its group rank.
  - Ticker pages: earnings reaction study (options-implied vs historical moves, post-earnings drift, gap fade) and SEC
    fails-to-deliver analysis (needs SEC_USER_AGENT).
* Long only + crash rebound (2026-11-01):
  - `crash_study.py` event study: buying 0-2 days after a big multi-day crash beats random entries (+2.5% over 5 days,
    holds out of sample); per-stock timing does not; shorting after spikes does not pay.
  - Candidate lists are long only; new `crash_rebound` setup with its own stop / targets / scoring and a +15 bonus;
    long T1 capped at entry + 2 ATR; stock pages show each stock's crash history; Research page shows the study.
* News (2026-11-01): automatic RSS headlines (Yahoo Finance + Nasdaq for the whole universe, Google News for candidates
  and market topics), tagged and stored; refreshed after each run and hourly; News tab, stock-page news, crash news.
* Long-term robustness review (2026-11-01):
  - Fixed: provider fallback marked the requested session as done; dashboard server kept running old code after an
    update; unbounded analyst cache; disk growth (reports, FINRA, SEC, news, logs); news volume that could get the IP
    rate-limited; tracked generated universe file blocking `git pull`; FOMC list ending after 2026 (now warns); SQLite
    lock contention; unpinned major versions; unscheduled market closures flagged as data gaps; price alerts set during
    the Asian day skipped the next US session (UTC vs New York date); half-written files after a crash; truncated SEC
    downloads cached forever; universe rebuild could shrink on a screener glitch; failing backtest / rebuild retried on
    every run; alert / journal symbols stopped updating after leaving the universe; duplicate headlines.
  - Added: yfinance self-upgrade when downloads fail broadly, daily journal / alert export with `--restore-user-data`,
    delisted-symbol warning, bundled chart library. See docs/TERMINAL.md section 10.
* Strategy lab (2026-11-01): exits per setup and all 183 indicators tested walk-forward on 34,195 setup trades + 21,328
  random entries. New exits (1.5 ATR stop + 3-ATR trailing stop, 20 sessions; squeeze 2 ATR + 3R) raise the
  out-of-sample result from +0.05R to about +0.29R per trade; no indicator model passed (score buckets earn the same),
  so ranking uses each setup's tested edge over random entries. See docs/TERMINAL.md 7e.

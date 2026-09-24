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

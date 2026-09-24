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

## Currently Working On

* Nothing - waiting for a live run on a computer with internet access (see Remaining).

## Remaining

* Run `python analyze.py AVAV` locally with live Yahoo Finance data and check the report
  (the live code path is verified against simulated Yahoo responses, not yet against Yahoo itself).

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

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

## Currently Working On

* Nothing - MVP complete; waiting for network access to Yahoo Finance to run the real AVAV analysis.

## Remaining

* Run `python analyze.py AVAV` with real data once Yahoo Finance is reachable, and inspect the output
  (the code paths for live yfinance responses - price history, company info, statements, earnings
  calendar - are written against yfinance 1.7's documented formats and tested with stand-ins, but
  have not yet seen a live response).

## Known Issues

* The cloud environment used for development blocks Yahoo Finance
  (`query1/query2.finance.yahoo.com`, `fc.yahoo.com`, `guce.yahoo.com`) at the network proxy.
  Every other market-data host tried (Stooq, SEC EDGAR, Nasdaq, Alpha Vantage, ...) is blocked too.
  To enable it: open the cloud environment settings (environment menu in the session title bar ->
  Edit -> Network access) and choose a broader access level or allow `*.yahoo.com`.
  On a normal computer with internet access this does not apply.
* The earnings calendar in yfinance is scraped from a Yahoo web page and may break if Yahoo changes
  the page; the analysis continues without it (noted in the report).

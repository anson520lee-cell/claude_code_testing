# Event-Driven Stock Research

A reusable research workflow for US-listed stocks. Give it a ticker:

```bash
python analyze.py AVAV
```

and it will:

1. download daily market data (and a benchmark, SPY by default),
2. calculate quantitative statistics (returns, Z-scores, volatility, drawdowns ...),
3. detect abnormal price / volume days ("events"),
4. store prices, events and fundamentals in a local SQLite database,
5. retrieve fundamental data (revenue, margins, cash flow, valuation multiples ...),
6. draw charts,
7. write a structured research report,
8. save everything in `reports/<TICKER>/<YYYY-MM-DD>/`.

**What it is not:** this is *not* a price-prediction bot. It produces no buy/sell
signals, price targets or "confidence scores". It helps you investigate:

> *What happened, how unusual was the move, what evidence is there about why,
> and what happened after similar events in the past?*

```
Market Data -> Detect Anomaly -> Identify Event -> Collect Evidence -> Quantitative Analysis
            -> Fundamental Context -> Historical Comparison -> Research Report
```

Anything the program cannot know from its data is shown as **Unknown** or
**Not available**. It never invents explanations, news, or financial numbers.

---

## Contents

1. [Quick start](#1-quick-start)
2. [What you get](#2-what-you-get)
3. [Command-line options](#3-command-line-options)
4. [Configuration](#4-configuration-configsettingsyaml)
5. [Architecture](#5-architecture)
6. [Methodology and formulas](#6-methodology-and-formulas)
7. [Event detection](#7-event-detection)
8. [The database](#8-the-database)
9. [Recording your own research](#9-recording-your-own-research)
10. [Studying events across many stocks](#10-studying-events-across-many-stocks)
11. [Fundamental data](#11-fundamental-data)
12. [Reliability and error handling](#12-reliability-and-error-handling)
13. [Tests](#13-tests)
14. [Limitations](#14-limitations)
15. [Possible future upgrades](#15-possible-future-upgrades)
16. [How to read the code (learning path)](#16-how-to-read-the-code-learning-path)

---

## 1. Quick start

Requirements: Python 3.10 or newer and an internet connection (for Yahoo Finance).

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install the dependencies
pip install -r requirements.txt

# 3. run an analysis
python analyze.py AVAV

# 4. run the tests
python -m pytest
```

Open `reports/AVAV/<today's date>/summary.md` (any Markdown viewer, VS Code,
or GitHub shows the charts inline).

### Dependencies

| Package | Why |
|---|---|
| `yfinance` | Free daily prices, company info, financial statements and earnings dates from Yahoo Finance |
| `pandas`, `numpy` | Tables and maths |
| `matplotlib` | Charts |
| `PyYAML` | Reads `config/settings.yaml` |
| `pytest` | Runs the automated tests |

SQLite (the database) is built into Python, so it needs no installation.

---

## 2. What you get

```
reports/
└── AVAV/
    └── 2026-09-24/
        ├── summary.md               <- the research report (start here)
        ├── events.csv               <- every detected event, one row each
        ├── fundamentals.csv         <- normalised fundamental data
        ├── price_chart.png          <- close, 20/50-day MAs, event markers
        ├── returns_distribution.png <- histogram with mean, median, std, normal curve
        ├── volume_chart.png         <- volume, previous-20-day average, abnormal-volume days
        ├── event_chart.png          <- Z-score timeline + price-move-vs-volume scatter
        ├── benchmark_chart.png      <- cumulative return vs SPY and relative performance
        ├── analysis.json            <- all results in machine-readable form
        └── run.log                  <- detailed log of this run
```

Running the same ticker again on the same day overwrites that day's folder;
different days get their own folder, so you keep a history of reports.

`summary.md` has these sections:

| Section | Contents |
|---|---|
| 1. Company / Ticker | Name, exchange, sector, data source and date range **[Reported]** |
| 2. Market Overview | Last close, 52-week range, moving averages, drawdown, volume **[Calculated]** |
| 3. Recent Performance | Returns over 1 day ... 1 year, YTD, full period **[Calculated]** |
| 4. Volatility | Rolling / annual volatility, distribution shape, max drawdown, fat-tail check **[Calculated]** |
| 5. Abnormal Events | Rules used, counts, event table, evidence, unknowns **[Calculated]** |
| 6. Fundamental Snapshot | Valuation, profitability, cash, debt, annual statements **[Reported]** + margins **[Calculated]** |
| 7. Benchmark Comparison | Stock vs benchmark returns, beta, correlation **[Calculated]** |
| 8. Historical Event Outcomes | What happened 1/5/20 days after events, vs. an all-days baseline **[Calculated]** |
| 9. Data Limitations | Standard caveats plus every warning raised during this run |

---

## 3. Command-line options

```bash
python analyze.py TICKER [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--period` | `5y` | History to analyse: `1y`, `2y`, `5y`, `10y`, `ytd`, `max` |
| `--benchmark` | `SPY` | Benchmark ticker, e.g. `QQQ`, `ITA`, `^GSPC` |
| `--no-benchmark` | | Skip the benchmark comparison |
| `--sigma` | `2.5` | Return Z-score threshold for events |
| `--volume-ratio` | `2.0` | Volume ratio threshold for events |
| `--offline` | | No internet: use prices/fundamentals already stored in the database |
| `--prices-csv FILE` | | Read the stock's daily prices from your own CSV instead of downloading |
| `--benchmark-csv FILE` | | Same for the benchmark |
| `--no-fundamentals` | | Skip fundamental data |
| `--config FILE` | `config/settings.yaml` | Use another settings file |
| `--db FILE` | `data/stock_research.db` | Use another database file |
| `--reports-dir DIR` | `reports` | Write reports somewhere else |
| `-v`, `--verbose` | | Show debug messages |

Examples:

```bash
python analyze.py AVAV --period 10y
python analyze.py AVAV --benchmark ITA          # compare with an aerospace & defence ETF
python analyze.py AVAV --sigma 3 --volume-ratio 3
python analyze.py KTOS --no-fundamentals
python analyze.py AVAV --offline                # re-analyse stored data without internet
```

A CSV passed with `--prices-csv` needs a `Date` column and a `Close` column;
`Open`, `High`, `Low`, `Adj Close` and `Volume` are used when present. Formats
such as Yahoo's download and Nasdaq.com's (`Close/Last`, `$` signs) are understood.

Exit codes: `0` success, `2` invalid input or settings, `3` no price data could
be obtained, `1` unexpected error (full traceback shown).

Other scripts:

| Script | Purpose |
|---|---|
| `python event_stats.py` | Pool stored events across all tickers ([section 10](#10-studying-events-across-many-stocks)) |
| `python import_research.py FILE` | Import your research notes from an edited `events.csv` ([section 9](#9-recording-your-own-research)) |

---

## 4. Configuration (`config/settings.yaml`)

Every threshold and window lives in one file, so you can experiment without
touching the code. The most important settings:

```yaml
data:
  period: "5y"
  benchmark: "SPY"
statistics:
  zscore_window: 60       # baseline days for the return Z-score
  volume_window: 20       # baseline days for average volume / volume ratio
  volatility_window: 20   # days for rolling volatility
events:
  return_zscore_threshold: 2.5   # |Z| >= 2.5
  volume_ratio_threshold: 2.0    # volume >= 2x average
  volume_zscore_threshold: null  # off
  abs_return_threshold: null     # off (e.g. 0.10 for |return| >= 10%)
  earnings_match_window_days: 1
```

Set a rule to `null` to switch it off. The settings are validated when the
program starts; an invalid value (e.g. a negative threshold) stops the run with
a clear message.

---

## 5. Architecture

```
analyze.py                 Command-line entry point (parses options, prints results)
event_stats.py             Cross-ticker event statistics from the database
import_research.py         Imports manual research notes from an edited events CSV
config/settings.yaml       All thresholds, windows, paths
src/
├── config.py              Loads + validates settings, applies CLI overrides
├── pipeline.py            Runs the whole workflow, step by step (read this first)
├── data/
│   ├── provider.py        ALL internet access (Yahoo Finance via yfinance), with retries
│   └── prices.py          Standardises, cleans and validates price tables; CSV import
├── analysis/
│   ├── statistics.py      Returns, Z-scores, volatility, volume, MAs, drawdown, forward returns
│   ├── benchmark.py       Abnormal returns, relative performance, beta
│   └── event_study.py     "What happened after events?" summary tables
├── events/
│   ├── detector.py        Flags abnormal days using the configured rules
│   └── evidence.py        Links events to earnings-release dates (date match only)
├── fundamentals/
│   └── fundamentals.py    Normalises provider fundamentals; calculates margins/growth
├── database/
│   ├── schema.sql         Table definitions
│   └── db.py              Connect, save (without duplicates), load
├── reporting/
│   └── report.py          summary.md, events.csv, fundamentals.csv, analysis.json
└── visualization/
    └── charts.py          The PNG charts (matplotlib)
tests/                     Automated tests (pytest)
data/                      The SQLite database (created on first run, not committed to git)
reports/                   Generated reports
```

Design principles:

* **One job per module.** Calculations never download data; the report writer
  never calculates statistics. Each piece can be tested on its own.
* **All internet access in one class** (`YahooFinanceProvider`). Tests replace it
  with a fake provider, so the full workflow is tested without internet. A
  different data source can be added later by writing a class with the same four
  methods.
* **Non-essential steps may fail without stopping the run.** If the benchmark,
  fundamentals, earnings calendar or a chart fail, the problem is recorded in
  the report's *Data Limitations* section and the analysis continues. Only
  missing price data stops a run.

---

## 6. Methodology and formulas

Notation: `P_t` = price on trading day `t`, `r_t` = daily return, `V_t` = volume,
`W` = window length in trading days.

### Price series

Returns use the **adjusted close** (adjusted for splits and dividends, i.e. a
total-return series). If the adjusted close is missing on any day, the plain
close is used for *every* day, so adjusted and unadjusted prices are never mixed
(the report says which one was used). Charts and moving averages show the plain
close, like a normal price chart.

### Returns

| Quantity | Formula |
|---|---|
| Daily return | `r_t = P_t / P_(t-1) - 1` |
| Log return | `ln(P_t / P_(t-1))` |
| 5-day / 20-day return (trailing) | `P_t / P_(t-5) - 1`, `P_t / P_(t-20) - 1` |
| Forward h-day return (h = 1, 5, 20) | `P_(t+h) / P_t - 1` |
| Annualised return over the period | `(P_end / P_start)^(252 / N) - 1`, N = number of daily returns |

### Baseline, Z-score and volatility

| Quantity | Formula |
|---|---|
| Rolling mean return | `mean_t = average(r_(t-W) ... r_(t-1))`, W = `zscore_window` (60) |
| Rolling standard deviation | `std_t = sample std(r_(t-W) ... r_(t-1))` |
| Return Z-score | `Z_t = (r_t - mean_t) / std_t` |
| Rolling volatility (annualised) | `std(r_(t-19) ... r_t) x sqrt(252)` |
| Full-period volatility | `std(all daily returns) x sqrt(252)` |

**Why the baseline excludes day t:** if today's +20% move were part of its own
baseline, it would inflate the standard deviation and shrink its own Z-score
(with a 20-day window the Z-score could never exceed about 4.3). Using only the
previous days measures "how unusual was today compared with recent normal
behaviour" and avoids look-ahead.

A standard deviation of (almost) zero, e.g. a stock whose price did not move for
60 days, gives no Z-score (n/a) rather than an enormous meaningless number.

### Volume

| Quantity | Formula |
|---|---|
| Average volume | `avgV_t = average(V_(t-20) ... V_(t-1))` (previous 20 days) |
| Volume ratio | `V_t / avgV_t` |
| Volume Z-score | `(V_t - avgV_t) / std(V_(t-20) ... V_(t-1))` |

### Trend and risk

| Quantity | Formula |
|---|---|
| Moving averages | `MA20_t = average(Close_(t-19) ... Close_t)`, same for 50 days |
| Drawdown | `P_t / max(P_0 ... P_t) - 1` |
| Maximum drawdown | the most negative drawdown, with its peak, trough and recovery dates |
| Fat-tail check | share of days with `|Z| >= k` vs. the normal-distribution value `erfc(k / sqrt(2))` (1.24% for k = 2.5) |

### Benchmark comparison

The benchmark is first aligned to the stock's trading days, *then* returns are
calculated, so both returns on a row always cover exactly the same dates.

| Quantity | Formula |
|---|---|
| Benchmark return | `rb_t = B_t / B_(t-1) - 1` |
| Abnormal return | `ar_t = r_t - rb_t` |
| Forward abnormal return | `(P_(t+h)/P_t - 1) - (B_(t+h)/B_t - 1)` |
| Cumulative return | `P_t / P_0 - 1` (and `B_t / B_0 - 1`) |
| Cumulative relative performance | `(P_t / P_0) / (B_t / B_0) - 1` |
| Beta | `cov(r, rb) / var(rb)` over daily returns |
| Correlation | Pearson correlation of daily returns |

Use a different benchmark with `--benchmark QQQ` or in `settings.yaml`.

---

## 7. Event detection

A trading day becomes an **event** when **any** enabled rule is true:

| Rule | Condition | Default |
|---|---|---|
| `return_zscore` | `|Z_t| >= threshold` | 2.5 |
| `volume_ratio` | `V_t / avgV_t >= threshold` | 2.0 |
| `volume_zscore` | volume Z-score `>= threshold` | off |
| `abs_return` | `|r_t| >= threshold` | off |

A day without enough history for a rule (e.g. the first 60 days for the
Z-score) can never trigger that rule.

For every event the program stores the facts it can measure: close, return,
Z-score, volume, volume ratio and Z-score, benchmark and abnormal return,
1/5/20-day forward returns (raw and abnormal), direction (`up`/`down`/`flat`),
which rules fired, and a data-derived **anomaly type**:

* *Abnormal price move with abnormal volume*
* *Abnormal price move (volume not flagged)*
* *Abnormal volume (price move not flagged)*

It also stores `trading_days_since_prev_event`, so clusters of events (e.g. an
earnings day followed by a second big day) can be filtered out in later
statistics.

### Research fields (the "why")

| Field | Starts as | Filled by |
|---|---|---|
| `event_category` | `Unknown` | evidence matching or you |
| `event_description` | empty | evidence matching or you |
| `source` | empty | evidence matching or you |
| `verification_status` | `Unverified` | evidence matching or you |
| `notes` | empty | you |

**Evidence currently collected automatically: earnings-release dates** from
Yahoo Finance's earnings calendar. For each release, the first trading session
that could react is worked out (a release after 16:00 New York time -> next
trading day; before the open or unknown time -> that day). An event on that
session or up to `earnings_match_window_days` (default 1) trading days later is
labelled:

* category `Earnings release (date match)`
* description with the release date and the reported EPS / estimate / surprise
* verification status `Unverified (automatic date match)`

This is a **date coincidence, not proof of cause**, and the report says so. All
other events stay `Unknown` until you research them.

### Historical event outcomes

Section 8 of the report (and `event_stats.py`) groups events and shows the
average and median 1/5/20-day forward return, the share of positive outcomes,
and the average forward *abnormal* return:

* all events; up moves; down moves;
* up/down moves **with** and **without** abnormal volume;
* volume-only events;
* **all trading days (baseline)** - compare every group with this row.

For up-move groups, "% positive" is the **continuation** rate; for down-move
groups it is the **reversal** rate. With small groups, overlapping windows and
no significance testing, these are descriptive numbers, not predictions.

---

## 8. The database

**Why SQLite (not DuckDB or CSV files)?** SQLite is built into Python (nothing
to install), stores everything in one file, enforces unique keys (which is how
duplicates are prevented), handles millions of rows, and can be opened with free
tools such as [DB Browser for SQLite](https://sqlitebrowser.org/). DuckDB is
faster for very large analytical queries, but it would be an extra dependency
this project does not need yet.

The database file is `data/stock_research.db`. It is created automatically and
is listed in `.gitignore` (it is your local research history; back it up if it
matters to you).

### Tables

| Table | One row per | Key | Main columns |
|---|---|---|---|
| `stocks` | ticker | `ticker` | name, exchange, sector, industry, currency, first/last analysed |
| `price_data` | ticker + trading day | `(ticker, date)` | open, high, low, close, adj_close, volume, source |
| `events` | ticker + event day | `(ticker, event_date)` | all event statistics, forward returns, research fields, run ids |
| `fundamentals` | ticker + period + metric | `(ticker, period_type, period_end, metric)` | value, unit, source, retrieved_at |
| `analysis_runs` | run of `analyze.py` | `run_id` | status, parameters used, date range, n_events, warnings, error |

`fundamentals` uses a "long" layout (one metric per row) so new metrics can be
added without changing the table, and repeated runs build a history:
`period_type = 'snapshot'` rows are dated by retrieval day (valuation over time);
`period_type = 'annual'` rows are dated by fiscal year end.

### Duplicate handling

Saving uses SQLite *upserts* (`INSERT ... ON CONFLICT DO UPDATE`):

* **Prices:** re-downloading updates existing days (e.g. after a split
  adjustment) instead of duplicating them.
* **Events:** statistics are refreshed; forward returns fill in as time passes
  (a known value is never replaced by "unknown"); benchmark figures are kept if
  a later run has no benchmark.
* **Research fields are protected:** the program only updates research fields
  that it wrote itself (`Unknown` / earnings match). Once you change an event's
  category, source or verification status, automatic runs never touch that
  event's research fields again, and `notes` are never overwritten.
* Events stored by an earlier run with different thresholds are kept (each
  stores the settings it was detected with in `detection_settings`), and their
  forward returns are still refreshed.

### Querying the database

With Python / pandas:

```python
import sqlite3, pandas as pd
conn = sqlite3.connect("data/stock_research.db")

# All big up-moves with high volume, across every ticker analysed so far
pd.read_sql("""
    SELECT ticker, event_date, daily_return, return_zscore, volume_ratio,
           fwd_return_5d, fwd_return_20d, event_category
    FROM events
    WHERE return_zscore >= 2.5 AND volume_ratio >= 2
    ORDER BY event_date
""", conn)
```

Useful SQL examples:

```sql
-- Average 20-day forward return after >2.5 sigma up-moves, per ticker
SELECT ticker, COUNT(*) AS n, AVG(fwd_return_20d) AS avg_fwd_20d
FROM events WHERE return_zscore >= 2.5 AND fwd_return_20d IS NOT NULL
GROUP BY ticker;

-- Did high-volume down moves continue or reverse?
SELECT AVG(fwd_return_5d > 0) AS share_reversed, COUNT(*) AS n
FROM events WHERE direction = 'down' AND is_price_event = 1 AND is_volume_event = 1
  AND fwd_return_5d IS NOT NULL;

-- History of the analysis runs
SELECT run_id, ticker, started_at, status, n_events FROM analysis_runs ORDER BY run_id DESC;
```

---

## 9. Recording your own research

The research fields are where your investigation goes (news found, filings
read, analyst actions ...). The easiest way to fill them:

1. Open `reports/AVAV/<date>/events.csv` in Excel, LibreOffice or Google Sheets.
2. For the events you researched, fill in `event_category` (e.g. `Contract award`,
   `Guidance cut`, `Analyst downgrade`), `event_description`, `source` (e.g. a URL
   or "8-K filed 2025-06-24"), `verification_status` (e.g. `Verified`) and `notes`.
3. Save as CSV and import:

```bash
python import_research.py reports/AVAV/2026-09-24/events.csv --dry-run   # preview
python import_research.py reports/AVAV/2026-09-24/events.csv             # apply
```

Only research columns are imported (never prices or statistics), empty cells
never delete existing text, and rows that match no stored event are skipped and
listed. If you fill in research but leave the status as `Unverified`, the status
becomes `Researched manually` so later runs will not overwrite it. The next
report for that ticker shows your research. You can also edit the `events`
table directly with an SQLite tool.

*The repository did not contain earlier CSV event files, so no data migration
was needed; this importer is also the path for bringing in event notes you keep
in spreadsheets.*

---

## 10. Studying events across many stocks

Every analysed ticker adds its events to the database, so questions like
*"what usually happens after a >2.5 sigma positive move?"* can be answered over
many stocks:

```bash
python analyze.py AVAV
python analyze.py KTOS
python analyze.py RKLB
python event_stats.py --min-sigma 2.5
python event_stats.py --tickers AVAV KTOS --min-volume-ratio 3 --csv out.csv
```

`event_stats.py` prints the same group table as report section 8, pooled over
the selected tickers, with an all-trading-days baseline calculated from the
stored prices.

---

## 11. Fundamental data

Retrieved from Yahoo Finance (via `yfinance`) and saved to `fundamentals.csv` and
the `fundamentals` table:

* **Snapshot [Reported]:** market cap, enterprise value, revenue (TTM), revenue
  growth (latest quarter vs. a year earlier), gross margin, operating margin, net
  income, EPS, free cash flow (Yahoo's "levered" FCF), EBITDA, cash, debt, shares
  outstanding, trailing and forward P/E, EV/Sales, EV/EBITDA.
* **Annual statements [Reported], about 4 fiscal years:** revenue, gross profit,
  operating income, net income, diluted EPS, EBITDA, operating cash flow, capex,
  free cash flow, cash, total debt, shares outstanding.
* **[Calculated] from the statements:** revenue growth vs. the previous fiscal
  year (only between consecutive years), gross, operating and net margin.

Every value records its source (e.g. `info['marketCap']` or the statement row
used). Values the provider does not supply, and non-numeric values such as
"Infinity", are simply left out and shown as **Not available**. Nothing is
estimated. If the download fails, the last stored copy is shown with a warning.

---

## 12. Reliability and error handling

| Situation | What happens |
|---|---|
| Invalid ticker format (e.g. `AV AV`) | Rejected before any download (exit code 2) |
| Unknown / delisted ticker | Clear message, not retried; no empty report folder left behind (exit code 3) |
| Network or API failure | Tried up to 3 times with increasing waits; then the **cached prices in the database** are used with a warning, or the run stops with a clear message if nothing is cached |
| Benchmark / fundamentals / earnings calendar unavailable | Run continues; noted under *Data Limitations* (stored fundamentals are used if available) |
| Missing or bad price rows | Removed and reported (missing/negative close, duplicate dates); suspicious values (high < low, long gaps, zero volume) are reported but not altered |
| Not enough history for a statistic | The statistic is `n/a`; it is never computed from too little data |
| A chart fails | The rest of the report is still written; the failure is noted |
| Running twice | No duplicate rows (see section 8) |
| Stale data | A warning if the latest price is more than 7 days old |

Every run writes `run.log` in its report folder, and every run (including failed
ones and their error message) is recorded in the `analysis_runs` table.

---

## 13. Tests

```bash
python -m pytest            # all tests
python -m pytest -v tests/test_statistics.py   # one file, verbose
```

The tests need no internet. They cover:

| File | What is tested |
|---|---|
| `test_statistics.py` | daily/log/multi-day returns, Z-scores (baseline excludes the day itself, full-window rule, zero-variance case), volume ratio and Z-score, volatility annualisation, moving averages, forward returns, drawdowns, summary statistics |
| `test_events.py` | planted events are found, rule combinations and thresholds, no events without history, default Unknown fields, spacing between events, earnings-date matching (after-close, pre-market, weekend releases) |
| `test_benchmark.py` | abnormal returns, date alignment, relative performance, beta = 2 for a 2x stock |
| `test_database.py` | table creation, insertion, duplicate handling, protection of manual research, forward-return refresh, research CSV import |
| `test_prices.py` | yfinance and CSV formats, cleaning and validation warnings |
| `test_fundamentals.py` | no invented values, calculated margins and growth, failures handled |
| `test_provider.py` | retries on network errors, invalid tickers not retried |
| `test_pipeline.py` | the complete workflow with a fake provider: every output file, database contents, re-runs, invalid tickers, network failure + cache fallback, offline mode, the CLI |

---

## 14. Limitations

* **Data source:** Yahoo Finance through the unofficial `yfinance` library. It is
  free and convenient but not an official record; data can have gaps, errors or
  revisions, and Yahoo can change or rate-limit the service at any time.
* **Causes of events are mostly Unknown.** Only earnings-release dates are
  matched automatically. News, SEC filings, analyst actions, index changes,
  contract announcements and sector moves are not checked.
* **Earnings calendar:** comes from scraping a Yahoo web page, which is fragile;
  release times can be approximate.
* **Z-score baseline:** a 60-day window adapts to regime changes, so after a
  quiet period ordinary moves can score high, and in volatile periods large
  moves can score low.
* **Abnormal return** is a simple difference from the benchmark, not adjusted
  for beta (no market model).
* **Event statistics** are descriptive: small samples, overlapping windows,
  no significance tests, survivorship bias (only tickers you chose to analyse).
* **Daily data only** - no intraday timing of moves.
* **Fundamentals** reflect the provider's definitions at retrieval time; about
  4 years of annual statements; no quarterly history yet.

---

## 15. Possible future upgrades

* **SEC EDGAR filings as evidence:** match events to 8-K filings (with item
  codes, e.g. 2.02 results, 1.01 material agreements) from the free SEC API.
* **News headlines** from a news API, stored with source links, as evidence.
* **Market-model abnormal returns** (`r - (alpha + beta x rb)`) and sector ETF
  benchmarks.
* **Significance tests** for event outcomes (t-tests, bootstrap), and
  de-clustering of overlapping events.
* **Quarterly fundamentals** and valuation history charts.
* **Batch mode:** `python analyze.py AVAV KTOS RKLB` or a watch-list file.
* **HTML report** with interactive charts.

---

## 16. How to read the code (learning path)

1. `analyze.py` - how command-line options are read.
2. `src/pipeline.py` - the whole workflow in order; each numbered step calls one module.
3. `src/analysis/statistics.py` - the core maths, with each formula written above its code.
4. `src/events/detector.py` - how rules become events.
5. `src/database/schema.sql` then `src/database/db.py` - how data is stored without duplicates.
6. `tests/test_statistics.py` - small hand-checked examples of every formula; a
   good way to see exactly what each function returns.

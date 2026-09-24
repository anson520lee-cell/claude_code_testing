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
8. describe the short-term **swing technical state** for a 2-7 trading-day
   holding period with the trader's chart settings (EMA 9/21/50/200/250, RSI 6/14,
   MACD 12/26/9, MAVOL20, BOLL 20/1.8): trend, momentum, volume, volatility,
   support / resistance, relative strength, earnings event risk and what happened
   after similar days,
9. save everything in `reports/<TICKER>/<YYYY-MM-DD>/`.

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
8. [Swing technical analysis (2–7 trading days)](#8-swing-technical-analysis-27-trading-days)
9. [The database](#9-the-database)
10. [Recording your own research](#10-recording-your-own-research)
11. [Studying events across many stocks](#11-studying-events-across-many-stocks)
12. [Fundamental data](#12-fundamental-data)
13. [Reliability and error handling](#13-reliability-and-error-handling)
14. [Tests](#14-tests)
15. [Limitations](#15-limitations)
16. [Possible future upgrades](#16-possible-future-upgrades)
17. [How to read the code (learning path)](#17-how-to-read-the-code-learning-path)

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
        ├── technical_analysis.md    <- swing technical report for a 2-7 trading-day horizon
        ├── technical_chart.png      <- 6-month swing chart: candles, EMA 9/21/50/200/250, BOLL 20/1.8, RSI 6/14, MACD 12/26/9, volume + MAVOL20
        ├── analysis.json            <- all results in machine-readable form
        └── run.log                  <- detailed log of this run
```

Running the same ticker again on the same day overwrites that day's folder (charts
the new run does not produce, e.g. the benchmark chart after `--no-benchmark`, are
removed; `run.log` keeps the log of every run that day). Different days get their
own folder, so you keep a history of reports.

`summary.md` has these sections:

| Section | Contents |
|---|---|
| Swing Technical Snapshot | Compact 2-7 day technical state and Swing Technical Condition; links to `technical_analysis.md` ([section 8](#8-swing-technical-analysis-27-trading-days)) **[Calculated]** |
| 1. Company / Ticker | Name, exchange, sector, data source and date range **[Reported]** |
| 2. Market Overview | Last close, 52-week range, moving averages, drawdown, volume **[Calculated]** |
| 3. Recent Performance | Returns over 1 day ... 1 year, YTD, full period **[Calculated]** |
| 4. Volatility | Rolling / annual volatility, distribution shape, max drawdown, fat-tail check **[Calculated]** |
| 5. Abnormal Events | Rules used, counts, event table, evidence, unknowns **[Calculated]** |
| 6. Fundamental Snapshot | Valuation, profitability, cash, debt, annual statements **[Reported]** + margins **[Calculated]** |
| 7. Benchmark Comparison | Stock vs benchmark returns, beta, correlation **[Calculated]** |
| 8. Historical Event Outcomes | What happened 2/3/5/7/20 days after events (incl. MFE / MAE), vs. an all-days baseline **[Calculated]** |
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
`--period` still applies (by default the most recent 5 years of the file are
analysed); use `--period max` to analyse the whole file.

Exit codes: `0` success, `2` invalid input or settings, `3` no price data could
be obtained, `1` unexpected error (full traceback shown).

Other scripts:

| Script | Purpose |
|---|---|
| `python event_stats.py` | Pool stored events across all tickers ([section 11](#11-studying-events-across-many-stocks)) |
| `python import_research.py FILE` | Import your research notes from an edited `events.csv` ([section 10](#10-recording-your-own-research)) |

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
technical:                # swing analysis, see section 8
  structure_lookback_days: 60
  event_risk_days: 7
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
├── technical/             Swing technical analysis (2-7 trading days)
│   ├── indicators.py      EMA 9/21/50/200/250, RSI 6/14, MACD 12/26/9, MAVOL20, BOLL 20/1.8, ATR, OBV, VWAP, ADX, ...
│   ├── classification.py  The documented rules that turn indicators into labels + Swing Technical Condition
│   ├── structure.py       Swing points, support / resistance, breakout / pullback / range states
│   ├── swing_stats.py     Historical 2-7 day outcomes (incl. MFE / MAE) after similar days
│   ├── analysis.py        Runs the steps above; earnings event risk
│   ├── reporting.py       technical_analysis.md and the summary.md snapshot
│   └── chart.py           technical_chart.png
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
  fundamentals, earnings calendar, the swing technical analysis or a chart fail, the problem is recorded in
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
| Forward h-day return (h = 1, 2, 3, 5, 7, 20) | `P_(t+h) / P_t - 1` |
| Maximum favourable / adverse excursion (h = 3, 5, 7) | `max(High_(t+1..t+h)) / P_t - 1` / `min(Low_(t+1..t+h)) / P_t - 1` (dividend-adjusted highs and lows) |
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
1/2/3/5/7/20-day forward returns (raw and abnormal), 3/5/7-day maximum
favourable / adverse excursion (MFE / MAE), direction (`up`/`down`/`flat`),
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
average 2/3/5/7-day forward return (20-day as a secondary horizon), the median
5-day return, the share of positive outcomes after 5 and 7 days, the average
forward *abnormal* return and the median MFE / MAE:

* all events; up moves; down moves;
* up/down moves **with** and **without** abnormal volume;
* volume-only events;
* **all trading days (baseline)** - compare every group with this row.

For up-move groups, "% positive" is the **continuation** rate; for down-move
groups it is the **reversal** rate. With small groups, overlapping windows and
no significance testing, these are descriptive numbers, not predictions.

## 8. Swing technical analysis (2–7 trading days)

Every run also describes the stock's **short-term technical state** for a
typical holding period of 2–7 trading days, using the trader's own chart
settings:

* `technical_analysis.md` - the full swing report (layout below),
* `technical_chart.png` - price, RSI, MACD and volume for about 6 months,
* a compact **Swing Technical Snapshot** near the top of `summary.md`,
* a `technical` block in `analysis.json` (all values, labels, levels and the
  parameters used).

### The indicator setup

| Indicator | Setting | Role for a 2–7 day swing |
|---|---|---|
| EMA9 | 9 | immediate momentum / very short-term trend |
| EMA21 | 21 | primary swing trend |
| EMA50 | 50 | medium-term structure - secondary confirmation |
| EMA200 / EMA250 | 200 / 250 | long-term / ~1-year trend - **context only** |
| RSI | 6 (primary), 14 (secondary) | short-term momentum |
| MACD | 12, 26, 9 | momentum and, above all, its *change* |
| MAVOL | 20 | volume reference |
| BOLL | 20 periods, 1.8 standard deviations | volatility envelope; band walk vs. extension |
| ATR | 14 | volatility and swing risk |

The parameters are fixed in `src/technical/indicators.py` (`EMA_SPANS`,
`RSI_PERIODS`, `MACD_FAST/SLOW/SIGNAL`, `MAVOL_WINDOW`, `BOLL_WINDOW`,
`BOLL_STD`) and written to `analysis.json` with every run.

**Priority.** The indicators are not weighted equally. In order: (1) price
structure, (2) EMA9 / EMA21, (3) volume vs MAVOL20, (4) RSI6 / RSI14, (5) MACD
12/26/9, (6) BOLL (20, 1.8), (7) ATR / volatility, (8) support and resistance,
(9) relative strength vs the benchmark, (10) EMA50 / EMA200 / EMA250 context.
The report text follows this order, and the Swing Technical Condition gives the
most points to the top of the list (EMA200 / EMA250 get none).

**Descriptive, not a recommendation.** Every label comes from the fixed rules
below, so the same data always gives the same result. The program never
outputs Buy / Sell / Hold, price targets or probabilities. RSI > 70 is not read
as "sell" nor RSI < 30 as "buy"; touching the upper Bollinger Band is not
bearish by itself; a moving-average crossover is described, not treated as a
trade signal.

### Price basis

All indicators use **dividend-adjusted** daily Open / High / Low / Close:
every value of a day is multiplied by `adj_close / close` of that day. On the
latest day the factor is 1, so the current price is the actual traded price.
Indicators are causal (day `t` only uses data up to day `t`) and are `n/a` until
enough history exists (EMA250 needs 250 sessions; the ATR percentile 126). With
fewer than 500 sessions of history the report notes that EMA200 / EMA250 still
depend a little on the first price in the data.

### Indicators

Notation: `C`, `H`, `L`, `V` = close, high, low, volume; `t` = today.

| Indicator | Formula |
|---|---|
| EMA9 / 21 / 50 / 200 / 250 | `EMA_t = a x C_t + (1 - a) x EMA_(t-1)`, `a = 2 / (n + 1)`, started at the first close; the first `n - 1` values are n/a |
| EMA slope (9, 21, 50) | `EMA_t / EMA_(t-3) - 1` (change over the last 3 sessions) |
| RSI6 (primary), RSI14 | Wilder: `gain = max(C_t - C_(t-1), 0)`, `loss = max(C_(t-1) - C_t, 0)`; first average = simple mean of `n` values, then `avg_t = (avg_(t-1) x (n - 1) + x_t) / n`; `RSI = 100 - 100 / (1 + avgGain / avgLoss)` (100 with no losses, 50 with no movement). Reported: RSI6 now and its 1 / 3 / 5-day change; RSI14 now and its 5-day change |
| MACD 12/26/9 | `DIF = EMA12 - EMA26`, `DEA (signal) = EMA9 of DIF`, `Histogram = DIF - DEA`; reported with the histogram's 1-day and 3-day change |
| MAVOL20 | average volume of the last 20 sessions **including today** (as drawn on charting platforms) |
| Volume ratio | `V_t / MAVOL20_t`; secondary: 5-day average volume / MAVOL20 |
| On-balance volume (OBV) - secondary | `OBV_t = OBV_(t-1) + sign(C_t - C_(t-1)) x V_t`; slope over 10 / 20 sessions = least-squares slope / average volume of the window |
| BOLL (20, 1.8) | `Middle = SMA20`, `Upper / Lower = Middle +/- 1.8 x std(C, last 20 days)`; `%B = (C - Lower) / (Upper - Lower)`; `Width = (Upper - Lower) / Middle`; width change = `W_t / W_(t-5) - 1`. `std` is the population standard deviation (Bollinger's definition, as on TradingView); a platform that uses the sample standard deviation draws bands about 2.6% wider |
| Band-width percentile | share of the previous 126 sessions with a width <= today's (needs 60) |
| True range, ATR(14) | `TR = max(H - L, abs(H - C_(t-1)), abs(L - C_(t-1)))`; `ATR` = Wilder average of TR over 14 days; `ATR% = ATR / C x 100` |
| ATR% percentile | share of the previous 252 sessions with an ATR% <= today's (needs 126) |
| Realised volatility | sample std of daily returns over 20 / 60 days x `sqrt(252)` |
| Returns / ROC | `C_t / C_(t-n) - 1` for n = 2, 3, 5, 7, 10, 20; `ROC(n)` = the same value x 100 |
| 20D / 60D high and low | highest High / lowest Low of the last 20 / 60 sessions (including today). The **previous 20D high / low** (the 20 sessions before today) is the breakout / breakdown reference |
| Relative return vs benchmark | `(C_t / C_(t-n) - 1) - (B_t / B_(t-n) - 1)` for n = 2, 5, 7, 20 |
| Swing risk range | `C +/- 1, 1.5 and 2 ATR` - a volatility reference around the price, **not a target** |
| Secondary references | VWAP20 = `sum(TP x V) / sum(V)` over 20 sessions, `TP = (H + L + C) / 3` (an approximation from daily bars); ADX(14) (Wilder) as trend-strength context |

### Classification rules

Rules are checked from the top; the first one that is true gives the label.
A label whose inputs are missing is `n/a`.

**Swing trend (EMA9 / EMA21)** - slopes are the 3-session EMA slopes

| Label | Rule |
|---|---|
| Strong short-term uptrend | `C > EMA9 > EMA21`, EMA9 slope > 0 and EMA21 slope > 0 |
| Strong short-term downtrend | `C < EMA9 < EMA21`, EMA9 slope < 0 and EMA21 slope < 0 |
| Short-term uptrend | `EMA9 > EMA21`, `C > EMA21` and EMA21 slope > 0 |
| Short-term downtrend | `EMA9 < EMA21`, `C < EMA21` and EMA21 slope < 0 |
| Improving | `C > EMA9` and EMA9 slope > 0 (the fast EMA turns first) |
| Deteriorating | `C < EMA9` and EMA9 slope < 0 |
| Mixed | anything else |

EMA50 is secondary confirmation: *Price above rising EMA50* (`C > EMA50`, slope
> 0), *Price below falling EMA50* (`C < EMA50`, slope < 0), otherwise *Mixed*.
EMA200 / EMA250 are only reported as distances (context).

**RSI6 zone:** `>= 80` Extremely strong / extended · `70-80` Strong momentum ·
`60-70` Positive momentum · `50-60` Mild positive momentum · `40-50` Mild negative
momentum · `30-40` Weak momentum · `< 30` Oversold (a value on a boundary belongs
to the higher zone). RSI6 / RSI14 relationships: RSI6 above / below RSI14; RSI6
**crossed** above / below RSI14 within the last 3 sessions; **recovering from
oversold** (RSI6 was < 30 in the previous 5 sessions and is now at least 10 points
above that low); **losing momentum from an extended level** (RSI6 was >= 80 in the
previous 5 sessions and is now at least 10 points below that high). With RSI6
>= 70 or < 30 the report adds one sentence reading it together with the trend,
structure and volume.

**MACD 12/26/9** (`H` = histogram, `d1` / `d3` = its change over 1 / 3 sessions;
the change is checked first):

| Label | Rule |
|---|---|
| Improving | `d1 > 0` and `d3 > 0` |
| Weakening | `d1 < 0` and `d3 < 0` |
| Positive | otherwise, DIF at or above DEA (`H >= 0`) |
| Negative | otherwise, DIF below DEA (`H < 0`) |

Details reported with it: **bullish / bearish crossover** (DIF crossed DEA within
the last 3 sessions, with how many sessions ago); **histogram expanding /
contracting** (`abs(H)` larger / smaller than 3 sessions ago, same sign); **MACD
below zero but improving** (DIF < 0 and `H` higher than 3 sessions ago); **MACD
above zero but deteriorating** (DIF > 0 and `H` lower than 3 sessions ago).

**Volume vs MAVOL20** - always read together with the day's direction (close vs
previous close): `>= 2.0x` Very high participation · `1.5-2.0x` Strong
participation · `0.8-1.5x` Normal · `< 0.8x` Weak participation. Heavy volume on
an up day is described as buying pressure confirmed by volume; heavy volume on a
down day as **selling pressure, not confirmation of strength**. OBV (and a
divergence between OBV and the 5-day move) is reported as a secondary sentence.

**Short-term momentum (returns)** (`P` = how many of the 2D, 3D, 5D, 7D returns
are positive; `M5 = (C_t - C_(t-5)) / ATR`): Strong positive `P = 4` and `M5 >= 1.5` ·
Strong negative `P = 0` and `M5 <= -1.5` · Positive `P >= 3` and `M5 > 0` ·
Negative `P <= 1` and `M5 < 0` · otherwise Mixed.

**BOLL (20, 1.8)**

| Measure | Rule |
|---|---|
| Position (`%B`) | `> 1` Above upper band · `>= 0.9` Near upper band · `> 0.6` Between middle and upper band · `>= 0.4` Near middle band · `> 0.1` Between middle and lower band · `>= 0` Near lower band · `< 0` Below lower band |
| Band width | width change `>= +20%` and the width percentile was <= 20 in the last 10 sessions: Expansion after squeeze · `>= +20%`: Band expansion · percentile <= 20: Volatility squeeze · `<= -15%`: Band contraction · otherwise Stable |
| Upper-band walk (strong momentum) | `%B >= 0.9` today and on at least 3 of the last 5 sessions, `C > EMA9 > EMA21` and EMA21 rising |
| Close above the upper band with trend and volume | `%B > 1`, `EMA9 > EMA21`, EMA21 rising and volume >= MAVOL20 (not a walk yet) |
| Extension above the upper band without trend / volume confirmation | `%B > 1` and none of the above |
| Lower band | the mirror rules: lower-band walk (strong downside momentum), close below the lower band with trend and volume, drop below the lower band without confirmation |

**Volatility and context**

| Measure | Rule |
|---|---|
| Volatility regime (ATR% percentile) | `>= 95` Extreme · `>= 80` High · `>= 60` Elevated · `>= 20` Normal · `< 20` Low |
| VWAP20 position (secondary) | `(C - VWAP20) / ATR`: > 0.5 Above, < -0.5 Below, otherwise Near |
| ADX(14) (context only) | `>= 25` Trending environment · `< 20` Weak / range-bound environment · otherwise Borderline |

**Relative strength** (5D and 7D matter most): `score = (rel5 / (s x sqrt(5)) +
rel7 / (s x sqrt(7))) / 2`, where `s` = std of daily (stock - benchmark) returns
over the previous 60 sessions. `>= 1.25` Strongly outperforming · `>= 0.5`
Outperforming · `> -0.5` Neutral · `> -1.25` Underperforming · otherwise Strongly
underperforming.

**Structure states** - descriptions of the chart, not signals. All true states
are listed; the first in this order is the "primary" structure.

| State | Rule |
|---|---|
| Breakout with volume confirmation | `C >` previous 20D high and volume >= 1.5x MAVOL20 |
| Breakout on normal volume | `C >` previous 20D high and volume 0.8-1.5x MAVOL20 |
| Breakout without volume confirmation | `C >` previous 20D high and volume < 0.8x MAVOL20 |
| Short-term breakdown with heavy volume / Short-term breakdown | `C <` previous 20D low with volume >= 1.5x MAVOL20 / otherwise |
| Price losing EMA21 / Price reclaiming EMA21 | the close crossed below / above EMA21 today (it was on the other side yesterday) |
| EMA9 crossed below EMA21 / EMA9 crossed above EMA21 | within the last 3 sessions |
| Pullback to EMA9 / Pullback to EMA21 | a pullback within a swing uptrend (next row) with `C` within 0.5 ATR of EMA9 / EMA21 (the nearer one) |
| Pullback within swing uptrend | `EMA9 > EMA21`, EMA21 rising, 3D return < 0, `C > EMA21` (and, when not near either EMA, `C < EMA9`) |
| Price losing EMA9 / Price reclaiming EMA9 | the close crossed below / above EMA9 today |
| Near breakout | not a breakout and `C` within 1 ATR below the previous 20D high |
| Momentum deterioration | `EMA9 > EMA21`, RSI6 fell >= 20 points in 3 sessions and the MACD histogram is below its level 3 sessions ago |
| EMA9 / EMA21 compression | `abs(EMA9 - EMA21) <= 0.3 ATR` |
| Volatility squeeze | band-width percentile <= 20 |
| Range / consolidation | `20D high - 20D low <= 4 ATR` and ADX < 20 |
| Price > EMA9 > EMA21 / Price < EMA9 < EMA21 | the stacked order (lowest priority) |

**Swing Technical Condition** - a transparent points score (-12 ... +12) whose
weights follow the priority list; the report shows every component:

| Component (priority) | Points |
|---|---|
| 1. Price structure (primary state) | breakout with volume confirmation +2 · breakout on normal volume +1 · reclaiming EMA21 +1 · EMA9 crossed above EMA21 +1 · losing EMA21 -1 · EMA9 crossed below EMA21 -1 · momentum deterioration -1 · breakdown -1 · breakdown with heavy volume -2 · any other state 0 |
| 2. Swing trend (EMA9 / EMA21) | strong uptrend +3 · uptrend +2 · improving +1 · mixed 0 · deteriorating -1 · downtrend -2 · strong downtrend -3 |
| 3. Volume vs MAVOL20 | `>= 2.0x`: +2 on an up day / -2 on a down day · `1.5-2.0x`: +1 / -1 · otherwise 0 |
| 4. RSI6 / RSI14 | RSI6 >= 60 and RSI6 >= RSI14: +1 · RSI6 < 40 and RSI6 <= RSI14: -1 · otherwise 0 |
| 5. MACD 12/26/9 | improving +1 · weakening -1 · positive / negative 0 |
| 6. BOLL (20, 1.8) | upper-band walk +1 · lower-band walk -1 · otherwise 0 |
| 9. Relative strength | (strongly) outperforming +1 · neutral / n/a 0 · (strongly) underperforming -1 |
| 10. EMA50 | price above rising EMA50 +1 · below falling EMA50 -1 · otherwise 0 |
| EMA200 / EMA250, ATR, support / resistance | 0 - context, risk and location are reported separately |

**Strong** = score >= 6 *and* an uptrend label · **Weak** = score <= -6 *and* a
downtrend label · **Improving** = score >= 2 · **Weakening** = score <= -2 ·
**Mixed** = anything else. It summarises the labels above; it is not a forecast.

### Support, resistance and recent range

* **Candidates:** swing highs and swing lows of the last `structure_lookback_days`
  (60) sessions - a High / Low that is the highest / lowest of `pivot_bars` (3)
  sessions on each side (the last 3 sessions cannot be confirmed yet) - plus the
  20D and 60D high and low.
* **Resistance** = candidates more than 0.1 ATR above the close, nearest first;
  **support** = candidates more than 0.1 ATR below it. So an old swing low above
  the price acts as resistance and an old swing high below it as support.
* **No duplicates:** a candidate within 0.5 ATR of a level already chosen is
  merged into it (its origin, and its price if different, are shown next to the
  level). Two levels per side (R1, R2, S1, S2).
* Distances: `level / C - 1` and `(level - C) / ATR`.
* The report also lists the 20D / 60D high and low, EMA9, EMA21, EMA50 and VWAP20
  with their distances and marks: **Within 1 ATR** of the 20D high / low, a
  **close above the previous 20D high** / **below the previous 20D low**, and
  **Near** (within 0.5 ATR) EMA9, EMA21, EMA50 or VWAP20.

### Earnings event risk

The next earnings release after the last session in the data is taken from the
provider's earnings calendar (the same calendar used for event evidence). The
first session that can react is the release day, or the next weekday for a
release at or after 16:00 New York time (weekends roll to Monday). Trading days
are counted Monday-Friday, so exchange holidays are not removed and the count
can be one day too high around holidays. If that session is within
`event_risk_days` (7), both reports show **EVENT RISK WITHIN TYPICAL HOLDING
WINDOW**. Without a calendar the report says *Earnings date unknown*; dates are
never guessed.

### Historical 2–7 day context

For the conditions that describe **today**, the program finds every earlier day
in the stock's own history that met them and shows what happened next:

1. **Same structure** - today's primary structure state (if there is one);
2. **Same swing trend and RSI6 zone**;
3. **All trading days (baseline)** - compare the groups with this row.

For 2, 3, 5 and 7 days: average, median, % positive and average return vs the
benchmark; for 3, 5 and 7 days also the average and median **MFE / MAE**
(maximum favourable / adverse excursion):

| Quantity | Formula (adjusted prices) |
|---|---|
| Forward return | `C_(t+h) / C_t - 1` for h = 1, 2, 3, 5, 7, 20 |
| MFE (h = 3, 5, 7) | `max(H_(t+1) ... H_(t+h)) / C_t - 1` - the best price reached |
| MAE (h = 3, 5, 7) | `min(L_(t+1) ... L_(t+h)) / C_t - 1` - the worst price reached |

Only days with a complete 7-day future are used, so N is the same for every
column. **N and the number of separate episodes are always shown** (consecutive
days share most of their future, so N overstates the independent evidence).
Fewer than `min_history_samples` (20) days are marked *small sample: treat as
anecdotal*; fewer than 5 show no statistics. No significance test is applied;
these are descriptions of the past, not predictions or probabilities. Every
stored event also gets the 2/3/7-day forward returns and 3/5/7-day MFE / MAE.

### `technical_analysis.md` layout

| Part | Contents | Length |
|---|---|---|
| Header | data date, basis, benchmark, indicator settings, disclaimer, event-risk banner, data notes | |
| Snapshot | Price, EMA9, EMA21, EMA50, EMA200, EMA250, RSI6, RSI14, MACD 12/26/9, Volume (x MAVOL20), BOLL (20, 1.8), ATR(14), Relative Strength 5D, Nearest Resistance, Nearest Support, Event risk, Swing Condition (with its points) | 17 rows |
| 1. Short-Term Trend | EMA9 / EMA21 trend and events, EMA50, EMA200 / EMA250 context, ADX | <= 4 sentences |
| 2. Momentum | RSI6 (1/3/5-day change) and RSI14, MACD 12/26/9 with its details, 2-20 day returns | <= 4 sentences |
| 3. Volume & Participation | volume vs MAVOL20 with the day's direction, OBV (secondary) | <= 3 sentences |
| 4. Volatility & Swing Risk | ATR, ATR%, percentile, realised volatility, BOLL position / width / walk; ATR range table | <= 4 sentences |
| 5. Relative Strength | 2D/5D/7D/20D relative returns and label | <= 3 sentences |
| 6. Price Structure | R2 / R1 / Current / S1 / S2 table, reference table, detected structure | |
| 7. 2–7 Day Historical Context | the groups above | |
| 8. Swing Summary | EMA9 / EMA21, momentum (RSI6 / RSI14, MACD), volume vs MAVOL20, BOLL position, volatility, nearest levels, relative strength, event risk; ends with **Swing Technical Condition: Strong / Improving / Mixed / Weakening / Weak** | <= 150 words |

### `technical_chart.png`

About 6 months (`chart_days`, 126 sessions), one slot per trading day:
(1) daily candles with EMA9 (blue), EMA21 (orange) and EMA50 (violet), EMA200 /
EMA250 as thin grey context lines, BOLL (20, 1.8) upper / middle / lower and the
nearest support / resistance labelled on the right - when EMA200 / EMA250 are
far from the price they are not allowed to squash the chart; their value is
printed at the edge instead; (2) RSI6 (bold) and RSI14 with 70 / 50 / 30 lines;
(3) MACD 12/26/9: DIF, DEA and a blue / red histogram with the zero line;
(4) volume coloured by up / down day with MAVOL20. Event markers are left out
to keep it readable (`price_chart.png` has the events).

### Settings

```yaml
technical:
  chart_days: 126               # sessions shown in technical_chart.png
  structure_lookback_days: 60   # support / resistance only from this many recent sessions
  pivot_bars: 3                 # sessions on each side of a swing high / low
  event_risk_days: 7            # earnings within this many trading days = event risk
  min_history_samples: 20       # fewer comparable days -> "small sample"
```

---

## 9. The database

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
| `events` | ticker + event day | `(ticker, event_date)` | all event statistics, forward returns (1-20 days), MFE / MAE, research fields, run ids |
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
* **Older databases are upgraded automatically:** a database created before the
  swing columns existed gets the missing columns (2/3/7-day forward returns, MFE,
  MAE) added on the next run, without losing any data; they fill in as events
  are refreshed.

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

## 10. Recording your own research

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

## 11. Studying events across many stocks

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

## 12. Fundamental data

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

## 13. Reliability and error handling

| Situation | What happens |
|---|---|
| Invalid ticker format (e.g. `AV AV`) | Rejected before any download (exit code 2) |
| Unknown / delisted ticker | Clear message ("may be invalid or delisted"), not retried - Yahoo's "no data" answer and its HTTP 404 "Not Found" are both recognised, so a mistyped ticker is never reported as a network problem (and a network problem never as a bad ticker); no empty report folder left behind (exit code 3) |
| Network or API failure | Tried up to 3 times with increasing waits; then the **cached prices in the database** are used with a warning, or the run stops with a clear message if nothing is cached |
| Benchmark / fundamentals / earnings calendar unavailable | Run continues; noted under *Data Limitations* (stored fundamentals are used if available) |
| Missing or bad price rows | Removed and reported (missing/negative close, duplicate dates); suspicious values (high < low, long gaps, zero volume) are reported but not altered |
| Not enough history for a statistic | The statistic is `n/a`; it is never computed from too little data |
| A chart fails | The rest of the report is still written; the failure is noted |
| Swing technical analysis fails | `summary.md` and every other output are still written; the failure is noted |
| Earnings calendar unavailable | Swing event risk is shown as *Earnings date unknown* (never guessed) |
| Running twice | No duplicate rows (see section 9) |
| Stale data | A warning if the latest price is more than 7 days old |

Every run writes `run.log` in its report folder, and every run (including failed
ones and their error message) is recorded in the `analysis_runs` table.

---

## 14. Tests

```bash
python -m pytest            # all tests
python -m pytest -v tests/test_statistics.py   # one file, verbose
```

The tests need no internet. They cover:

| File | What is tested |
|---|---|
| `test_statistics.py` | daily/log/multi-day returns, Z-scores (baseline excludes the day itself, full-window rule, zero-variance case), volume ratio and Z-score, volatility annualisation, moving averages, forward returns (1/2/3/5/7/20 days), MFE / MAE (incl. negative MFE, dividend adjustment), drawdowns, summary statistics |
| `test_events.py` | planted events are found, rule combinations and thresholds, no events without history, default Unknown fields, spacing between events, earnings-date matching (after-close, pre-market, weekend releases) |
| `test_benchmark.py` | abnormal returns, date alignment, relative performance, beta = 2 for a 2x stock |
| `test_database.py` | table creation, insertion, duplicate handling, protection of manual research, forward-return refresh, upgrade of an older database without data loss, research CSV import |
| `test_prices.py` | yfinance and CSV formats, cleaning and validation warnings |
| `test_fundamentals.py` | no invented values, calculated margins and growth, failures handled |
| `test_provider.py` | retries on network errors, invalid tickers (incl. HTTP 404) not retried |
| `test_charts.py` | readable axis labels on edge cases: low-volatility stocks, very short histories, undefined statistics, extreme thresholds |
| `test_pipeline.py` | the complete workflow with a fake provider: every output file (incl. `technical_analysis.md`, `technical_chart.png`, the summary snapshot and the JSON block), database contents, re-runs, invalid tickers, network failure + cache fallback, offline mode, earnings event-risk banner, a failing technical step not stopping the report, the CLI |
| `test_technical.py` | the exact indicator settings, and every indicator against an independent step-by-step calculation (EMA9/21/50/200/250, RSI6/14, MACD 12/26/9, BOLL 20/1.8, MAVOL20 and volume ratio, ATR, rolling VWAP, OBV, 20D/60D highs and lows, relative strength), causality, every classification rule at its thresholds (EMA9/EMA21 trend, RSI6 zones and RSI6/RSI14 relationships, MACD states and crossovers, volume with price direction, BOLL position, squeeze, band walk vs extension), all 21 structure states, the priority-weighted Swing Technical Condition, support / resistance (nearest levels, merged duplicates, polarity, lookback, unconfirmed pivots), earnings event-risk counting, historical context (N, episodes, small samples), report rules (section order, sentence limits, <= 150-word summary incl. the longest case, condition line, no Buy / Sell / Hold, no obsolete indicators, event-risk banner), chart panels, legends and y-range, settings validation |

---

## 15. Limitations

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
* **Swing technical analysis:** daily bars only, so VWAP20 is an approximation
  and nothing is known about intraday levels or the open of the next session.
  Indicator values can differ slightly from a charting platform: EMAs start at
  the first price of the downloaded history (EMA200 / EMA250 settle only after a
  few hundred sessions), BOLL uses the population standard deviation, and
  MAVOL20 includes the current session.
  Classification thresholds are fixed, documented rules of thumb - not fitted or
  optimised - and different reasonable thresholds would label some days
  differently. Support / resistance come from a simple swing-point method. The
  2-7 day historical context uses one stock's own history: groups are often
  small, days overlap, and there is no significance test, so it describes the
  past and says nothing reliable about the next few days. Event-risk day counts
  ignore exchange holidays and depend on the provider's earnings calendar.

---

## 16. Possible future upgrades

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

## 17. How to read the code (learning path)

1. `analyze.py` - how command-line options are read.
2. `src/pipeline.py` - the whole workflow in order; each numbered step calls one module.
3. `src/analysis/statistics.py` - the core maths, with each formula written above its code.
4. `src/events/detector.py` - how rules become events.
5. `src/database/schema.sql` then `src/database/db.py` - how data is stored without duplicates.
6. `tests/test_statistics.py` - small hand-checked examples of every formula; a
   good way to see exactly what each function returns.
7. `src/technical/indicators.py`, then `classification.py` - the swing indicators
   and the exact rules behind every label; `tests/test_technical.py` shows each
   rule on hand-made examples.

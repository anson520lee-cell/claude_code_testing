# Quantitative Swing Research Terminal

A daily, automated **research and decision-support** platform for US swing trading
(1–10 trading days). After every completed US session it updates market data, checks
data quality, classifies the market regime, scans a stock universe for swing setups,
builds a risk plan for each, deep-dives the shortlist (fundamentals, options,
positioning, catalysts), ranks candidates with a transparent score, and publishes a
report and a web dashboard.

> **It never trades.** No broker connection, no order submission, no automated execution.
> Every number is a research input for your own decision.

---

## 1. Quick start

```bash
pip install -r requirements.txt
```

```bash
export SEC_USER_AGENT="Your Name you@example.com"   # required by SEC EDGAR's fair-access policy
```

```bash
python daily_run.py          # first run downloads ~2 years of history (2-4 minutes)
```

```bash
python serve.py              # dashboard at http://127.0.0.1:8000
```

Outputs per session: `reports/daily/<YYYY-MM-DD>/report.html` (printable), `report.md`, `data.json`
(everything the dashboard shows). All history accumulates in `data/stock_research.db`.

### macOS: one-click app in the Dock

Everything runs on your own Mac (the dashboard is a local web page, `http://127.0.0.1:8000`).
One-time setup in Terminal (needs Python 3.10+; macOS's built-in `python3` may be older - install from
python.org or `brew install python@3.12`):

```bash
git clone https://github.com/anson520lee-cell/claude_code_testing.git ~/swing-terminal
```

```bash
cd ~/swing-terminal && bash scripts/install_mac.sh
```

The installer creates `.venv`, installs the dependencies, asks for your SEC contact (saved in `.env`,
never committed), builds `~/Applications/Swing Terminal.app`, optionally adds it to the Dock, optionally
installs an hourly background job (`launchd`, runs the analysis once per new US session) and runs the
first data download.

Clicking **Swing Terminal** in the Dock starts the dashboard server if needed, starts a background data
update when a newer session has closed (the page shows progress and refreshes when done) and opens
the browser. `bash scripts/stop_terminal.sh` stops the server. Logs: `data/server.log`, `data/daily_run.log`.
Only one analysis can run at a time (file lock), whichever of Dock, button or schedule starts it.

### Scheduling (after the close)

The run identifies the latest *completed* session itself (NYSE calendar incl. holidays, early
closes and a 30-minute settle time), so it is safe to schedule it at any time after the close.
`--if-new` makes repeated triggers free:

```bash
# crontab, server clock in UTC: 21:45 UTC = 16:45 EST / 17:45 EDT, Monday-Friday
45 21 * * 1-5  cd /path/to/repo && .venv/bin/python daily_run.py --if-new >> data/daily_run.log 2>&1
```

| Option | Meaning |
|---|---|
| `--session YYYY-MM-DD` | analyse a specific past session |
| `--offline` | no downloads; re-analyse stored data |
| `--skip-deep` | technical scan only (no fundamentals / options / positioning / catalysts) |
| `--if-new` | exit immediately if the session was already processed successfully |
| `--config FILE` | alternative to `config/terminal.yaml` |

Exit codes: 0 success, 2 settings error, 3 no market data, 1 unexpected error.

---

### Interface language and focus

The dashboard and the daily report are in English (readable labels for internal keys live in
`src/i18n/zh.py`; the methodology shown on the Research page comes straight from the module docstrings). The universe is focused on AI / semiconductors / high-growth / high-beta names
plus leveraged ETFs (SOXL, TQQQ, NVDL, TSLL, LABU). Every run applies a focus filter (`universe_filter` in
`config/terminal.yaml`): 120-session beta vs SPY ≥ 1.3, 60-session annualised volatility ≥ 35%, market cap
≥ US$2B (leveraged ETFs exempt; caps cached weekly) and 20-day average dollar volume ≥ US$100M. Stocks that
fail a rule are hidden from candidates, the universe table and the setup statistics, and listed with the
reason on the Data page.

## 2. Daily workflow

```
latest completed session ─► batch price update ─► data-quality gate ─► market regime + sectors
      ─► features + setups (whole universe) ─► historical simulation of every setup
      ─► stage-1 technical score (+ liquidity filter) ─► shortlist (20 long + 20 short)
      ─► stage-2 deep dive: SEC fundamentals · options · positioning · catalysts
      ─► penalties + supporting / contradicting evidence ─► final ranking ─► DB + reports + dashboard
```

| Step | Module |
|---|---|
| Trading calendar, latest completed session | `src/market/calendar.py` |
| Batch download, incremental refresh, corporate-action detection | `src/data/market_data.py` |
| Data-quality checks | `src/data/quality.py` |
| Market regime, sector rotation, index table | `src/market/regime.py` |
| Features (reuses `src/technical` indicators) | `src/scanner/features.py` |
| Setup rules | `src/scanner/setups.py` |
| Risk plan (entry / stop / targets) | `src/scanner/plan.py` |
| Historical simulation | `src/scanner/backtest.py` |
| Composite score, penalties, evidence | `src/scanner/scoring.py` |
| Point-in-time fundamentals (SEC XBRL; Yahoo fallback) | `src/fundamentals/sec_fundamentals.py`, `src/data/sec.py` |
| Options analysis | `src/derivatives/options.py` |
| Positioning & ownership (short interest, 13F, Form 4, FINRA) | `src/positioning/positioning.py` |
| Catalysts (earnings, SEC filings, analyst actions, macro) | `src/catalysts/catalysts.py` |
| Orchestration, storage, reports | `src/daily/pipeline.py`, `store.py`, `report.py`, `enrichment.py` |
| Dashboard API + single-page app | `src/web/app.py`, `src/web/static/` |

Every rule is written out in the module docstrings; the dashboard's **Research** tab shows them verbatim.

---

## 3. Data sources and their limits

| Data | Source | Timing / caveat |
|---|---|---|
| Daily OHLCV (stocks, ETFs, VIX, VIX3M) | Yahoo Finance via `yfinance` (unofficial) | End of day; adjusted for splits / dividends |
| Treasury yields (2Y, 10Y, 3M) | FRED (H.15) | ~1 business day lag |
| Fundamentals | SEC EDGAR XBRL company facts | Point-in-time: period end **and** filing date kept |
| Fundamentals fallback | Yahoo quarterly statements | Filing date **estimated** as quarter end + 45 days |
| SEC filings (8-K items, S-3, 424B, 13D…) | SEC EDGAR submissions | Acceptance timestamp kept |
| Earnings dates, EPS surprises, analyst actions | Yahoo Finance | Unconfirmed dates flagged "Estimated" |
| Options chains | Yahoo Finance | End of day; OI is prior-day; IV unreliable on illiquid strikes |
| Short interest, float, holders % | Yahoo Finance (FINRA settlement data) | 1–4 weeks old; date shown |
| 13F institutional holders | Yahoo Finance | Up to ~4.5 months old, long positions only |
| Insider transactions | Yahoo Finance (SEC Form 4) | Within 2 business days |
| Short-sale volume | FINRA Reg SHO daily file | Mostly market-maker activity; **not** short interest |
| Off-exchange volume | FINRA OTC Transparency (weekly) | Published 2–4 weeks after the week |
| Block trades, FDA calendar, contract awards | — | No free reliable source: shown as **not available** |

Without `SEC_USER_AGENT`, SEC requests are not sent (EDGAR refuses anonymous clients with HTTP 403):
fundamentals fall back to Yahoo and SEC filing / dilution checks are missing. The run and the
dashboard banner say so explicitly.

---

## 4. Data engineering

* **Incremental, idempotent updates.** First run: 2 years per symbol. Later runs: only the last
  15 calendar days, upserted on `(ticker, date)`. Running twice changes nothing.
* **Corporate actions.** Yahoo re-adjusts history after splits / dividends. Each refresh compares the
  overlap with stored bars; a mismatch > 0.5% triggers a full re-download of that symbol, so adjusted
  and unadjusted prices are never mixed.
* **No partial bars.** Bars dated after the target session (a run during market hours) are discarded.
* **Provider lag.** If SPY has no bar for the target session yet, the run analyses the latest
  published session and says so.
* **Quality gate** per symbol: stale last bar, missing sessions vs the exchange calendar, too little
  history, inconsistent OHLC, zero volume, frozen quotes, > 50% daily jumps. Stale / broken symbols
  are excluded from scoring; warnings cost score points.
* **Schema upgrades are non-destructive.** The terminal adds tables (`terminal_runs`, `regime_history`,
  `scan_candidates`, `options_snapshots`, `catalyst_events`, `data_quality`) next to the existing ones.
  Before the first upgrade of an existing database a copy is written to `data/backups/`.

---

## 5. Market regime

Six components, each scored −2…+2 by explicit formulas, weighted into a composite −100…+100:

| Component | Weight | Inputs |
|---|---|---|
| Index trend | 25 | SPY/QQQ/IWM/DIA vs EMA21/EMA50, EMA50 vs EMA200, EMA21 slope |
| Index momentum | 15 | 20-day return of SPY and QQQ in units of their own volatility |
| Breadth | 20 | % of universe above EMA50 / EMA200, net 20-day highs − lows (universe proxy) |
| Volatility | 20 | VIX level, VIX/VIX3M term structure, 5-day VIX spike |
| Risk appetite | 15 | HYG−IEF (credit), offensive − defensive sectors, IWM − SPY |
| Rates | 5 | 10-year yield change over 20 sessions (FRED) |

Labels: ≥ +45 Strong Bullish · ≥ +15 Bullish · > −15 Neutral · > −45 Bearish · else Strong Bearish
(Strong labels require SPY on the matching side of its EMA50). Each label maps to written guidance
and an illustrative risk budget for longs / shorts. The regime is computed for every date, so the
dashboard shows its history and the historical simulation reports results *by regime*.

Sector rotation uses a transparent relative-rotation analogue: RS = ETF / SPY,
RS-trend = RS / SMA50(RS) − 1, RS-momentum = 10-session change of RS-trend →
Leading / Weakening / Lagging / Improving.

---

## 6. Setups, plans and scores

**Setups** combine trend context, location, volume and momentum (never a lone crossover):
long — breakout, pullback to EMA9/21, volatility squeeze near highs (conditional entry), EMA21 reclaim,
momentum leader near highs; short — breakdown, failed rally into a falling EMA21, EMA21 loss.

**Risk plan** (identical in today's report and in the historical simulation): structural stop clamped to
1–3 ATR, Target 1 = nearest 20/60/252-session high (low for shorts) at least 0.75 ATR away (else 2 ATR),
Target 2 = max(T1 + 1 ATR, 2R), 7-session time stop.

**Composite score** = Σ weight × sub-score / 100 − penalties (0–100), weights in `config/terminal.yaml`:
trend 20, momentum 15, relative strength 15, volume 10, setup quality 15, reward/risk 10, liquidity 5,
regime alignment 10. Penalties: earnings inside the holding window, extension > 3 ATR from EMA21, extreme
RSI6, recent dilution filing, financial-distress flags, reward/risk < 1.5, wide structural stop,
data-quality warning, options pricing an event premium. Tiers A ≥ 75, B ≥ 62, C ≥ 50, else Watch.

Every candidate lists **supporting** and **contradicting** evidence, so a high score never hides a
problem (e.g. earnings in 2 sessions, trading below EMA200, insider selling, crowded short).

---

## 7. Historical simulation — read this before trusting any number

Each setup's plan is simulated on the stored history of the whole universe: signal at the close,
entry at the next open (+ slippage), stop / T1 / time stop on daily bars (stop assumed first if both are
hit in one bar), slippage on exit. Results: N, win rate, average / median R, profit factor, target / stop
rates, MFE / MAE, forward returns — overall and by regime, compared with an all-days baseline.

Known biases: **survivorship** (today's universe), **in-sample** (rules and data overlap), overlapping
correlated signals, no commissions / borrow costs, daily bars only. On the first real run (2024–2026,
a strong bull market) most long setups were close to break-even in R after slippage and short setups
lost money — the terminal reports this as is. Treat the statistics as a sanity check of the rules,
not as expected performance.

### 7b. SL / TP validation backtest (`python backtest.py`)

A separate, deeper study of whether the stops and targets themselves are any good
(`src/backtest/research.py`, results on the dashboard's Research page, refreshed monthly by
`daily_run.py`, or on demand with the "Re-run backtest" button / `python backtest.py`). It downloads 5 years
of daily bars once.

* **Signals**: every setup episode (first active day) over the focus universe, with the live plan, the
  live score and its components, and ~20 raw indicators recorded at the signal date.
* **Grid**: 8 stops (live structural, 1.0-3.0 ATR, last swing, volume-profile node) x 8 targets (live
  resistance, 2-4 ATR, 2R, 3R, volume-profile node / vacuum top, 3-ATR chandelier trail) x 3 time stops
  (5 / 7 / 10 sessions), long and short.
* **Walk-forward honesty**: everything is chosen on the first 60% of signal dates (in-sample) and judged
  on the last 40% (out-of-sample), which never takes part in the selection.
* **Random-entry baseline**: the same exit rules applied to an entry every 5th session of every stock,
  with no setup. *Setup edge = setup result - baseline result*. If the baseline also makes money, the
  profit comes from market drift / the exit rule, not from the setups.
* **Accuracy diagnostics** of the live plan: outcome rates, "stopped then reached the target anyway",
  heat (MAE) taken by eventual winners, MFE in R, planned R:R vs actual hit rate vs break-even, score
  quintiles, per setup / regime x direction / quarter.
* **Factor study**: Spearman IC of each score component and raw indicator with the trade's R, in- and
  out-of-sample, plus an IC-weighted score fitted in-sample and checked out-of-sample by quintile.
* Options cannot be back-tested (no free historical implied volatility); survivorship bias applies.

First result (2021-2026, 233 stocks, 54k signals): the live plan has no out-of-sample edge (+0.00R,
t 0.2); targets are too far (every planned-R:R bucket hits below break-even); stops are not too tight
(7% stopped-then-target). The best long exit was 1.5 ATR stop + 3-ATR trail + 10 days (+0.17R OOS), but
random entries with the same exit earn +0.14R, so the setups add only ~+0.03R. Short setups lose in
this high-beta universe (best combination -0.05R OOS). The score has no predictive power
(Spearman -0.03); the planned-R:R component is *negatively* related to outcomes. The live plan is not
changed automatically - the dashboard shows the backtest's preferred plan next to every candidate.

### 7c. Crash rebound and spike fade (`python crash_study.py`)

An event study over the same 5-year history (`src/backtest/crash_study.py`). Crash events: 1-day drop
beyond 3σ, 1-day drop ≥ 8%, 3-day drop ≥ 15%, 5-day drop ≥ 20%, close ≥ 3 ATR below EMA21 (one per stock
per 10 sessions). Entries from the next day to +10 days, or after a first up day / a close above the prior
high; outcomes over 3-20 days and as 2-ATR-stop trades with close targets; everything compared with
random entries, in-sample vs out-of-sample. Spikes (the mirror image) test shorting after surges.

First result (233 stocks, ~10,600 crash events): buying 0-2 days after a 5-day drop ≥ 20% or a 3-day drop
≥ 15% beat random entries by about +2.5% over 5 days in and out of sample (win rate 55-58%); ≥ 3 ATR
below EMA21 with a 2-ATR stop and a 1.5-2 ATR target earned about +0.2R per trade out of sample (win
rate ~65%). The edge is gone by day 4-7. A stock's own "best entry day" did worse out of sample than
the universe-wide rule, so per-stock timing is not used (each stock page shows its own crash history).
Shorting after surges did not make money (only less than random shorts in a rising universe).

Consequences in the live terminal: candidate lists are **long only** (`scanner.directions`); a new
**crash rebound** setup (first crash in 10 sessions, entry window 0-2 days, stop 2 ATR, T1 1.5 ATR, T2
EMA21 capped at 3 ATR, 10-day time stop) is scored on long-term trend and how oversold it is and gets
a +15 bonus (`scanner.setup_bonus`); long Target 1 is capped at entry + 2 ATR (`max_target1_atr`).

---

## 8. Dashboard

| Tab | Contents |
|---|---|
| Market | regime gauge, component bars with evidence, regime history, indices / VIX / credit, breadth, rates, sector rotation map + table, top candidates, next events, cross-asset (DXY, WTI, gold, Bitcoin), advance / decline line + McClellan oscillator, industry-group RS ranking |
| Candidates | cards (or a table) with a mini K-line preview per stock, score bars, tiers, plan levels, R/R, RS percentile, earnings, evidence counts, penalties, setup history; filters |
| Ticker (click any row) | candlestick chart with EMA 9/21/50/200/250, BOLL(20, 1.8), plan levels and setup markers; volume + MAVOL20, RSI 6/14, MACD 12/26/9; thesis, evidence, score breakdown, simulation, risk plan, technical snapshot, S/R, fundamentals (quarters, flags), options (IV, expected move, OI by strike, unusual activity, GEX estimate), positioning, catalysts, candidate history |
| Universe | every screened stock, sortable |
| Catalysts | upcoming earnings, macro calendar, SEC filings, analyst actions |
| Portfolio | position sizing (account, risk %, position cap, heat cap; saved in the browser), position correlation matrix, total heat, effective number of bets, correlated clusters, price alerts, trade journal with R-based review |
| Research | how SL / TP are set, the SL / TP validation backtest (section 7b), setup simulation tables, full methodology |
| Data | quality checks, update summary, run log, report archive |

Ticker pages also show the **earnings reaction study** (options-implied move of the first expiry after
the next report vs the median absolute move over the last ~16 reports, gap / close / 5- and 10-day
drift per event, post-earnings-drift and gap-fade tendencies) and **SEC fails-to-deliver** (last ~month
of published half-month files: max / latest fails, max fails vs 20-day average volume, trend; needs
`SEC_USER_AGENT`; the data lag two to four weeks). Every risk plan has one-click alert buttons
(stop / target 1 / trigger).

**Industry groups**: each stock's Yahoo industry (cached 120 days) - groups with at least two members
ranked by the median RS score (0.5 x rel 20d + 0.3 x rel 60d + 0.2 x rel 5d), with the change in rank
over 5 sessions. **Advance / decline** is computed on the terminal's own universe, not the whole NYSE
(no free reliable source). **Cross-asset** series are context only and are not quality-gated (they trade
on other calendars; Bitcoin trades on weekends).

**Portfolio maths**: shares = account x risk % x regime budget / |entry - stop|, capped at the position
limit; heat = sum of dollar risk; position correlation uses 60-session daily returns with short positions
sign-flipped; effective bets = n^2 / sum of the correlation matrix. **Alerts** (`above`, `below`,
`close_above`, `close_below`) are checked against stored daily bars after every run and when the
dashboard opens; on macOS a triggered alert raises a system notification. The journal and alerts live in
the database (`trade_journal`, `price_alerts`). Nothing is ever sent to a broker.

The **Run** button starts the daily pipeline in the background (data download + analysis only).
Edit `config/macro_events.yaml` to add CPI / PCE / GDP dates; FOMC 2026 dates, monthly options
expiration and the (estimated) employment report are included.

---

## 9. Configuration

`config/terminal.yaml`: universe (ticker → sector / industry ETF; quote tickers YAML reads as booleans,
e.g. `"ON"`), market symbols, liquidity limits, scanner parameters, score weights (must sum to 100),
penalties and tiers. Settings are validated at start-up.

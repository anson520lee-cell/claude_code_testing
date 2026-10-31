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

---

## 8. Dashboard

| Tab | Contents |
|---|---|
| Market | regime gauge, component bars with evidence, regime history, indices / VIX / credit, breadth, rates, sector rotation map + table, top candidates, next events |
| Candidates | long / short tables with score bars, tiers, plan levels, R/R, RS percentile, earnings, evidence counts, penalties, setup history; filters |
| Ticker (click any row) | candlestick chart with EMA 9/21/50/200/250, BOLL(20, 1.8), plan levels and setup markers; volume + MAVOL20, RSI 6/14, MACD 12/26/9; thesis, evidence, score breakdown, simulation, risk plan, technical snapshot, S/R, fundamentals (quarters, flags), options (IV, expected move, OI by strike, unusual activity, GEX estimate), positioning, catalysts, candidate history |
| Universe | every screened stock, sortable |
| Catalysts | upcoming earnings, macro calendar, SEC filings, analyst actions |
| Research | setup simulation tables (overall and by regime), full methodology |
| Data | quality checks, update summary, run log, report archive |

The **Run** button starts the daily pipeline in the background (data download + analysis only).
Edit `config/macro_events.yaml` to add CPI / PCE / GDP dates; FOMC 2026 dates, monthly options
expiration and the (estimated) employment report are included.

---

## 9. Configuration

`config/terminal.yaml`: universe (ticker → sector / industry ETF; quote tickers YAML reads as booleans,
e.g. `"ON"`), market symbols, liquidity limits, scanner parameters, score weights (must sum to 100),
penalties and tiers. Settings are validated at start-up.

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

### Intraday prices every 30 minutes (display only)

`install_mac.sh` can also install a second `launchd` job (`com.swingterminal.intraday`) that runs
`intraday_run.py` every 5 minutes. The script works in New York time (daylight saving and the Mac's own
time zone do not matter) and only downloads when a new half-hour slot has started on a trading day:
9:30, 10:00, ... 16:00 (13:00 on early-close days). A slot missed while the Mac slept is fetched within
25 minutes, otherwise skipped. Each slot is one small yfinance batch of 5-minute bars for the latest
candidates, open journal trades, active price alerts and SPY / QQQ / IWM (at most 80 symbols; ~14
batches a day), written to `data/intraday/latest.json`. Log: `data/intraday.log`.

**It changes the displayed prices only.** Candidate cards and the stock page show "Live hh:mm ET" with
the change against the last stored close; scores, setups, plans, alerts and the price database are not
touched, and no provisional candle is added to the chart. The validated setups are after-close signals
entered at the next open, so they are still computed once per session by `daily_run.py --if-new` from the
final daily bar. Settings: `intraday:` in `config/terminal.yaml` (`enabled: false` turns it off);
`python intraday_run.py --force` fetches the latest slot of today by hand.

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

### 6a. Live strategy (agreed 2026-11-01)

What the daily scan does now (long only, never places orders). The Research page shows the same table
("Live strategy") with the live numbers.

| Setup | Entry | Exit (strategy lab) | Evidence |
|---|---|---|---|
| Crash rebound | **Signal day only** (`scanner.fresh_only_setups`): buy before that close or at the next open, not later | 1.5 ATR stop + 3-ATR trailing stop, max 20 sessions | +0.1 to +0.2R per trade over random entries (+0.08 to +0.17R with live-like sizing), positive in both halves, **t below 2.5: not proven** |
| Squeeze | Buy-stop at the 20-day high + 0.1 ATR, next session only (a market buy did worse than random) | 2 ATR stop + 3R target, max 20 sessions | +0.14R over random out of sample, −0.02R in sample: not proven |
| Breakout | Next open | 1 ATR stop + 3-ATR trailing stop, max 20 sessions | **Unproven**: +0.01 to +0.03R over random at every entry timing (bonus 0); kept by the user's choice |

* Live setups: `scanner.enabled_setups: [crash_rebound, squeeze, breakout]`. Momentum, pullback and reclaim are still
  simulated in the backtests but never make a candidate.
* Score bonus per setup = 150 points × its edge over random entries (−10..+20, 0 inside ±0.05R).
* CEO / CFO / President open-market buy ≥ $100k within 5 calendar days = +10 points (`scanner.insider_exec_buy`).
* One full entry; risk per trade `portfolio.risk_per_trade` of the account; shares computed after the close from
  |entry − stop| (Portfolio page).
* Tested and **not used**: scaling in (§7h), 2x leveraged ETFs (§7i), the Bullish-only regime filter for momentum /
  pullback (§7e), cross-asset lead-lag signals (no condition improved the three setups), shorting, the 183-indicator
  model.
* The backtest's 60–70% a year is mostly survivorship and a bull market (random entries +59%); the strategy's own
  edge is the excess over random entries.

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
**crash rebound** setup (first crash in 10 sessions) is scored on long-term trend and how oversold it is;
long Target 1 is capped at entry + 2 ATR (`max_target1_atr`). Since 2026-11-01 it is listed on the signal
day only and uses the strategy lab's exit (see §6a; the original 0-2 day window, 2 ATR stop, 1.5 ATR target
and 10-day time stop are only used by this study and the old SL/TP simulation).

### 7e. Strategy lab (`python strategy_lab.py`, refreshed monthly)

The question behind it: are the exits right, and does any weighting of ALL the indicators pick better trades?
Every long setup episode over 5 years (34,195) plus a random-entry sample (21,328 entries, every 10th session of every
stock) is tested; choices are made on the first 60% of dates and judged on the last 40%.

* **Exits (350 combinations per setup):** stops of 1-3 ATR, swing low or structural; targets of 1.5-4 ATR or 1.5-3R, or
  a 2 / 2.5 / 3-ATR trailing stop; time stops of 5-20 sessions. Result (2026-10-30): the old plan (structural stop,
  resistance target, 7 days) earned +0.05R out of sample; **1.5 ATR stop + 3-ATR trailing stop, up to 20 sessions earns
  +0.29R** (random entries with the same exit +0.25R). Squeeze uses 2 ATR + 3R target (+0.44R), breakout a 1 ATR stop.
  The live plan now uses each setup's chosen exit (`exit_rule` in the plan); trailing exits show no fixed target.
  Trailing exits win only 30-50% of the time: many small losses, fewer but larger winners.
* **Indicators (183 per trade):** every `build_features` column, distances to the moving averages / highs / lows in ATR,
  the 49 analyst techniques, the volume profile (position vs POC / VAH / VAL, dense-chip nodes, vacuum, holder profit
  ratio) and market context (`src/scanner/model_features.py`, shared with the live run). Two models are fitted
  walk-forward (sign-consistent IC weights; ridge regression) and must lift the top fifth by >= 0.05R with t >= 2 in
  most test years. Result: **neither passed** - the five score buckets earn about the same average R. Single
  indicators have consistent rank correlations of 0.10-0.15 (higher volatility, further below the 52-week high, more
  holders under water), but they shift the typical outcome, not the average profit. The candidate score is therefore
  context; the evidence-based part of the ranking is the setup bonus = the setup's edge over random entries with its
  exit (crash rebound +0.20R, squeeze +0.14R; breakout and reclaim about 0). Over the whole period an edge smaller
  than 0.05R either way (`setup_bonus_from_backtest.min_edge_r`) counts as noise and gives a bonus of 0: with the
  2026-10-30 results crash rebound +20, squeeze +8.2, and breakout, reclaim, momentum and pullback 0 (they track the
  market, so they neither lift nor sink a candidate).
* **Regime filter (momentum / pullback only in Bullish or Strong Bullish markets):** tested 2026-11-01, **not
  adopted**. Out of sample the kept regimes beat random entries (+0.06R momentum, +0.09R pullback), but in sample they
  were worse (-0.04R / -0.05R), so the filter fails the "both halves" rule. The dropped regimes lag random entries
  (random entries do very well after market falls), yet those trades still made money (+0.02R to +0.33R), so dropping
  them would cut profit. Shown on the Research page under "Only in Bullish markets?" (`regime_split` in the lab).
* Live candidates (user decision 2026-11-01): only crash rebound, squeeze and breakout (`scanner.enabled_setups`).
* The results of the 2026-10-30 run ship as `config/strategy_lab.default.json`, used until this machine's own monthly
  run writes `reports/backtest/<session>/strategy_lab.json`.
* Not testable: options positioning (no free option-chain history) and fundamentals over time. Survivorship bias
  applies (today's universe only).

### 7f. Strategy discovery (`python discover.py`, refreshed monthly)

Fourteen trading styles the terminal did not use, taken from published research and common practice (none designed on
this data): RSI(2) pullbacks, closes at the day's low (internal bar strength), three lower closes, idiosyncratic dips vs
the sector, VIX spikes and term-structure inversions, breadth washouts and thrusts, turn of the month, pre-holiday,
post-event drift, gap-down reversals, 52-week-high breakouts and NR7 inside days. Same method as the strategy lab
(exit chosen in sample, judged out of sample against random entries with the same exit); t-statistics are counted by
signal date, and a style must beat random entries by >= 0.05R in AND out of sample with t >= 2.5.

* **High-beta stocks (5 years):** none passed. The single-stock versions of the mean-reversion styles earn no more
  than random entries; market-wide styles fire on too few days in 5 years to be judged.
* **SPY since 1993 and QQQ since 1999 (out of sample from 2013):** on the index ETFs short-term mean reversion and the
  turn of the month do hold - QQQ "close at the day's low" passed (+0.07R edge, 225 trades, t 3.9 - but not on SPY);
  RSI(2) pullbacks beat random entries by about +0.2R on both ETFs in both periods but with only ~60 trades each; the
  turn of the month earned a small, steady +0.06R (SPY positive in 24 of 33 years).
* The Market page's **Index timing** card shows which of these fire at today's close on SPY / QQQ, with a plan and the
  long-history result (passing styles and near passes with >= 0.05R edge in both periods and t >= 2).
* Fact about these stocks: on average almost all of the return came overnight (close to open), not during the session.
* The results of the 2026-10-30 run ship as `config/discovery.default.json`.

### 7g. Insider buying (`python insider_study.py`, run by hand)

Do a company's own officers and directors buying shares in the open market predict the next weeks? Source: the SEC
insider transactions data sets (every Form 4, one zip per quarter, 2020 onwards, ~300 MB cached in
`data/cache/sec/files/`); needs `SEC_USER_AGENT`. Only open-market purchases (code P) by officers and directors worth
>= $5,000; the signal day is the filing date (when the public could know), entry at the next open. Three styles fixed
in advance: >= 3 insiders buying within 30 days, >= 2 buying >= $250k in total, and a CEO / CFO / President buying
>= $100k. Tested on every liquid company with such filings (price >= $5, >= $5M daily dollar volume; ~1,800 stocks,
bars from Yahoo) with the discovery study's method and pass rule.

* Result (2026-11-01): **none passed**. The CEO / CFO buy is a near pass: edge over random entries +0.13R in sample and
  +0.16R out of sample (724 trades), t 2.2 (the bar is 2.5), and the edge holds with all three fixed exits (+0.04R to
  +0.14R). The classic "cluster" signal (>= 3 insiders) adds almost nothing (+0.04R, t 0.6).
* **Independent check (2010-2019 filings, never used to design the study):** the CEO / CFO / President buy beat
  random entries in the same stocks by +0.15R (t 3.5) with the exit the study had chosen (1 ATR stop + 3-ATR trailing
  stop, 20 sessions), and by +0.09R with both other fixed exits (t 3.7 / 4.3); 2020 onwards +0.05R to +0.13R. It
  passes in both periods, so it is **used live**: a deep-dive candidate whose CEO, CFO or President bought >= $100k
  in the open market in the last 5 calendar days gets **+10 points** (`scanner.insider_exec_buy`; 150 points/R x
  +0.07R, the smaller edge). The window is short on purpose: entering 7 days after the filing lost the edge in
  2010-2019. Source of the live data: Yahoo's insider-transaction table (transaction date, position, value).
* Cross-asset lead-lag (2026-11-01, research notes, not in the code): 63 assets (sector ETFs, oil, gold, copper,
  rates, dollar, yen, bitcoin, VIX, Asian and European indices), 22,104 "A moved > 2 sd today -> B over the next 1 /
  3 / 5 days" tests, first 60% of history to find, last 40% to check, false-discovery control. 17 held up, all
  one-day effects (bonds give back the day after an equity sell-off; after a big Hang Seng fall US stocks open lower
  and recover during the session). None of 251 cross-asset conditions changed the result of crash rebound / squeeze
  / breakout trades, so none is in the score. Oil up -> tech down: no stable relation.
* In the terminal's own 248-stock universe there are too few signals (62 out of sample) to judge.
* Delisted companies are missing (survivorship bias), which flatters every style. Not used in the live ranking.
* The 2026-11-01 results ship as `config/insider.default.json`; the Research page shows the newest run.

### 7h. Scaling in (`python scale_in_study.py`, run by hand, ~2 minutes)

Does entering a trade in tranches beat one entry? Tested on the three setups the terminal keeps (crash rebound,
squeeze, breakout) and on random entries, each with the exit the strategy lab chose for the setup. Risk is fixed: the
whole trade has a 1R (1% of the account) budget; tranche i gets w_i of it and is sized on its own fill and the stop
in force (`w_i / (fill - stop)`); an unfilled tranche leaves its budget unused. 206 schemes: 2 tranches with the first
at 25 / 33 / 50 / 67%, or 3 tranches (34/33/33, 50/30/20, 20/30/50); adds by time (open of day +1 / +2 / +3 / +5), on a
pullback (limit 0.5 / 1 / 1.5 ATR below the first fill), on strength (buy stop 0.5 / 1 / 1.5 ATR above) or on a
confirming close (next open after a close >= +0.5 / 1 / 1.5 ATR), valid 3 or 10 sessions; all with one shared stop
and again with each tranche's own stop. A scheme is chosen in sample by the paired difference against a single
entry on the same trades and must beat it out of sample with t >= 2.5 (t on per-date means).

* Result (2026-11-01): **no scheme beats a single full entry** on any of the three setups. Crash rebound and breakout:
  0 of 206 schemes are better in sample or out of sample; squeeze: 8 are better in sample, 8 out of sample, none in
  both, best +0.03R out of sample with t 1.4. The best in-sample choices lose -0.05R (crash rebound), -0.06R
  (squeeze) and -0.03R (breakout) per trade out of sample.
* Why: the edge is at the first open. All-in a few days later earns much less (crash rebound +0.51R at the next open,
  +0.22R one day later; breakout +0.31R vs +0.18R). Pullback adds fill mostly on the trades that go on to lose
  (crash rebound: the -0.5 ATR add fills on 57% of trades whose single-entry result is -0.20R vs +1.43R for the rest);
  strength adds fill on winners but at a worse price, and the risk they save on losers is smaller than the profit
  they give up on winners.
* Not used live: the plan stays a single entry. The 2026-11-01 compact results ship as `config/scale_in.default.json`.

### 7i. 2x leveraged ETFs and tighter stops (`python leverage_study.py`, run by hand)

Would trading the validated setups (crash rebound, squeeze, breakout) through a 2x daily long ETF, with tighter stops,
or with the entry split into tranches, earn more? Everything is in R with the SAME risk budget per trade, so a 2x ETF
does not double a trade's R: it halves the cash the position needs. What leverage changes is the path (daily reset:
volatility decay), the cost, the size of gaps through the stop, and how many positions the cash allows.

* Synthetic 2x ETF = 2x each day's move from the previous close (open, high, low and close), compounded at each close,
  minus a daily cost. Checked on 28 real 2x single-stock ETFs (NVDL, TSLL, CONL, MSTU, ...): daily correlation 0.999,
  but the **real cost is ~17% a year** (median; swap financing, far above the ~1% expense ratio). Over 20 sessions a
  2x ETF returns ~2.6% less than twice the stock (median).
* Sizing is done the evening before from the plan (entry reference - stop), as in live use. (The strategy lab sized
  from the actual fill; a fill just above the stop then gives an extreme R - one +142R crash-rebound trade carried a
  large part of its out-of-sample edge.)
* Stops 0.5-2 ATR (in UNDERLYING ATR) x 5 exits x 2 horizons on the stock and on the ETF; edge over random entries
  with the same plan, also against random stocks on the SAME signal dates; pass = edge >= 0.05R in and out of sample
  and out-of-sample t >= 2.5 by date.
* Scaling in (a cross-check of section 7h, same conclusion): 36 plans fixed in advance (add on strength +0.5 / +1 / +1.5 ATR, buy dips -0.25 / -0.5 / -1 ATR, next
  day, 3 or 5 days, after a confirmation close; 2 or 3 tranches, several splits). Each tranche risks its share of
  the 1R budget to the current stop; an add and a stop on the same bar = add first, then the stop; a dip-buy never
  fills on a bar that also reached the target (that ordering bias first produced a fake +0.27R for squeeze).
  Adopted only if it beats one entry by >= 0.05R in and out of sample with t >= 2.5.
* Account simulation: daily mark-to-market, cash account (or 2x margin where stated), at most 10 positions, each
  trade risks 1% (or 2%) of equity; medians over 12 random orders of same-day signals; random = a random stock on
  the same day with the same plan. Also SPY and 2x SPY buy & hold.

Result (2026-11-01, 2021-11 to 2026-10, in sample to 2025-01):

* The same trades on the 2x ETF earn 0.06-0.09R LESS in sample (decay + cost); nothing passes the t >= 2.5 bar -
  not even the unleveraged setups once sizing is realistic (crash rebound edge +0.13R / +0.08R, t 0.8).
* Tighter stops ruin squeeze (1 ATR: -0.31R vs random, t -2.8) and do nothing for breakout. Only crash rebound
  likes a tight stop (0.5-0.75 ATR: +0.19 to +0.31R vs same-day random, t 0.8-2.0, not significant).
* No scaling-in plan beats one entry in both halves (best in-sample plans: -0.01 to -0.08R out of sample).
* Accounts, whole period: crash rebound on the stock 37% a year, max drawdown -46%; on a 2x ETF with the same % stop
  (= half the stop distance) 72%, -66%, Sharpe unchanged (1.10 vs 1.09); the same thing on the stock with 2x margin
  84%, -64% (no ETF cost, and every stock has one - only ~15% of the signals were on stocks with a real 2x ETF, most
  launched in 2024-2025). With 5 positions instead of 10 the leveraged versions have the LOWER Sharpe. All three
  setups together on a 2x ETF: max drawdown -85% or worse. SPY: 10% a year, -25%.
* Nothing changes in the live scoring. The 2026-11-01 results ship as `config/leverage.default.json`; the Research
  page shows the newest run.

### 7d. News (automatic)

`src/catalysts/news.py` fetches free RSS headlines - Yahoo Finance and Nasdaq for **every stock in the
universe**, Google News for the listed candidates and for market topics (stock market, the Fed,
semiconductors / AI, Nasdaq / tech, the economy). Headlines are de-duplicated across feeds, kept for 21
days in the database (`news` table, so history accumulates), tagged by keyword rules (earnings, guidance,
analyst, M&A, offering, legal, FDA, contract, product, insider, macro, big move) and given a keyword
tone. They are refreshed after every daily run and **every hour by the background job** while the Mac is
on (`news.hourly_refresh`; `python daily_run.py --news-only` does it by hand); each feed is cached for 50
minutes. The News tab shows the market feed, today's candidates' news and every universe headline with
filters; stock pages fetch their headlines live and, for crash rebounds, show the news around the crash
("why did it crash?"). Offering / legal / guidance headlines appear as contradicting evidence on a
candidate; news never changes the score (there is no historical news to back-test that).

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

## 10. Long-term operation (what maintains itself)

The terminal is meant to run unattended for years. What it does on its own, and what is left to you:

| Risk over months / years | What the program does |
|---|---|
| Yahoo changes its site and old yfinance versions return nothing | When ≥ 80% of downloads fail, the hourly job runs `pip install --upgrade yfinance` in `.venv` (at most once a day); the Dock launcher then restarts the dashboard. `install_mac.sh` also upgrades on every update. |
| Provider has not published the latest session yet | The run analyses the latest available session, records it, and `--if-new` keeps retrying (at most 6 times per session). |
| A stock is delisted, acquired or renamed | Excluded by the quality check, named in its own warning; the monthly universe rebuild drops it (core list: edit `config/terminal.yaml`). |
| Screener glitch returns only a few stocks | The monthly universe rebuild refuses to shrink the list below half and keeps the previous one; a failed rebuild waits 3 days before retrying. |
| A backtest keeps failing | Retried after 3 days instead of on every run (`reports/backtest/last_failure`). |
| Disk growth | After every run: reports older than 30 sessions are gzipped, 6 backtests kept, FINRA / SEC / FTD / news caches pruned, news rows older than 180 days deleted, logs over 5 MB trimmed. |
| Database damaged or deleted | The trade journal and price alerts are exported daily to `data/backups/user_data_YYYYMMDD.json` (14 kept). Restore with `python daily_run.py --restore-user-data data/backups/user_data_YYYYMMDD.json`. Prices are simply downloaded again. |
| Crash or power loss in the middle of writing | Reports, snapshots, backtests and SEC files are written to a temporary file and renamed, so the dashboard never reads half a file; leftovers are deleted. |
| Two updates at the same time (Dock app + hourly job) | A file lock lets only one run write; news refreshes have their own lock. |
| Alerts or open trades on stocks that left the universe | Their prices keep updating, so alerts still trigger and open trades stay marked to market. |
| Internet / CDN outage | The chart library is bundled (`src/web/static/vendor/`), so the dashboard works offline on stored data. |
| News sources rate-limit | Each host is paused for an hour after a 403 / 429 / 503 so price downloads are never put at risk. |

What still needs you: add the 2028 FOMC meeting dates to `config/macro_events.yaml` when the Fed
publishes them (the run warns 45 days before the list runs out), and `git pull` + `bash scripts/install_mac.sh`
to take updates.

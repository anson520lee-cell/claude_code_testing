# reality/ — "can anyone really make 90%/yr?" study (research only, no live rule touched)

Report (Chinese PDF, charts, HTML source and full numbers) is committed in `reality/report/`.

Run order (caches live in `/tmp/rc`, created by the first two steps):

1. `python -m reality.membership` — rebuild point-in-time S&P 500 membership from the Wikipedia change log
   (needs `/tmp/rc/hist.wiki`, the raw wikitext of "Historical components of the S&P 500").
2. `python reality/download.py` — Yahoo adjusted prices for every ticker that was ever a member (+ SPY/QQQ/TQQQ/SOXL/SMH...).
3. `python -m reality.run1` momentum, three universes (today's members / point-in-time / hindsight winners), random baselines
4. `python -m reality.run2` leveraged ETFs (real + synthetic), 200-day timing, daily-reversal cost sensitivity
5. `python -m reality.run3` calendar/rolling windows, winner persistence, SMA parameter sweep, walk-forward gradient-boosting stock picker
6. `python -m reality.run4` the terminal's own (2026-screened) universe vs random picks
7. `python -m reality.report <outdir>` charts + HTML (print to PDF with Chromium)

Conventions: weights decided at the close of day t are traded one day later (t+2 returns), 10 bps per side, names whose
data end are converted to cash. Missing delisted names (coverage 71% in 2005, 92% in 2020) mean the point-in-time
numbers are still slightly optimistic.

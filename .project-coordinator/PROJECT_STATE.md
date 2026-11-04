# Trading terminal research
<!-- project: 8c1e3595-439a-4844-8db8-3a3c567643a0 -->

Goal: Reproducible, after-cost, cross-regime 80–90% annualized candidate research; none qualified.
Code baseline: ae57c9af46f112627cdae716680219759c980b6d; parent 2f015a09a95f7d81e568cd594114dbaa91aa3113. Legacy handoff trading-terminal/main revision 7 is preserved, not a coordinator digest.
Status: Joint-cap bug OPEN. The 10026-config search is incomplete; the old ~4000 unsaved trials remain lost. No new strategy or trial was produced during this model-switch save.
Done: Per-trial SQLite checkpoints, fingerprints, bounded batches, missing-return hindsight fix, initial-capital drawdowns, portable paths, resumable collection, public OHLCV snapshot.
Evidence: Pinned docs/RESEARCH_RESUME_20261104.md records previous 17 research tests + 16 targeted Mac reruns; not Windows revalidation. Current 315 local tracked files exactly match baseline; PREREG and snapshot hashes verified.
Next: Resume task-6ae0aace40e049c893af83df887074fb; first fix joint caps, explicit infeasibility and rebalance/band/cash regressions. Then restore real original data and continue batches with --workers 2 --max-new 250.
Blockers: Missing original 30-sleeve data; Library snapshot recovery requires receiver verification. Windows code not synchronized; session inventory responded but downloads/execution remain unverified.
Data: Public 350-symbol / 361601-row cache, 2021-11-01..2026-11-03, NOT PIT membership or original pool. Archive SHA256 4e1d2291e69fbac080f6e6e0c6f1b4791c92ef91d849bd7d6413ceeb74ad10ca.
Limits: 2025–26 was already observed; historical reused validation only. Daily MTM drawdown/margin/liquidation qualifications remain unfinished. No live changes, orders, broker, deployments, PR merge, scaling-in or return guarantees.
Background: Existing recurring task retained; recorded completion has no latest result and proves no strategy progress. RDC reports no active tool sessions; Mac OS process inventory was denied. Check again before compute.

Read TASKS.md and DECISIONS.md only as needed; detailed sources are pinned in the task handoff.

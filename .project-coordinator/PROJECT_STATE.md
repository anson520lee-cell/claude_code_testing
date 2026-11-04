# Trading terminal research
<!-- project: 8c1e3595-439a-4844-8db8-3a3c567643a0 -->

Goal: After-cost, cross-regime 80-90% annualized candidate research; none qualified.
Task: task-6ae0aace40e049c893af83df887074fb; original handoff 0f3a79db3158d2e952e331e939f23550f21796e2dab7254979b2443ea1d4922b remains immutable/incomplete.
Code: ff9aca77ba1d7e109d7ec5f45db00950978c89d9; joint-cap fix cb9ff8ff11f7b005d508fe9f4475b52054f4c421; ETF preregistration cf62d6a30df62c470a56fbe608747f0085acde90.
Status: Joint-cap repaired and tested. Original 10026-config / 30-sleeve search still blocked by incomplete original inputs; lost ~4000 old trials remain lost. Separate ETF expansion COMPLETE: 1728 saved trials, 0 reach 80% in all four selection subperiods.
Evidence: 68 targeted Windows tests passed together (16.03s), including original benchmark parity, joint feasibility, monthly/band drift, cash liquidation cost, checkpoint restart, lagged execution and bounded archive retries. Details: docs/RESEARCH_CONTINUATION_20261104.md.
ETF result: Frozen pre-2025 trial 332: 2008-24 CAGR 18.78%, daily maxDD -39.13%; reused 2025-26 CAGR 34.22%, maxDD -27.98%. Double-cost 2008-24 CAGR 16.64%, maxDD -40.38%. All metrics and protocol in reports/etf_expansion/. No stable-return qualification.
Data: Frozen 2026-11-04 exclusive collection: 50 Yahoo symbols + 10 Cboe histories + 2 panels, 62 successful manifest entries. Four public x80 intermediate studies rebuilt. Cboe histories are benchmark indexes, not executable option quotes.
Preserved snapshot: 350 symbols / 361601 daily rows; archive SHA256 4e1d2291e69fbac080f6e6e0c6f1b4791c92ef91d849bd7d6413ceeb74ad10ca and Parquet SHA256 5476d4902a8e5a1dd810e6aa4eedcbb103ff7cb1317e467c8a140e23de43e40f reverified before Mac disconnection. Not PIT or the original pool.
Next: Continue bounded Binance spot/funding/premium restoration (source catalogue includes delisted pairs; spot catalogue 700 symbols), then original PIT/flow/fundamental/FX/derivative inputs. Only after completeness/audit resume the ORIGINAL search, <=2 workers and <=250 new trials per batch.
Execution: Windows C:/Users/jordan/trading-research is verified; Mac execution host became unavailable. A two-worker, six-symbol initial spot batch was launched; read its real status before another batch. Keep the live swing-terminal checkout unchanged.
Blockers: Original 30-sleeve inputs incomplete; archive coverage/partial-month tails must be inspected, never filled with invented zero returns. Full ETF trial-return checkpoint and frozen data are on Windows (public all-trial metrics in Git).
Limits: 2025-26 is reused historical validation, not untouched. No live changes, orders, broker, deployment, merge, scaling-in, paid data or guarantees. Daily executable margin/option liquidity and future validation remain required.
Background: Existing recurring task is retained; do not infer completed work from its scheduling metadata. Avoid duplicate processes; no claim that an LLM runs independently of this session.

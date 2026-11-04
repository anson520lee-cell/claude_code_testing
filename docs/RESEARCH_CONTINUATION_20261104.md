# Research continuation after the versioned takeover (2026-11-04)

Starting state: task-6ae0aace40e049c893af83df887074fb, handoff
0f3a79db3158d2e952e331e939f23550f21796e2dab7254979b2443ea1d4922b,
state/code branch codex/research-resume-20261104 at dc1c9d356d50e2ae285d1de64838ba95ffc28f2d.

## Verified this continuation
- Windows research checkout C:/Users/jordan/trading-research was fast-forwarded to the saved commit.
  The different pre-existing untracked checkpoint.py was preserved byte-for-byte under
  research-preserved-666559515fec/checkpoint.py (SHA256
  666559515fec6a5fdc9ec7c6308d3c8b54e6ad1d4f7aa710cab9d25915d66425).
  The live swing-terminal checkout was inspected only; no live settings, orders or deployment changed.
- Yahoo, Cboe and PyPI were tested successfully from Windows. statsmodels was installed in the research environment.
- Frozen collection succeeded for all 50 requested Yahoo symbols and 8 Cboe strategy-index histories,
  plus the 2 combined Yahoo panels (60 manifest entries, 0 failed). Exclusive as-of date: 2026-11-04.
  Exact source hashes are in research_data/frozen_20261104/collection.json.
  Cboe histories are index benchmarks, not executable historical option quotes.
- Original recovery artifacts were checked before the Mac execution host disconnected:
  archive SHA256 4e1d2291e69fbac080f6e6e0c6f1b4791c92ef91d849bd7d6413ceeb74ad10ca,
  prices.parquet SHA256 5476d4902a8e5a1dd810e6aa4eedcbb103ff7cb1317e467c8a140e23de43e40f.
  They match the handoff. That snapshot remains a present-universe cache, not PIT membership.
- 54 targeted Windows tests passed (10.89s): test_optimize, test_optimize_resume,
  test_joint_caps, test_levscore, test_x80. These include restart-without-recomputation
  and rejection of changed inputs, not a claim of a full Windows application-suite rerun.

## Joint-cap correction and changed semantics
- constraints.py enforces asset weight and crypto variance-contribution limits together.
  Unconstrained weights, including hedged risk contributions greater than 100%, remain unchanged.
  Sparse scores can allocate residual weight to zero-score eligible sleeves to obey an asset cap.
- The four-asset counterexample has exactly one full-investment allocation at a 25% cap.
  Its crypto variance share is 75%; a 35% crypto cap is mathematically infeasible.
  The old 44.0339% non-crypto allocation is rejected.
- Proven infeasibility is code 1. Numerical non-convergence is code 2 and is NOT called
  mathematical infeasibility. Both hold cash for that week and are counted separately in search results.
  No limit is silently relaxed. SLSQP results are checked against both actual constraints.
- Monthly/band drift is rechecked using past covariance. Risk limits override the scheduled
  rebalance when needed. Held-weight and crypto-share diagnostics describe the allocation actually used.
- Liquidation into cash now pays the model's transaction cost. The account no longer exits free.
- The constraints source is included in the study fingerprint. Changed semantics require a new
  output directory; do not resume an old fingerprint or overwrite any saved trial.
- Original PREREG.md, STATUS.md, previous reports and the historical audit counterexample are preserved.

## Remaining
- The original 30-sleeve inputs are still incomplete (PIT equities, flow/fundamentals,
  crypto universe including delistings, funding, derivative histories and FX/rates).
  The 50-symbol collection must not be substituted for that original pool.
- Rebuild public-only intermediate sleeves from the frozen real dataset, preserving earlier
  reports and all trials. Then restore the remaining sources before the original 10026-config search.
- No strategy has newly qualified for stable 80-90% annual returns. 2025-26 remains reused
  historical validation. Daily MTM, executable margins and liquidation checks are still required.
- Mac execution environment disconnected during this work; do not depend on its temporary helpers.
  Windows is the verified active research execution host.

# Project state
<!-- project: 51579ace-6062-55f7-b90d-ebfb9c9ce054 -->

Goal: Maintain ZeroX Labs, Anson's English single-page personal site for tutoring, AI-score detection and website services, with 3D liquid glass visuals.
Status: 2026-11-04 report: restored and hash-verified 14 files from legacy project-handoff my-website-zerox revision 2 (V13.4); read source/README.md and the bundled design skill. GitHub and the public live site match each other but differ from the older handoff source.
Next: Await the user's theme choice or another specific edit. Before rebuilding, reconcile restored source/template.html and source/stage.js with the newer published button interaction changes.
Blockers: No theme selected. Restored editable source has not been reconciled with the newer published bundle. Vercel connector returned 403; public HTTP verification succeeded, but deployment access remains unverified. Cloudflare deployment was historically incomplete and was not rechecked.
Code baseline: https://github.com/anson520lee-cell/claude_code_testing/tree/b9520cb2e3ff42b357ce250ae9b06df94938b1a2 (zerox-site); public/index.html SHA-256 86626296ae8de93cf897901f931468b35e91763041a139a0714d4e40b58ad6bc, 832340 bytes.
Evidence: 2026-11-04 HTTP 200 from https://zerox-labs.vercel.app; byte-identical to pinned public/index.html. Legacy handoff index.html SHA-256 1adc8871870c01c9fa18e9892a3a671bb870e11b357ba9792a0f7f69c1ef0645, 832695 bytes. Source/content comparison completed; no new browser rendering or animation tests were run.
Published differences observed: smaller hover movement (0.04/0.06), smooth non-elastic return, no separate label movement, and removed cursor-driven glass bulge. A newer snapshot and comparison are local only at my-website-zerox/ref/current-live-2026-11-04/.
Artifacts: Legacy source and design files remain at https://github.com/anson520lee-cell/ai-shared-memory/tree/dd6d273922f22d8fd3137efd9054b28cb1942e2c/projects/my-website-zerox (revision 2). Local restored files/snapshots are not uploaded by this state report. No code edit or deployment was performed.

Load TASKS.md for work tracking; DECISIONS.md for design rationale.
Git commit from resume is the state revision; do not duplicate a timestamp on every read.

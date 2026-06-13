# autoresearch — motion-rep-hota

## Goal
Maximize `pooled_hota` on `gmc_link/alignment.py`. Higher is better.

## What the Agent Can Change
- Only `gmc_link/alignment.py` — this is the single file being optimized.
- Everything inside that file is fair game unless constrained below.

## What the Agent Cannot Change
- The evaluation script (`evaluate.py` or the eval command). It is read-only.
- Dependencies — do not add new packages or imports that aren't already available.
- Any other files in the project unless explicitly noted here.
- Additional constraints: Edit ONLY gmc_link/alignment.py (shared_weight aligner). ONE change per iteration. GOAL: raise iKUN MOVING DetRe+DetPr (oracle headroom: ship MOVING 20.5/26.1 -> ceiling 69.2/80.3, +6.13 pooled reachable WITHOUT new tracker) while holding STATIC. Metric pooled_hota already bakes the STATIC>=43.2 guardrail (penalized -100 on miss). Single-seed = diagnostic screen ONLY; n=3 STATIC>=43.2 is the real promotion gate, run manually before any SOTA claim. DO NOT touch the fusion recipe (exhausted: 24 NEG levers) or eval scripts. Aligner levers: trunk width/depth, norm (LN/BN), dropout, activation, residual, projection dim, output L2-norm, InfoNCE temperature. Prior arch (Exp36 transformer / MLP-25D) NEG at AUC 0.779 -- HOTA-direct keep/discard is the NEW variable, re-screen them. Iteration ~22min (train 2m + 3 GMC caches 18m + fuse 1m). Baseline: seed0 pooled 44.561 / STATIC 43.240 / MOVING 28.885.

## Strategy
1. First run: establish baseline. Do not change anything.
2. Profile/analyze the current state — understand why the metric is what it is.
3. Try the most obvious improvement first (low-hanging fruit).
4. If that works, push further in the same direction.
5. If stuck, try something orthogonal or radical.
6. Read the git log of previous experiments. Don't repeat failed approaches.

## Simplicity Rule
A small improvement that adds ugly complexity is NOT worth it.
Equal performance with simpler code IS worth it.
Removing code that gets same results is the best outcome.

## Stop When
You don't stop. The human will interrupt you when they're satisfied.
If no improvement in 20+ consecutive runs, change strategy drastically.

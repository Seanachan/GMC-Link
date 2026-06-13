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

## Weak-point ledger (run 10, 2026-06-13)

### Ship now
`9c000d6` = Dropout 0.05 + trunk 768 → pooled **44.739** (+0.178 vs baseline). Single-seed; n=3 STATIC≥43.2 gate not yet run.

### Probed axes (DEAD = won't revisit)
- **Width**: 512→768 POS, 768→1024 NEG. **Sweet at 768.** DEAD beyond 768 at current reg.
- **Dropout dose at 512 trunk**: 0→0.1 (component+ pooled tied) → 0.05 (pooled +0.135). **Sweet 0.05.**
- **Activation swap**: ReLU→GELU NEG (MOVING −0.778). DEAD smooth-activation axis.
- **Adapter capacity**: deepen 13→128→256 MLP NEG (STATIC crash). DEAD.
- **Per-modality LN before trunk**: NEG (MOVING −0.763 / over-normalized motion variance). DEAD.
- **Residual skip in trunk (post-LN repositioned)**: catastrophic MOVING −1.973 (raw adapter dominates summed signal). DEAD residual-skip without scale matching.

### Pattern
- Capacity adds without reg → STATIC crash (overfit).
- Smooth activation (GELU) → MOVING drop (loses sparse residual signal).
- Asymmetric per-modality changes (LN, deeper adapter) → MOVING drop (over-normalizes or scale-shifts motion-distinctive features).
- Sweet spot is **narrow** around the symmetric trunk with mild reg.

### Bottleneck hypothesis
MOVING DetRe (oracle 20.5→69.2) is not reachable via aligner-only architectural tweaks at this representational ceiling. Current 30.959 vs oracle 69.2 = 38pt gap; aligner perturbations only swing ±2pt. Real lever may be **input representation** (motion features themselves), but locked to alignment.py.

### Next-3 candidate directions
1. **Dropout 0.04 at trunk 768** — finer dose-probe between known POS 0.05 and zero. May find sub-0.05 sweet spot for current width.
2. **LeakyReLU(0.01) in trunk** — preserve signed residual-velocity activations without GELU's smoothness penalty.
3. **Trunk depth +1 layer** (768×3 hidden) — capacity in depth instead of width. Tests whether 768 saturation is width-specific or general.

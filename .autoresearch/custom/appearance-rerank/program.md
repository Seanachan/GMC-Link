# autoresearch — appearance-rerank (phase 3)

## Goal
Maximize 3-seq pooled `pooled_hota` for the CLIP-L/14 + spatial-gate APPEARANCE
re-ranker stacked on the iKUN ship. Higher is better. Baseline (posqual 34-expr
subset, single-seed) = **45.089**. Beat it by EXPANDING which appearance exprs get reranked.

## Context (why this loop exists)
Motion-rep loop (phases 1+2) CONVERGED NEG — aligner arch (18 runs) + motion features
(5 families) all dilute; the MOVING oracle gap is classification-bound, unreachable
within GMC-Link+iKUN+additive-fusion (native-veto). The ONE live positive signal source
is APPEARANCE re-rank: CLIP-L/14 zero-shot track-median cosine REPLACES native admit on
listed exprs (non-additive => dodges native-veto). Spatial gate (cx side, for left/right
exprs) was the +0.690 lever. Shipped stack (rerank ⊕ Path B, n=3) = 45.612.

## What the Agent Can Change
- **ONLY `diagnostics/appearance_rerank/seqmap_active.txt`** — the set of `seq+expr` keys
  that get CLIP-L/14 spatial rerank. This is the optimization surface (subset expansion).
- Candidate exprs to ADD live in: `seqmap_catastrophic.txt` (46 catastrophic-admit exprs,
  oracle prize +4.4), `seqmap_color.txt` (18 color), `exprs_catastrophic.txt`. Add exprs
  that are CLIP-L/14-SEPARABLE (color/appearance) and SKIP non-separable ones (full
  catastrophic was NEG −0.34: non-color exprs aren't CLIP-separable → FP-flood).
- Format: one `seq+expr` per line (e.g. `0011+left-cars-in-black`). seqs = 0005/0011/0013.

## What the Agent Cannot Change
- `autoresearch/eval_appearance_rerank.py` (the eval) — read-only.
- `run_ikun_linear_additive.py` (rerank mechanism), fusion recipe, CLIP-L/14 caches, GMC caches.
- tau (0.205) and spatial gate are fixed for this loop (subset is the lever). Changing tau is a
  separate experiment.

## How it works (no sync hazard — single eval path)
Eval reads `seqmap_active.txt`, runs `gen_predicts(mode="rerank_clipb32", rerank_set=...,
rerank_tau=0.205)` with `RERANK_SPATIAL=1` on PRE-BUILT ship GMC caches (no retrain), then
3-seq pooled HOTA. ~1-2 min/iter. No train/inference sync issue (inference-only rerank).

## Iteration
1. Read results.tsv — which subsets tried, what helped/hurt. Don't repeat.
2. Hypothesis: which UNREANKED appearance expr is CLIP-L/14-separable + currently mis-admitted?
   (Prefer color/appearance exprs from the catastrophic/color pools; avoid action/non-visual.)
3. Edit seqmap_active.txt (add — or remove a hurting — expr; small deltas to attribute effect).
4. git add diagnostics/appearance_rerank/seqmap_active.txt && git commit -m "experiment: <subset change + hypothesis>"
5. Run: python /home/seanachan/.claude/plugins/cache/claude-code-skills/autoresearch-agent/2.1.2/scripts/run_experiment.py --experiment custom/appearance-rerank --single
6. Read KEEP/DISCARD. Eval resets seqmap_active.txt on discard (reset --hard HEAD~1).
7. Every 10 runs: append a ledger subsection here (separate commit).

## Promotion gate
Single-seed = screen. Real ship = n=3 + re-confirm the stack (rerank ⊕ Path B) before any claim.
Cross-host: rerank is iKUN-only, deficit-conditional (FH V2 NEG — no appearance deficit).

## Stop When
Human interrupts, or no improvement in 15+ consecutive runs (subset space is finite ~46 candidates).

## Ledger (CONVERGED, 2026-06-14) — subset expansion exhausted
Best = `3a32819` pooled **45.241** single-seed (seed posqual-34 + silver-2 + back-to-camera-4 = 36 active exprs), +0.152 over 34-expr baseline 45.089.
All catastrophic-46 + color-18 candidates tested:
| candidate group | type | Δ pooled | verdict |
|-----------------|------|----------|---------|
| silver (cars/vehicles-in-silver) | rare color | +0.148 | KEEP — the win |
| back-to-the-camera ×4 | pedestrian orientation | +0.004 | KEEP (noise-level) |
| carrying/holding-a-bag ×2 | fine-action object | 0.000 | DISCARD (neutral) |
| in-front-of-ours ×2 | center-cx geometry | 0.000 | DISCARD (neutral) |
| horizon-direction ×2 | directional | −0.047 | DISCARD (mild FP-flood) |

Conclusion: only rare-color (silver) exprs were genuinely mis-admitted + CLIP-L/14-separable. Pedestrian-fine ≈ neutral; geometry/directional position-blind → neutral/NEG (crop-CLIP can't see position/direction; matches `project_range_rerank_falsified`). Pool exhausted → cron 471c4f40 KILLED. Real deliverable = n=3 + stack (rerank ⊕ Path B) re-confirm of the 36-expr subset before any claim.

#!/usr/bin/env python3
"""autoresearch evaluator — motion-representation -> iKUN pooled HOTA.

GROUND TRUTH. The autoresearch loop edits the *target* file (the aligner /
motion representation), never this script. Each iteration this wrapper:
  1. retrains the seed0 shared_weight aligner on the current representation,
  2. rebuilds the 3 V1 GMC caches (0005/0011/0013) from those weights,
  3. runs the LOCKED iKUN linear-additive ship recipe + TrackEval,
  4. parses pooled / MOVING / STATIC / APPEAR HOTA from the summary row,
  5. emits `pooled_hota: <value>` for run_experiment.py to parse.

The STATIC>=floor guardrail (pre-committed >=43.2 at n=3) is baked into the
emitted metric: on a guardrail miss the value is penalized (-100) so the
runner can never KEEP a STATIC-regressing config. RAW numbers are always
printed for the agent's report.

AR-private artifact names (gmc_link_weights_AR_sw_seed0.pth /
_AR_sw_seed0_rawcos caches) keep the ship's seed0 weights + caches untouched.

Single-seed = diagnostic screen (per project decision rules). Promotion to a
KEEP still requires the n=3 STATIC>=43.2 gate, run manually outside the loop.
"""
import argparse
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SEQS = ["0005", "0011", "0013"]
WEIGHTS = str(REPO / "gmc_link_weights_AR_sw_seed0.pth")
SUFFIX = "_AR_sw_seed0_rawcos"
# Ship iKUN recipe (CLAUDE.md, locked). Tag prefix used to find the summary row.
RECIPE = ("--alpha 1.0 --gmc_scale 0.9 --thr 0.17 "
          "--alpha_appear 1.0 --gmc_scale_appear 0.30 --thr_appear 0.10")
ROW_RE = re.compile(r"^a1\.0_scale0\.9")


def sh(cmd, env_extra=None, label=""):
    print(f">>> {label}: {cmd}", flush=True)
    env = {**os.environ, **(env_extra or {})}
    r = subprocess.run(cmd, shell=True, cwd=str(REPO), env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:] + "\n" + r.stderr[-4000:] + "\n")
        sys.exit(f"STEP FAILED ({label}) rc={r.returncode}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-floor", type=float, default=43.2,
                    help="STATIC HOTA guardrail; miss => penalized metric.")
    ap.add_argument("--skip-train", action="store_true",
                    help="Reuse existing AR weights (baseline/measurement check).")
    args = ap.parse_args()

    if not args.skip_train:
        sh(f"python -m gmc_link.train --split v1 --stage 1 "
           f"--architecture shared_weight --seed 0 --save-path {WEIGHTS}",
           label="train")

    # run_build_gmc_cache.py SKIPS rebuild when the cache file already exists.
    # Delete first so the freshly-trained weights actually produce a new cache —
    # otherwise the fuse reads a stale cache from a prior architecture and every
    # iteration silently returns the baseline (CACHE_VER default "v1").
    for s in SEQS:
        (REPO / "gmc_link" / f"gmc_scores_v1_{s}{SUFFIX}_cache.json").unlink(missing_ok=True)

    cache_env = {"GMC_WEIGHTS": WEIGHTS, "GMC_SUFFIX": SUFFIX, "GMC_RAW_COS": "1"}
    for s in SEQS:
        sh(f"python run_build_gmc_cache.py {s}", env_extra=cache_env,
           label=f"cache_{s}")

    out = sh(f"python run_ikun_linear_additive.py {RECIPE}",
             env_extra={"GMC_SUFFIX": SUFFIX, "GMC_RAW_COS": "1"}, label="fuse")

    row = next((ln.split() for ln in out.splitlines()
                if ROW_RE.match(ln.strip())), None)
    if not row or len(row) < 4:
        sys.exit("PARSE FAIL: no iKUN summary row found in fusion stdout")
    pooled, appear, moving, static = (float(x) for x in row[-4:])

    passed = static >= args.static_floor
    print(f"RAW pooled={pooled:.3f} STATIC={static:.3f} "
          f"MOVING={moving:.3f} APPEAR={appear:.3f}")
    print(f"guardrail STATIC>={args.static_floor}: "
          f"{'PASS' if passed else 'FAIL (penalized -100)'}")
    # Penalized so a STATIC-regressing config can never beat baseline.
    print(f"pooled_hota: {pooled if passed else pooled - 100.0:.3f}")


if __name__ == "__main__":
    main()

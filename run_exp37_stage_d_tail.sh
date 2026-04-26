#!/usr/bin/env bash
# Wait for current bare arm PID to finish, then run alpha arms sequentially.
# Usage: bash run_exp37_stage_d_tail.sh <bare_pid>

set -euo pipefail

BARE_PID="${1:-1351942}"
GMC_LINK=/home/seanachan/GMC-Link
OUT="${GMC_LINK}/diagnostics/results/exp37/stage_d"

echo "[tail] waiting for bare PID ${BARE_PID} ..."
while kill -0 "${BARE_PID}" 2>/dev/null; do
  sleep 30
done
echo "[tail] bare done at $(date +%H:%M:%S); launching alpha arms"

for ARM in beta05 beta10 beta20; do
  echo "[tail] launching ${ARM} at $(date +%H:%M:%S)"
  bash "${GMC_LINK}/run_exp37_stage_d.sh" "${ARM}" \
    >"${OUT}/${ARM}_outer.log" 2>&1
  echo "[tail] ${ARM} finished at $(date +%H:%M:%S)"
done

echo "[tail] all arms complete"

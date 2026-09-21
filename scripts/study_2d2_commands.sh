#!/usr/bin/env bash
# The exact command matrix of the 2D-2 geometry / temporal / spatial convergence study.
# Each run writes results/2d2_<label>.{txt,json}, results/2d2_timeseries_<label>.npz and
# figures/2d2_<label>.png. Collect tables with scripts/collect_2d2_study.py.
# Usage: bash scripts/study_2d2_commands.sh <stage>   (stages below; run in order, each stage
# needs the states written by the previous ones). MAXJOBS runs are executed concurrently.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
MAXJOBS=${MAXJOBS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-2} \
       OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
mkdir -p results/states logs

run() {  # run <label> <h_far> <h_cyl> <dt> <t_end> <order> [extra args...]
  # Skips labels that already have results; --resume continues a killed run from its checkpoint.
  # caffeinate keeps the machine awake: a sleep/network-interface loss kills MPICH runs.
  local label=$1 hf=$2 hc=$3 dt=$4 te=$5 order=$6; shift 6
  [ -f "results/2d2_$label.json" ] && { echo "skip $label (done)"; return 0; }
  caffeinate -i $PY scripts/run_2d2_unsteady.py --h-far "$hf" --h-cyl "$hc" --dt "$dt" --t-end "$te" \
      --geometry-order "$order" --label "$label" --resume "$@" >> "logs/$label.log" 2>&1
}
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done; }

case "${1:?stage}" in
  geometry)   # same vertices/cells/dofs/dt, polygon vs curved, from rest to t = 8, states saved
    run geo_p1_full 0.03 0.005 0.005 8 1 --save-state results/states/p1_base_t8.npz &
    run geo_p2_full 0.03 0.005 0.005 8 2 --save-state results/states/p2_base_t8.npz &
    wait ;;
  temporal)   # curved base mesh, dt halved, all continued from the t = 8 state to t = 12
    for spec in 0.01:t_p2_dt01 0.005:t_p2_dt005 0.0025:t_p2_dt0025 0.00125:t_p2_dt00125; do
      throttle; run "${spec##*:}" 0.03 0.005 "${spec%%:*}" 12 2 --init-state results/states/p2_base_t8.npz &
    done; wait ;;
  spatial)    # curved ladder at dt = 0.0025 (ratio 1.5 in both sizes), then decoupled meshes
    for spec in L5:0.0133:0.00222 L4:0.02:0.00333 L2:0.045:0.0075 L1:0.06:0.01 \
                hc3:0.03:0.00333 hf2:0.02:0.005; do
      IFS=: read -r tag hf hc <<< "$spec"
      throttle; run "sp_$tag" "$hf" "$hc" 0.0025 12 2 --init-state results/states/p2_base_t8.npz &
    done; wait ;;
  polygon)    # straight-sided counterparts of the ladder, dt = 0.0025, from the polygon t = 8 state
    for spec in L4:0.02:0.00333 L3:0.03:0.005 L2:0.045:0.0075; do
      IFS=: read -r tag hf hc <<< "$spec"
      throttle; run "pg_$tag" "$hf" "$hc" 0.0025 12 1 --init-state results/states/p1_base_t8.npz &
    done; wait ;;
  *) echo "unknown stage"; exit 1 ;;
esac

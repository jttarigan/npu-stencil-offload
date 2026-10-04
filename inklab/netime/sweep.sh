#!/bin/zsh
# Generality sweep: time every workload at K = 1, 2, 4, 8 on the neural engine,
# GPU and CPU (back to back), plus the neural engine paced at the game cadence
# (submissions at 30/K Hz, ~15 s). Three repeats each; medians are taken later
# by costmodel.py. Output: one CSV row per measurement.
#   ./sweep.sh out.csv [workload ...]
# Workloads are model-name stems; the K suffix is appended here.
cd "$(dirname "$0")"
out=${1:-sweep.csv}; shift
stems=(heat_128x256_s8_fp16 heat_128x256_s32_fp16 grayscott_128x256_s8_fp16 ink_unroll_128x256_j26_fp16)
(( $# )) && stems=("$@")
[[ -f $out ]] || echo "workload,K,units,mode,rep,ms_per_submission" > $out
for stem in $stems; do
  for k in 1 2 4 8; do
    m=${stem}_u${k}.mlmodelc
    [[ -d $m ]] || { echo "missing $m" >&2; continue; }
    for units in ne gpu cpu; do
      for rep in 1 2 3; do
        ms=$(./netime $m $units 300 | sed -E 's/.* ([0-9.]+) ms\/step.*/\1/')
        echo "$stem,$k,$units,b2b,$rep,$ms" >> $out
      done
    done
    hz=$(printf "%.4f" $(( 30.0 / k ))); calls=$(( 450 / k ))
    for rep in 1 2 3; do
      ms=$(./netime $m ne $calls $hz | sed -E 's/.*: ([0-9.]+) ms busy.*/\1/')
      echo "$stem,$k,ne,paced,$rep,$ms" >> $out
    done
    echo "done $stem K=$k" >&2
  done
done

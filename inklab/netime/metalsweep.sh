#!/bin/zsh
# Metal baseline through the sweep.sh protocol. Two rows per measurement:
# units "metal" = wall clock per submission incl. commit + wait (comparable to
# netime), units "metal-exec" = GPU execution time alone (the marginal cost when
# the work rides in an already-submitted frame, which is how the game runs it).
cd "$(dirname "$0")"
out=${1:?out.csv}
for spec in "ink 0 ink_unroll_128x256_j26_fp16" "heat 32 heat_128x256_s32_fp16" \
            "heat 8 heat_128x256_s8_fp16" "grayscott 8 grayscott_128x256_s8_fp16"; do
  set -- ${=spec}
  for k in 1 2 4 8; do
    for rep in 1 2 3; do
      r=$(./metaltime $1 $2 $k 300)
      echo "$3,$k,metal,b2b,$rep,$(echo $r | sed -E 's/.* ([0-9.]+) ms\/step.*/\1/')" >> $out
      echo "$3,$k,metal-exec,b2b,$rep,$(echo $r | sed -E 's/.*gpu ([0-9.]+).*/\1/')" >> $out
    done
    hz=$(printf "%.4f" $(( 30.0 / k )))
    for rep in 1 2 3; do
      r=$(./metaltime $1 $2 $k $(( 450 / k )) $hz)
      echo "$3,$k,metal,paced,$rep,$(echo $r | sed -E 's/.*: ([0-9.]+) ms busy.*/\1/')" >> $out
      echo "$3,$k,metal-exec,paced,$rep,$(echo $r | sed -E 's/.*gpu ([0-9.]+).*/\1/')" >> $out
    done
    echo "done $1 s$2 K=$k" >&2
  done
done

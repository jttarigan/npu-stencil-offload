#!/bin/zsh
# Energy per fluid step on this Mac: runs each condition as a steady load
# for DUR seconds while powermetrics samples CPU/GPU/ANE power once a
# second, then reports average power and energy per step.
# Needs sudo for powermetrics:   sudo ./measure.sh [DUR=20]
set -e
cd "$(dirname "$0")"
DUR=${1:-20}
T=../trace; N=../netime
[ -x $T/.build/trace_harness ] || (cd $T && ./run.sh /tmp/warm 1 1 >/dev/null)
[ -x $N/netime ] || (cd $N && swiftc -O main.swift -o netime)
[ -d $N/ink_step_128x256_j26_fp16.mlmodelc ] || xcrun coremlcompiler compile $T/ink_step_128x256_j26_fp16.mlpackage $N >/dev/null
mkdir -p out
sample() {   # $1 label; samples power for DUR s into out/$1.txt
  powermetrics -i 1000 -n $DUR --samplers cpu_power,gpu_power 2>/dev/null \
    | grep -E "CPU Power|GPU Power|ANE Power|Combined Power" > out/$1.txt
}
report() {   # $1 label $2 steps/s
  awk -v sps=$2 -v lbl=$1 '
    /CPU Power/ {c+=$3; n++} /GPU Power/ {g+=$3} /ANE Power/ {a+=$3} /Combined Power/ {t+=$(NF-1)}
    END { printf "%-10s CPU %6.0f mW  GPU %6.0f mW  ANE %6.0f mW  total %6.0f mW  -> %.1f uJ/step (total) at %.0f steps/s\n",
          lbl, c/n, g/n, a/n, t/n, (t/n)*1000/sps, sps }' out/$1.txt
}
echo "idle baseline ($DUR s)..."; sample idle; report idle 1
HZ=30   # the game's fluid cadence; every load is paced to it, so the
        # energy per step includes the accelerator's idle/wake transitions
STEPS=$(( (DUR + 6) * HZ ))
echo "Metal kernels, paced at $HZ Hz..."
( PACE_HZ=$HZ $T/.build/trace_harness /tmp/ink_energy $STEPS 1000000000 >/dev/null & )
sleep 2; sample metal; pkill -f trace_harness || true
echo "Core ML fp16 on the Neural Engine, paced at $HZ Hz..."
( $N/netime $N/ink_step_128x256_j26_fp16.mlmodelc ne $STEPS $HZ >/dev/null & ); sleep 2; sample ne; pkill -f "netime" || true
echo "Core ML fp16 on the CPU, paced at $HZ Hz..."
( $N/netime $N/ink_step_128x256_j26_fp16.mlmodelc cpu $STEPS $HZ >/dev/null & ); sleep 2; sample cpu; pkill -f "netime" || true
echo
echo "All loads ran at $HZ steps/s, so energy per step = average power / $HZ."
report metal $HZ; report ne $HZ; report cpu $HZ
echo "Subtract the idle line from each to get the load's own power. Raw samples in out/."

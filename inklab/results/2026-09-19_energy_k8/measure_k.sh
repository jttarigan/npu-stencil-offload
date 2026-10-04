#!/bin/zsh
# Energy per fluid step with and without batching (K = 1 vs K = 8), on this Mac.
# Every load runs the ink step at the game's 30 steps per second: K=1 submits
# 30 times a second, K=8 submits 3.75 times a second, so energy per STEP is
# comparable across conditions and includes the idle/wake transitions.
#   sudo ./measure_k.sh [DUR=60]
# Conditions (DUR s each, after a 3 s settle):
#   idle, Metal K=1, Metal K=8 (standalone submissions, metaltime),
#   neural engine K=1, neural engine K=8, CPU K=1 (Core ML, netime)
# Output: out_k/<label>.txt raw samples, and a summary table on stdout
# (average CPU / GPU / ANE / combined power, and mJ per step above idle).
# Protocol: mains power, lid open, no other apps, Low Power Mode off.
set -e
cd "$(dirname "$0")"
[[ $EUID -eq 0 ]] || { echo "run with sudo (powermetrics needs it)"; exit 1; }
DUR=${1:-60}
N=../netime
for m in ink_unroll_128x256_j26_fp16_u1 ink_unroll_128x256_j26_fp16_u8; do
  [ -d $N/$m.mlmodelc ] || { echo "missing $N/$m.mlmodelc"; exit 1; }
done
[ -x $N/netime ] && [ -x $N/metaltime ] || { echo "build netime and metaltime first"; exit 1; }
mkdir -p out_k
HZ=30
CALLS1=$(( (DUR + 8) * HZ )); CALLS8=$(( (DUR + 8) * HZ / 8 ))

sample() {   # $1 label
  powermetrics -i 1000 -n $DUR --samplers cpu_power,gpu_power 2>/dev/null \
    | grep -E "CPU Power|GPU Power|ANE Power|Combined Power" > out_k/$1.txt
}
run() {      # $1 label, rest = load command (run as the invoking user, not root)
  local label=$1; shift
  echo "$label ..."
  ( sudo -u "${SUDO_USER:-$USER}" "$@" >/dev/null 2>&1 & )
  sleep 3; sample $label
  pkill -f "netime|metaltime" 2>/dev/null || true; sleep 5
}

echo "idle ..."; sleep 3; sample idle; sleep 2
run metal_k1 $N/metaltime ink 0 1 $CALLS1 $HZ
run metal_k8 $N/metaltime ink 0 8 $CALLS8 3.75
run ne_k1    $N/netime $N/ink_unroll_128x256_j26_fp16_u1.mlmodelc ne  $CALLS1 $HZ
run ne_k8    $N/netime $N/ink_unroll_128x256_j26_fp16_u8.mlmodelc ne  $CALLS8 3.75
run cpu_k1   $N/netime $N/ink_unroll_128x256_j26_fp16_u1.mlmodelc cpu $CALLS1 $HZ

echo
awk -v hz=$HZ '
  FNR == 1 { if (f) done(); f = FILENAME; c = g = a = t = n = 0 }
  /CPU Power/ { c += $3; n++ } /GPU Power/ { g += $3 } /ANE Power/ { a += $3 }
  /Combined Power/ { t += $(NF-1) }
  function done() {
    lbl = f; sub(/.*\//, "", lbl); sub(/\.txt$/, "", lbl)
    C[lbl] = c/n; G[lbl] = g/n; A[lbl] = a/n; T[lbl] = t/n; S[lbl] = n; order[++k] = lbl
  }
  END {
    done()
    printf "%-9s %4s %8s %8s %8s %9s %12s\n", "load", "n", "CPU mW", "GPU mW", "ANE mW", "total mW", "mJ/step>idle"
    for (i = 1; i <= k; i++) { l = order[i]
      e = (l == "idle") ? 0 : (T[l] - T["idle"]) / hz
      printf "%-9s %4d %8.0f %8.0f %8.0f %9.0f %12.2f\n", l, S[l], C[l], G[l], A[l], T[l], e }
    print "\nmJ/step>idle = (total - idle total) / 30 steps per second. Raw samples in out_k/."
  }' out_k/idle.txt out_k/metal_k1.txt out_k/metal_k8.txt out_k/ne_k1.txt out_k/ne_k8.txt out_k/cpu_k1.txt \
  | tee out_k/summary.txt
chown -R "${SUDO_USER:-$USER}" out_k

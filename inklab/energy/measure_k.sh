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
# Summary: GPU Power is printed by two samplers, so each rail is averaged over
# its own line count; medians are reported too (background spikes skew means).
python3 - out_k $HZ <<'PY' | tee out_k/summary.txt
import re, sys, statistics as st
d, hz = sys.argv[1], float(sys.argv[2])
rows = {}
for l in ["idle", "metal_k1", "metal_k8", "ne_k1", "ne_k8", "cpu_k1"]:
    r = {"CPU": [], "GPU": [], "ANE": [], "Combined": []}
    for line in open(f"{d}/{l}.txt"):
        m = re.match(r"(CPU|GPU|ANE|Combined) Power.*?([\d.]+) mW", line)
        if m: r[m.group(1)].append(float(m.group(2)))
    rows[l] = r
idle = st.median(rows["idle"]["Combined"])
print(f"{'load':9s} {'n':>3s} {'CPU':>7s} {'GPU':>6s} {'ANE':>6s} {'total mean':>11s} {'total median':>13s} {'mJ/step>idle (median)':>22s}")
for l, r in rows.items():
    g = lambda k: st.mean(r[k]) if r[k] else 0.0
    med = st.median(r["Combined"])
    e = 0.0 if l == "idle" else (med - idle) / hz
    print(f"{l:9s} {len(r['Combined']):3d} {g('CPU'):7.0f} {g('GPU'):6.1f} {g('ANE'):6.1f} {g('Combined'):11.0f} {med:13.0f} {e:22.2f}")
iq = st.quantiles(rows["idle"]["Combined"], n=4)
print(f"\nidle interquartile range {iq[0]:.0f}-{iq[2]:.0f} mW: totals closer than that are noise.")
PY
chown -R "${SUDO_USER:-$USER}" out_k

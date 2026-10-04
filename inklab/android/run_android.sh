#!/bin/zsh
# One InkBench session on the connected Android phone.
#   [CHECK=1] ./run_android.sh [backends] [workloads] [ks]
# CHECK=1 adds an output check of every accelerated LiteRT run against fp32
# XNNPACK (after timing, so the timings are unchanged); results in check.txt.
# SKIP_INSTALL=1 reuses the APK already on the phone (no build, no install tap).
# DUMP=1 (with CHECK=1) also saves inputs, reference and outputs per condition to dump/.
# Defaults run everything: gles,litert-cpu,litert-gpu,nnapi,qnn-htp x
# ink,heat_s32,heat_s8,grayscott_s8 x K=1,2,4,8 (roughly 1-2 h on a mid-range phone).
#
# Protocol (keep identical across phones, note deviations in the results README):
#   - phone at >= 50% battery, NOT charging (unplug after install if the phone
#     allows adb over Wi-Fi; otherwise note "charging" in the README),
#   - airplane mode on, brightness at minimum, no other apps, screen stays on,
#   - let the phone cool to room temperature between sessions.
#
# Output: results/<model>_<date>/ with the CSV, the device properties, and the
# LiteRT placement lines ("Replacing N out of M node(s) with delegate ...").
set -e
cd "$(dirname "$0")"
ADB=${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}
backends=${1:-gles,litert-cpu,litert-gpu,nnapi,qnn-htp}
workloads=${2:-ink,heat_s32,heat_s8,grayscott_s8}
ks=${3:-1,2,4,8}
check=false; [ -n "$CHECK" ] && check=true
dump=false; [ -n "$DUMP" ] && dump=true

$ADB get-state >/dev/null || { echo "no device (adb devices)"; exit 1; }
model=$($ADB shell getprop ro.product.model | tr -d '\r' | tr ' ' '_')
out=results/${model}_$(date +%Y-%m-%d_%H%M)
mkdir -p $out
for p in ro.product.model ro.soc.manufacturer ro.soc.model ro.board.platform ro.build.version.release ro.build.fingerprint; do
  echo "$p=$($ADB shell getprop $p | tr -d '\r')"
done > $out/device.txt
$ADB shell dumpsys battery | grep -E "level|temperature|AC powered|USB powered" >> $out/device.txt

if [ -z "$SKIP_INSTALL" ]; then
  ./build.sh :app:assembleRelease
  $ADB install -r app/build/outputs/apk/release/app-release.apk
fi
$ADB logcat -c
$ADB shell rm -rf /sdcard/Android/data/org.inklab.bench/files/dump
$ADB shell am force-stop org.inklab.bench
$ADB shell am start -n org.inklab.bench/.MainActivity \
  --es backends $backends --es workloads $workloads --es ks $ks --ez check $check --ez dump $dump
echo "running; follow with: $ADB logcat -s InkBench"

# Keep the full log (placement lines come from the runtime's own tags).
# The stream dies when the cable is pulled or Wi-Fi debugging drops; restart it
# from the last line we hold so a lost connection cannot hang the session.
startlog() { $ADB logcat -v time ${1:+-T "$1"} >> $out/logcat.txt & logpid=$! }
: > $out/logcat.txt
startlog
# A Wi-Fi stream can also stall without dying, so ask the device's own buffer too.
while ! grep -q "InkBench.*DONE" $out/logcat.txt 2>/dev/null; do
  sleep 10
  if $ADB logcat -d -v time -s InkBench 2>/dev/null | grep "DONE " >> $out/logcat.txt; then
    $ADB logcat -d -v time > $out/logcat_device_buffer.txt; break
  fi
  if ! kill -0 $logpid 2>/dev/null; then
    $ADB wait-for-device
    startlog "$(grep -E '^[0-9]{2}-[0-9]{2} ' $out/logcat.txt | tail -1 | cut -c1-18)"
  fi
done
kill $logpid 2>/dev/null || true
csv=$(grep -o "DONE .*" $out/logcat.txt | tail -1 | cut -d' ' -f2)
$ADB pull $csv $out/
[ "$dump" = true ] && $ADB pull "$(dirname $csv)/dump" $out/
# Placement report: each delegate's own claim, between our BEGIN/END markers.
cat $out/logcat.txt $out/logcat_device_buffer.txt 2>/dev/null | LC_ALL=C sort -u > $out/logcat_all.txt
grep -E "PLACEMENT-(BEGIN|END)|Replacing [0-9]+ out of [0-9]+ node|delegate.*(partition|fallback)|QNN|NNAPI|nnapi" \
  $out/logcat_all.txt > $out/placement.txt || true
grep -E "SKIP|FAIL" $out/logcat_all.txt > $out/skipped.txt || true
grep -E "InkBench.*CHECK" $out/logcat_all.txt > $out/check.txt || true
echo "results in $out"

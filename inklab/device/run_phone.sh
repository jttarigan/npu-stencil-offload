#!/bin/zsh
# Device runs for the NPU-ink study on any paired iPhone.
#
#   run_phone.sh check            is a phone connected, paired, in Developer Mode?
#   run_phone.sh bench [quick]    build + install + launch InkBench (inklab/ios): the
#                                 standalone K-sweep / cost-model protocol, ~15 min,
#                                 runs by itself; the screen shows DONE at the end
#   run_phone.sh bench-pull       copy InkBench's CSVs to inklab/ios/results/<model>/
#
# The phone is found automatically (the single connected, paired iPhone); set
# DEVICE=<udid or name> to pick one when several are connected.
#
# The in-game runs in results/ used builds of the game, which is not part of
# this archive; device/game_frames.py analyses their frame CSVs.
# Analysis after pulling:
#   python3 inklab/netime/costmodel.py inklab/ios/results/<model>/bench_*.csv --md
#
# Release configuration throughout: Debug frame times on this renderer are
# several times slower and are not comparable with the published numbers.
# Free-provisioning signatures last 7 days; rebuild to refresh.
set -e
cd "$(dirname "$0")/../.."
TEAM=${TEAM:-}   # your Apple development team ID: TEAM=<id> ./run_phone.sh bench

pick_device() {
  if [[ -n "$DEVICE" ]]; then echo "$DEVICE"; return; fi
  local json=$(mktemp)
  xcrun devicectl list devices --json-output "$json" >/dev/null 2>&1
  python3 - "$json" <<'PY'
import json, sys
devs = json.load(open(sys.argv[1]))["result"]["devices"]
ok = [d for d in devs
      if d.get("hardwareProperties", {}).get("reality") == "physical"
      and d.get("hardwareProperties", {}).get("platform") == "iOS"
      and d.get("connectionProperties", {}).get("tunnelState") != "unavailable"
      and d.get("connectionProperties", {}).get("pairingState") == "paired"]
wired = [d for d in ok if d.get("connectionProperties", {}).get("transportType") == "wired"]
if len(wired) == 1: ok = wired   # a device still reachable over Wi-Fi must not win over the cable
if len(ok) != 1:
    sys.stderr.write(f"need exactly one connected, paired iPhone, found {len(ok)}; set DEVICE=...\n")
    sys.exit(1)
print(ok[0]["hardwareProperties"]["udid"])
PY
  rm -f "$json"
}

model_of() {   # marketing-independent model id, e.g. iPhone18,2
  local json=$(mktemp)
  xcrun devicectl list devices --json-output "$json" >/dev/null 2>&1
  python3 - "$json" "$1" <<'PY'
import json, sys
for d in json.load(open(sys.argv[1]))["result"]["devices"]:
    h = d["hardwareProperties"]
    if sys.argv[2] in (h.get("udid"), d.get("identifier"), d.get("deviceProperties", {}).get("name")):
        print(h.get("productType", "unknown")); break
else:
    print("unknown")
PY
  rm -f "$json"
}

build_install() {   # project scheme bundle [extra build settings...]
  local proj=$1 scheme=$2 bundle=$3; shift 3
  [[ -n "$TEAM" ]] || { echo "set TEAM=<your Apple development team ID>"; exit 1; }
  xcodebuild -project "$proj" -scheme "$scheme" -configuration Release \
    -destination "platform=iOS,id=$UDID" -allowProvisioningUpdates -allowProvisioningDeviceRegistration DEVELOPMENT_TEAM="$TEAM" \
    "$@" build > /tmp/run_phone_build.log 2>&1 \
    || { grep -E "error:" /tmp/run_phone_build.log | sort -u | head -5; echo "BUILD FAILED (log: /tmp/run_phone_build.log)"; exit 1; }
  tail -1 /tmp/run_phone_build.log
  local app=$(xcodebuild -project "$proj" -scheme "$scheme" -configuration Release \
    -destination "platform=iOS,id=$UDID" -showBuildSettings 2>/dev/null \
    | awk -F' = ' '/ BUILT_PRODUCTS_DIR /{d=$2} / FULL_PRODUCT_NAME /{n=$2} END{print d"/"n}')
  echo "== installing $app =="
  xcrun devicectl device install app --device "$UDID" "$app" 2>&1 | tail -2
}

pull() {   # bundle destination
  mkdir -p "$2"
  xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
    --domain-identifier "$1" --source Documents --destination "$2" 2>&1 | tail -2
  echo "CSVs now under $2:"; find "$2" -name "*.csv" | tail -10
}

UDID=$(pick_device)
MODEL=$(model_of "$UDID")
echo "== device $UDID ($MODEL) =="

case "${1:-}" in
  check)
    xcrun devicectl device info details --device "$UDID" 2>&1 \
      | grep -iE "name|productType|osVersion|developerModeStatus|pairingState|tunnelState" | head -12
    exit 0 ;;
  bench)
    ./inklab/ios/prepare.sh
    build_install inklab/ios/InkBench.xcodeproj InkBench com.smoketest.inkbench
    xcrun devicectl device process launch --device "$UDID" com.smoketest.inkbench ${2:+$2} 2>&1 | tail -2
    echo "== InkBench running (~15 min; ~3 min with quick). Keep the phone unlocked, screen on."
    echo "   When the screen shows DONE: ./inklab/device/run_phone.sh bench-pull"
    exit 0 ;;
  bench-pull)
    pull com.smoketest.inkbench "inklab/ios/results/$MODEL"; exit 0 ;;
  *) echo "usage: run_phone.sh check|bench [quick|placement]|bench-pull"; exit 1 ;;
esac


#!/bin/zsh
# Build the InkBench APK (JDK, Gradle and Android SDK from the environment).
# Usage: ./build.sh [gradle tasks...]   (default: :app:assembleRelease)
set -e
export JAVA_HOME=${JAVA_HOME:-$HOME/dev/jdk/Contents/Home}
export ANDROID_HOME=${ANDROID_HOME:-$HOME/Library/Android/sdk}
cd "$(dirname "$0")"
[ -f local.properties ] || echo "sdk.dir=$ANDROID_HOME" > local.properties
exec ${GRADLE:-gradle} "${@:-:app:assembleRelease}"

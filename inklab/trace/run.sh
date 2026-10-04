#!/bin/zsh
# Builds the golden-trace harness: wraps the game's ink kernels
# (../ink_kernels.metal) in a Swift string, compiles it with main.swift here,
# then runs it.
#   ./run.sh <outDir> [steps] [dumpEvery] [seed] [script.json]
set -e
cd "$(dirname "$0")"
mkdir -p .build
{ echo 'let shaderSource = """'; cat ../ink_kernels.metal; echo '"""'; } > .build/shader.swift
swiftc -O .build/shader.swift main.swift -o .build/trace_harness
.build/trace_harness "${1:-trace_out}" "${2:-300}" "${3:-30}" "${4:-12345}" ${5:+"$5"}

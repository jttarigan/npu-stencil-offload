# Where Core ML places each model (MLComputePlan), cpuAndNeuralEngine

Tool: inklab/netime/placement.swift (Mac); the same function runs at the start
of every InkBench sweep and writes Documents/placement_<model>_<time>.csv
(launch argument "placement" runs only that).

## M5 (placement_M5.csv)
Placement is all or nothing per model: every graph runs entirely on the neural
engine except heat s8 at K = 1 and 2, which Core ML keeps entirely on the CPU.
This confirms the timing-based observation that light graphs stay on the CPU
until batching makes them heavy enough (heat s8 moves at K >= 4).

## Devices
- iPad mini 6 (A15), iPad Pro 3rd gen (A12X): swept before placement logging
  existed; rerun `run_phone.sh bench` with launch argument placement (seconds)
  when the tablets are back. The A12X check is needed to read its NE anomaly
  (results/2026-09-19_ipadpro_A12X).
- iPhone 17 Pro Max: included automatically in tonight's bench.

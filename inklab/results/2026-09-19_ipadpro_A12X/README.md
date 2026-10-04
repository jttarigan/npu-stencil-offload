# iPad Pro 12.9-inch 3rd gen (iPad8,5, Apple A12X, 8-core NE), InkBench sweep (2026-09-19)

Full run, 304 rows, iPadOS 26.7, plugged in, thermal state 0 on every row.
Fits in costmodel.md.

## Per-step cost b and fixed cost a, back to back (ms)
workload        NE a / b        Core ML gpu b   CPU b    Metal a / b     Metal exec b
ink step        0.39 / 3.91     4.08            1.94     0.65 / 0.61     0.48
heat s32        0.53 / 1.44     0.53            0.84     0.82 / 0.43     0.40
heat s8         0.32 / 0.46     0.52            0.22     0.45 / 0.19     0.12
Gray-Scott s8   -0.18 / 2.51    0.75            0.67     0.38 / 0.17     0.12

## Reading
- ANOMALY: the `ne` path (cpuAndNeuralEngine) is SLOWER per step than the CPU
  on every workload (ink 3.9 vs 1.9 ms) and gains nothing from batching
  (ink K=8 per-step 1.00x). Its a is small and b large, i.e. the cost scales
  with steps, not submissions. Most likely the fp16 graphs do not (fully) run
  on the A12X neural engine and Core ML splits them, paying transfers per
  sub-step. UNVERIFIED: needs a placement check (MLComputePlan) on this iPad
  before the paper reads it as a generation effect.
- Paced NE busy time is 3-13 ms per submission (vs 0.8-3.9 b2b): waking the
  path after each 33 ms idle gap is expensive on this chip.
- Standalone Metal: fixed cost 0.38-0.82 ms with GPU execution a ~ 0, same
  pattern as the A15 and the M5.
- CPU control: a ~ 0, gain 1.07-1.32x.

## Placement (2026-10-04, placement_iPad8,5_1791048508.csv): the anomaly explained

`DEVICE=<udid> run_phone.sh bench placement` over Wi-Fi,
same iPadOS 26.7 and the same compiled models as the 2026-09-19 sweep
(netime/*.mlmodelc all dated before it). MLComputePlan with
cpuAndNeuralEngine, as the `ne` path uses:

- Core ML plans EVERY graph onto the CPU. On the neural engine: ink 11 of 136
  operations at K=1, 0 at K=2, 4, 8; heat s32, heat s8, Gray-Scott 1 op at
  K=1 and 0 beyond. The rest is CPU (2-13 ops per model unplaced: I/O).
  Off-NE op types include everything (conv, pad, add, mul, sub, resample, ...),
  so it is not one unsupported operator as on the A14.
- So the `ne` rows of this sweep are a second CPU path (Core ML's CPU
  execution of the fp16 graph), 2x slower per step than the `cpu` path for
  the ink (3.91 vs 1.94 ms), with no fixed cost to amortise: the anomaly
  is the planner, not a slow accelerator.
- Same API on the A15 (iPad14,1): all 16 models 100% on the NE; A14: all but
  the resample ops. Ladder: 2018 nothing, 2020 all but resampling, 2021+ all.
- Not established: whether this is the A12X's neural engine generation
  (H11) lacking support for these graphs or an OS policy; cost estimates are
  unavailable on this device (cost_* columns 0).

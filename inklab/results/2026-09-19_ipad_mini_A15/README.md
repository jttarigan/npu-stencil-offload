# iPad mini 6th gen (iPad14,1, Apple A15), InkBench standalone sweep (2026-09-19)

Full run (not quick): 304 rows, iPadOS 26.7, plugged in, thermal state 0
(nominal) on every row. Fits in costmodel.md (`costmodel.py --md`). The CSV was
written by an InkBench build that did not quote the model id ("iPad14,1"
splits into two fields); costmodel.py rejoins it.

## Fixed cost a per submission, back to back (ms)
workload        NE      standalone Metal   Metal exec only   | M5: NE    Metal
ink step        0.53    0.67               0.03              |    0.22   0.22
heat s32        0.43    0.73               0.05              |    0.15   0.20
heat s8         0.42    0.59               0.00              |    0.14   0.21
Gray-Scott s8   0.42    0.63               0.01              |    0.14   0.22

## Reading
- NE fixed cost is workload-independent again (0.42-0.53 ms) and about 3x the
  M5's: the A15's submission is dearer, as the Apple ladder asked.
- Standalone Metal pays a fixed cost at least as large as the NE's
  (0.59-0.73 ms) while its GPU execution has a ~ 0: the M5 reframe holds on a
  mobile chip, and here the standalone GPU submission is the MORE expensive one.
- Ink step K=8 per-step gain: NE 1.92x (M5 1.84x), Metal 1.70x; CPU control
  0.96-1.20x (a ~ 0), as on the M5.
- Paced at 30 Hz the NE's a rises to 1.0-1.5 ms (clock drop, as on the M5).
- Core ML's `gpu` compute-unit path is erratic (non-monotonic in K, R^2
  0.65-0.93); not a path the paper uses.

## Placement (added 2026-09-24, placement_iPad14,1_1790251570.csv)
`run_phone.sh bench placement` over Wi-Fi (iPadOS 26.7), Core ML compute plan
with .cpuAndNeuralEngine: ALL 16 models (ink, heat s32, heat s8, Gray-Scott at
K = 1,2,4,8) are 100% on the neural engine, no op on the CPU or GPU, including
the ink step's resample ops, which the A14 (iPhone 12) sends to the CPU. The
paper's "the A15 and later place resampling on the neural engine" is now
measured, not inferred. (The placement-only mode also writes an empty bench
CSV header; it was deleted.)

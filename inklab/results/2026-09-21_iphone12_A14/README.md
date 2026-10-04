# iPhone 12 (iPhone13,2, Apple A14), InkBench sweep (2026-09-21)

Full run, 304 rows, iOS 26.6.1, plugged in, thermal state 0 on every row.
Fits in costmodel.md; placement (MLComputePlan) in placement_*.csv.

## Placement: NOT all or nothing on this chip
- Heat and Gray-Scott: every op on the neural engine except the three
  float32<->float16 input/output casts (CPU). Harmless.
- Ink step: the two `resample` ops per step (the advection's grid sample) run
  on the CPU, plus a few casts/sub/mul/slices around them: 22 of 136 ops at
  K=1, 38 of 1041 at K=8 (16 of them resamples). Every sub-step inside a batch
  therefore crosses NE -> CPU -> NE.
- The M5 and A19 Pro place the whole ink graph, resampling included, on the NE,
  so between A14 (2020) and A15 (2021) Core ML started running this op on the
  accelerator. Likely also the A12X anomaly (unverified: needs `bench placement`
  on the iPad Pro).

## Fits, back to back (ms)
workload        path          t_sub K=1/2/4/8               a       b      gain@K=8
ink step        NE            1.447 / 4.296 / 12.500 / 29.473  -3.32   4.07   0.39x  (WORSE with K)
ink step        Metal         1.557 / 2.350 / 3.987 / 7.145    0.76    0.80   1.74x
ink step        Metal exec    0.746 / 1.483 / 2.945 / 5.858    0.02    0.73   1.02x
ink step        CPU           1.526 / 2.984 / 5.945 / 12.233  -0.07    1.53   1.00x
heat s32        NE            0.365 / 0.476 / 0.682 / 1.016    0.29    0.09   2.87x
heat s8         NE            0.326 / 0.360 / 0.398 / 0.534    0.30    0.03   4.88x
Gray-Scott s8   NE            0.452 / 0.568 / 0.789 / 1.451    0.28    0.14   2.49x

- The ink NE path is superlinear in K, so the linear model's a is negative and
  meaningless here; report the per-step cost (1.45 -> 3.68 ms) instead.
- Where the graph stays on the NE (the stencils), the fixed cost is 0.28-0.30 ms
  and batching gains as on every other chip.
- Standalone Metal a = 0.76 ms with GPU execution a ~ 0: the same pattern again.

## In-game, five backends (game_full/, game_full.md; second sample game_short/)
One full game per backend at Sparks x1, played by hand. Play phase:
backend     p95    p99    GPU ms   fluid on render thread   long frames
off         17.8   20.2   10.7     -                        0.0%
gpu         17.7   20.4   12.4     -                        0.0%
npu (K=1)   20.1   32.8   13.7     4.7 ms/frame (max 13)    3.2%
npu4 (K=4)  50.3   66.0   12.2     5.4 ms/frame (max 36)   16.3%
npua (async, K=1)  17.6   20.5   12.0   0.17 ms/frame (max 1.4)  0.5%
- npua was played on 2026-09-22 (perf_1790056567), one day after the other
  four, after the purple-pellet blackout landed in the game (a few extra
  velocity-only ink pushes per second while a blackout is on; the fluid
  solver and the Core ML graph are unchanged). Longer game (6313 play frames
  vs 3934-4773).
- ASYNC PAYS on this phone: the same Core ML step, taken off the render
  thread, brings the tail back to the GPU prepass (p95 17.6 vs 17.7, p99 20.5
  vs 20.4), long frames 0.5% vs 0.0%, GPU time 12.0 vs 12.4 ms (the prepass's
  ~1.7 ms over "off" is NOT given back, within variation). So the in-game cost
  of the NE on the A14 was entirely the synchronous call, not the accelerator.
- NOT measured: how often the fluid ran late (lateTicks is counted in the app
  but not written to the CSV); the fluid's slowdown under load is unknown.
- All four hold the 16.7 ms median; the difference is the tail.
- The GPU prepass costs ~1.7 ms GPU per frame over "off"; the NE backends do
  NOT give it back (GPU time within run-to-run variation of the prepass).
- The Core ML call is synchronous on the render thread, 4.7-5.4 ms per frame
  on average; long frames go 0% -> 3% (K=1) -> 16% (K=4).
- K=4 is WORSE here because on the A14 four steps cost more than four times one
  (resampling on the CPU, see placement above).
- The short first runs (one per backend, ~25 s each, nothing eaten) give the
  same ordering: npu 6%/npu4 18% long frames at the finale vs 0% for gpu.
- METRIC WARNING: the "CPU ms" column wraps view.currentDrawable and includes
  the wait for a free drawable (16 ms with the fluid off, because the CPU
  finishes early and waits). It is not CPU work; game_frames.py's verdict no
  longer uses it. The first verdict lines printed before that fix were wrong.
- Verdict for RQ4 on this phone: the SYNCHRONOUS NPU backends do not pay;
  the asynchronous one does, on tail latency, without returning GPU time.
  This is the least favourable Apple chip (it splits the fluid graph); an
  A15/A19 Pro game run would show whether a chip that keeps the whole graph
  on the NE also gives GPU time back.

# iPhone 17 Pro Max (iPhone18,2, Apple A19 Pro), InkBench sweep (2026-09-19 night)

Full run, 304 rows, iOS 26.6.2, plugged in. CAVEAT: thermal state 1 (fair) on
every row, not 0 as on the Mac and the tablets; a rerun on a cool phone would
settle whether any figure is throttled. Fits in costmodel.md, placement in
placement_iPhone18,2_*.csv.

## Ink step, back to back (ms)
path                a fixed   b per step   K=8 per-step gain
neural engine       0.222     0.190        1.92x
standalone Metal    0.165     0.121        2.10x
Metal exec only     -0.005    0.107        0.98x
CPU (control)       -0.092    0.941        1.02x

M5 for comparison: NE 0.220 / 0.189, gain 1.84x. The phone reproduces the
laptop to within a percent on both terms: the cost model is not a laptop
artefact.

## Other workloads (NE a / b)
heat s32 0.221 / 0.076; heat s8 0.129 / 0.029; Gray-Scott s8 0.147 / 0.067.
a is workload-independent again (0.13-0.22 ms).

## Placement
All 16 models fully on the neural engine, including heat s8 at K=1 and 2 which
the M5 keeps entirely on the CPU. Placement is all or nothing on both chips,
but the threshold is per chip.

## Not collected
The in-game runs (off / gpu / npu / npu4): the phone was disconnected after the
`off` build was installed. No game CSV was pulled.

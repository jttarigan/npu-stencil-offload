# K-step unrolled Core ML graph — validated timing (2026-09-15)

Environment validated before the run: AC power, Low Power Mode off, no
simulator booted, OneDrive / Time Machine / Spotlight idle, 78% CPU idle,
no thermal warnings. Supersedes results/2026-09-15_unroll_prelim, which was
taken with an iOS simulator running a game in the background.

Apple M5. Model ink_unroll_128x256_j26_fp16_uK.mlpackage (trace/unroll_graph.py):
K simulation steps per submission, splat fields stacked (K,2,H,W)/(K,3,H,W),
pressure warm-started across sub-steps as in the game. Harness: netime.

## Back to back, neural engine, 500 calls x 3 (ms)
K   repeats                  median   per simulation step
1   0.416  0.380  0.378      0.380    0.380
2   0.571  0.565  0.567      0.567    0.284
4   0.956  0.951  0.953      0.953    0.238
8   1.720  1.715  1.720      1.720    0.215
Linear fit: 0.188 ms fixed per submission + 0.192 ms per step.
Speedup per step, K=1 -> K=8: 1.77x.

## Paced at the game's 30 Hz cadence (submissions at 30/K Hz), ~15 s x 3
K   repeats                  median   per step   CPU busy per second of play
1   1.405  1.367  1.390      1.390    1.390      41.7 ms/s
2   1.976  1.975  1.983      1.976    0.988      29.6 ms/s
4   2.998  2.922  2.979      2.979    0.745      22.3 ms/s
8   5.093  5.164  5.177      5.164    0.646      19.4 ms/s
Linear fit: ~0.85 ms fixed per submission + ~0.54 ms per step.
Speedup per step, K=1 -> K=8: 2.15x. Occupancy falls 4.2% -> 1.9% of one
second of wall clock.

## Control: is the gain amortisation or arithmetic? (back to back, 300 x 3)
path              K=1 median   K=8 median   per step K=8   speedup
Core ML on NE     0.380        1.720        0.215          1.77x
Core ML on GPU    0.973        2.500        0.313          3.11x
Core ML on CPU    0.963        7.527        0.941          1.02x
The CPU gains nothing. The win is amortisation of a per-submission cost
that exists only on the accelerators, not reduced arithmetic.

## Against the shipped Metal kernels (0.089 ms per step, gameplay trace)
Best unrolled NE per step: 0.215 ms back to back (2.4x Metal), 0.646 ms
paced (7.3x Metal). Unrolling narrows the paced gap from 15.6x to 7.3x but
does not close it on the M5, where the GPU is uncontended.

## Equivalence (trace/unroll_equiv.py, CPU+NE)
One 8-step submission vs eight chained 1-step submissions, after 8 steps:
vel max|diff| 0.797 (2.2% of range), dye 2.7e-3 (0.27%), pr 0.148 (0.43%).
Within float16 noise; unrolling does not change the simulation.

## Cost of the trick
Splats batch by K frames, so ink lags the player by K/30 s (0.27 s at K=8).
Needs a perceptual check before it is proposed as a shipping mode.

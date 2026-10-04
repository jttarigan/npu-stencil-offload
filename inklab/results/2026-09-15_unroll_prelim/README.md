# K-step unrolled Core ML graph — preliminary timing (2026-09-15)

PRELIMINARY: taken while the Mac may have been in light interactive use.
Re-run on a quiet machine before the numbers go in the paper.

Model: ink_unroll_128x256_j26_fp16_uK.mlpackage (trace/unroll_graph.py).
K simulation steps in one submission; splat fields stacked (K,2,H,W)/(K,3,H,W);
pressure warm-starts across sub-steps exactly as in the game.
Apple M5, battery, Low Power Mode off. netime, 300 calls back to back;
paced runs at 30/K Hz for ~10 s.

## Back to back, neural engine (ms)
K   per submission   per simulation step
1   0.389            0.389
2   0.592            0.296
4   0.972            0.243
8   1.753            0.219
Linear fit: 0.194 ms fixed per submission + 0.195 ms per step.

## Paced at the game's 30 Hz cadence, neural engine (busy ms)
K   per submission   per simulation step   CPU busy per second of play
1   1.404            1.404                 42 ms/s
2   1.981            0.991                 30 ms/s
4   2.735            0.684                 21 ms/s
8   4.317            0.540                 16 ms/s
Linear fit: ~1.0 ms fixed per submission + ~0.40 ms per step.

## Control: does unrolling help every processor? (back to back, ms per step)
path              K=1     K=8     speedup
Core ML on NE     0.389   0.219   1.8x
Core ML on GPU    0.957   0.346   2.8x
Core ML on CPU    0.980   0.942   1.0x
The CPU shows no gain, so the win is amortisation of a per-submission cost
that exists only on the accelerators, not reduced arithmetic.

## Equivalence (trace/unroll_equiv.py, CPU+NE)
One 8-step submission vs eight chained 1-step submissions, after 8 steps:
  vel max|diff| 0.797 (2.2% of range)   dye 2.7e-3 (0.27%)   pr 0.148 (0.43%)
Within float16 noise; unrolling does not change the simulation.

## Fidelity against the float32 NumPy reference (from a GPU state, step 20)
K=1 vel 6.0e-1  dye 3.9e-3  pr 3.0e-1  dye mass 0.9999
K=2 vel 1.5e+0  dye 7.1e-3  pr 3.8e-1  dye mass 0.9998
K=4 vel 1.4e+0  dye 1.3e-2  pr 4.6e-1  dye mass 0.9995
K=8 vel 3.2e+0  dye 2.4e-2  pr 1.1e+0  dye mass 0.9990
Growth is ordinary accumulation over more steps, not an unrolling artefact.

## Cost of the trick
Splats are batched: the host hands over K frames of splat fields at once, so
the ink responds to the player K frames late (K=8 at 30 Hz is 0.27 s). For a
background lava lamp this is a design knob, not a defect, but it needs a
perceptual check.

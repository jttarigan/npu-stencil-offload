# Energy per step, K = 1 vs K = 8 (2026-09-19, Apple M5, mains power)

energy/measure_k.sh, 60 one-second powermetrics samples per condition, every
load at 30 simulation steps per second (K=8 submits 3.75 times a second).
The raw files here are the QUIET rerun (18:14-18:19, other apps closed, machine
left alone). The first attempt, which was inside background noise, is kept in
`first_run_noisy/`.

## Validity
- Idle median 22 mW, interquartile range 18-38 mW (first run: 52-1469 mW).
- netime/metaltime pace with usleep, not a spin loop, so harness CPU is not
  counted as workload.
- Numbers below drop the first 8 samples of each condition: ne_k1's first ~5
  samples were before the model finished loading and compiling (near idle,
  then a 1.6 W compile burst). ne_k8 had a 5 s burst up to 8.8 W at samples
  25-29 (ANE rail also briefly 9-11 mW); medians are robust to it.

## Results (medians after trimming; energy = (median - idle median) / 30)
condition            CPU     accel rail    total    total IQR   mJ/step above idle
Metal standalone K=1  21 mW  GPU 7.3 mW     28 mW    25-33       0.20
Metal standalone K=8  20 mW  GPU 6.7 mW     27 mW    24-33       0.17
NE, K=1              300 mW  ANE 7.8 mW    307 mW   228-400      9.5
NE, K=8              360 mW  ANE 3.6 mW    362 mW   274-426     11.3
CPU (Core ML), K=1   447 mW  -             447 mW   415-481     14.2

Accelerator rail per step: NE 0.26 -> 0.12 mJ (2.2x less with batching;
reproduces the first run's 0.29 -> 0.12). GPU rail 0.24 -> 0.22 mJ.

## Reading
1. Batching cuts the neural engine's own energy per step as much as its time.
2. It does NOT cut package energy: the NE path costs ~300 mW on the CPU rail, about
   40x the ANE rail, and K=8 is no cheaper than K=1 (quartiles overlap, so
   "no saving", not "worse"). The first run's "NE K=8 costs more" was dismissed
   as noise; on a quiet machine it holds as "no saving".
3. Standalone Metal is nearly free in energy (~6 mW above idle, ~50x less per
   step than the NE path), even though it pays the same ~0.2 ms fixed time.
4. The NE path's CPU-rail power is NOT the runtime's CPU work (system-wide
   Time Profiler, 15 s each, see profile.md): netime's own CPU time is 28 ms/s
   at K=1 (0.93 ms/step) and 6.3 ms/s at K=8 (0.21 ms/step), 4.4x less with
   batching; aned 2-4 ms per 15 s; kernel equal to idle. ~28 ms/s of CPU cannot
   draw 300 mW, and it shrinks with K while the rail does not. Read as a
   platform power state held while the neural engine is in use; powermetrics
   does not resolve it further. (Input copying, MLE5BindInputBufferObjectBy-
   CopyingMultiArray -> memmove, is ~20% of netime's CPU: real, but small.)

Supersedes the 2026-09-13 paced run (results/2026-09-13_energy_paced_clean),
whose idle baseline was ~1.13 W and whose Metal load was trace_harness, not
metaltime; its "every path ~0.1 W above idle" is withdrawn from the paper.

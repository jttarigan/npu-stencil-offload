# Galaxy Tab S7+ (SM-T975, Qualcomm SM8250 / Snapdragon 865+), Android 13, 2026-09-21

First Qualcomm device. QUICK PASSES ONLY (K = 1 and 8, 3 reps): use them for
placement and orders of magnitude, NOT for cost-model fits. DEVIATIONS: ink pass
at 19-22% battery while charging over USB; stencil pass over Wi-Fi debugging,
battery rising to 90% (on the charger). Thermal status 0 throughout, 30-31 C.
A full sweep (K = 1,2,4,8, unplugged, >= 50%) is still owed.

## Why this device matters
NNAPI has vendor drivers here, unlike the Galaxy A56 (Android 15), where no
HAL service existed at all. `nnapi_hal_services.txt`: qti-default, qti-dsp,
qti-gpu, qti-hta (neuralnetworks HAL 1.0-1.3, build_kona). The NNAPI runtime
logs "Found interface" for all four and compiles on qti-default.

## Medians, ms per submission, back to back (paced in brackets)
workload      backend      K=1            K=8              per step @8   gain
ink           gles          1.41 (2.55)    14.97 (23.50)    1.87         0.75x
ink           litert-cpu   52.66 (52.82)  407.19 (407.75)  50.9          1.03x
ink           nnapi        FAIL           732.75 (725.65)  91.6          -
heat_s8       gles          0.62 (1.22)     4.72 (13.75)    0.59         1.04x
heat_s8       litert-cpu   11.76 (22.19)  102.00 (109.80)  12.75         0.92x
heat_s8       litert-gpu   21.55 (25.29)  174.90 (161.48)  21.86         0.99x
heat_s8       nnapi        24.75 (30.86)  128.73 (153.25)  16.09         1.54x
grayscott_s8  gles          0.47 (1.13)     4.21 (14.71)    0.53         0.89x
grayscott_s8  litert-cpu   32.24 (33.17)  234.41 (239.99)  29.30         1.10x
grayscott_s8  litert-gpu   43.87 (44.86)  402.25 (319.80)  50.28         0.87x
grayscott_s8  nnapi        64.35 (63.93)  327.28 (344.48)  40.91         1.57x
qnn-htp: SKIP everywhere (delegate will not apply; Hexagon 698 is a v66 DSP,
the QNN HTP backend needs v68+). litert-gpu: SKIP on ink (MIRROR_PAD replicate
padding, as on the A56).

## Reading
- NNAPI reaches a vendor driver but takes only a sliver of every graph:
  ink 45/188 (K=1) and 55/1497 (K=8); heat 6/26 and 15/208; Gray-Scott 45/146
  and 48/1168, always in 6-7 partitions on qti-default. The rest runs on the
  plain CPU kernels, so NNAPI is SLOWER than XNNPACK at K=1 everywhere
  (2.0-2.1x) and 1.3-1.8x slower at K=8.
- NNAPI is nevertheless the one path here that shows a fixed cost being
  amortised (1.5-1.6x at K=8): the 6-7 partition hand-offs are paid per
  submission, not per step. Two points only; needs the full K sweep for (a, b).
- NNAPI FAILS the ink graph at K=1: "gather index out of bounds", node 29
  GATHER. Unexplained. Guess (unverified): a vendor partition hands wrong
  indices back to the CPU GATHER. K=8 runs; its output was not checked.
- The LiteRT GPU delegate accepts the stencils in name only: 2/26 and 9/208
  nodes (heat), 16/146 and 17/1168 (Gray-Scott). It is the slowest path.
- GLES compute is 20-70x faster than every LiteRT path and shows NO batching
  gain back to back (0.9-1.0x): its fixed cost is small next to the step here,
  unlike the A56 (3.0 ms). Paced at 30/K Hz it slows 2-3.5x (DVFS), as on
  every other device. The ink GLES K=8 reps drift upward (13.8 -> 16.1 ms), low
  battery suspected; do not read the 0.75x as a batching result.

## Files
bench_ink_quick.csv, bench_stencils_quick.csv, device.txt,
nnapi_hal_services.txt, placement_ink.txt, placement_stencils.txt.
Raw logcats stay in the gitignored inklab/android/results/SM-T975_*.

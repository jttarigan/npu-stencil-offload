# Galaxy Tab S7+ (SM-T975, Qualcomm SM8250 / Snapdragon 865+), Android 13, 2026-09-24

FULL SWEEP, supersedes the 2026-09-21 quick passes (../2026-09-21_tabS7plus_SD865plus).
ink, heat_s32, heat_s8, grayscott_s8 x K = 1,2,4,8 x gles, litert-cpu, nnapi;
3 reps back to back + 3 paced. 282 rows, 16:14-18:23. litert-gpu and qnn-htp
not run (quick passes: GPU delegate refuses ink and is the slowest route on the
stencils; QNN HTP cannot apply to the v66 Hexagon).

Conditions (full protocol): unplugged, Wi-Fi adb, 72 -> 50% battery (ended
exactly at the protocol floor), 29.0-30.8 C, thermal status 0 throughout.
Set by adb before launch: power saving OFF (it was ON), brightness 1 manual
(was 78 auto), `am kill-all` (21 apps in recents). Settings restored after.
APK not reinstalled (installed 2026-09-21; app source unchanged since 09-20).
The first launch attempt died on `logcat -c` before the bench started.

## Placement (placement.txt): NNAPI takes a sliver, 6-7 partitions at every K
ink 45/188, 48/375, 51/749, 55/1497; heat_s32 6/98 ... 15/784; heat_s8 6/26,
8/52, 11/104, 15/208; Gray-Scott 45/146, 46/292, 48/584, 48/1168. The claimed
count barely grows with K and the partition count not at all.

## Fits (costmodel.md; medians of 3, ms per submission, back to back)
workload      backend     K=1 / 2 / 4 / 8                      a       b      R2     gain@8
ink           gles         1.30 / 2.99 / 6.31 / 11.05          0.23    1.38   0.992  0.94x
ink           litert-cpu  54.20 / 96.63 / 195.96 / 417.42     -5.24   52.34   0.998  1.04x
ink           nnapi        FAIL / 198.85 / 337.13 / 657.69    38.57   77.00   0.999  1.21x (K=2->8)
heat_s32      gles         2.52 / 3.04 / 6.65 / 9.75           1.47    1.07   0.965  2.07x
heat_s32      litert-cpu  46.20 / 96.88 / 191.63 / 382.36     -0.26   47.87   1.000  0.97x
heat_s32      nnapi       71.54 / 136.80 / 279.59 / 582.30    -7.47   73.34   0.999  0.98x
heat_s8       gles         1.38 / 1.77 / 2.37 / 3.63           1.09    0.32   0.999  3.04x
heat_s8       litert-cpu  11.27 / 22.41 / 46.15 / 93.35       -0.80   11.76   1.000  0.97x
heat_s8       nnapi       22.88 / 38.74 / 74.31 / 141.39       5.58   17.00   1.000  1.29x
grayscott_s8  gles         1.20 / 1.54 / 1.91 / 3.40           0.84    0.31   0.984  2.81x
grayscott_s8  litert-cpu  28.13 / 60.22 / 108.86 / 223.71      1.50   27.66   0.999  1.01x
grayscott_s8  nnapi       60.21 / 98.40 / 185.21 / 355.86     15.67   42.47   1.000  1.35x
!! costmodel.md's ink/nnapi rows are WRONG: K=1 failed, so the script took the
K=2,4,8 medians as K=1,2,4. The ink nnapi line above is refitted on K=2,4,8.

## Reading
- NNAPI is the slowest route on every workload: 1.5-2.1x the CPU at K=1
  (stencils), 1.5-1.6x at K=8 (all four). It still shows a fixed cost that
  batching amortises (heat_s8 a = 5.6 ms, 1.29x; Gray-Scott 15.7 ms, 1.35x;
  ink 39 ms): the 6-7 driver/CPU hand-overs are paid per submission. heat_s32's
  73 ms per step hides any fixed term (a < 0, gain 0.98x).
- ink NNAPI fails at K=1 again (gather index out of bounds), runs K=2,4,8.
- GLES is far steadier than on the POCO (reps mostly within 30%; one outlier,
  ink K=1 2.12 vs 1.30/1.14) and shows the standalone
  submission cost: stencils a = 0.84-1.47 ms, gains 2.07-3.04x at K=8; ink
  a = 0.23 ms against b = 1.38, no gain. GLES is 8-42x faster than the CPU at K=1.
- CONTRADICTS the quick pass: it had GLES heat_s8 0.62 / 4.72 ms (1.04x) vs now
  1.38 / 3.63 (3.04x). The quick passes were charging / on 19-22% battery and
  their power-saving state was not recorded; use this sweep.
- The CPU control has no fixed term and no batching gain, as everywhere.

## Files
bench_SM-T975_1790241277.csv, costmodel.md, device.txt, placement.txt,
skipped.txt. Raw logcats in the gitignored inklab/android/results/SM-T975_2026-09-24_1614/.

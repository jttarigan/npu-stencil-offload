# POCO X6 Pro (2311DRK48G, MediaTek Dimensity 8300 / MT6897), Android 16 (HyperOS 3), 2026-09-24

Second session; completes the owed part of the 2026-09-22 partial sweep
(../2026-09-22_pocox6pro_D8300). heat_s8 and grayscott_s8, K = 1,2,4,8,
3 reps back to back + 3 paced, backends gles, litert-cpu, nnapi. 144 rows,
15:15-15:47. The heat_s8 series here SUPERSEDES the K = 1,2 partial of
2026-09-22 (all four K now from one session; the old K = 1,2 agree within 7%).

Conditions: unplugged from the start, 95 -> 88% battery, 33.7-34.8 C
(warmer than the 29-31 C of 2026-09-22), thermal status 0 throughout, recent
apps closed before launch, Wi-Fi adb (airplane mode off). DEVIATION:
brightness not lowered by adb (HyperOS denies WRITE_SETTINGS to the shell,
it was 44 with auto-brightness on); the author lowered it by hand from
the quick-settings panel during the run (confirmed; exact time not recorded,
so the first conditions may have run at the original brightness). The APK was NOT reinstalled (same
build as 2026-09-22); the run used `run_android.sh`'s steps minus
build/install, with the full logcat streamed to the Mac.

## Placement (placement.txt)
NNAPI takes EVERY operation of both graphs in ONE partition at every K
(heat_s8 26/26, 52/52, 104/104, 208/208; Gray-Scott 146/146 ... 1168/1168),
"compilation finished successfully on mtk-neuron" for all eight models.
With heat_s32 (2026-09-22) all three stencils run whole on the APU. No
bench SKIP/FAIL (the one FAIL line in skipped.txt is an unrelated Google
WorkManager job).

## Fits (costmodel.md; medians of 3, ms per submission, back to back)
workload      backend      K=1 / 2 / 4 / 8                     a      b      R2     gain@8  ceiling
heat_s8       nnapi (APU)   1.09 / 1.53 / 2.34 / 4.20           0.62   0.44   0.999  2.08x   2.40x
heat_s8       litert-cpu    5.12 / 11.84 / 19.57 / 42.75        0.02   5.28   0.995  0.96x
heat_s8       gles          2.11 / 3.61 / 5.79 / 5.28           (R2 0.57, not usable)
grayscott_s8  nnapi (APU)   3.17 / 5.72 / 10.05 / 18.29         1.28   2.14   0.999  1.39x   1.60x
grayscott_s8  litert-cpu   33.42 / 67.58 / 137.17 / 276.56     -1.69  34.77   1.000  0.97x
grayscott_s8  gles          2.09 / 3.52 / 6.49 / 6.30           (R2 0.68, not usable)
Paced (30/K Hz): nnapi heat_s8 a = 2.03, gain 3.35x; nnapi grayscott_s8
a = 2.23, gain 1.66x.

## Reading
- The APU shows a + bK on all three stencils (with heat_s32: a = 0.79,
  b = 1.32, 1.45x). Fixed cost 0.62-1.28 ms, less uniform across workloads
  than the Apple neural engines (0.14-0.22 ms on the M5), and 5-9x the M5's
  for the same stencils (heat_s32 0.79 vs 0.145; Gray-Scott 1.28 vs 0.139).
- APU vs CPU at K = 1: heat_s8 4.7x, Gray-Scott 10.5x faster.
- The CPU control again has no fixed term and no batching gain.
- GLES is erratic AGAIN, now at 95% battery: the three reps of one condition
  differ by up to 3x (Gray-Scott K=1: 1.29 / 2.09 / 4.27) and K=8 is cheaper
  than K=4 in both workloads. So the non-monotonic GLES series of 2026-09-22
  was NOT the drained battery; it is this phone's GPU path (clocking is a
  guess, not tested). APU reps agree within 5%.
- heat_s8 CPU is slower than on 2026-09-22 (K=1 5.12 vs 3.38 ms) with a
  warmer phone; the APU numbers moved by < 7%.

## Still owed on this phone (optional)
- nnapi ink with setAllowFp16(false) (needs an APK rebuild + one tap on the
  phone for the install), and a check of the K = 1 ink output.

## Files
bench_2311DRK48G_1790237736.csv (144 rows), costmodel.md, device.txt,
placement.txt, skipped.txt. Raw logcats (24 + 15 MB) stay in the gitignored
inklab/android/results/2311DRK48G_2026-09-24_1515/.

# POCO X6 Pro (2311DRK48G, MediaTek Dimensity 8300 / MT6897), Android 16 (HyperOS 3), 2026-09-22

FIRST ANDROID DEVICE WHERE NNAPI REACHES THE NPU. Partial sweep (session cut
short): ink and heat_s32 complete at K = 1,2,4,8 (3 reps each); heat_s8 only
K = 1 and 2 (nnapi K=2 paced missing); grayscott_s8 NOT run. Backends gles,
litert-cpu, nnapi everywhere; litert-gpu only for ink (refused) and heat_s32
K = 1,2 (it was cut: 160-760 ms per submission, 3-5 of 196-392 nodes).
DEVIATIONS: 52% battery, charging over USB at the start, unplugged after the
first item, on Wi-Fi adb (airplane mode off), 38% at the end; thermal status 0
throughout, 29-31 C. The remaining items are owed: heat_s8 K=4,8 (+ nnapi K=2
paced), grayscott_s8 all K. Launch:
  ANDROID_SERIAL=<ip:5555> adb shell am start -n org.inklab.bench/.MainActivity \
    --es backends gles,litert-cpu,nnapi --es workloads heat_s8,grayscott_s8 --es ks 1,2,4,8
(HyperOS demands a tap on the phone for every adb install; the APK is already
installed, so relaunching needs no install.)

## NNAPI placement (placement.txt, nnapi_services.txt)
`lshal` lists NO neuralnetworks HAL, but `service list` shows three AIDL
devices: android.hardware.neuralnetworks.IDevice/mtk-neuron_shim, mtk-mdla_shim,
mtk-dsp_shim. LESSON: on Android 15+/16 check `service list`, not `lshal`
(the Galaxy A56 "no HAL service" note was drawn from lshal and should be
rechecked with `service list` if that phone returns; its behavioural evidence,
no partition line and 4.5x slower than the CPU, still stands).
- heat_s32: 196/196, 392/392, 784/784 nodes in ONE partition, "compilation
  finished successfully on mtk-neuron" at every K. Whole graph on the APU.
- ink: the Neuron compiler rejects MIRROR_PAD (MVPU converter fails, EDPA
  unsupported), FLOOR, GATHER, RESHAPE, CAST (MDLA unsupported); 269/375 (K=2),
  375/749 (K=4), 375/1497 (K=8) nodes in 6-7 partitions; then the interpreter
  FAILS at K = 2, 4, 8 with "gather index out of bounds". CORRECTED 2026-09-24:
  K = 1 did NOT fail ("done ink K=1 nnapi", 40.67 ms b2b, 40.70 paced, vs
  litert-cpu 47.26); its placement is lost (ring buffer rolled) and its output
  was not checked. The Tab S7+ is the mirror image (fails K=1, runs K=8). HYPOTHESIS (untested):
  setAllowFp16(true) lets the APU compute the resampling indices in half
  precision, which cannot represent flattened indices above 2048 on a
  32768-cell grid. Test: rerun nnapi ink with setAllowFp16(false). The Tab S7+
  K=1 failure may share the cause.

## Fits (costmodel.md; medians of 3, ms per submission, back to back)
workload  backend     K=1 / 2 / 4 / 8                    a      b      R2     gain@8
heat_s32  nnapi (APU)  2.05 / 3.44 / 6.18 / 11.32        0.79   1.32   1.000  1.45x
heat_s32  gles         8.47 / 13.66 / 12.01 / 27.04      6.07   2.46   0.879  2.51x
heat_s32  litert-cpu  10.85 / 21.85 / 45.65 / 90.79     -0.66  11.45   1.000  0.96x
heat_s32  litert-gpu 200.7 / 387.0 / 763.1 / (cut)      12.6  187.6   1.000  1.05x
ink       gles         2.73 / 11.25 / 19.40 / 27.71      2.83   3.32   0.917  0.79x
ink       litert-cpu  47.26 / 97.50 / 194.4 / 395.5     -2.71  49.70   1.000  0.96x
ink       nnapi        40.67 at K=1 (placement lost), FAIL at K=2,4,8; litert-gpu refused (MIRROR_PAD)
heat_s8   nnapi (APU)  1.05 / 1.43 (K=4,8 owed)
heat_s8   gles         2.19 / 2.89
heat_s8   litert-cpu   3.38 / 5.63
Paced (30/K Hz): nnapi heat_s32 2.86 / 4.60 / 7.03 / 12.17 (a = 1.76, gain
1.88x); every other path slows 1.5-3x paced, as on all devices (DVFS).

## Reading
- The APU runs the heat stencil 5.3x faster than XNNPACK and 4.1x faster than
  GLES compute at K=1, with a fixed cost a = 0.79 ms (1.76 ms paced) that
  batching amortises to 1.45x (1.88x paced) at K=8: the a + bK shape of the
  Apple neural engines, on Android, through NNAPI. Ceiling (a+b)/b = 1.6x.
- The ink graph does not run on this route: same replicate-padding and
  index-gather ops that the LiteRT GPU delegate and the A14 fall over.
- GLES ink K=1..8 is erratic (2.7 / 11.3 / 19.4 / 27.7, R2 0.92, K=1 far below
  the line) and the K=8 per-step figure is WORSE than K=1 (0.79x); the heat_s32
  gles series is also non-monotonic (K=4 below K=2). Battery fell 52 -> 38%
  across the session; treat the GLES fits on this phone as provisional.
- The CPU control again shows no fixed term and no gain from batching.

## Files
bench_part1_ink_heat32_k12.csv (117 rows), bench_part2_heat32_k48.csv (36),
bench_part3_heat8_k1_partial.csv (33), costmodel.md, device.txt,
nnapi_services.txt, placement.txt, skipped.txt. Raw logcats in the gitignored
inklab/android/results/2311DRK48G_2026-09-22_1529/ (logcat*.txt; the main
ring buffer rolled, so ink K=1 nnapi placement survives only in logcat.txt).

# POCO X6 Pro (2311DRK48G, MediaTek Dimensity 8300 / MT6897), Android 16 (HyperOS 3), 2026-09-30

Third session on this phone, and the first output checks of any Android
accelerator run. Before today, InkBench timed every backend but never compared
what it computed. New tooling (all opt-in, timings unchanged):

- `CHECK=1 run_android.sh ...`: after each accelerated LiteRT condition has been
  timed, the same model runs once on litert-cpu (fp32 XNNPACK) from the same
  seeded inputs, and `CHECK` lines log max / RMS error as a fraction of the fp32
  output's range, plus the count of non-finite values (`check.txt`).
- `DUMP=1` (with CHECK=1): inputs, fp32 reference and backend outputs as raw
  little-endian float32 per condition (`dump/<condition>/`, `meta.txt` names the
  tensors). `../../android/dump_check.py` recomputes padtest and the stencils in
  NumPy from those inputs with edge-replicate and with REFLECT padding.
- `SKIP_INSTALL=1`: reuse the installed APK (HyperOS needs a tap per install).
- Backend `nnapi-fp32`: NNAPI with `setAllowFp16(false)`; `nnapi` unchanged.
- Model variants from `trace/tflite_graph.py`: `ink --int-index` (`ink_i32`, the
  gather index built in int32), `heat|grayscott --edge-concat` (`*_ec`, the
  width-1 replicate padding built from edge slices + CONCATENATION instead of
  MIRROR_PAD), and `padtest` (one MIRROR_PAD SYMMETRIC, nothing else). The
  original models are unchanged (byte-identical to HEAD).

## Conditions

Unplugged over Wi-Fi adb, airplane mode on with Wi-Fi back on, brightness at
minimum by hand (auto off), power saving off, recent apps closed, thermal status
0 in every row. DEVIATIONS: the phone came off the charger at 54% and 36.6 C
(battery), warmer than 2026-09-22 (29-31 C) and 2026-09-24 (34-35 C); runs 2-3
ran at 39-41 C. The adb connection dropped around 19:41 during run 1 (the bench kept
running on the phone; wireless debugging had switched off) and was re-enabled at 20:29 with the phone recharged to
61%. Runs 1 and 2 were stopped early on purpose (below). Correctness does not
depend on any of this; the timings are from a warm phone.

| run | folder | start | batt C | what |
|---|---|---|---|---|
| 1 | 1_ink | 18:31, 54% | 35.7-40.7 | ink, nnapi / nnapi-fp32 / litert-cpu, K=1,2,4,(8); stopped during K=8 nnapi-fp32 |
| 2 | 2_ink_i32 | 20:30, 60% | 40.4-41.0 | ink_i32, nnapi / nnapi-fp32, K=1,2; stopped after K=2 nnapi |
| 3 | 3_stencils | 20:41, 56% | 39.1-41.0 | heat_s32, heat_s8, grayscott_s8 as in the paper, nnapi, K=1-8 |
| 4 | 4_padtest_dump | 21:13, 62% | 35.5 | padtest + the three stencils at K=1, with dumps |
| 5 | 5_stencils_edgeconcat | 21:17, 62% | 34.9-35.5 | the three stencils with edge-concat padding, nnapi, K=1-8, with dumps |

## Finding 1: the APU executes MIRROR_PAD SYMMETRIC as REFLECT

The stencils (run 3) run whole on the APU as before (one NNAPI partition, every
node, `mtk-neuron`; timings within 5% of 2026-09-22/24 for 10 of 12 conditions,
12-13% for heat_s8 K=1,2) but their outputs are wrong: max error 12-18% of range for heat, 2.0-5.4%
for Gray-Scott; no non-finite values.

padtest (run 4) isolates it: one MIRROR_PAD through NNAPI is off by 0.189 (95% of
range) on the border ring and exact inside. Recomputing from the dumped inputs
(`4_padtest_dump/dump_check.txt`, absolute errors, border band = 4 cells):

| condition | vs edge-replicate: border / interior | vs REFLECT: border / interior |
|---|---|---|
| padtest | 0.189 / 3.1e-5 | 3.1e-5 / 3.1e-5 |
| heat_s32 K=1 | 0.0066 / 0.0026 | 2.0e-5 / 1.8e-5 |
| heat_s8 K=1 | 0.022 / 6.5e-4 | 3.6e-5 / 3.8e-5 |
| grayscott_s8 K=1 | 0.017 / 3.8e-4 | 3.9e-4 / 3.5e-4 |

The phone's output equals the REFLECT-padded computation everywhere to fp16
rounding (3.05e-5 is half the fp16 spacing for values up to 0.1). The NumPy
edge model reproduces the fp32 CPU output to 1e-8, so the comparison is sound.
Apart from that one operator the APU computes the stencils correctly at fp16
precision. Before the dumps, a statistical prediction per condition (REFLECT
padding vs fp16 rounding alone, which differ by ~100x) was consistent with all 12
run-3 conditions: same order for heat, two-digit agreement for Gray-Scott at K>=2.

## Finding 2: with edge-concat padding the stencils are correct, still whole on the APU, same cost

Run 5: every `_ec` model is claimed whole by NNAPI (heat_s32 K=8: 2064/2064
nodes, one partition) and compiled on `mtk-neuron` (12/12). Output max error
0.03-0.04% of range (heat), 0.11-0.20% (Gray-Scott), no non-finite values; the
dumps match edge padding, not REFLECT (`5_stencils_edgeconcat/dump_check.txt`;
only the K=1 dumps are kept here, the K>=2 dumps, 13 MB, stay in the gitignored
android/results/2311DRK48G_2026-09-30_2117/).

Fits, same session, back to back (`costmodel_stencils.md`):

| workload | a ms (MIRROR_PAD -> edge-concat) | b ms/step | gain at K=8 |
|---|---|---|---|
| heat_s32 | 0.94 -> 0.86 | 1.27 -> 1.32 | 1.51x -> 1.57x |
| heat_s8 | 0.85 -> 0.71 | 0.41 -> 0.43 | 2.40x -> 2.22x |
| grayscott_s8 | 1.25 -> 1.24 | 2.19 -> 2.13 | 1.41x -> 1.40x |

For reference, the paper's POCO numbers (2026-09-22/24, cooler phone, old
graphs): heat_s32 a 0.79 / b 1.32 / 1.45x, heat_s8 0.62 / 0.44 / 2.08x,
Gray-Scott 1.28 / 2.14 / 1.39x. The a + bK shape and the APU placement survive
the fix; the numbers for the paper should come from the edge-concat graphs,
ideally a clean (cool, charged) session.

## Finding 3: the ink step does not work through NNAPI on this phone

| run | model / backend | runs at | output vs fp32 |
|---|---|---|---|
| 1 | ink / nnapi (fp16) | K=1 only; K=2,4,8 "gather index out of bounds" | velocity 67% and pressure 65% non-finite; dye max 100% of range |
| 1 | ink / nnapi-fp32 | K=1,2,4 (K=8: 10.7 s, 2.7 s, 3.1 s per submission) | no NaN; RMS 8-21% of range, max 41-72% |
| 2 | ink_i32 / nnapi (fp16) | K=1,2 (the crash is gone) | velocity 67-93%, pressure 65-92% non-finite; dye RMS 13-21% |
| 2 | ink_i32 / nnapi-fp32 | K=1 | same as ink / nnapi-fp32 (RMS 10-21%) |

Reading, from strongest to weakest evidence:
- The "gather index out of bounds" failure is the float index: y*W + x reaches
  32767, which fp16 cannot hold (above 16384 it rounds to multiples of 16;
  32767 rounds to 32768). An emulation on the Mac with only the index in fp16
  produces out-of-bounds indices, and the int32 index removes the crash on the
  phone.
- The remaining error is not the index. The dye output at K=1 depends only on
  the advection (the pressure solve never touches it), and its RMS error, 21% of
  range, is what fully uncorrelated values give on this input
  (sqrt(2) x 0.0577 / 0.398 = 0.205). So the gather returns values from the
  wrong cells, in both precision modes and with either index. Mechanism
  (GATHER or the RESHAPE around it in the driver) not isolated; a gather-only
  padtest-style model would settle it.
- The fp16 NaNs are separate (unchanged by the index fix). Candidate, untested:
  `curl / max(ln, 1e-12)` in the vorticity term, where 1e-12 underflows to 0 in
  fp16 and 0/0 = NaN where ln = 0.
- The ink graph also contains MIRROR_PAD (every conv3), so Finding 1 applies to
  it as well.

The paper's "POCO NNAPI ink ran at K=1 (40.67 ms)" was an invalid output.
Today's K=1 nnapi time was 39.1 ms (40.4 / 39.1 / 39.1).

Ink timings, back to back, median of 3 (ms per submission; outputs wrong for
every NNAPI row):

| K | litert-cpu | nnapi | nnapi-fp32 | ink_i32 nnapi | ink_i32 nnapi-fp32 |
|---|---|---|---|---|---|
| 1 | 44.3 | 39.1 | 97.4 | 53.2 | 171.7 |
| 2 | 94.4 | fails | 226.7 | 69.2 | - |
| 4 | 199.0 | fails | 390.9 | - | - |
| 8 | - | fails | 3062 (reps 10699 / 2660 / 3062) | - | - |

In fp32 mode NNAPI claims 99 of 188 nodes in 6 partitions at K=1 (178/375 at K=2).

## What this changes elsewhere

- Paper (not edited today): POCO rows and text for heat/Gray-Scott should use
  the edge-concat numbers and mention the MIRROR_PAD fault; the POCO ink row
  becomes "invalid output at every K"; the limitation "NNAPI outputs of
  partial-graph runs were not checked" is superseded on the POCO.
- Not checked yet: the Tab S7+ NNAPI runs (ink at K=8, the partial stencil
  graphs) and anything on the A56. GLES uses its own shaders and litert-cpu is
  the reference, so neither is affected by the NNAPI faults.

## Files

Per run: bench CSV, device.txt, check.txt, placement.txt (NNAPI / Neuron lines),
skipped.txt where something failed. 4_padtest_dump/dump and
5_stencils_edgeconcat/dump (K=1 only) hold the raw float dumps. Raw logcats
(up to 0.5 GB per run) stay in the gitignored
inklab/android/results/2311DRK48G_2026-09-30_{1831,2030,2041,2113,2117}/.

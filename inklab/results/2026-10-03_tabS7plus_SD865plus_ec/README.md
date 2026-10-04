# Galaxy Tab S7+ (SM-T975, Qualcomm SM8250 / Snapdragon 865+), Android 13, 2026-10-03

Third session on this tablet, first with the edge-padded graphs and with
output checks. Purpose: (1) CPU rows free of
the MIRROR_PAD handicap (footnote c / limitation), (2) check the Qualcomm NNAPI
outputs (paper: "timed but not checked"), (3) the LiteRT GPU delegate on a
second device. Same APK as the POCO's 2026-10-02 session (built 2026-10-02
10:10, no app source change since); installed over Wi-Fi without a tap
(Samsung, unlike HyperOS).

## Conditions

Unplugged, Wi-Fi adb from the start, 99% battery at 18:12, 28.9-31.9 C,
thermal status 0 in 175 rows and 1 ("light") in the last 23 (ink K=4 and 8,
31.6-31.9 C; their times stay linear in K, no sign of throttling). Set by adb:
brightness 1 manual (was 37 auto), `am kill-all`; power saving was already
off. DEVIATION: airplane mode OFF (switching it on over adb would drop the
Wi-Fi link). Settings restored after.

The tablet's adb transport changed name during run 1 (two mdns entries, one
vanished); the run itself completed (DONE 19:20) but run_android.sh's log poll
hung on the dead serial, so the CSV pull and the placement/check extraction
were done by hand at 22:10 with the surviving serial. The streamed
logcat.txt stops at 19:24; the device buffer pulled at 22:10 still held every
InkBench line. Raw logcats in the gitignored
inklab/android/results/SM-T975_2026-10-03_1812/.

| run | folder | start | what |
|---|---|---|---|
| 1 | 1_sweep_ec | 18:12 | heat_s8_ec, grayscott_s8_ec, ink_i32_ec_fs x litert-cpu, nnapi, litert-gpu x K=1,2,4,8, CHECK (no install) |
| 2 | 2_ink_k1_fp16_fp32 | 22:44 | ink_i32_ec_fs x nnapi, nnapi-fp32 x K=1, CHECK + DUMP (65-68%, 28.5 C) |

GOTCHA (run 2): the bench started THREE times in one process (three
"InkBench on" banners within 0.5 s, every line triplicated about 1 s apart).
MainActivity starts a bench thread in every onCreate and is locked to
portrait; the tablet had been left lying in landscape, so the launch
recreated the activity twice. Run 2's TIMINGS ARE INVALID (three concurrent
benches); its outputs are deterministic and identical across the three copies
(and identical to run 1's K=1 nnapi check to every digit), so the checks and
dumps stand. Run 1 had one banner and exactly 24 rows per condition. Before a
tablet run: stand it upright (portrait) before launching, or guard onCreate
(static started flag / configChanges) in the next APK.

## Run 1: edge-padded graphs, CPU + NNAPI + GPU delegate, checked

### Fits (costmodel.md; medians of 3, ms per submission, back to back)

| workload | path | K=1 / 2 / 4 / 8 | a | b | R2 | gain K=8 |
|---|---|---|---|---|---|---|
| heat_s8_ec | litert-cpu | 2.159 / 3.955 / 7.981 / 16.201 | 0.010 | 2.017 | 1.000 | 1.07x |
| heat_s8_ec | nnapi | 27.359 / 34.407 / 63.425 / 124.797 | 8.915 | 14.289 | 0.994 | 1.75x |
| heat_s8_ec | litert-gpu | 1.632 / 1.968 / 3.180 / 5.999 | 0.800 | 0.639 | 0.993 | 2.18x |
| grayscott_s8_ec | litert-cpu | 7.817 / 15.946 / 34.458 / 71.084 | -1.759 | 9.089 | 1.000 | 0.88x |
| grayscott_s8_ec | nnapi | 61.680 / 101.013 / 166.328 / 312.161 | 27.008 | 35.543 | 1.000 | 1.58x |
| grayscott_s8_ec | litert-gpu | 3.498 / 5.259 / 14.253 / 18.653 | 2.066 | 2.227 | 0.908 | 1.50x |
| ink_i32_ec_fs | litert-cpu | 12.021 / 25.767 / 52.539 / 105.420 | -1.022 | 13.322 | 1.000 | 0.91x |
| ink_i32_ec_fs | nnapi | 95.779 / 158.862 / 285.049 / 584.804 | 18.261 | 70.097 | 0.998 | 1.31x |
| ink_i32_ec_fs | litert-gpu | 17.126 / refused / refused / refused | | | | |

GPU delegate fits are noisy (Gray-Scott R2 0.908, K=4 above the line); read
its a as order of magnitude only.

### Placement (placement.txt)

- CPU (XNNPACK): every stencil graph in ONE partition at every K (66/66 ...
  528/528 heat, 226/226 ... 1808/1808 Gray-Scott); ink 318/342 in 5, 630/678
  in 9, 1254/1350 in 17, 2502/2694 in 33 partitions (gathers etc. stay on the
  plain kernels), as on the POCO.
- NNAPI: still a sliver, 6-7 partitions at every K: heat 12/66, 13/132,
  16/264, 20/528; Gray-Scott 54/226, 55/452, 57/904, 57/1808; ink 43/342,
  52/678, 54/1350, 59/2694.
- GPU delegate: stencils whole, one partition, every K; ink K=1 229 of 342 in
  2 partitions (same as the POCO); ink K=2,4,8 refused at init ("Batch size
  mismatch", 263 ops would go to the GPU in 3 partitions).

### Output checks (check.txt; max / rms error as % of the fp32 CPU output's range)

- heat_s8_ec: NNAPI max 0.03-0.05%, GPU delegate 0.12-0.14%, every K.
- grayscott_s8_ec: NNAPI max 0.05-0.15%, GPU delegate 0.13-0.24%, every K.
- ink_i32_ec_fs, GPU delegate K=1: velocity 0.06%, dye exact, pressure 0.20%.
- ink_i32_ec_fs, NNAPI: runs at EVERY K (the int32 index removed the K=1
  "gather index out of bounds" of 2026-09-21/24). K=4 and 8 agree within
  0.09-0.36% max (rms 0.01%); K=1 and 2 are off by up to 4.0-8.4% max, rms
  0.6-1.6% (velocity, dye, pressure alike). Followed up in run 2.
- No non-finite value anywhere.

### Reading

- The old Tab S7+ CPU rows (2026-09-24, MIRROR_PAD graphs) were 3.6-5.2x too
  slow at K=1: ink 54.20 -> 12.02 ms, heat_s8 11.27 -> 2.16, Gray-Scott
  28.13 -> 7.82. Against the fixed CPU, NNAPI is 8.0x (ink), 12.7x (heat_s8),
  7.9x (Gray-Scott) slower at K=1 (paper said 1.5-2.1x).
- NNAPI still shows a per-submission fixed cost that batching amortises
  (heat 8.9 ms, 1.75x; Gray-Scott 27.0 ms, 1.58x; ink 18.3 ms, 1.31x), the
  6-7 driver/CPU hand-overs, as on 2026-09-24.
- Qualcomm's NNAPI computes the stencils correctly (unlike MediaTek's
  MIRROR_PAD, which these graphs avoid anyway).
- GPU delegate on a second device: stencils whole and correct, FASTER than
  the CPU here (heat_s8 1.63 vs 2.16 ms at K=1, 6.00 vs 16.20 at K=8;
  Gray-Scott 3.50 vs 7.82, 18.65 vs 71.08), unlike the POCO where it lost at
  K=1. GLES (2026-09-24) is still faster for heat_s8 (1.38 / 3.63). Ink K=1
  correct but 1.4x slower than the CPU (17.13 vs 12.02); batched ink refused.
  "No route runs the fluid faster than the CPU" still holds on this device.

## Run 2: is the K=1 ink error half precision or a driver fault?

Same graph, same inputs, the bench's `nnapi` setting (setAllowFp16(true),
as in every NNAPI row) and `nnapi-fp32` (setAllowFp16(false)). Dumps in
2_ink_k1_fp16_fp32/dump, analysis `android/ink_dump.py` -> ink_dump.txt.

| mode | velocity max / rms | dye max / rms | pressure max / rms |
|---|---|---|---|
| nnapi (fp16 allowed) | 7.25% / 1.27% | 8.35% / 1.55% | 5.63% / 0.79% |
| nnapi-fp32 | 3.5e-8 / 6e-9 | exact | 9.6e-8 / 1e-8 |

(% of the fp32 CPU output's range; the fp16 row reproduces run 1's K=1 check
to every digit.)

- In float32 the Qualcomm path matches the CPU EXACTLY (dye bit-identical).
  No driver fault: unlike MediaTek's, Qualcomm's GATHER (or whatever ops it
  claims) computes the right cells.
- The fp16 error is spread over the whole grid (border and interior alike;
  18-51% of cells off by more than 1% of range) and is far coarser than
  per-op rounding (median 3-5% of each cell's own value). It GROWS WITH THE
  COORDINATE: rms by row band 0-16 / 16-32 / 32-64 / 64-128 / 128-256 =
  0.52 / 0.61 / 0.69 / 0.96 / 1.60% (velocity), dye and pressure alike, and
  more weakly with the column (the grid is 128 wide). That is the signature
  of back-traced positions held in half precision, whose spacing doubles
  every octave (1/8 cell above 128), plus a floor of about 0.5%.
- The check's inputs are random with |velocity| <= 0.1 cell per step, so the
  back-trace moves each position by less than one fp16 step in the upper
  rows: a worst case for this effect. Game velocities are larger.
- Why K=4 and 8 come out 20-100x closer (max error) than K=1 and 2 is not established; the
  driver claims a different set of operations at each K (43, 52, 54, 59) and
  op-level placement is not logged.

## Reading for the paper

- Tab S7+ CPU rows: replace with run 1 (edge-padded; footnote c and the
  "CPU rows ... flatter every path" limitation then apply to the A56 only).
- Tab S7+ NNAPI rows: replace with run 1's edge-padded rows (same sliver,
  now checked); NNAPI is 8-13x slower than the CPU at K=1.
- Tab S7+ ink through NNAPI: computes the fluid correctly in float32 mode;
  with fp16 allowed its error is that of half-precision coordinates, not a
  wrong read. So "no run produces a valid fluid" is the POCO only; on the
  tablet the fluid is valid but costs 8x the CPU (95.8 vs 12.0 ms at K=1).
- GPU delegate: a second device, stencils correct and faster than the CPU
  here; ink K=1 correct but slower than the CPU; batched ink refused again.
- Run 2 timings are not usable (see gotcha); nnapi-fp32 was not timed cleanly.

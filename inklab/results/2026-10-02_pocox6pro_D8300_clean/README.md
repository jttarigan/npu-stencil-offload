# POCO X6 Pro (2311DRK48G, MediaTek Dimensity 8300 / MT6897), Android 16 (HyperOS 3), 2026-10-02

Fourth session on this phone. Purpose: (1) the clean re-sweep of the
edge-concat stencils for the paper numbers, (2) pin down why the APU's ink
output is wrong (2026-09-30 found wrong cells in the gather and fp16 NaN),
(3) the ink with every fix of our own, on the APU and on the CPU.
Tooling as in ../2026-09-30_pocox6pro_D8300_checks; new today (APK rebuilt and
installed once, one tap):

- `trace/tflite_graph.py gathertest`: the advection's index chain alone
  (back-trace of up to +-4 cells, floor, clip, int32 index) and GATHERs of
  1-, 2- and 3-channel tables with it, plus a 3-channel gather with a constant
  index (shift by 3 rows, 5 columns). Outputs the clipped floor coordinates too.
- `--fp16-safe` (`_fs`): vorticity normalised as g / (max(ln, 1e-12) + 1 - gate)
  * gate instead of curl * gate / max(ln, 1e-12); identical in fp32 (Mac
  fidelity unchanged at K = 1, 2, 4, 8), cannot divide 0 by 0 in fp16.
- `--gather-c1` (`_g1`): one rank-1 gather table per channel. Built, in the
  APK, NOT run (the gathertest showed 1-channel tables fail the same way).
- MainActivity accepts combined variant suffixes (`ink_i32_ec_fs`, ...).
- `android/dump_check.py` analyses gathertest dumps: per output, wrong cells,
  and which table element each wrong value came from.

## Conditions

Unplugged over Wi-Fi adb from the start (95% at 10:04, 85% at 10:39),
airplane mode on with Wi-Fi back on, brightness 1 with auto off (set by the
author), power saving off, Gmail and Settings force-stopped before run 1,
thermal status 0 in every row, battery 33.2-34.5 C during run 1 (cooler than
2026-09-30's 35-41 C, like 2026-09-24's 33.7-34.8 C).

| run | folder | start | what |
|---|---|---|---|
| 1 | 1_stencils_ec | 10:04 | heat_s32_ec, heat_s8_ec, grayscott_s8_ec x litert-cpu, nnapi x K=1,2,4,8, CHECK (no install) |
| 2 | 2_gathertest | 10:30 | APK install (tap); gathertest x nnapi-fp32, nnapi, CHECK + DUMP |
| 3 | 3_ink_fixed | 10:37 | ink_i32_ec_fs x litert-cpu, nnapi x K=1,2,4,8, CHECK + DUMP |

## Run 1: the edge-concat stencils, clean (paper numbers)

Every NNAPI model claimed whole, one partition (258/258 ... 2064/2064 for
heat_s32; 66 ... 528 heat_s8; Gray-Scott likewise). APU output vs fp32 CPU:
max 0.03-0.04% of range (heat), 0.11-0.20% (Gray-Scott), no non-finite value.
Fits (`costmodel_stencils_ec.md`, b2b, medians of 3):

| workload | path | K=1 / 2 / 4 / 8 ms | a | b | R2 | gain K=8 (paced) |
|---|---|---|---|---|---|---|
| heat_s32_ec | nnapi (APU) | 2.180 / 3.502 / 5.874 / 11.344 | 0.826 | 1.306 | 0.999 | 1.54x (1.73x) |
| heat_s32_ec | litert-cpu | 8.790 / 18.183 / 36.270 / 74.080 | -0.615 | 9.319 | 1.000 | 0.95x |
| heat_s8_ec | nnapi (APU) | 1.072 / 1.638 / 2.388 / 4.096 | 0.709 | 0.424 | 0.998 | 2.09x (2.83x) |
| heat_s8_ec | litert-cpu | 2.320 / 4.654 / 9.237 / 18.515 | 0.011 | 2.312 | 1.000 | 1.00x |
| grayscott_s8_ec | nnapi (APU) | 3.129 / 5.592 / 9.531 / 17.995 | 1.174 | 2.103 | 0.999 | 1.39x (1.63x) |
| grayscott_s8_ec | litert-cpu | 10.303 / 20.100 / 36.038 / 81.434 | -1.065 | 10.142 | 0.995 | 1.01x |

APU vs CPU at K=1: 4.0x, 2.2x, 3.3x (the paper had 5.3x, 4.7x, 10.5x).

**Side finding: the CPU path was handicapped by MIRROR_PAD.** XNNPACK does not
take MIRROR_PAD, so with the original graphs it claimed 18 of 26 nodes in 17
partitions (heat_s8 K=1) up to 264 of 392 in 257 partitions (heat_s32 K=8,
2026-09-30), leaving every pad to the builtin kernel. The edge-concat graphs
run whole in one XNNPACK partition, and the CPU gets faster at K=1: heat_s32
10.85 -> 8.79 ms (1.2x), heat_s8 5.12 -> 2.32 (2.2x), Gray-Scott 33.42 ->
10.30 (3.2x) (old numbers 2026-09-22/24). Every Android CPU row measured
with MIRROR_PAD graphs (A56, Tab S7+, earlier POCO) carries this handicap.
The CPU's shape is unchanged (a ~ 0, no batching gain).

## Run 2: gathertest, the gather fault isolated

NNAPI claimed 29 of 30 nodes (fp32 mode) and 27 of 30 (fp16), 3 partitions;
the GATHER partition compiled on mtk-neuron (the driver rejects only the
int32 MUL/ADD of the index: "MDLA: Cannot support Int32 input/output").
`2_gathertest/dump_check.txt`:

- The index chain is right: the clipped floor coordinates are exact in fp32;
  in fp16, 2-4% of cells are off by one cell (fp16 rounding of the back-traced
  coordinate near integers), as expected.
- The GATHER is wrong. In fp32 mode EVERY output row equals table row 0
  (cell 0,0), for the 1-, 2- and 3-channel tables with the computed index
  (1 distinct output row each); the constant-index gather gives 27 distinct
  rows, nearly all row 0. In fp16 mode the outputs take 2 distinct values
  per table, and the constant-index gather returns integer-like garbage
  (127, 125, 126, 148, ... up to 175) that is not in the table.
- So the "wrong cells" of 2026-09-30 are the driver's GATHER, not our graph
  and not the RESHAPE around it; per-channel gathers (`_g1`) cannot help.

## Run 3: the ink with every fix of our own

`ink_i32_ec_fs` (int32 index, edge-concat padding, fp16-safe vorticity).
NNAPI claims 335 of 342 nodes in 5 partitions at K=1.
- fp16 NaN GONE: nonfinite = 0 in velocity, dye and pressure (2026-09-30:
  65-93% non-finite). With the edge padding changed at the same time the
  attribution is by mechanism, not by a single-variable test: 1e-12 is 0 in
  fp16, the old form divides 0 by 0 where the gradient vanishes.
- Output still wrong: dye RMS 21% of range (uncorrelated values), velocity
  13%, pressure 10%: the gather fault. At K >= 2 the dye blows up (K=2 max
  36x the range; K=4 and 8 values near 1e38, still finite), velocity and
  pressure stay at 9-13% RMS.
- The int32 index removed the "gather index out of bounds": NNAPI runs at
  every K. Claimed nodes: 335/342 (5 partitions), 556/678 (6), 777/1350 (7),
  777/2694 (7) at K = 1, 2, 4, 8; the claim stops growing at K = 4.
- CPU (XNNPACK, edge-padded ink, correct): 11.80 ms at K=1, 4x faster than
  the 47.26 ms of the MIRROR_PAD graph in the paper (2026-09-22). No
  batching gain (0.82x at K=8, negative a: per-step cost grows with K, cache).
  NOTE: the A56's "odd" CPU ink of 13.5 ms (2026-09-20) is close to this;
  possibly that build's CPU path took the padding. Not verified.
- NNAPI (invalid output): 52.5 / 70.1 / 76.6 / 220.1 ms at K = 1, 2, 4, 8,
  4.5x the CPU at K=1 (`costmodel_ink_fixed.md`; fit meaningless, R2 0.91).
Dumps: only K=1 kept here (2.9 MB); K=2,4,8 dumps stay in the gitignored
android/results/2311DRK48G_2026-10-02_1037/dump.

Dump tensor order (ink): in0 pr, in1 dyeAdd, in2 velAdd, in3 dye, in4 vel;
out0 vel, out1 dye, out2 pr.

## What this changes in the paper (edited 2026-10-02)

New subsection "Output checks on Android" (sec:checks); POCO stencil rows and
text from run 1 (CPU and APU from the same session); table footnote c on the
MIRROR_PAD CPU handicap; abstract, contribution (6), RQ1 fluid paragraph,
discussion and limitations updated. Not checked yet: the Tab S7+ NNAPI runs.

## Run 4: LiteRT GPU delegate on the edge-padded graphs (11:20 and 11:25)

`4_gpu_delegate/` (k1 = K=1 for all four workloads, k248 = K=2,4,8);
`costmodel_gpu_delegate.md`. Phone at 75% and 29.6-29.8 C, thermal 0, no install.
The GPU delegate refused every MIRROR_PAD graph before; with edge padding:
- Stencils whole on the GPU (1 partition, all nodes) at every K, output correct
  (max 0.04-0.24% of range). Fits (b2b): heat_s32 a 6.00 [5.31-6.56] b 3.78,
  gain 2.34x; heat_s8 a 4.94 [4.78-5.34] b 1.50, 2.64x; Gray-Scott a 8.00
  [7.46-8.13] b 3.36, 2.44x (bootstrap 95%, netime/bootstrap_ci.py). The
  largest fixed cost of any path in the study; slower than the CPU at K=1.
- Ink K=1: 229 of 342 ops on the GPU in 2 partitions (gathers on the CPU),
  output correct (vel 0.045%, dye exact, pr 0.06% of range), 32.7 ms vs CPU
  11.8 ms. K=2,4,8 refused at init: "TfLiteGpuDelegate Init: Batch size
  mismatch" (the K-step graphs stack event inputs along the batch axis);
  also unsupported: CAST from bool, int32 ADD/MUL/RESHAPE.

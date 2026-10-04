# Offloading real-time stencil simulations to mobile NPUs: code, models and measurements

Archive for the paper

> J. T. Tarigan and R. F. Rahmat. Offloading Real-Time Stencil Simulations to
> Mobile Neural Processing Units: Overhead, Placement, and Correctness.
> Submitted to the Journal of Systems Architecture.

Archive DOI: assigned on publication

It holds the benchmark harnesses for macOS, iOS and Android, the scripts that
build the Core ML and LiteRT graphs, the compiled models, the recorded fluid
traces, and every measurement the paper reports, together with the analysis
scripts that turn the measurements into the paper's tables and figures.

## Layout

Everything sits under `inklab/`, the folder the scripts expect.

| Path | Contents |
| --- | --- |
| `inklab/ink_kernels.metal` | The six Metal compute kernels of the game's ink bath (stable fluids on a 128x256 grid), copied verbatim. This is the GPU path every other path is compared with. |
| `inklab/trace/` | Graph builders: `coreml_graph.py` (one step), `unroll_graph.py` (K steps per submission), `stencil_graph.py` (heat, Gray-Scott), `tflite_graph.py` (LiteRT graphs and their Android variants), `surrogate.py` (learned pressure solve). `reference.py` is the float32 NumPy step. `main.swift` and `run.sh` run the Metal kernels headlessly and record golden traces (`trace_*/`). `gameplay_*.json` are recorded gameplay event streams. Also the Core ML packages (`*.mlpackage`) and the fidelity, lag and figure scripts. |
| `inklab/netime/` | Mac harnesses: `main.swift` (Core ML, time per submission), `metaltime.swift` (standalone Metal), `placement.swift` (Core ML compute plan), the sweep scripts, the compiled models (`*.mlmodelc`) and the analysis: `costmodel.py`, `ladder.py`, `bootstrap_ci.py`. |
| `inklab/ios/` | InkBench for iPhone and iPad (the Mac protocol on device); `prepare.sh` bundles the models and kernels. |
| `inklab/device/` | `run_phone.sh` builds, installs, runs and pulls InkBench; `game_frames.py` analyses the in-game frame-time CSVs. |
| `inklab/android/` | InkBench for Android: LiteRT on the CPU (XNNPACK), NNAPI, the GPU delegate, the QNN delegate, and a hand-written OpenGL ES 3.1 port of the kernels. Also the LiteRT models (`models/`), `run_android.sh`, and the output-check tools `ink_dump.py` and `dump_check.py`. |
| `inklab/energy/` | `powermetrics` scripts for the energy measurements on the Mac. |
| `inklab/results/` | Every measurement used in the paper, one folder per session, each with its own README. See the index below. |
| `inklab/paper/make_costfig.py` | Regenerates the paper's cost-model figure from `results/`. |

Each script documents its usage in its header.

## Not included

- **The game.** Only its ink kernels are here. The in-game measurements (the
  `*_game*` and `*iphone12_A14*` folders in `results/`) were taken with builds
  of the game carrying the different fluid backends. Their frame-time CSVs
  and the analysis script are included; the game is not.
- **Raw device logs.** The full Android system logs are left out because they
  record unrelated apps on the devices. The placement and check extracts in
  `results/` keep only the lines written by the benchmark, LiteRT and NNAPI.
- **Python environments and build outputs.**

## Requirements (as used)

- A Mac with Apple silicon, macOS 26.6, Xcode 27 (Swift, Metal, Core ML).
- Python 3.9 with coremltools 9.0, NumPy 2.0 and PyTorch 2.8 for the Core ML
  graphs, the reference step and the surrogate.
- Python 3.9 with TensorFlow 2.20 for the LiteRT graphs.
- Android: a JDK and Gradle (set `JAVA_HOME` and `GRADLE`), the Android SDK
  (`ANDROID_HOME`, `ADB`), LiteRT 1.4.2 and the Qualcomm QNN delegate 2.50.0.
- iPhone and iPad: an Apple development team for signing
  (`TEAM=<team id> DEVICE=<udid> ./inklab/device/run_phone.sh bench`).

## Measurements

| Folder | Session |
| --- | --- |
| `results/2026-09-13_battery_lpm_off/` | Latencies, Apple M5, ON BATTERY, Low Power Mode OFF (2026-09-13 night) |
| `results/2026-09-13_energy_paced_clean/` | Energy at the game's cadence, Apple M5 (2026-09-13): idle, Metal kernels, Core ML on the neural engine and on the CPU, each paced at 30 steps/s; powermetrics, 20 x 1 s per condition |
| `results/2026-09-13_lowpower_battery/` | Measurements of 2026-09-13, Apple M5 MacBook Air, ON BATTERY with Low Power Mode ON |
| `results/2026-09-15_splat_lag/` | Perceptual cost of batching splats for the unrolled graph (2026-09-15) |
| `results/2026-09-15_unroll_prelim/` | K-step unrolled Core ML graph: preliminary timing (2026-09-15) |
| `results/2026-09-15_unroll_validated/` | K-step unrolled Core ML graph: validated timing (2026-09-15) |
| `results/2026-09-19_energy_k8/` | Energy per step, K = 1 vs K = 8 (2026-09-19, Apple M5, mains power) |
| `results/2026-09-19_generality/` | Generality sweep and cost model (2026-09-19, Apple M5, battery, LPM off) |
| `results/2026-09-19_ipad_mini_A15/` | iPad mini 6th gen (iPad14,1, Apple A15), InkBench standalone sweep (2026-09-19) |
| `results/2026-09-19_ipadpro_A12X/` | iPad Pro 12.9-inch 3rd gen (iPad8,5, Apple A12X, 8-core NE), InkBench sweep (2026-09-19) |
| `results/2026-09-19_iphone17pro_A19/` | iPhone 17 Pro Max (iPhone18,2, Apple A19 Pro), InkBench sweep (2026-09-19 night) |
| `results/2026-09-19_placement/` | Where Core ML places each model (MLComputePlan), cpuAndNeuralEngine |
| `results/2026-09-20_galaxyA56_exynos1580/` | Galaxy A56 (SM-A566B, Samsung Exynos 1580), Android 15, 2026-09-20 |
| `results/2026-09-21_iphone12_A14/` | iPhone 12 (iPhone13,2, Apple A14), InkBench sweep (2026-09-21) |
| `results/2026-09-21_tabS7plus_SD865plus/` | Galaxy Tab S7+ (SM-T975, Qualcomm SM8250 / Snapdragon 865+), Android 13, 2026-09-21 |
| `results/2026-09-22_pocox6pro_D8300/` | POCO X6 Pro (2311DRK48G, MediaTek Dimensity 8300 / MT6897), Android 16 (HyperOS 3), 2026-09-22 |
| `results/2026-09-24_ipadmini_A15_game/` | iPad mini 6 (A15), in-game runs, 2026-09-24: the fair RQ4 test |
| `results/2026-09-24_pocox6pro_D8300/` | POCO X6 Pro (2311DRK48G, MediaTek Dimensity 8300 / MT6897), Android 16 (HyperOS 3), 2026-09-24 |
| `results/2026-09-24_tabS7plus_SD865plus/` | Galaxy Tab S7+ (SM-T975, Qualcomm SM8250 / Snapdragon 865+), Android 13, 2026-09-24 |
| `results/2026-09-30_pocox6pro_D8300_checks/` | POCO X6 Pro (2311DRK48G, MediaTek Dimensity 8300 / MT6897), Android 16 (HyperOS 3), 2026-09-30 |
| `results/2026-10-02_pocox6pro_D8300_clean/` | POCO X6 Pro (2311DRK48G, MediaTek Dimensity 8300 / MT6897), Android 16 (HyperOS 3), 2026-10-02 |
| `results/2026-10-03_tabS7plus_SD865plus_ec/` | Galaxy Tab S7+ (SM-T975, Qualcomm SM8250 / Snapdragon 865+), Android 13, 2026-10-03 |

## Licenses

- **Code** (`*.swift`, `*.metal`, `*.py`, `*.sh`, `*.kt`, `*.kts`, and the Xcode
  and Gradle project files): PolyForm Noncommercial License 1.0.0, see
  `LICENSE.txt`.
- **Data, models, traces and documentation** (everything else): Creative
  Commons Attribution 4.0 International, see `LICENSE-DATA.txt`.

## Citation

Please cite the paper and this archive (DOI above).

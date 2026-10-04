# CPU profile of the NE path (2026-09-19, M5, xctrace Time Profiler)

netime ink_unroll u1 ne at 30 Hz, and u8 ne at 3.75 Hz (both 30 steps/s),
steady state (recording starts 4 s after launch), --all-processes, 15 s, 1 ms
samples. The machine was NOT quiet (Spotlight, mediaanalysisd, Terminal,
Codex); per-process on-CPU time is still attributable.

process    K=1       K=8       idle
netime     418 ms    94 ms     -
aned       4 ms      2 ms      2 ms
kernel     588 ms    463 ms    582 ms

In-process (netime-only launch trace, K=1): 87% under -[MLModel prediction];
~22% ANE execute-sync (mach_msg waits), ~22% input copy
(MLE5BindInputBufferObjectByCopyingMultiArray -> _platform_memmove).

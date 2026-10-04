# Generality sweep and cost model (2026-09-19, Apple M5, battery, LPM off)

sweep_battery.csv: netime/sweep.sh (Core ML ne/gpu/cpu, back to back 3 x 300;
ne paced at 30/K Hz 3 x ~15 s), the paced cpu/gpu control (2 x ~10 s), and
netime/metalsweep.sh (Metal, K steps per command buffer: `metal` = wall per
submission incl. commit + wait, `metal-exec` = GPU execution time only).
costmodel.md: netime/costmodel.py fit t_sub(K) = a + bK per workload and path.

Workloads (trace/stencil_graph.py, trace/unroll_graph.py): heat s32, heat s8,
Gray-Scott s8, the ink step (26 Jacobi). All 128x256, fp16 on the NE.

Findings
1. NE fixed cost per submission is ~0.14-0.22 ms back to back for every
   workload: a property of the path, not the work. Per-step gain at K=8:
   NE 1.8-2.3x; Core ML CPU 1.0-1.2x.
2. Pacing at 30 Hz slows EVERY path (CPU ink 0.96 -> 4.28 ms, standalone
   Metal 0.32 -> 1.16 ms): clocks drop between submissions. Paced gains
   therefore mix dispatch amortisation with power management; the
   back-to-back control is the clean evidence.
3. Standalone Metal has the same kind of fixed cost as the NE (~0.20-0.22 ms
   back to back) and batching helps it just as much (ink 2.9x). What the NE
   cannot do is ride in the renderer's command buffer: the GPU path's marginal
   cost is execution only (ink: a ~0.01 ms, b ~0.075 ms per step).
   => The offload's price is a standalone submission, which batching buys back.
4. Per-step arithmetic on the M5: NE 0.19 ms vs GPU execution 0.075 ms (ink).
5. Core ML's own GPU path is not a fair GPU baseline: it gets slower per step
   with K on the pure-conv stencils (0.7x).
6. Placement: Core ML keeps light graphs on the CPU (heat s1 at every K; heat
   s8 at K = 1, 2, then NE at K = 4, 8). Batching can decide whether the
   accelerator is used at all.
Battery vs AC does not matter: ink NE paced 1.394 / 5.134 ms at K = 1 / 8 here
vs 1.390 / 5.164 on AC (2026-09-15).

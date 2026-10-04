# Measurements of 2026-09-13, Apple M5 MacBook Air, ON BATTERY with Low Power Mode ON

Kept for reference; superseded by the plugged-in, Low Power Mode OFF run.

Metal kernels (trace harness, gpu start-to-end per full step, 300 steps):
  median 0.127 ms, max 5.28 ms   (metal_gpu_ms_trace_ref.csv)

Core ML step, Swift timing harness (netime, 500 calls, preallocated inputs):
  128x256 j26 fp16:  CPU 1.952 ms  GPU 1.776 ms  NE 0.565 ms
  256x512 j26 fp16:  CPU 5.961 ms  GPU 2.260 ms  NE 1.242 ms
  128x256 j26 fp32:  CPU 2.355 ms  GPU 2.316 ms  NE 2.511 ms (all ops on CPU)

Core ML step, Python predict (framework overhead dominated; coreml_*.json):
  128x256 j26 fp16: CPU 4.09  GPU 3.86  NE 2.49  ALL 3.36 ms
  128x256 j26 fp32: CPU 5.03  GPU 4.29  NE 5.94  ALL 6.12 ms
  256x512 j26 fp16: CPU 6.63  GPU 3.84  NE 3.43  ALL 3.94 ms

Fidelity results (fp16 vs GPU trace, run-level) are power-state independent
and are reported in the paper (fidelity section).

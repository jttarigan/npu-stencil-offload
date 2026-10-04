# Latencies, Apple M5, ON BATTERY, Low Power Mode OFF (2026-09-13 night)

netime (Swift, MLModel, preallocated inputs, 500 calls):
  128x256 j26 fp16:  CPU 1.001  GPU(CoreML) 0.958  NE 0.364 ms
  256x512 j26 fp16:  CPU 2.988  GPU(CoreML) 0.911  NE 0.910 ms
  128x256 j26 fp32:  CPU 1.260  GPU 1.241  NE 1.258 ms (all ops on CPU)
  Jacobi ladder, 128x256 fp16 (decay-every-8 variants), NE:
    j8 0.326  j12 0.330  j16 0.342  j26 0.366 ms
Metal kernels (trace harness): synthetic script median 0.092 ms (p95 0.39);
  recorded gameplay median 0.089 ms (p95 0.17).
Compare the Low Power Mode set in ../2026-09-13_lowpower_battery (NE 0.565,
CPU 1.95, Metal 0.127): Low Power Mode slowed the NE by 1.55x, the CPU by
1.95x, the GPU by 1.4x.

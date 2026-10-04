| device | workload | path | mode | t_sub K=1..8 (ms) | a fixed | b per step | R2 | per-step gain K=8 meas | pred | ceiling (a+b)/b |
|---|---|---|---|---|---|---|---|---|---|---|
| 2311DRK48G | ink_i32_ec_fs | litert-cpu | b2b | 11.801 / 27.561 / 55.928 / 115.346 | -2.583 | 14.731 | 1.000 | 0.82x | 0.84x | 0.82x |
| 2311DRK48G | ink_i32_ec_fs | litert-cpu | paced | 14.448 / 29.402 / 57.159 / 116.898 | -0.296 | 14.606 | 1.000 | 0.99x | 0.98x | 0.98x |
| 2311DRK48G | ink_i32_ec_fs | nnapi | b2b | 52.535 / 70.106 / 76.574 / 220.067 | 15.176 | 23.905 | 0.912 | 1.91x | 1.51x | 1.63x |
| 2311DRK48G | ink_i32_ec_fs | nnapi | paced | 52.504 / 69.945 / 81.439 / 239.939 | 10.090 | 26.898 | 0.920 | 1.75x | 1.31x | 1.38x |

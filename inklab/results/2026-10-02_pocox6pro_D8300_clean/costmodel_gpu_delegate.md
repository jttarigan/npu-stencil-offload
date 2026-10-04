| device | workload | path | mode | t_sub K=1..8 (ms) | a fixed | b per step | R2 | per-step gain K=8 meas | pred | ceiling (a+b)/b |
|---|---|---|---|---|---|---|---|---|---|---|
| 2311DRK48G | grayscott_s8_ec | litert-gpu | b2b | 10.445 / 14.820 / 22.878 / 34.253 | 7.995 | 3.361 | 0.990 | 2.44x | 2.60x | 3.38x |
| 2311DRK48G | grayscott_s8_ec | litert-gpu | paced | 14.359 / 20.099 / 29.252 / 51.954 | 8.900 | 5.338 | 0.998 | 2.21x | 2.21x | 2.67x |
| 2311DRK48G | heat_s32_ec | litert-gpu | b2b | 10.566 / 12.170 / 21.838 / 36.145 | 5.999 | 3.782 | 0.993 | 2.34x | 2.16x | 2.59x |
| 2311DRK48G | heat_s32_ec | litert-gpu | paced | 14.981 / 20.992 / 29.698 / 44.464 | 12.082 | 4.120 | 0.993 | 2.70x | 2.88x | 3.93x |
| 2311DRK48G | heat_s8_ec | litert-gpu | b2b | 5.402 / 8.362 / 12.079 / 16.356 | 4.935 | 1.497 | 0.957 | 2.64x | 3.04x | 4.30x |
| 2311DRK48G | heat_s8_ec | litert-gpu | paced | 9.961 / 12.626 / 15.233 / 23.030 | 8.404 | 1.816 | 0.994 | 3.46x | 3.57x | 5.63x |

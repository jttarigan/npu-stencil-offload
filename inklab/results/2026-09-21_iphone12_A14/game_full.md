| phase | backend | fx | frames | frame p50 | p95 | p99 | GPU ms | CPU ms | ink ms (mean) | ink ms (max) | % frames >1.5x period |
|---|---|---|---|---|---|---|---|---|---|---|---|
| finale | off | x1 | 500 | 16.74 | 17.15 | 17.62 | 11.509 | 16.581 | 0.000 | 0.00 | 0.0 |
| finale | gpu | x1 | 452 | 16.73 | 16.80 | 16.81 | 12.409 | 3.975 | 0.000 | 0.00 | 0.0 |
| finale | npu | x1 | 451 | 16.73 | 25.80 | 32.93 | 13.117 | 10.915 | 4.155 | 12.29 | 8.0 |
| finale | npu4 | x1 | 366 | 16.75 | 50.21 | 65.75 | 12.448 | 12.487 | 4.890 | 35.66 | 15.6 |
| finale | npua | x1 | 483 | 16.72 | 17.12 | 17.23 | 12.319 | 16.582 | 0.046 | 0.85 | 0.0 |
| play | off | x1 | 4747 | 16.72 | 17.76 | 20.15 | 10.687 | 16.128 | 0.000 | 0.00 | 0.0 |
| play | gpu | x1 | 4773 | 16.73 | 17.66 | 20.36 | 12.356 | 2.092 | 0.000 | 0.00 | 0.0 |
| play | npu | x1 | 3934 | 16.73 | 20.06 | 32.79 | 13.671 | 7.456 | 4.713 | 13.22 | 3.2 |
| play | npu4 | x1 | 3979 | 16.75 | 50.25 | 66.04 | 12.182 | 11.031 | 5.406 | 35.90 | 16.3 |
| play | npua | x1 | 6313 | 16.73 | 17.63 | 20.50 | 12.004 | 11.207 | 0.165 | 1.41 | 0.5 |

finale npu  : GPU time saved -0.709 ms/frame; fluid on the render thread 4.16 ms/frame (max 12.3); long frames 0.0% -> 8.0% -> does not pay
finale npu4 : GPU time saved -0.040 ms/frame; fluid on the render thread 4.89 ms/frame (max 35.7); long frames 0.0% -> 15.6% -> does not pay
finale npua : GPU time saved +0.090 ms/frame; fluid on the render thread 0.05 ms/frame (max 0.8); long frames 0.0% -> 0.0% -> pays
play   npu  : GPU time saved -1.315 ms/frame; fluid on the render thread 4.71 ms/frame (max 13.2); long frames 0.0% -> 3.2% -> does not pay
play   npu4 : GPU time saved +0.173 ms/frame; fluid on the render thread 5.41 ms/frame (max 35.9); long frames 0.0% -> 16.3% -> does not pay
play   npua : GPU time saved +0.351 ms/frame; fluid on the render thread 0.17 ms/frame (max 1.4); long frames 0.0% -> 0.5% -> pays

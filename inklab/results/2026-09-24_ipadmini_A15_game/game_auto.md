| phase | backend | fx | frames | frame p50 | p95 | p99 | GPU ms | CPU ms | ink ms (mean) | ink ms (max) | % frames >1.5x period |
|---|---|---|---|---|---|---|---|---|---|---|---|
| finale | off | x1 | 479 | 16.67 | 16.76 | 16.82 | 4.215 | 16.435 | 0.000 | 0.00 | 0.0 |
| finale | gpu | x1 | 476 | 16.67 | 16.73 | 16.75 | 5.245 | 3.596 | 0.000 | 0.00 | 0.0 |
| finale | npu | x1 | 473 | 16.67 | 16.72 | 16.76 | 4.381 | 5.097 | 1.254 | 3.37 | 0.0 |
| finale | npu4 | x1 | 503 | 16.67 | 16.71 | 16.75 | 4.438 | 4.531 | 0.726 | 6.05 | 0.0 |
| finale | npua | x1 | 499 | 16.67 | 16.75 | 16.85 | 4.580 | 16.327 | 0.043 | 0.78 | 0.0 |
| play | off | x1 | 5106 | 16.67 | 16.74 | 16.79 | 2.703 | 7.121 | 0.000 | 0.00 | 0.1 |
| play | gpu | x1 | 4303 | 16.67 | 16.71 | 16.76 | 3.715 | 1.567 | 0.000 | 0.00 | 0.0 |
| play | npu | x1 | 5112 | 16.67 | 16.71 | 16.76 | 3.145 | 3.535 | 1.453 | 8.60 | 0.0 |
| play | npu4 | x1 | 5544 | 16.67 | 16.72 | 16.76 | 3.186 | 3.167 | 0.841 | 7.81 | 0.0 |
| play | npua | x1 | 5549 | 16.67 | 16.78 | 16.89 | 3.175 | 16.488 | 0.144 | 1.16 | 0.1 |

finale npu  : GPU time saved +0.864 ms/frame; fluid on the render thread 1.25 ms/frame (max 3.4); long frames 0.0% -> 0.0% -> does not pay
finale npu4 : GPU time saved +0.808 ms/frame; fluid on the render thread 0.73 ms/frame (max 6.0); long frames 0.0% -> 0.0% -> pays
finale npua : GPU time saved +0.665 ms/frame; fluid on the render thread 0.04 ms/frame (max 0.8); long frames 0.0% -> 0.0% -> pays
play   npu  : GPU time saved +0.570 ms/frame; fluid on the render thread 1.45 ms/frame (max 8.6); long frames 0.0% -> 0.0% -> does not pay
play   npu4 : GPU time saved +0.529 ms/frame; fluid on the render thread 0.84 ms/frame (max 7.8); long frames 0.0% -> 0.0% -> does not pay
play   npua : GPU time saved +0.540 ms/frame; fluid on the render thread 0.14 ms/frame (max 1.2); long frames 0.0% -> 0.1% -> pays

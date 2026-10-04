| phase | backend | fx | frames | frame p50 | p95 | p99 | GPU ms | CPU ms | ink ms (mean) | ink ms (max) | % frames >1.5x period |
|---|---|---|---|---|---|---|---|---|---|---|---|
| play | off | x1 | 4979 | 16.67 | 16.72 | 16.76 | 2.698 | 1.476 | 0.000 | 0.00 | 0.1 |
| play | gpu | x1 | 5113 | 16.67 | 16.74 | 16.80 | 3.666 | 16.253 | 0.000 | 0.00 | 0.1 |
| play | npu | x1 | 5117 | 16.67 | 16.72 | 16.76 | 3.131 | 3.509 | 1.452 | 6.59 | 0.0 |
| play | npu4 | x1 | 5550 | 16.67 | 16.72 | 16.76 | 3.180 | 3.165 | 0.842 | 7.52 | 0.0 |
| play | npua | x1 | 5104 | 16.67 | 16.72 | 16.76 | 3.140 | 1.853 | 0.139 | 1.06 | 0.0 |
play   npu  : GPU time saved +0.535 ms/frame; fluid on the render thread 1.45 ms/frame (max 6.6); long frames 0.1% -> 0.0% -> does not pay
play   npu4 : GPU time saved +0.486 ms/frame; fluid on the render thread 0.84 ms/frame (max 7.5); long frames 0.1% -> 0.0% -> does not pay
play   npua : GPU time saved +0.526 ms/frame; fluid on the render thread 0.14 ms/frame (max 1.1); long frames 0.1% -> 0.0% -> pays

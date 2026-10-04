| phase | backend | fx | frames | frame p50 | p95 | p99 | GPU ms | CPU ms | ink ms (mean) | ink ms (max) | % frames >1.5x period |
|---|---|---|---|---|---|---|---|---|---|---|---|
| finale | off | x1 | 468 | 16.73 | 16.99 | 17.17 | 11.408 | 16.581 | 0.000 | 0.00 | 0.0 |
| finale | gpu | x1 | 454 | 16.73 | 17.38 | 18.61 | 12.626 | 16.517 | 0.000 | 0.00 | 0.0 |
| finale | npu | x1 | 453 | 16.73 | 25.28 | 32.39 | 13.267 | 11.628 | 4.219 | 14.92 | 6.0 |
| finale | npu4 | x1 | 359 | 16.75 | 50.21 | 66.36 | 12.826 | 13.704 | 4.956 | 35.75 | 17.8 |
| play | off | x1 | 1031 | 16.73 | 17.18 | 27.68 | 7.180 | 16.672 | 0.000 | 0.00 | 1.2 |
| play | gpu | x1 | 784 | 16.73 | 17.20 | 33.34 | 9.309 | 16.597 | 0.000 | 0.00 | 1.5 |
| play | npu | x1 | 2133 | 16.73 | 16.85 | 27.09 | 12.899 | 7.074 | 4.644 | 12.59 | 1.3 |
| play | npu4 | x1 | 553 | 16.74 | 50.23 | 66.02 | 8.277 | 11.693 | 5.439 | 35.77 | 16.5 |

finale npu  : GPU time saved -0.641 ms/frame, CPU-side time added -4.890 ms/frame, net +4.249 ms/frame -> pays on the GPU-bound side
finale npu4 : GPU time saved -0.200 ms/frame, CPU-side time added -2.813 ms/frame, net +2.613 ms/frame -> pays on the GPU-bound side
play   npu  : GPU time saved -3.590 ms/frame, CPU-side time added -9.523 ms/frame, net +5.933 ms/frame -> pays on the GPU-bound side
play   npu4 : GPU time saved +1.032 ms/frame, CPU-side time added -4.904 ms/frame, net +5.936 ms/frame -> pays on the GPU-bound side

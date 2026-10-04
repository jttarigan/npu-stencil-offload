# Galaxy A56 (SM-A566B, Samsung Exynos 1580), Android 15, 2026-09-20

First non-Apple device. Reduced protocol (15-minute window): ink workload only,
K = 1,2,4,8, backends nnapi and litert-cpu. 48 rows. Thermal status 0-1,
battery 32-34 C, 82%. DEVIATIONS from the protocol: the phone was charging over
USB (adb), and other apps were live (YouTube, Chrome background traffic in the
logcat). Not airplane mode.

NOTE the device is a Galaxy A56, not the A57 the plan and paper had recorded.
Chip confirmed from the device: ro.soc.model = s5e8855 = Exynos 1580.

## Fits (medians of 3, ms per submission)
backend      K=1/2/4/8                          a      b      R2     gain@K=8
nnapi        60.6 / 121.5 / 241.2 / 479.1       1.7    59.7   1.000  1.01x
litert-cpu   13.5 /  28.6 /  87.5 / 126.9       2.3    16.5   0.940  0.85x
(paced at 30/K Hz: nnapi 61.3 / 121.8 / 241.4 / 477.4; litert-cpu 70.9 / 126.4
/ 273.8 / 548.0, i.e. the CPU path is ~5x slower paced than back to back.)

## Reading
- NNAPI DID NOT REACH THE NPU. Evidence: TFLite logged "Created ... delegate for
  NNAPI" but never a "Replacing N out of M node(s)" line, while the CPU backend
  logged one for every K (140/188 at K=1 up to 1120/1497 at K=8); the cost is
  exactly linear in K (R2 = 1.000) with a ~ 0, which is the signature of no
  per-submission cost at all; and it is 4.5x SLOWER than the XNNPACK CPU path.
  Android 15 deprecated NNAPI. Not yet proven: query the NNAPI device list
  directly, or run with delegate verbose logging, before the paper asserts it.
- The CPU path's fit is poor (R2 0.94, K=4 out of line at 87 ms) and gains
  nothing from batching, as the CPU control does everywhere else.
- CAUTION: a CSV-writing bug (below) means the numbers above come from a
  repaired parse. Reruns after the fix are clean.

## Second session 18:17 (bench_gles_SM-A566B.csv, after the locale fix)
backend      K=1/2/4/8                       a      b      R2     gain@K=8
gles         3.5 / 6.6 / 11.5 / 15.6        3.01   1.68   0.938  1.78x
gles paced  10.3 / 14.5 / 18.0 / 21.4      10.62   1.45   0.896  3.87x
litert-gpu   REFUSED at every K.

- GLES compute is the Android analogue of the standalone Metal submission and
  behaves the same way: a real fixed cost per submission, amortised by
  batching (1.78x at K=8). The fixed cost is 3.0 ms, an order of magnitude
  above Apple's 0.2 ms, so the submission-cost argument is stronger here.
- GLES is 3.9x faster per step than the XNNPACK CPU path and 17x faster than
  the NNAPI fallback.
- The LiteRT GPU delegate refuses the ink graph: MIRROR_PAD supports only
  reflective padding (the simulation wraps with replicate), plus a RESHAPE
  type/shape incompatibility and a batch mismatch; 42 of 188 ops would run on
  the GPU and TFLite aborts rather than split. This is the operator-coverage
  risk the paper predicted, observed.
- Thermal status 1 and battery 35 C by this session (it was 0-1, 32 C before).

## Bug found and fixed
MainActivity.row formatted ms with the default locale, and this phone's locale
uses a comma decimal separator: "60,2586" split the CSV column and shifted
thermal and battery. Fixed with Locale.ROOT; the APK must be rebuilt before the
next Android session. Any Android CSV written before 2026-09-20 needs the
rejoin (see the parser note in this folder's analysis or costmodel.py's
equivalent for Apple ids).

"""Two-parameter cost model of a K-step submission, fitted per workload and path.

    t_sub(K) = a + b*K        a: fixed cost per submission (dispatch, sync, I/O)
                              b: cost per simulation step (arithmetic + memory)

so the per-step cost is a/K + b and batching can gain at most (a+b)/b.
Reads the CSV written by sweep.sh (medians of the repeats), fits a and b by
least squares, and reports the fit quality and the predicted vs measured
per-step speedup at K=8. The CPU should show a ~ 0: nothing to amortise.

  python3 costmodel.py sweep.csv [more.csv ...] [--md]

Several CSVs may be given (the Mac sweep, InkBench CSVs pulled from phones);
rows carrying a `device` column are fitted per device.
"""
import csv, sys
from collections import defaultdict
from statistics import median

rows = defaultdict(list)
for path in [a for a in sys.argv[1:] if not a.startswith("--")]:
    f = csv.reader(open(path)); hdr = next(f)
    for vals in f:
        # InkBench builds before 2026-09-19 wrote the model id unquoted, and
        # Apple model ids contain a comma (iPad14,1): rejoin the split field.
        if len(vals) == len(hdr) + 1 and hdr[0] == "device":
            vals = [vals[0] + "," + vals[1]] + vals[2:]
        r = dict(zip(hdr, vals))
        if r["ms_per_submission"]:
            dev = r.get("device") or "M5"
            rows[(dev, r["workload"], r["units"], r["mode"], int(r["K"]))].append(float(r["ms_per_submission"]))

def fit(ks, ts):
    n = len(ks); mk = sum(ks) / n; mt = sum(ts) / n
    b = sum((k - mk) * (t - mt) for k, t in zip(ks, ts)) / sum((k - mk) ** 2 for k in ks)
    a = mt - b * mk
    ss = sum((t - mt) ** 2 for t in ts); se = sum((t - a - b * k) ** 2 for k, t in zip(ks, ts))
    return a, b, 1 - se / ss if ss else 1.0

md = "--md" in sys.argv
series = sorted({k[:4] for k in rows})
hdr = ["device", "workload", "path", "mode", "t_sub K=1..8 (ms)", "a fixed", "b per step", "R2",
       "per-step gain K=8 meas", "pred", "ceiling (a+b)/b"]
print(("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr)) if md else "\t".join(hdr))
for dev, w, u, m in series:
    ks = sorted(k for (d2, w2, u2, m2, k) in rows if (d2, w2, u2, m2) == (dev, w, u, m))
    ts = [median(rows[(dev, w, u, m, k)]) for k in ks]
    if len(ks) < 3:
        continue
    a, b, r2 = fit(ks, ts)
    meas = ts[0] / (ts[-1] / ks[-1])
    pred = (a + b) / (a / ks[-1] + b)
    ceil = (a + b) / b if b > 0 else float("inf")
    cells = [dev, w.replace("_128x256", "").replace("_fp16", ""), u, m, " / ".join(f"{t:.3f}" for t in ts),
             f"{a:.3f}", f"{b:.3f}", f"{r2:.3f}", f"{meas:.2f}x", f"{pred:.2f}x", f"{ceil:.2f}x"]
    print(("| " + " | ".join(cells) + " |") if md else "\t".join(cells))

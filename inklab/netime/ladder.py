"""The paper's per-chip table (tab:ladder), rebuilt from whatever device CSVs exist.

    python3 ladder.py [--workload ink_unroll_j26] [--tex]

Scans inklab/results/*/bench_*.csv plus the Mac sweep, fits t_sub(K) = a + b*K per
device and path for one workload, and prints the table body. With --tex it prints
the LaTeX rows to paste into inklab/paper/main.tex; new phones appear as soon as
their CSV is committed. Chips are named by CHIPS below, keyed on the device id the
CSV carries (Apple model id, or Android "SOC_MANUFACTURER SOC_MODEL").
"""
import csv, glob, os, sys
from collections import defaultdict
from statistics import median

CHIPS = {   # device id in the CSV -> (chip as the paper names it, sort key = year)
    "M5": ("M5 (2025)", 2025),
    "iPhone18,2": ("A19 Pro (2025)", 2025.1),
    "iPad14,1": ("A15 (2021)", 2021),
    "iPhone13,2": ("A14 (2020)", 2020),
    "iPad8,5": ("A12X (2018)", 2018),
}
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PATHS = ["ne", "metal", "cpu"]


def fit(ks, ts):
    n = len(ks); mk = sum(ks) / n; mt = sum(ts) / n
    den = sum((k - mk) ** 2 for k in ks)
    if not den: return 0.0, 0.0
    b = sum((k - mk) * (t - mt) for k, t in zip(ks, ts)) / den
    return mt - b * mk, b


def rows(workload):
    vals = defaultdict(list)   # (device, path, K) -> ms
    files = glob.glob(f"{ROOT}/results/*/bench_*.csv") + glob.glob(f"{ROOT}/results/*/sweep*.csv")
    for path in files:
        with open(path) as fh:
            r = csv.reader(fh); hdr = next(r)
            for v in r:
                if len(v) == len(hdr) + 1 and hdr[0] == "device":   # unquoted "iPad14,1"
                    v = [v[0] + "," + v[1]] + v[2:]
                row = dict(zip(hdr, v))
                if row.get("mode") != "b2b" or not row.get("ms_per_submission"):
                    continue
                w = row["workload"].replace("_128x256", "").replace("_fp16", "").replace("_s8", "_s8")
                if workload not in (row["workload"], w):
                    continue
                vals[(row.get("device") or "M5", row["units"], int(row["K"]))].append(
                    float(row["ms_per_submission"]))
    out = []
    devices = {d for d, _, _ in vals}
    for dev in sorted(devices, key=lambda d: CHIPS.get(d, (d, 9999))[1]):
        cells = {}
        for p in PATHS:
            ks = sorted(k for (d, u, k) in vals if (d, u) == (dev, p))
            if len(ks) < 2: continue
            cells[p] = fit(ks, [median(vals[(dev, p, k)]) for k in ks])
        if cells:
            out.append((CHIPS.get(dev, (dev, 0))[0], cells))
    return out


if __name__ == "__main__":
    wl = sys.argv[sys.argv.index("--workload") + 1] if "--workload" in sys.argv else "ink_unroll_j26"
    table = rows(wl)
    if not table:
        sys.exit(f"no rows for workload {wl}; known: run costmodel.py to list them")
    if "--tex" in sys.argv:
        for chip, c in table:
            f = lambda p, i: f"{c[p][i]:.3f}" if p in c else "--"
            print(f"{chip:14s} & {f('ne',0)} & {f('ne',1)} & {f('metal',0)} & {f('metal',1)} \\\\")
    else:
        print(f"{'chip':16s} {'NE a':>7s} {'NE b':>7s} {'Metal a':>8s} {'Metal b':>8s} {'CPU a':>7s} {'CPU b':>7s}")
        for chip, c in table:
            f = lambda p, i: f"{c[p][i]:7.3f}" if p in c else "      -"
            print(f"{chip:16s} {f('ne',0)} {f('ne',1)} {f('metal',0)} {f('metal',1)} {f('cpu',0)} {f('cpu',1)}")

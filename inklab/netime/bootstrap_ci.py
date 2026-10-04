"""Bootstrap intervals for the cost model t_sub(K) = a + b*K.

For every (device, workload, path) series in back-to-back mode, the repeats at
each K are resampled with replacement, the median per K is refitted, and the
2.5th and 97.5th percentiles of a, b and the K=8 per-step gain are reported.
With three repeats per K the intervals are coarse; they bound run-to-run noise
within one session, not session-to-session variation.

  python3 bootstrap_ci.py file.csv [more.csv ...] [--md] [--n 4000]
"""
import csv, sys, random
from collections import defaultdict
from statistics import median

KS = (1, 2, 4, 8)

def load(paths):
    reps = defaultdict(lambda: defaultdict(list))
    for p in paths:
        with open(p) as fh:
            r = csv.reader(fh); hdr = next(r)
            for v in r:
                if len(v) == len(hdr) + 1 and hdr[0] == "device":
                    v = [v[0] + "," + v[1]] + v[2:]
                if len(v) == len(hdr) + 1 and hdr[0] == "device" and "soc" in hdr:
                    v = v[:7] + [v[7] + "." + v[8]] + v[9:]
                row = dict(zip(hdr, v))
                if row.get("mode") != "b2b" or not row.get("ms_per_submission"):
                    continue
                key = (row.get("device") or "M5", row["workload"], row["units"])
                reps[key][int(row["K"])].append(float(row["ms_per_submission"]))
    return reps

def fit(ks, ts):
    n = len(ks); mk = sum(ks) / n; mt = sum(ts) / n
    b = sum((k - mk) * (t - mt) for k, t in zip(ks, ts)) / sum((k - mk) ** 2 for k in ks)
    return mt - b * mk, b

def pct(xs, q):
    xs = sorted(xs); i = q * (len(xs) - 1); lo = int(i)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (i - lo)

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 4000
    if "--n" in sys.argv: args.remove(str(n))
    md = "--md" in sys.argv
    rnd = random.Random(1)
    rows = []
    for key, byk in sorted(load(args).items()):
        ks = [k for k in KS if byk.get(k)]
        if len(ks) < 3:
            continue
        med = [median(byk[k]) for k in ks]
        a0, b0 = fit(ks, med)
        A, B, G = [], [], []
        for _ in range(n):
            ts = [median(rnd.choices(byk[k], k=len(byk[k]))) for k in ks]
            a, b = fit(ks, ts); A.append(a); B.append(b)
            if 1 in ks and 8 in ks:
                G.append(ts[ks.index(1)] / (ts[ks.index(8)] / 8))
        g0 = med[ks.index(1)] / (med[ks.index(8)] / 8) if 1 in ks and 8 in ks else float("nan")
        rows.append((*key, a0, pct(A, .025), pct(A, .975), b0, pct(B, .025), pct(B, .975),
                     g0, pct(G, .025) if G else float("nan"), pct(G, .975) if G else float("nan")))
    if md:
        print("| device | workload | path | a | a 95% | b | b 95% | gain K=8 | gain 95% |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} to {r[5]:.3f} | {r[6]:.3f} | "
                  f"{r[7]:.3f} to {r[8]:.3f} | {r[9]:.2f} | {r[10]:.2f} to {r[11]:.2f} |")
    else:
        for r in rows:
            print(*r)

if __name__ == "__main__":
    main()

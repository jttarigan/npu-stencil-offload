"""Where does an NNAPI ink output differ from the fp32 CPU, and does it look like rounding?

  python3 ink_dump.py <dump dir of one condition> [more...]
Per output: max/rms error as % of range, border band vs interior, the error
relative to each cell's own magnitude (rounding scales with |value|, a wrong
read does not), the share of cells off by more than 1% of range, and whether
the bad cells cluster in rows/columns.
"""
import sys, os
import numpy as np

BAND = 4
NAMES = {"out0": "velocity", "out1": "dye", "out2": "pressure"}

def load(d):
    meta = {}
    for line in open(os.path.join(d, "meta.txt")):
        key, name, shape = line.split()
        meta[key] = tuple(int(s) for s in shape.split("x"))
    outs = {}
    for k in sorted(meta):
        if k.startswith("out") and os.path.exists(os.path.join(d, "got" + k[3:] + ".f32")):
            w = np.fromfile(os.path.join(d, "want" + k[3:] + ".f32"), "<f4").reshape(meta[k])[0]
            g = np.fromfile(os.path.join(d, "got" + k[3:] + ".f32"), "<f4").reshape(meta[k])[0]
            outs[k] = (w, g)
    return outs

for d in sys.argv[1:]:
    print(f"== {os.path.basename(d.rstrip('/'))}")
    for k, (w, g) in load(d).items():
        e = np.abs(g - w)
        rng = float(w.max() - w.min())
        H, W = w.shape[:2]
        border = np.zeros((H, W), bool)
        border[:BAND, :] = border[-BAND:, :] = True
        border[:, :BAND] = border[:, -BAND:] = True
        eb, ei = e[border], e[~border]
        rel = e / (np.abs(w) + 1e-3 * rng)
        big = (e > 0.01 * rng).any(axis=-1)
        print(f"  {k} {NAMES.get(k, k):9s} range {rng:.3g}  max {100*e.max()/rng:.3f}%  rms {100*np.sqrt((e**2).mean())/rng:.4f}%"
              f"  | border max {100*eb.max()/rng:.3f}%  interior max {100*ei.max()/rng:.3f}%"
              f"  | rel-to-value median {np.median(rel):.2e} p99 {np.quantile(rel, 0.99):.2e}"
              f"  | cells >1% of range: {big.sum()} ({100*big.mean():.2f}%)")
        if big.any():
            ys, xs = np.nonzero(big)
            print(f"      bad cells: rows {ys.min()}-{ys.max()} ({len(set(ys))} distinct), cols {xs.min()}-{xs.max()} ({len(set(xs))} distinct)")
            i = np.argmax(e.max(axis=-1))
            y, x = divmod(int(i), W)
            print(f"      worst cell (y={y}, x={x}): want {w[y, x]} got {g[y, x]}")

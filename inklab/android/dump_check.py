"""Where do a delegate's outputs differ from fp32, and which padding explains them?

Reads the per-condition dumps written by InkBench with CHECK=1 DUMP=1
(results/<run>/dump/<workload>_K<k>_<backend>/: meta.txt, in*.f32, want*.f32,
got*.f32) and, for padtest and the heat / Gray-Scott stencils, recomputes the
output in NumPy from the dumped inputs with edge-replicate (what the graph
asks for) and with REFLECT padding. Prints, per condition, the error of the
delegate's output against each, split into a border band and the interior.

For gathertest it recomputes the index and every gather from the dumped inputs,
reports per output how many cells are wrong, and for wrong gathered values finds
which source element the delegate actually read (the inputs are random, so a
value identifies its source; exact in fp32, ambiguous where fp16 rounding makes
values collide), then prints the most common (source - expected) offsets.

  python3 dump_check.py results/<run>/dump [more dump dirs...]
"""
import sys, os, glob
import numpy as np

H, W = 256, 128
ALPHA = 0.2
DU, DV, F, KILL = 0.16, 0.08, 0.035, 0.065
BAND = 4  # cells from any edge counted as "border"

def load(d):
    meta = {}
    for line in open(os.path.join(d, "meta.txt")):
        key, name, shape = line.split()
        meta[key] = (name, tuple(int(s) for s in shape.split("x")))
    arr = lambda key, f: np.fromfile(os.path.join(d, f), "<f4").reshape(meta[key][1])
    ins = {meta[k][0]: arr(k, k + ".f32") for k in meta if k.startswith("in")}
    outs = [(arr(k, "want" + k[3:] + ".f32"), arr(k, "got" + k[3:] + ".f32"))
            for k in sorted(meta) if k.startswith("out") and os.path.exists(os.path.join(d, "got" + k[3:] + ".f32"))]
    return ins, outs

def pad(x, mode):  # x is (H, W, C)
    return np.pad(x, ((1, 1), (1, 1), (0, 0)), mode=mode)

def stencil(name, S, field, src, mode):
    """field (1,H,W,C), src (K,H,W,C) -> (1,H,W,C), float32, same order of operations as tflite_graph.py."""
    x = field[0].astype(np.float32)
    for s in range(src.shape[0]):
        x = x + src[s]
        for _ in range(S):
            p = pad(x, mode)
            lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * x
            if name == "heat":
                x = x + np.float32(ALPHA) * lap
            else:
                u, v = x[..., 0], x[..., 1]; uvv = u * v * v
                x = np.stack([u + DU * lap[..., 0] - uvv + F * (1 - u),
                              v + DV * lap[..., 1] + uvv - (F + KILL) * v], -1).astype(np.float32)
    return x[None]

GSHIFT = 40.0
def gather_ref(vel, x1, x2, x3):
    """Same as tflite_graph.py's gather_ref; returns (outputs by key, index, constant index)."""
    ys, xs = np.mgrid[0:H, 0:W]
    PX = (xs + 0.5).astype(np.float32)[None, ..., None]; PY = (ys + 0.5).astype(np.float32)[None, ..., None]
    bx = PX - vel[..., 0:1] * np.float32(GSHIFT); by = PY - vel[..., 1:2] * np.float32(GSHIFT)
    x0 = np.clip(np.floor(bx - np.float32(0.5)), 0, W - 1).astype(np.float32)
    y0 = np.clip(np.floor(by - np.float32(0.5)), 0, H - 1).astype(np.float32)
    ii = (y0.astype(np.int64) * W + x0.astype(np.int64)).reshape(H * W)
    gc = (np.clip(ys - 3, 0, H - 1) * W + np.clip(xs - 5, 0, W - 1)).reshape(H * W)
    g = lambda x, idx: x.reshape(H * W, -1)[idx].reshape(1, H, W, -1)
    outs = {"a_y0": y0, "b_x0": x0, "c_g1": g(x1, ii), "d_g2": g(x2, ii), "e_g3": g(x3, ii), "f_gc3": g(x3, gc)}
    return outs, ii, gc

def gathertest(tag, ins, outs):
    vel, x1, x2, x3 = (ins[f"serving_default_{n}:0"] for n in ("vel", "x1", "x2", "x3"))
    ref, ii, gc = gather_ref(vel, x1, x2, x3)
    tables = {"c_g1": (x1, ii), "d_g2": (x2, ii), "e_g3": (x3, ii), "f_gc3": (x3, gc)}
    fp16 = "fp32" not in tag
    for want, got in outs:
        # which output is this? the fp32 CPU reference equals exactly one recomputed output
        key = min((k for k in ref if ref[k].shape == want.shape), key=lambda k: np.abs(ref[k] - want).max())
        e_cpu = np.abs(ref[key] - want).max()
        err = np.abs(got - ref[key])
        tol = 2e-3 * max(np.abs(ref[key]).max(), 1e-6) if fp16 else 0.0
        wrong = ~(err <= tol)
        print(f"{tag:30s} {key:6s} wrong {int(wrong.sum()):6d} of {got.size} "
              f"({wrong.mean():.1%})  max|err| {np.nanmax(err) if np.isfinite(err).any() else float('nan'):.3g}"
              f"  nonfinite {int((~np.isfinite(got)).sum())}  (NumPy vs fp32 cpu {e_cpu:.1g})")
        if key not in tables or not wrong.any():
            continue
        src, idx = tables[key]
        C = src.shape[-1]
        flat = src.reshape(-1)
        cast = (lambda a: a.astype(np.float16).astype(np.float32)) if fp16 else (lambda a: a)
        vals = cast(flat)
        order = np.argsort(vals, kind="stable"); sv = vals[order]
        g = got.reshape(-1); w = wrong.reshape(-1)
        cells = np.nonzero(w)[0]
        lo = np.searchsorted(sv, g[cells], "left"); hi = np.searchsorted(sv, g[cells], "right")
        unique = (hi - lo) == 1
        found = order[np.minimum(lo, len(order) - 1)]
        exp = idx[cells // C] * C + cells % C           # element the graph asked for
        off = (found - exp)[unique]
        print(f"{'':30s}        wrong values found in the table uniquely: {unique.mean():.0%}"
              + (f"; not in the table at all: {((hi - lo) == 0).mean():.0%}" if ((hi - lo) == 0).any() else ""))
        if off.size:
            # named hypotheses for how the read went wrong (fractions of the uniquely found values)
            f_, cell, ch = found[unique], cells[unique] // C, cells[unique] % C
            want = idx[cell]
            hyp = {"right cell, wrong channel": (f_ // C == want) & (f_ % C != ch),
                   "row width ignored (idx + c)": f_ == want + ch,
                   "planar table (c*H*W + idx)": f_ == ch * H * W + want,
                   "no gather (own cell)": f_ == cell * C + ch,
                   "same channel": f_ % C == ch}
            print(f"{'':30s}        " + ", ".join(f"{k} {np.mean(v):.0%}" for k, v in hyp.items()))
            dc, n = np.unique(f_ // C - want, return_counts=True)
            top = np.argsort(-n)[:6]
            print(f"{'':30s}        source cell - requested cell (one image row = {W} cells):",
                  ", ".join(f"{int(dc[t])} x{int(n[t])}" for t in top))

def split(err, h, w):
    b = np.zeros((h, w), bool); b[:BAND] = b[-BAND:] = True; b[:, :BAND] = b[:, -BAND:] = True
    e = err.reshape(err.shape[0], h, w, -1) if err.ndim == 4 else err
    e = np.abs(e[0])
    return e[b].max(), e[~b].max() if (~b).any() else 0.0

def main(dirs):
    conds = sorted(d for root in dirs for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    print(f"{'condition':30s} {'vs':8s} {'border max':>11s} {'interior max':>13s}   (absolute; range of fp32 output in brackets)")
    for d in conds:
        tag = os.path.basename(d)
        ins, outs = load(d)
        if tag.startswith("gathertest"):
            gathertest(tag, ins, outs)
            continue
        want, got = outs[0]
        rng = float(want.max() - want.min())
        cands = {"fp32 cpu": want}
        if tag.startswith("padtest"):
            x = ins["serving_default_x:0"][0]
            cands = {"fp32 cpu": want, "edge": pad(x, "edge")[None], "reflect": pad(x, "reflect")[None]}
        elif tag.startswith(("heat", "grayscott")):
            name = "heat" if tag.startswith("heat") else "grayscott"
            S = int(tag.split("_s")[1].split("_")[0])
            field = ins["serving_default_field:0"]; src = ins["serving_default_src:0"]
            cands = {"fp32 cpu": want, "edge": stencil(name, S, field, src, "edge"),
                     "reflect": stencil(name, S, field, src, "reflect")}
        h, w = got.shape[1], got.shape[2]
        nonfinite = int((~np.isfinite(got)).sum())
        for label, ref in cands.items():
            bmax, imax = split(got - ref, h, w)
            print(f"{tag:30s} {label:8s} {bmax:11.3g} {imax:13.3g}   [{rng:.3g}]" + (f"  nonfinite {nonfinite}" if nonfinite and label == "fp32 cpu" else ""))
        if "edge" in cands:
            e = np.abs(cands["edge"] - want).max()
            print(f"{'':30s} (NumPy edge model vs fp32 cpu: {e:.2g}, so the model is the graph)")

if __name__ == "__main__":
    main(sys.argv[1:])

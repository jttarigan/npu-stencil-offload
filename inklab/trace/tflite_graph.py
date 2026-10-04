"""The ink step (and the two generality stencils) as LiteRT / TFLite models for Android.

Same programs as unroll_graph.py / stencil_graph.py, K steps per submission,
but written so every operator is a TFLite builtin that the Android delegates
(GPU, NNAPI, Qualcomm QNN) can claim:

  * advection: Core ML's `resample` (ONNX GridSample) has no TFLite builtin and
    no NNAPI op, so the bilinear clamp-to-edge read is spelled out as floor /
    frac of the back-traced coordinate and four GATHERs on flattened indices
    (GATHER is an NNAPI op since API 29; GATHER_ND is TFLite-only).
  * replicate padding of width 1 == MIRROR_PAD in SYMMETRIC mode.
  * everything else is CONV_2D / DEPTHWISE_CONV_2D and elementwise.

Layout is NHWC. The converter is restricted to TFLITE_BUILTINS, so a model that
would need the Flex (full TensorFlow) runtime fails to convert instead of
silently shipping ops no delegate can run.

--int-index builds the flat gather index y*W + x in int32 (file suffix _i32).
The default builds it in float and casts, which is exact in fp32 but not on an
accelerator that computes in fp16: integers above 2048 are not representable,
indices up to H*W-1 = 32767 round to multiples of 16, and some land past the end
("gather index out of bounds" through NNAPI on the POCO X6 Pro and Tab S7+).
The float version is kept because every Android timing so far used it.

--fp16-safe rewrites the vorticity confinement so that fp16 cannot make NaN
(file suffix _fs). The original divides curl * gate by max(ln, 1e-12); in fp16
1e-12 is 0, so wherever ln == 0 that is 0 / 0. The rewrite normalises the
gradient by max(ln, 1e-12) + (1 - gate), which equals ln wherever gate is 1, so
fp32 results are the same up to rounding order.

--gather-c1 does the bilinear read with one rank-1 GATHER table per channel
instead of one (H*W, C) table per field (file suffix _g1), in case an
accelerator mishandles gathers of rows wider than one value.

--edge-concat builds the width-1 replicate padding from edge slices and
CONCATENATION instead of MIRROR_PAD (file suffix _ec). On the POCO X6 Pro the
APU's MIRROR_PAD output matches REFLECT rather than SYMMETRIC (2026-09-30 output
check), so the stencils were wrong near the borders. padtest is that one op alone.

  .venv-tf/bin/python tflite_graph.py ink [K] [--emit-all] [--int-index] [--edge-concat]
                                         [--fp16-safe] [--gather-c1]
  .venv-tf/bin/python tflite_graph.py heat|grayscott [S] [K] [--edge-concat]
  .venv-tf/bin/python tflite_graph.py padtest
  .venv-tf/bin/python tflite_graph.py gathertest
     gathertest is the advection's index chain and GATHER alone: inputs vel, x1, x2, x3;
     the int32 index of each cell's lower-left corner after a back-trace of up to
     +-4 cells (vel * 40), and gathers of 1-, 2- and 3-channel tables with it, plus a
     3-channel gather with a constant index (a shift by 3 rows and 5 columns).
     -> ../android/models/*.tflite, a K-step fidelity check against the NumPy
        reference, and the operator list.
"""
import os, sys, json, numpy as np
import tensorflow as tf

W, H = 128, 256
DT, VDAMP, DDAMP, VORT, ITERS = 1 / 30, 0.3, 0.03, 2.5, 26
NAME = sys.argv[1]
EMIT_ALL = "--emit-all" in sys.argv
INT_INDEX = "--int-index" in sys.argv
EDGE_CONCAT = "--edge-concat" in sys.argv
FP16_SAFE = "--fp16-safe" in sys.argv
GATHER_C1 = "--gather-c1" in sys.argv
pos = [a for a in sys.argv[2:] if not a.startswith("--")]
if NAME == "ink":
    K = int(pos[0]) if pos else 4
    S = ITERS
elif NAME in ("padtest", "gathertest"):
    K, S = 1, 0
else:
    S = int(pos[0]) if pos else 8
    K = int(pos[1]) if len(pos) > 1 else 4
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "android", "models")

def pad1(x):
    if EDGE_CONCAT:
        x = tf.concat([x[:, :1], x, x[:, -1:]], axis=1)
        return tf.concat([x[:, :, :1], x, x[:, :, -1:]], axis=2)
    return tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")

def conv3(x, k):
    """3x3 conv with replicate padding; k is (3,3,Cin,Cout) in HWIO."""
    return tf.nn.conv2d(pad1(x), tf.constant(k), strides=1, padding="VALID")

def taps(pattern, cin=1, cout=1):
    k = np.zeros((3, 3, cin, cout), np.float32)
    for (dy, dx, ci, co), w in pattern.items():
        k[dy + 1, dx + 1, ci, co] = w
    return k

# The same stencils as the Metal kernels (x = right, y = down).
KDIV = taps({(0, 1, 0, 0): 0.5, (0, -1, 0, 0): -0.5, (1, 0, 1, 0): 0.5, (-1, 0, 1, 0): -0.5}, 2, 1)
KCURL = taps({(0, 1, 1, 0): 0.5, (0, -1, 1, 0): -0.5, (1, 0, 0, 0): -0.5, (-1, 0, 0, 0): 0.5}, 2, 1)
KGRAD = taps({(0, 1, 0, 0): 0.5, (0, -1, 0, 0): -0.5, (1, 0, 0, 1): 0.5, (-1, 0, 0, 1): -0.5}, 1, 2)
KJAC = taps({(0, 1, 0, 0): 0.25, (0, -1, 0, 0): 0.25, (1, 0, 0, 0): 0.25, (-1, 0, 0, 0): 0.25})
KLAP = taps({(0, 1, 0, 0): 1, (0, -1, 0, 0): 1, (1, 0, 0, 0): 1, (-1, 0, 0, 0): 1, (0, 0, 0, 0): -4})

ys, xs = np.mgrid[0:H, 0:W]
PX = (xs + 0.5).astype(np.float32)[None, ..., None]
PY = (ys + 0.5).astype(np.float32)[None, ..., None]
WALL = np.ones((1, H, W, 1), np.float32)
WALL[:, 0], WALL[:, -1], WALL[:, :, 0], WALL[:, :, -1] = 0, 0, 0, 0

def bilinear(fields, bx, by):
    """Clamp-to-edge bilinear read of each (1,H,W,C) field at texel coords (bx, by)."""
    u = bx - 0.5; v = by - 0.5
    x0 = tf.floor(u); y0 = tf.floor(v)
    fx = u - x0; fy = v - y0
    xi0 = tf.clip_by_value(x0, 0, W - 1); xi1 = tf.clip_by_value(x0 + 1, 0, W - 1)
    yi0 = tf.clip_by_value(y0, 0, H - 1); yi1 = tf.clip_by_value(y0 + 1, 0, H - 1)
    if INT_INDEX:
        idx = lambda yy, xx: tf.reshape(tf.cast(yy, tf.int32) * W + tf.cast(xx, tf.int32), [H * W])
    else:
        idx = lambda yy, xx: tf.reshape(tf.cast(yy * W + xx, tf.int32), [H * W])
    corners = [(idx(yi0, xi0), (1 - fx) * (1 - fy)), (idx(yi0, xi1), fx * (1 - fy)),
               (idx(yi1, xi0), (1 - fx) * fy), (idx(yi1, xi1), fx * fy)]
    out = []
    for f in fields:
        c = f.shape[-1]
        if GATHER_C1:
            chans = []
            for ch in range(c):
                flat = tf.reshape(f[..., ch:ch + 1], [H * W])
                acc = None
                for ii, wgt in corners:
                    g = tf.reshape(tf.gather(flat, ii), [1, H, W, 1]) * wgt
                    acc = g if acc is None else acc + g
                chans.append(acc)
            out.append(tf.concat(chans, axis=-1))
            continue
        flat = tf.reshape(f, [H * W, c])
        acc = None
        for ii, wgt in corners:
            g = tf.reshape(tf.gather(flat, ii), [1, H, W, c]) * wgt
            acc = g if acc is None else acc + g
        out.append(acc)
    return out

def ink_program(vel, dye, pr, velAdd, dyeAdd):
    v, d, p = vel, dye, pr
    seq = []
    for s in range(K):
        va = velAdd[s:s + 1]; da = dyeAdd[s:s + 1]
        vx = v[..., 0:1]; vy = v[..., 1:2]
        bx = PX - vx * DT; by = PY - vy * DT
        av, ad = bilinear([v, d], bx, by)
        nv = (av * float(np.exp(-DT * VDAMP)) + va) * WALL
        nd = tf.minimum(ad * float(np.exp(-DT * DDAMP)) + da, 1.6)
        seq.append(nd)
        curl = conv3(nv, KCURL)
        g = conv3(tf.abs(curl), KGRAD)
        gx = g[..., 0:1]; gy = g[..., 1:2]
        ln = tf.sqrt(gx * gx + gy * gy)
        gate = tf.cast(ln > 1e-5, tf.float32)
        if FP16_SAFE:
            den = tf.maximum(ln, 1e-12) + (1 - gate)
            nx = gx / den * gate; ny = gy / den * gate
            nv = nv + tf.concat([ny * curl, -nx * curl], axis=-1) * (VORT * DT)
        else:
            f = curl * gate / tf.maximum(ln, 1e-12) * (VORT * DT)
            nv = nv + tf.concat([gy * f, -gx * f], axis=-1)
        div = conv3(nv, KDIV)
        q = p
        for _ in range(ITERS):
            q = conv3(q, KJAC) - div * 0.25
        p = q
        v = (nv - conv3(p, KGRAD)) * WALL
        d = nd
    dye_out = tf.concat(seq, axis=0) if EMIT_ALL else d
    return {"vel_out": v, "dye_out": dye_out, "pr_out": p}

GSHIFT = 40.0  # gathertest: vel in (-0.1, 0.1) moves the read by up to 4 cells
GCONST = (np.clip(ys - 3, 0, H - 1) * W + np.clip(xs - 5, 0, W - 1)).reshape(H * W).astype(np.int32)

def gather_program(vel, x1, x2, x3):
    """The advection's index chain (int32 index, lower-left corner only) and GATHER alone.
    Outputs are sorted by name in the model: a_y0, b_x0, c_g1, d_g2, e_g3, f_gc3."""
    bx = PX - vel[..., 0:1] * GSHIFT; by = PY - vel[..., 1:2] * GSHIFT
    x0 = tf.clip_by_value(tf.floor(bx - 0.5), 0, W - 1)
    y0 = tf.clip_by_value(tf.floor(by - 0.5), 0, H - 1)
    ii = tf.reshape(tf.cast(y0, tf.int32) * W + tf.cast(x0, tf.int32), [H * W])
    g1 = tf.reshape(tf.gather(tf.reshape(x1, [H * W]), ii), [1, H, W, 1])
    g2 = tf.reshape(tf.gather(tf.reshape(x2, [H * W, 2]), ii), [1, H, W, 2])
    g3 = tf.reshape(tf.gather(tf.reshape(x3, [H * W, 3]), ii), [1, H, W, 3])
    gc3 = tf.reshape(tf.gather(tf.reshape(x3, [H * W, 3]), tf.constant(GCONST)), [1, H, W, 3])
    return {"a_y0": y0, "b_x0": x0, "c_g1": g1, "d_g2": g2, "e_g3": g3, "f_gc3": gc3}

def gather_ref(vel, x1, x2, x3):
    """NumPy model of gather_program, (1,H,W,C) arrays in, dict out."""
    bx = PX - vel[..., 0:1] * np.float32(GSHIFT); by = PY - vel[..., 1:2] * np.float32(GSHIFT)
    x0 = np.clip(np.floor(bx - np.float32(0.5)), 0, W - 1).astype(np.float32)
    y0 = np.clip(np.floor(by - np.float32(0.5)), 0, H - 1).astype(np.float32)
    ii = (y0.astype(np.int32) * W + x0.astype(np.int32)).reshape(H * W)
    g = lambda x, idx: x.reshape(H * W, -1)[idx].reshape(1, H, W, -1)
    return {"a_y0": y0, "b_x0": x0, "c_g1": g(x1, ii), "d_g2": g(x2, ii), "e_g3": g(x3, ii),
            "f_gc3": g(x3, GCONST)}

DU, DV, F, KILL, ALPHA = 0.16, 0.08, 0.035, 0.065, 0.2

def stencil_program(field, src):
    x = field
    for s in range(K):
        x = x + src[s:s + 1]
        for _ in range(S):
            if NAME == "heat":
                x = x + ALPHA * conv3(x, KLAP)
            else:
                u = x[..., 0:1]; v = x[..., 1:2]
                lu = conv3(u, KLAP); lv = conv3(v, KLAP)
                uvv = u * v * v
                x = tf.concat([u + DU * lu - uvv + F * (1 - u), v + DV * lv + uvv - (F + KILL) * v], -1)
    return {"field_out": x}

# --- NumPy references, copied from unroll_graph.py / stencil_graph.py (both
# parse sys.argv and import coremltools at import time, so they cannot be imported here) ---
def splat_fields(ref, splats):
    nv = np.zeros((H, W, 2), np.float32); nd = np.zeros((H, W, 3), np.float32)
    for s in splats:
        pv = np.array(s["posVel"], np.float32); col = np.array(s["color"], np.float32)
        pa = np.array(s["params"], np.float32)
        dd = ref.p - pv[:2]; r = max(pa[0], 0.5)
        g = np.exp(-(dd * dd).sum(-1) / (r * r)).astype(np.float32)
        nv += pv[2:4] * g[..., None]
        dl = np.sqrt((dd * dd).sum(-1))
        nv += np.where((dl > 0.001)[..., None], dd / np.maximum(dl, 1e-9)[..., None] * (pa[2] * g)[..., None], 0)
        nd += col * (pa[1] * g)[..., None]
    return nv, nd

def ref_step(x, add):
    x = x + add
    for _ in range(S):
        p = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
        l = p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:] - 4 * x
        if NAME == "heat":
            x = x + ALPHA * l
        else:
            u, v = x[0], x[1]; uvv = u * v * v
            x = np.stack([u + DU * l[0] - uvv + F * (1 - u), v + DV * l[1] + uvv - (F + KILL) * v])
    return x.astype(np.float32)

def initial(rng):
    if NAME == "heat":
        return rng.uniform(0, 1, (1, H, W)).astype(np.float32)
    x = np.stack([np.ones((H, W)), np.zeros((H, W))]).astype(np.float32)
    for _ in range(12):
        cy, cx = rng.integers(8, H - 8), rng.integers(8, W - 8)
        x[0, cy - 4:cy + 4, cx - 4:cx + 4] = 0.5; x[1, cy - 4:cy + 4, cx - 4:cx + 4] = 0.25
    return x

def convert(fn, specs):
    mod = tf.Module()
    mod.f = tf.function(fn, input_signature=specs)
    conc = mod.f.get_concrete_function()
    conv = tf.lite.TFLiteConverter.from_concrete_functions([conc], mod)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return conv.convert()

def op_list(model_bytes):
    from collections import Counter
    it = tf.lite.Interpreter(model_content=model_bytes)
    ops = Counter(d["op_name"] for d in it._get_ops_details())
    return dict(sorted(ops.items(), key=lambda kv: -kv[1]))

def run(model_bytes, feeds):
    it = tf.lite.Interpreter(model_content=model_bytes)
    runner = it.get_signature_runner()
    return runner(**feeds)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if NAME == "ink":
        specs = [tf.TensorSpec([1, H, W, 2], tf.float32, name="vel"),
                 tf.TensorSpec([1, H, W, 3], tf.float32, name="dye"),
                 tf.TensorSpec([1, H, W, 1], tf.float32, name="pr"),
                 tf.TensorSpec([K, H, W, 2], tf.float32, name="velAdd"),
                 tf.TensorSpec([K, H, W, 3], tf.float32, name="dyeAdd")]
        blob = convert(ink_program, specs)
        path = os.path.join(OUT, f"ink_128x256_j{ITERS}" + ("_i32" if INT_INDEX else "")
                            + ("_ec" if EDGE_CONCAT else "") + ("_fs" if FP16_SAFE else "")
                            + ("_g1" if GATHER_C1 else "") + f"_u{K}" + ("_seq" if EMIT_ALL else "") + ".tflite")
    elif NAME == "padtest":
        blob = convert(lambda x: {"padded": pad1(x)}, [tf.TensorSpec([1, H, W, 1], tf.float32, name="x")])
        path = os.path.join(OUT, "padtest_128x256" + ("_ec" if EDGE_CONCAT else "") + ".tflite")
    elif NAME == "gathertest":
        specs = [tf.TensorSpec([1, H, W, 2], tf.float32, name="vel"),
                 tf.TensorSpec([1, H, W, 1], tf.float32, name="x1"),
                 tf.TensorSpec([1, H, W, 2], tf.float32, name="x2"),
                 tf.TensorSpec([1, H, W, 3], tf.float32, name="x3")]
        blob = convert(gather_program, specs)
        path = os.path.join(OUT, "gathertest_128x256.tflite")
    else:
        C = {"heat": 1, "grayscott": 2}[NAME]
        specs = [tf.TensorSpec([1, H, W, C], tf.float32, name="field"),
                 tf.TensorSpec([K, H, W, C], tf.float32, name="src")]
        blob = convert(stencil_program, specs)
        path = os.path.join(OUT, f"{NAME}_128x256_s{S}" + ("_ec" if EDGE_CONCAT else "") + f"_u{K}.tflite")
    open(path, "wb").write(blob)
    print("saved", os.path.relpath(path), f"{len(blob) / 1e6:.2f} MB")
    print("ops:", op_list(blob))

    # --- fidelity against the NumPy references (float32 on the Mac's TFLite CPU kernels) ---
    if NAME == "padtest":
        x = np.random.default_rng(1).random((1, H, W, 1)).astype(np.float32)
        err = np.abs(run(blob, {"x": x})["padded"] - np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")).max()
        print(f"padtest max|err| vs edge pad {err:.3e}")
    elif NAME == "gathertest":
        rng = np.random.default_rng(1)
        feeds = {"vel": rng.uniform(-0.1, 0.1, (1, H, W, 2)).astype(np.float32)}
        for c in (1, 2, 3):
            feeds[f"x{c}"] = rng.uniform(-0.1, 0.1, (1, H, W, c)).astype(np.float32)
        out, ref = run(blob, feeds), gather_ref(**feeds)
        it = tf.lite.Interpreter(model_content=blob)
        print("output tensors in order:", [d["name"] for d in it.get_output_details()],
              "signature keys:", sorted(out))
        for k in sorted(ref):
            print(f"  {k}: max|err| vs NumPy {np.abs(out[k] - ref[k]).max():.3e}")
    elif NAME == "ink":
        from reference import InkRef
        m = json.load(open("trace_1step/manifest.json")); n = 20
        ref = InkRef(m["W"], m["H"], m["dt"], m["velDamp"], m["dyeDamp"], m["vorticity"], m["jacobiIters"])
        ref.vel = np.fromfile(f"trace_1step/vel_{n:04d}.f32", np.float32).reshape(H, W, 2)
        ref.dye = np.fromfile(f"trace_1step/dye_{n:04d}.f32", np.float32).reshape(H, W, 3)
        ref.pr = np.fromfile(f"trace_1step/pr_{n:04d}.f32", np.float32).reshape(H, W)
        scripts = [m["script"][n + s]["splats"] for s in range(K)]
        va = np.zeros((K, H, W, 2), np.float32); da = np.zeros((K, H, W, 3), np.float32)
        for s in range(K):
            va[s], da[s] = splat_fields(ref, scripts[s])
        feeds = {"vel": ref.vel[None], "dye": ref.dye[None], "pr": ref.pr[None, ..., None],
                 "velAdd": va, "dyeAdd": da}
        for s in range(K):
            ref.step(scripts[s])
        out = run(blob, feeds)
        dye = out["dye_out"][-1]
        ev = np.abs(out["vel_out"][0] - ref.vel).max(); ed = np.abs(dye - ref.dye).max()
        ep = np.abs(out["pr_out"][0, ..., 0] - ref.pr).max()
        print(f"after {K} steps  max|err| vel {ev:.3e} (of {np.abs(ref.vel).max():.1f}) "
              f"dye {ed:.3e} pr {ep:.3e}  dye mass ratio {dye.sum() / ref.dye.sum():.5f}")
    else:
        rng = np.random.default_rng(1)
        x0 = initial(rng)
        C = x0.shape[0]
        src = np.zeros((K, C, H, W), np.float32)
        for s in range(K):
            cy, cx = rng.integers(4, H - 4), rng.integers(4, W - 4)
            src[s, -1, cy - 2:cy + 2, cx - 2:cx + 2] = 0.2
        ref = x0.copy()
        for s in range(K):
            ref = ref_step(ref, src[s])
        out = run(blob, {"field": x0.transpose(1, 2, 0)[None], "src": src.transpose(0, 2, 3, 1)})
        err = np.abs(out["field_out"][0].transpose(2, 0, 1) - ref).max()
        print(f"after {K} steps x {S} sweeps  max|err| {err:.3e}")

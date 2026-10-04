"""Direct Core ML graph of the ink step (feasibility spike).

Builds the step with the coremltools MIL builder: advection as `resample`,
vorticity / divergence / projection as fixed 3x3 convolutions with replicate
padding (clamp-to-edge), the pressure solve as 26 repeated 5-tap
convolutions warm-started from the previous pressure. Splat contributions
(gaussian dye drops and impulses) enter as two input fields computed on the
CPU, since they are cheap and change every frame.

  .venv/bin/python coreml_graph.py [W H] [iters]   -> ink_step.mlpackage,
     fidelity vs the NumPy reference on the trace_1step states, and latency
     on CPU-only, CPU+GPU, CPU+NE compute units, plus op placement.
"""
import sys, time, json, numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from reference import InkRef, bilinear, neighbours

W = int(sys.argv[1]) if len(sys.argv) > 2 else 128
H = int(sys.argv[2]) if len(sys.argv) > 2 else 256
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 26
PREC = sys.argv[4] if len(sys.argv) > 4 else "fp16"
DECAY_K = int(sys.argv[5]) if len(sys.argv) > 5 else 1   # apply damping every K steps with the K-th power
DT, VDAMP, DDAMP, VORT = 1 / 30, 0.3, 0.03, 2.5

def conv3(x, k, name):
    """3x3 conv with replicate padding on a (1,C,H,W) tensor; k: (Cout,Cin,3,3)."""
    xp = mb.pad(x=x, pad=[0, 0, 0, 0, 1, 1, 1, 1], mode="replicate")
    return mb.conv(x=xp, weight=k.astype(np.float32), pad_type="valid", name=name)

def build():
    ys, xs = np.mgrid[0:H, 0:W]
    px = (xs + 0.5).astype(np.float32); py = (ys + 0.5).astype(np.float32)
    wall = np.ones((1, 1, H, W), np.float32); wall[:, :, 0, :] = 0; wall[:, :, -1, :] = 0; wall[:, :, :, 0] = 0; wall[:, :, :, -1] = 0
    # kernels (Cout, Cin, kh, kw); image axes: kh = y, kw = x
    kdx = np.zeros((1, 1, 3, 3), np.float32); kdx[0, 0, 1, 2] = 0.5; kdx[0, 0, 1, 0] = -0.5   # 0.5*(R-L)
    kdy = np.zeros((1, 1, 3, 3), np.float32); kdy[0, 0, 2, 1] = 0.5; kdy[0, 0, 0, 1] = -0.5   # 0.5*(D-U)
    kjac = np.zeros((1, 1, 3, 3), np.float32); kjac[0, 0, 1, 0] = kjac[0, 0, 1, 2] = kjac[0, 0, 0, 1] = kjac[0, 0, 2, 1] = 0.25
    kdiv = np.zeros((1, 2, 3, 3), np.float32); kdiv[0, 0] = kdx[0, 0]; kdiv[0, 1] = kdy[0, 0]     # d vx/dx + d vy/dy
    kcurl = np.zeros((1, 2, 3, 3), np.float32); kcurl[0, 1] = kdx[0, 0]; kcurl[0, 0] = -kdy[0, 0] # d vy/dx - d vx/dy
    kgrad = np.concatenate([kdx, kdy], 0)                                                       # (2,1,3,3): gradient of a scalar
    kproj = np.concatenate([kdx, kdy], 0)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, H, W)), mb.TensorSpec(shape=(1, 3, H, W)),
                             mb.TensorSpec(shape=(1, 1, H, W)), mb.TensorSpec(shape=(1, 2, H, W)),
                             mb.TensorSpec(shape=(1, 3, H, W))], opset_version=ct.target.iOS17)
    def prog(vel, dye, pr, velAdd, dyeAdd):
        # --- advection: back-trace, bilinear resample with clamp-to-edge ---
        vx = mb.slice_by_index(x=vel, begin=[0, 0, 0, 0], end=[1, 1, H, W])
        vy = mb.slice_by_index(x=vel, begin=[0, 1, 0, 0], end=[1, 2, H, W])
        bx = mb.sub(x=px[None, None], y=mb.mul(x=vx, y=DT))      # texel coords of the source
        by = mb.sub(x=py[None, None], y=mb.mul(x=vy, y=DT))
        # resample wants normalized coords in [-1, 1] with align_corners=False semantics:
        # texel center c -> (2c/W - 1)
        nx = mb.sub(x=mb.mul(x=bx, y=2.0 / W), y=1.0); ny = mb.sub(x=mb.mul(x=by, y=2.0 / H), y=1.0)
        grid = mb.concat(values=[nx, ny], axis=1)                 # (1,2,H,W)
        grid = mb.transpose(x=grid, perm=[0, 2, 3, 1])            # (1,H,W,2)
        adv_v = mb.resample(x=vel, coordinates=grid, sampling_mode="bilinear", padding_mode="border",
                            padding_value=0.0, coordinates_mode="normalized_minus_one_to_one", align_corners=False)
        adv_d = mb.resample(x=dye, coordinates=grid, sampling_mode="bilinear", padding_mode="border",
                            padding_value=0.0, coordinates_mode="normalized_minus_one_to_one", align_corners=False)
        # damping: with DECAY_K > 1 a separate "decay" model applies the K-th power
        # every K steps, so the step program itself does not damp.
        if DECAY_K == 1:
            nv = mb.add(x=mb.mul(x=adv_v, y=float(np.exp(-DT * VDAMP))), y=velAdd)
            nd = mb.add(x=mb.mul(x=adv_d, y=float(np.exp(-DT * DDAMP))), y=dyeAdd)
        else:
            nv = mb.add(x=adv_v, y=velAdd)
            nd = mb.add(x=adv_d, y=dyeAdd)
        nv = mb.mul(x=nv, y=wall)
        nd = mb.minimum(x=nd, y=1.6, name="dye_out")
        # --- vorticity confinement ---
        curl = conv3(nv, kcurl, "curl")                            # (1,1,H,W)
        g = conv3(mb.abs(x=curl), kgrad, "gradabs")                # (1,2,H,W)
        gx = mb.slice_by_index(x=g, begin=[0, 0, 0, 0], end=[1, 1, H, W])
        gy = mb.slice_by_index(x=g, begin=[0, 1, 0, 0], end=[1, 2, H, W])
        ln = mb.sqrt(x=mb.add(x=mb.mul(x=gx, y=gx), y=mb.mul(x=gy, y=gy)))
        inv = mb.real_div(x=1.0, y=mb.maximum(x=ln, y=1e-12))
        gate = mb.cast(x=mb.greater(x=ln, y=1e-5), dtype="fp32")
        f = mb.mul(x=mb.mul(x=curl, y=mb.mul(x=inv, y=gate)), y=VORT * DT)
        force = mb.concat(values=[mb.mul(x=gy, y=f), mb.mul(x=mb.mul(x=gx, y=f), y=-1.0)], axis=1)
        nv = mb.add(x=nv, y=force)
        # --- divergence, Jacobi x ITERS (warm start), projection ---
        div = conv3(nv, kdiv, "div")
        p = pr
        for i in range(ITERS):
            p = mb.sub(x=conv3(p, kjac, f"jac{i}"), y=mb.mul(x=div, y=0.25))
        p = mb.identity(x=p, name="pr_out")
        gp = conv3(p, kproj, "gradp")
        nv = mb.mul(x=mb.sub(x=nv, y=gp), y=wall, name="vel_out")
        return nv, nd, p
    return prog

def splat_fields(ref, splats):
    nv = np.zeros((H, W, 2), np.float32); nd = np.zeros((H, W, 3), np.float32)
    for s in splats:
        pv = np.array(s["posVel"], np.float32); col = np.array(s["color"], np.float32); pa = np.array(s["params"], np.float32)
        d = ref.p - pv[:2]; r = max(pa[0], 0.5)
        g = np.exp(-(d * d).sum(-1) / (r * r)).astype(np.float32)
        nv += pv[2:4] * g[..., None]
        dl = np.sqrt((d * d).sum(-1))
        nv += np.where((dl > 0.001)[..., None], d / np.maximum(dl, 1e-9)[..., None] * (pa[2] * g)[..., None], 0)
        nd += col * (pa[1] * g)[..., None]
    return nv, nd

if __name__ == "__main__":
    prog = build()
    mlmodel = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
                         compute_precision=ct.precision.FLOAT16 if PREC == "fp16" else ct.precision.FLOAT32)
    path = f"ink_step_{W}x{H}_j{ITERS}_{PREC}" + (f"_k{DECAY_K}" if DECAY_K > 1 else "") + ".mlpackage"
    mlmodel.save(path)
    print("saved", path)
    if DECAY_K > 1:
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, H, W)), mb.TensorSpec(shape=(1, 3, H, W))], opset_version=ct.target.iOS17)
        def decay(vel, dye):
            v = mb.mul(x=vel, y=float(np.exp(-DT * VDAMP * DECAY_K)), name="vel_out")
            d = mb.mul(x=dye, y=float(np.exp(-DT * DDAMP * DECAY_K)), name="dye_out")
            return v, d
        dm = ct.convert(decay, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
                        compute_precision=ct.precision.FLOAT16 if PREC == "fp16" else ct.precision.FLOAT32)
        dm.save(path.replace(".mlpackage", "_decay.mlpackage")); print("saved decay model")
        sys.exit(0)

    # fidelity: one step from a GPU-dumped state (trace_1step, step 20)
    m = json.load(open("trace_1step/manifest.json"))
    n = 20
    ref = InkRef(m["W"], m["H"], m["dt"], m["velDamp"], m["dyeDamp"], m["vorticity"], m["jacobiIters"])
    same_grid = (m["W"] == W and m["H"] == H and m["jacobiIters"] == ITERS)
    if same_grid:
        ref.vel = np.fromfile(f"trace_1step/vel_{n:04d}.f32", np.float32).reshape(H, W, 2)
        ref.dye = np.fromfile(f"trace_1step/dye_{n:04d}.f32", np.float32).reshape(H, W, 3)
        ref.pr = np.fromfile(f"trace_1step/pr_{n:04d}.f32", np.float32).reshape(H, W)
        splats = m["script"][n]["splats"]
    else:
        rng = np.random.default_rng(1)
        ref.vel = rng.normal(0, 5, (H, W, 2)).astype(np.float32); ref.dye = rng.uniform(0, 1, (H, W, 3)).astype(np.float32)
        ref.pr = rng.normal(0, 1, (H, W)).astype(np.float32); splats = []
    va, da = splat_fields(ref, splats)
    inp = {"vel": ref.vel.transpose(2, 0, 1)[None], "dye": ref.dye.transpose(2, 0, 1)[None], "pr": ref.pr[None, None],
           "velAdd": va.transpose(2, 0, 1)[None], "dyeAdd": da.transpose(2, 0, 1)[None]}
    ref.step(splats)
    results = {}
    for label, cu in [("CPU", ct.ComputeUnit.CPU_ONLY), ("CPU+GPU", ct.ComputeUnit.CPU_AND_GPU), ("CPU+NE", ct.ComputeUnit.CPU_AND_NE), ("ALL", ct.ComputeUnit.ALL)]:
        mdl = ct.models.MLModel(path, compute_units=cu)
        out = mdl.predict(inp)
        ev = np.abs(out["vel_out"][0].transpose(1, 2, 0) - ref.vel).max()
        ed = np.abs(out["dye_out"][0].transpose(1, 2, 0) - ref.dye).max()
        ep = np.abs(out["pr_out"][0, 0] - ref.pr).max()
        for _ in range(10): mdl.predict(inp)
        t0 = time.perf_counter(); N = 100
        for _ in range(N): mdl.predict(inp)
        ms = (time.perf_counter() - t0) / N * 1000
        results[label] = ms
        print(f"{label:8s} {ms:7.3f} ms/step   one-step max|err| vs NumPy: vel {ev:.2e} dye {ed:.2e} pr {ep:.2e}")
    # op placement on the NE configuration
    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(path=mdl.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
        prog = plan.model_structure.program
        from collections import Counter
        placement = Counter()
        for fn in prog.functions.values():
            for op in fn.block.operations:
                info = plan.get_compute_device_usage_for_mlprogram_operation(op)
                dev = type(info.preferred_compute_device).__name__ if info else "?"
                placement[(op.operator_name, dev)] += 1
        print("op placement (CPU+NE):")
        for (name, dev), c in sorted(placement.items()): print(f"  {name:14s} -> {dev:24s} x{c}")
    except Exception as e:
        print("compute plan unavailable:", e)
    json.dump({"W": W, "H": H, "iters": ITERS, "ms": results}, open(f"coreml_{W}x{H}_j{ITERS}_{PREC}.json", "w"))

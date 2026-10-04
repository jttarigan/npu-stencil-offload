"""K-step unrolled Core ML graph of the ink step (submission-cost experiment).

Motivation: on the neural engine the pressure
solve is nearly free (26 Jacobi sweeps cost 0.04 ms of a 0.37 ms step) and a
step paced at 30 Hz costs 1.15 ms of busy time against 0.38 ms back to back.
The dominant cost is therefore per submission, not per arithmetic operation.
This builds a program that runs K simulation steps in ONE submission, so the
fixed cost is amortised over K steps. The price is that splats are batched:
the host must hand over K frames of splat fields at once, so ink responds to
the player K frames late.

Inputs: the three state fields plus stacked per-sub-step splat fields
(velAdd (K,2,H,W), dyeAdd (K,3,H,W)); outputs the state after K steps.
Pressure warm-starts from the previous sub-step, exactly as in the game.

  .venv/bin/python unroll_graph.py [W H] [iters] [prec] [K]
     -> ink_unroll_{W}x{H}_j{iters}_{prec}_u{K}.mlpackage, plus a K-step
        fidelity check against the NumPy reference and op placement.

Timing is deliberately NOT done here (it is load sensitive); use netime on
the compiled model and divide by K, pacing at 30/K Hz.
"""
import sys, json, numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from reference import InkRef

W = int(sys.argv[1]) if len(sys.argv) > 2 else 128
H = int(sys.argv[2]) if len(sys.argv) > 2 else 256
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 26
PREC = sys.argv[4] if len(sys.argv) > 4 else "fp16"
K = int(sys.argv[5]) if len(sys.argv) > 5 else 4
EMIT_ALL = "--emit-all" in sys.argv   # also output every intermediate dye field
# no in-graph damping: the host applies it in float32 every 8 steps, as the game's
# neural-engine backend does (the fp16 per-step decay rounds badly)
NO_DECAY = "--no-decay" in sys.argv
DT, VDAMP, DDAMP, VORT = 1 / 30, 0.3, 0.03, 2.5

def conv3(x, k, name):
    """3x3 conv with replicate padding on a (1,C,H,W) tensor; k: (Cout,Cin,3,3)."""
    xp = mb.pad(x=x, pad=[0, 0, 0, 0, 1, 1, 1, 1], mode="replicate")
    return mb.conv(x=xp, weight=k.astype(np.float32), pad_type="valid", name=name)

def build(K):
    ys, xs = np.mgrid[0:H, 0:W]
    px = (xs + 0.5).astype(np.float32); py = (ys + 0.5).astype(np.float32)
    wall = np.ones((1, 1, H, W), np.float32)
    wall[:, :, 0, :] = 0; wall[:, :, -1, :] = 0; wall[:, :, :, 0] = 0; wall[:, :, :, -1] = 0
    kdx = np.zeros((1, 1, 3, 3), np.float32); kdx[0, 0, 1, 2] = 0.5; kdx[0, 0, 1, 0] = -0.5
    kdy = np.zeros((1, 1, 3, 3), np.float32); kdy[0, 0, 2, 1] = 0.5; kdy[0, 0, 0, 1] = -0.5
    kjac = np.zeros((1, 1, 3, 3), np.float32)
    kjac[0, 0, 1, 0] = kjac[0, 0, 1, 2] = kjac[0, 0, 0, 1] = kjac[0, 0, 2, 1] = 0.25
    kdiv = np.zeros((1, 2, 3, 3), np.float32); kdiv[0, 0] = kdx[0, 0]; kdiv[0, 1] = kdy[0, 0]
    kcurl = np.zeros((1, 2, 3, 3), np.float32); kcurl[0, 1] = kdx[0, 0]; kcurl[0, 0] = -kdy[0, 0]
    kgrad = np.concatenate([kdx, kdy], 0)
    kproj = np.concatenate([kdx, kdy], 0)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, H, W)), mb.TensorSpec(shape=(1, 3, H, W)),
                             mb.TensorSpec(shape=(1, 1, H, W)), mb.TensorSpec(shape=(K, 2, H, W)),
                             mb.TensorSpec(shape=(K, 3, H, W))], opset_version=ct.target.iOS17)
    def prog(vel, dye, pr, velAdd, dyeAdd):
        v, d, p = vel, dye, pr
        seq = []
        for s in range(K):
            last = (s == K - 1)
            t = lambda n: f"{n}_{s}"                    # unique op names per sub-step
            va = mb.slice_by_index(x=velAdd, begin=[s, 0, 0, 0], end=[s + 1, 2, H, W], name=t("va"))
            da = mb.slice_by_index(x=dyeAdd, begin=[s, 0, 0, 0], end=[s + 1, 3, H, W], name=t("da"))
            # --- advection: back-trace, bilinear resample with clamp-to-edge ---
            vx = mb.slice_by_index(x=v, begin=[0, 0, 0, 0], end=[1, 1, H, W], name=t("vx"))
            vy = mb.slice_by_index(x=v, begin=[0, 1, 0, 0], end=[1, 2, H, W], name=t("vy"))
            bx = mb.sub(x=px[None, None], y=mb.mul(x=vx, y=DT))
            by = mb.sub(x=py[None, None], y=mb.mul(x=vy, y=DT))
            nx = mb.sub(x=mb.mul(x=bx, y=2.0 / W), y=1.0)
            ny = mb.sub(x=mb.mul(x=by, y=2.0 / H), y=1.0)
            grid = mb.concat(values=[nx, ny], axis=1)
            grid = mb.transpose(x=grid, perm=[0, 2, 3, 1], name=t("grid"))
            adv_v = mb.resample(x=v, coordinates=grid, sampling_mode="bilinear", padding_mode="border",
                                padding_value=0.0, coordinates_mode="normalized_minus_one_to_one",
                                align_corners=False, name=t("advv"))
            adv_d = mb.resample(x=d, coordinates=grid, sampling_mode="bilinear", padding_mode="border",
                                padding_value=0.0, coordinates_mode="normalized_minus_one_to_one",
                                align_corners=False, name=t("advd"))
            fv = 1.0 if NO_DECAY else float(np.exp(-DT * VDAMP))
            fd = 1.0 if NO_DECAY else float(np.exp(-DT * DDAMP))
            nv = mb.add(x=mb.mul(x=adv_v, y=fv), y=va)
            nd = mb.add(x=mb.mul(x=adv_d, y=fd), y=da)
            nv = mb.mul(x=nv, y=wall, name=t("wall1"))
            nd = mb.minimum(x=nd, y=1.6, name=("dye_out" if last and not EMIT_ALL else t("dyec")))
            seq.append(nd)
            # --- vorticity confinement ---
            curl = conv3(nv, kcurl, t("curl"))
            g = conv3(mb.abs(x=curl), kgrad, t("gradabs"))
            gx = mb.slice_by_index(x=g, begin=[0, 0, 0, 0], end=[1, 1, H, W], name=t("gx"))
            gy = mb.slice_by_index(x=g, begin=[0, 1, 0, 0], end=[1, 2, H, W], name=t("gy"))
            ln = mb.sqrt(x=mb.add(x=mb.mul(x=gx, y=gx), y=mb.mul(x=gy, y=gy)), name=t("ln"))
            inv = mb.real_div(x=1.0, y=mb.maximum(x=ln, y=1e-12))
            gate = mb.cast(x=mb.greater(x=ln, y=1e-5), dtype="fp32", name=t("gate"))
            f = mb.mul(x=mb.mul(x=curl, y=mb.mul(x=inv, y=gate)), y=VORT * DT, name=t("f"))
            force = mb.concat(values=[mb.mul(x=gy, y=f), mb.mul(x=mb.mul(x=gx, y=f), y=-1.0)],
                              axis=1, name=t("force"))
            nv = mb.add(x=nv, y=force, name=t("vort"))
            # --- divergence, Jacobi x ITERS (warm start from the previous sub-step) ---
            div = conv3(nv, kdiv, t("div"))
            q = p
            for i in range(ITERS):
                q = mb.sub(x=conv3(q, kjac, t(f"jac{i}")), y=mb.mul(x=div, y=0.25))
            p = mb.identity(x=q, name="pr_out" if last else t("pr"))
            gp = conv3(p, kproj, t("gradp"))
            v = mb.mul(x=mb.sub(x=nv, y=gp), y=wall, name="vel_out" if last else t("wall2"))
            d = nd
        if EMIT_ALL:
            # every intermediate dye field, so the host can show one per frame
            # (the DELAY presentation; the HOLD presentation stutters, see lag_check.py)
            d = mb.concat(values=seq, axis=0, name="dye_seq")
        return v, d, p
    return prog

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

if __name__ == "__main__":
    prog = build(K)
    mlmodel = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
                         compute_precision=ct.precision.FLOAT16 if PREC == "fp16" else ct.precision.FLOAT32)
    path = (f"ink_unroll_{W}x{H}_j{ITERS}_{PREC}_u{K}" + ("_seq" if EMIT_ALL else "")
            + ("_nodecay" if NO_DECAY else "") + ".mlpackage")
    mlmodel.save(path)
    print("saved", path)

    # --- fidelity: K steps from a GPU-dumped state against the NumPy reference ---
    m = json.load(open("trace_1step/manifest.json"))
    n = 20
    ref = InkRef(m["W"], m["H"], m["dt"], m["velDamp"], m["dyeDamp"], m["vorticity"], m["jacobiIters"])
    if m["W"] == W and m["H"] == H and m["jacobiIters"] == ITERS:
        ref.vel = np.fromfile(f"trace_1step/vel_{n:04d}.f32", np.float32).reshape(H, W, 2)
        ref.dye = np.fromfile(f"trace_1step/dye_{n:04d}.f32", np.float32).reshape(H, W, 3)
        ref.pr = np.fromfile(f"trace_1step/pr_{n:04d}.f32", np.float32).reshape(H, W)
        scripts = [m["script"][n + s]["splats"] for s in range(K)]
    else:
        rng = np.random.default_rng(1)
        ref.vel = rng.normal(0, 5, (H, W, 2)).astype(np.float32)
        ref.dye = rng.uniform(0, 1, (H, W, 3)).astype(np.float32)
        ref.pr = rng.normal(0, 1, (H, W)).astype(np.float32)
        scripts = [[] for _ in range(K)]
    va = np.zeros((K, 2, H, W), np.float32); da = np.zeros((K, 3, H, W), np.float32)
    # splat fields depend only on the script, not on the evolving state
    for s in range(K):
        a, b = splat_fields(ref, scripts[s])
        va[s] = a.transpose(2, 0, 1); da[s] = b.transpose(2, 0, 1)
    inp = {"vel": ref.vel.transpose(2, 0, 1)[None], "dye": ref.dye.transpose(2, 0, 1)[None],
           "pr": ref.pr[None, None], "velAdd": va, "dyeAdd": da}
    if NO_DECAY:   # the reference must skip damping too
        ref.velDamp = ref.dyeDamp = 0.0
    for s in range(K):
        ref.step(scripts[s])
    for label, cu in [("CPU", ct.ComputeUnit.CPU_ONLY), ("CPU+NE", ct.ComputeUnit.CPU_AND_NE)]:
        mdl = ct.models.MLModel(path, compute_units=cu)
        out = mdl.predict(inp)
        ev = np.abs(out["vel_out"][0].transpose(1, 2, 0) - ref.vel).max()
        dye_final = out["dye_seq"][-1] if EMIT_ALL else out["dye_out"][0]
        ed = np.abs(dye_final.transpose(1, 2, 0) - ref.dye).max()
        ep = np.abs(out["pr_out"][0, 0] - ref.pr).max()
        mass = dye_final.sum() / max(ref.dye.sum(), 1e-9)
        print(f"{label:8s} after {K} steps  max|err| vel {ev:.3e} dye {ed:.3e} pr {ep:.3e}  dye mass ratio {mass:.4f}")
    try:
        from coremltools.models.compute_plan import MLComputePlan
        from collections import Counter
        plan = MLComputePlan.load_from_path(path=mdl.get_compiled_model_path(),
                                            compute_units=ct.ComputeUnit.CPU_AND_NE)
        placement = Counter()
        for fn in plan.model_structure.program.functions.values():
            for op in fn.block.operations:
                info = plan.get_compute_device_usage_for_mlprogram_operation(op)
                placement[type(info.preferred_compute_device).__name__ if info else "?"] += 1
        print("op placement (CPU+NE):", dict(placement))
    except Exception as e:
        print("compute plan unavailable:", e)

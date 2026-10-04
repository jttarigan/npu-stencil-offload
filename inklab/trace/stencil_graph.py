"""K-step unrolled Core ML graphs of two further stencil workloads (generality check).

The ink result (unroll_graph.py) says the accelerator's cost is per submission,
not per operation. If that is a property of the accelerator rather than of the
fluid, it must hold for other stencil PDEs too. Two workloads bracket the ink
step in arithmetic per step:

  heat       explicit diffusion u += a*lap(u) + src, S sweeps per step
             (pure 3x3 conv + add; the lightest possible stencil step)
  grayscott  Gray-Scott reaction-diffusion on two fields (u, v), S sweeps
             per step (conv + a nonlinear u*v*v term; medium)

Both use the same conventions as the ink graph: (1,C,H,W) tensors, replicate
padding (zero-flux boundary), a per-sub-step source input stacked (K,C,H,W)
so the host batches events exactly as it batches splats, fp16 compute.

  .venv/bin/python stencil_graph.py <heat|grayscott> [W H] [S] [K]
     -> {name}_{W}x{H}_s{S}_fp16_u{K}.mlpackage, a K-step fidelity check
        against the NumPy reference below, and op placement.

Timing is not done here; use netime on the compiled model (as for the ink).
"""
import sys, numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

NAME = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 3 else 128
H = int(sys.argv[3]) if len(sys.argv) > 3 else 256
S = int(sys.argv[4]) if len(sys.argv) > 4 else 8
K = int(sys.argv[5]) if len(sys.argv) > 5 else 4

ALPHA = 0.2                                  # heat: explicit, stable below 0.25
DU, DV, F, KILL = 0.16, 0.08, 0.035, 0.065   # Gray-Scott, dt = 1, 5-point Laplacian
C = {"heat": 1, "grayscott": 2}[NAME]

LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)

def lap_conv(x, ch, name):
    """Per-channel 5-point Laplacian with replicate padding on a (1,ch,H,W) tensor."""
    xp = mb.pad(x=x, pad=[0, 0, 0, 0, 1, 1, 1, 1], mode="replicate")
    k = np.repeat(LAP[None, None], ch, 0)          # (ch,1,3,3), grouped
    return mb.conv(x=xp, weight=k, pad_type="valid", groups=ch, name=name)

def build():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, C, H, W)), mb.TensorSpec(shape=(K, C, H, W))],
                opset_version=ct.target.iOS17)
    def prog(field, src):
        x = field
        for s in range(K):
            t = lambda n: f"{n}_{s}"
            add = mb.slice_by_index(x=src, begin=[s, 0, 0, 0], end=[s + 1, C, H, W], name=t("src"))
            x = mb.add(x=x, y=add, name=t("inject"))
            for i in range(S):
                l = lap_conv(x, C, t(f"lap{i}"))
                if NAME == "heat":
                    x = mb.add(x=x, y=mb.mul(x=l, y=ALPHA), name=t(f"h{i}"))
                else:
                    u = mb.slice_by_index(x=x, begin=[0, 0, 0, 0], end=[1, 1, H, W], name=t(f"u{i}"))
                    v = mb.slice_by_index(x=x, begin=[0, 1, 0, 0], end=[1, 2, H, W], name=t(f"v{i}"))
                    lu = mb.slice_by_index(x=l, begin=[0, 0, 0, 0], end=[1, 1, H, W], name=t(f"lu{i}"))
                    lv = mb.slice_by_index(x=l, begin=[0, 1, 0, 0], end=[1, 2, H, W], name=t(f"lv{i}"))
                    uvv = mb.mul(x=u, y=mb.mul(x=v, y=v), name=t(f"uvv{i}"))
                    # u += Du*lap u - u v^2 + F(1-u);  v += Dv*lap v + u v^2 - (F+k) v
                    nu = mb.add(x=mb.sub(x=mb.add(x=u, y=mb.mul(x=lu, y=DU)), y=uvv),
                                y=mb.mul(x=mb.sub(x=1.0, y=u), y=F), name=t(f"nu{i}"))
                    nv = mb.sub(x=mb.add(x=mb.add(x=v, y=mb.mul(x=lv, y=DV)), y=uvv),
                                y=mb.mul(x=v, y=F + KILL), name=t(f"nv{i}"))
                    x = mb.concat(values=[nu, nv], axis=1, name=t(f"x{i}"))
        return mb.identity(x=x, name="state_out")
    return prog

def ref_step(x, add):
    """NumPy float32 reference for one step; x: (C,H,W)."""
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
        return rng.uniform(0, 1, (C, H, W)).astype(np.float32)
    x = np.stack([np.ones((H, W)), np.zeros((H, W))]).astype(np.float32)
    for _ in range(12):                              # seeded squares of v, as in the classic demos
        cy, cx = rng.integers(8, H - 8), rng.integers(8, W - 8)
        x[0, cy - 4:cy + 4, cx - 4:cx + 4] = 0.5; x[1, cy - 4:cy + 4, cx - 4:cx + 4] = 0.25
    return x

if __name__ == "__main__":
    mlmodel = ct.convert(build(), convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
                         compute_precision=ct.precision.FLOAT16)
    path = f"{NAME}_{W}x{H}_s{S}_fp16_u{K}.mlpackage"
    mlmodel.save(path)
    print("saved", path)

    rng = np.random.default_rng(1)
    x0 = initial(rng)
    src = np.zeros((K, C, H, W), np.float32)
    for s in range(K):                               # one small event per sub-step, like a splat
        cy, cx = rng.integers(4, H - 4), rng.integers(4, W - 4)
        src[s, -1, cy - 2:cy + 2, cx - 2:cx + 2] = 0.2
    ref = x0.copy()
    for s in range(K):
        ref = ref_step(ref, src[s])
    scale = max(np.abs(ref).max(), 1e-6)
    for label, cu in [("CPU", ct.ComputeUnit.CPU_ONLY), ("CPU+NE", ct.ComputeUnit.CPU_AND_NE)]:
        mdl = ct.models.MLModel(path, compute_units=cu)
        out = mdl.predict({"field": x0[None], "src": src})["state_out"][0]
        err = np.abs(out - ref).max()
        print(f"{label:8s} after {K} steps x {S} sweeps  max|err| {err:.3e} ({100 * err / scale:.2f}% of range)")
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

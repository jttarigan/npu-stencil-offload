"""Float16 Neural Engine fidelity over the full 300-step script: runs the
Core ML step from zero with the trace's splat script and compares each
dumped state with the GPU trace (dye PSNR, dye mass, velocity RMS error)."""
import json, sys, numpy as np, coremltools as ct
from reference import InkRef
_argv = sys.argv; sys.argv = [sys.argv[0]]
from coreml_graph import splat_fields
sys.argv = _argv
d = sys.argv[1] if len(sys.argv) > 1 else "trace_ref"
path = sys.argv[2] if len(sys.argv) > 2 else "ink_step_128x256_j26_fp16.mlpackage"
units = {"ne": ct.ComputeUnit.CPU_AND_NE, "cpu": ct.ComputeUnit.CPU_ONLY}[sys.argv[3] if len(sys.argv) > 3 else "ne"]
m = json.load(open(f"{d}/manifest.json")); W, H = m["W"], m["H"]
ref = InkRef(W, H, m["dt"], m["velDamp"], m["dyeDamp"], m["vorticity"], m["jacobiIters"])  # only for .p and splat fields
mdl = ct.models.MLModel(path, compute_units=units)
import re, os
_k = re.search(r"_k(\d+)", path); DECAY_K = int(_k.group(1)) if _k else 1
decay = ct.models.MLModel(path.replace(".mlpackage", "_decay.mlpackage"), compute_units=units) if DECAY_K > 1 else None
vel = np.zeros((1, 2, H, W), np.float32); dye = np.zeros((1, 3, H, W), np.float32); pr = np.zeros((1, 1, H, W), np.float32)
print("step  dyePSNR(dB)  dyeMass ne/gpu   velRMSerr/velRMS")
for e in m["script"]:
    n = e["step"]
    va, da = splat_fields(ref, e["splats"])
    out = mdl.predict({"vel": vel, "dye": dye, "pr": pr, "velAdd": va.transpose(2, 0, 1)[None], "dyeAdd": da.transpose(2, 0, 1)[None]})
    vel, dye, pr = out["vel_out"].astype(np.float32), out["dye_out"].astype(np.float32), out["pr_out"].astype(np.float32)
    if decay is not None and (n + 1) % DECAY_K == 0:
        o = decay.predict({"vel": vel, "dye": dye}); vel, dye = o["vel_out"].astype(np.float32), o["dye_out"].astype(np.float32)
    if (n + 1) % m["dumpEvery"] == 0:
        t = f"{n+1:04d}"
        gv = np.fromfile(f"{d}/vel_{t}.f32", np.float32).reshape(H, W, 2).transpose(2, 0, 1)
        gd = np.fromfile(f"{d}/dye_{t}.f32", np.float32).reshape(H, W, 3).transpose(2, 0, 1)
        mse = ((dye[0] - gd) ** 2).mean(); psnr = 10 * np.log10(1.6 ** 2 / max(mse, 1e-12))
        vr = np.sqrt(((vel[0] - gv) ** 2).mean()) / max(np.sqrt((gv ** 2).mean()), 1e-9)
        print(f"{t}   {psnr:6.1f}      {dye.sum()/gd.sum():.4f}          {vr:.3f}")

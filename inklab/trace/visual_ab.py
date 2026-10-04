"""Side-by-side dye renders: GPU trace vs fp16 Neural Engine run at chosen
steps of the recorded game. Writes visual_ab.png (columns = steps; rows =
GPU, NE fp16, |difference| x4)."""
import json, sys, numpy as np, coremltools as ct
_argv = sys.argv; sys.argv = [sys.argv[0]]
from reference import InkRef; from coreml_graph import splat_fields
sys.argv = _argv
import zlib, struct
d = sys.argv[1] if len(sys.argv) > 1 else "trace_game"
path = sys.argv[2] if len(sys.argv) > 2 else "ink_step_128x256_j26_fp16_k8.mlpackage"
steps = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "400,1000,1600,1900").split(",")]
m = json.load(open(f"{d}/manifest.json")); W, H = m["W"], m["H"]
ref = InkRef(W, H, m["dt"], m["velDamp"], m["dyeDamp"], m["vorticity"], m["jacobiIters"])
mdl = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
import re; k = re.search(r"_k(\d+)", path); K = int(k.group(1)) if k else 1
decay = ct.models.MLModel(path.replace(".mlpackage", "_decay.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE) if K > 1 else None
vel = np.zeros((1, 2, H, W), np.float32); dye = np.zeros((1, 3, H, W), np.float32); pr = np.zeros((1, 1, H, W), np.float32)
cols = []
for e in m["script"]:
    n = e["step"]; va, da = splat_fields(ref, e["splats"])
    out = mdl.predict({"vel": vel, "dye": dye, "pr": pr, "velAdd": va.transpose(2, 0, 1)[None], "dyeAdd": da.transpose(2, 0, 1)[None]})
    vel, dye, pr = out["vel_out"].astype(np.float32), out["dye_out"].astype(np.float32), out["pr_out"].astype(np.float32)
    if decay is not None and (n + 1) % K == 0:
        o = decay.predict({"vel": vel, "dye": dye}); vel, dye = o["vel_out"].astype(np.float32), o["dye_out"].astype(np.float32)
    if (n + 1) in steps:
        gd = np.fromfile(f"{d}/dye_{n+1:04d}.f32", np.float32).reshape(H, W, 3)
        ne = dye[0].transpose(1, 2, 0)
        diff = np.abs(ne - gd) * 4
        cols.append([gd, ne, diff])
GAP = 3   # thin white line between panels, px (about 2 pt in the paper)
img = np.full((3 * (H + GAP) - GAP, len(cols) * (W + GAP) - GAP, 3), 255, np.uint8)
for j, c in enumerate(cols):
    for i, p in enumerate(c):   # 0.85 = the game's ink opacity, on black
        img[i * (H + GAP):i * (H + GAP) + H, j * (W + GAP):j * (W + GAP) + W] = (np.clip(p * 0.85, 0, 1) * 255).astype(np.uint8)
h, w, _ = img.shape
raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
def chunk(t, b): return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
open("visual_ab.png", "wb").write(png); print("wrote visual_ab.png", w, "x", h, "steps", steps)

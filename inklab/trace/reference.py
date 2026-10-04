"""NumPy reference of the game's ink step, kernel by kernel, in float32.

Validates the reading of the Metal kernels: replays the splat script in
<trace_dir>/manifest.json and compares each dumped field against the
Metal trace. Also the ground truth that the Core ML graph is built from.

  .venv/bin/python reference.py trace_ref
"""
import json, sys, numpy as np

def load_trace(d):
    m = json.load(open(f"{d}/manifest.json"))
    return m

def bilinear(field, back):
    """Sample field (H,W,C) at continuous texel coords back (H,W,2) with
    clamp-to-edge, pixel centers at +0.5 (Metal normalized-coord semantics)."""
    H, W, _ = field.shape
    x = back[..., 0] - 0.5; y = back[..., 1] - 0.5
    x0 = np.floor(x); y0 = np.floor(y)
    fx = (x - x0).astype(np.float32); fy = (y - y0).astype(np.float32)
    x0 = x0.astype(np.int64); y0 = y0.astype(np.int64)
    def at(xi, yi):
        return field[np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)]
    a = at(x0, y0); b = at(x0 + 1, y0); c = at(x0, y0 + 1); d = at(x0 + 1, y0 + 1)
    fx = fx[..., None]; fy = fy[..., None]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

def neighbours(f):
    """Clamp-to-edge L, R, U, D of a (H,W) field."""
    L = np.concatenate([f[:, :1], f[:, :-1]], 1); R = np.concatenate([f[:, 1:], f[:, -1:]], 1)
    U = np.concatenate([f[:1, :], f[:-1, :]], 0); D = np.concatenate([f[1:, :], f[-1:, :]], 0)
    return L, R, U, D

class InkRef:
    def __init__(self, W, H, dt, velDamp, dyeDamp, vort, iters):
        self.W, self.H, self.dt = W, H, np.float32(dt)
        self.velDamp, self.dyeDamp, self.vort, self.iters = np.float32(velDamp), np.float32(dyeDamp), np.float32(vort), iters
        self.vel = np.zeros((H, W, 2), np.float32); self.dye = np.zeros((H, W, 3), np.float32)
        self.pr = np.zeros((H, W), np.float32)
        ys, xs = np.mgrid[0:H, 0:W]
        self.p = np.stack([xs + 0.5, ys + 0.5], -1).astype(np.float32)

    def step(self, splats):
        W, H, dt = self.W, self.H, self.dt
        # ink_step: semi-Lagrangian advection + damping + splats
        v = self.vel
        back = self.p - v * dt
        nv = bilinear(self.vel, back) * np.exp(-dt * self.velDamp)
        nd = bilinear(self.dye, back) * np.exp(-dt * self.dyeDamp)
        for s in splats:
            pv = np.array(s["posVel"], np.float32); col = np.array(s["color"], np.float32); pa = np.array(s["params"], np.float32)
            d = self.p - pv[:2]; r = max(pa[0], 0.5)
            g = np.exp(-(d * d).sum(-1) / (r * r)).astype(np.float32)
            nv = nv + pv[2:4] * g[..., None]
            dl = np.sqrt((d * d).sum(-1))
            mask = dl > 0.001
            radial = np.where(mask[..., None], d / np.maximum(dl, 1e-9)[..., None] * (pa[2] * g)[..., None], 0)
            nv = nv + radial
            nd = nd + col * (pa[1] * g)[..., None]
        nv[0, :] = 0; nv[-1, :] = 0; nv[:, 0] = 0; nv[:, -1] = 0
        self.vel = nv.astype(np.float32); self.dye = np.minimum(nd, 1.6).astype(np.float32)
        # ink_vorticity
        vx, vy = self.vel[..., 0], self.vel[..., 1]
        L, R, U, D = neighbours(vy); _, _, Uu, Dd = neighbours(vx)
        curl = 0.5 * ((R - L) - (Dd - Uu))
        cL, cR, cU, cD = neighbours(curl)
        grad = 0.5 * np.stack([np.abs(cR) - np.abs(cL), np.abs(cD) - np.abs(cU)], -1)
        ln = np.sqrt((grad * grad).sum(-1))
        n = np.where(ln[..., None] > 1e-5, grad / np.maximum(ln, 1e-12)[..., None], 0)
        force = self.vort * curl[..., None] * np.stack([n[..., 1], -n[..., 0]], -1) * dt
        self.vel = (self.vel + np.where(ln[..., None] > 1e-5, force, 0)).astype(np.float32)
        # ink_divergence
        L, R, _, _ = neighbours(self.vel[..., 0]); _, _, U, D = neighbours(self.vel[..., 1])
        div = (0.5 * ((R - L) + (D - U))).astype(np.float32)
        # ink_jacobi x iters, warm-started from previous pressure
        pr = self.pr
        for _ in range(self.iters):
            L, R, U, D = neighbours(pr)
            pr = ((L + R + U + D - div) * 0.25).astype(np.float32)
        self.pr = pr
        # ink_project
        L, R, U, D = neighbours(pr)
        nv = self.vel - 0.5 * np.stack([R - L, D - U], -1)
        nv[0, :] = 0; nv[-1, :] = 0; nv[:, 0] = 0; nv[:, -1] = 0
        self.vel = nv.astype(np.float32)

def compare(d):
    m = load_trace(d)
    ref = InkRef(m["W"], m["H"], m["dt"], m["velDamp"], m["dyeDamp"], m["vorticity"], m["jacobiIters"])
    W, H = m["W"], m["H"]
    for entry in m["script"]:
        n = entry["step"]
        ref.step(entry["splats"])
        if (n + 1) % m["dumpEvery"] == 0 or n == m["steps"] - 1:
            tag = f"{n + 1:04d}"
            gv = np.fromfile(f"{d}/vel_{tag}.f32", np.float32).reshape(H, W, 2)
            gd = np.fromfile(f"{d}/dye_{tag}.f32", np.float32).reshape(H, W, 3)
            gp = np.fromfile(f"{d}/pr_{tag}.f32", np.float32).reshape(H, W)
            def err(a, b): return float(np.abs(a - b).max()), float(np.abs(b).max())
            ev, sv = err(ref.vel, gv); ed, sd = err(ref.dye, gd); ep, sp = err(ref.pr, gp)
            print(f"step {tag}: vel max|err| {ev:.2e} (scale {sv:.2f})  dye {ed:.2e} (scale {sd:.2f})  pr {ep:.2e} (scale {sp:.2f})"
                  f"  dye mass ref/gpu {ref.dye.sum():.1f}/{gd.sum():.1f}")

if __name__ == "__main__" and not (len(sys.argv) > 2 and sys.argv[2] == "onestep"):
    compare(sys.argv[1] if len(sys.argv) > 1 else "trace_ref")

def onestep(d):
    """From each dumped GPU state, take exactly one reference step and compare
    with the next GPU dump (needs a trace with dumpEvery=1)."""
    m = load_trace(d); W, H = m["W"], m["H"]
    assert m["dumpEvery"] == 1
    ref = InkRef(W, H, m["dt"], m["velDamp"], m["dyeDamp"], m["vorticity"], m["jacobiIters"])
    worst = [0, 0, 0]
    for entry in m["script"][1:]:
        n = entry["step"]; a = f"{n:04d}"; b = f"{n + 1:04d}"
        ref.vel = np.fromfile(f"{d}/vel_{a}.f32", np.float32).reshape(H, W, 2)
        ref.dye = np.fromfile(f"{d}/dye_{a}.f32", np.float32).reshape(H, W, 3)
        ref.pr = np.fromfile(f"{d}/pr_{a}.f32", np.float32).reshape(H, W)
        ref.step(entry["splats"])
        gv = np.fromfile(f"{d}/vel_{b}.f32", np.float32).reshape(H, W, 2)
        gd = np.fromfile(f"{d}/dye_{b}.f32", np.float32).reshape(H, W, 3)
        gp = np.fromfile(f"{d}/pr_{b}.f32", np.float32).reshape(H, W)
        e = [float(np.abs(ref.vel - gv).max()), float(np.abs(ref.dye - gd).max()), float(np.abs(ref.pr - gp).max())]
        worst = [max(w, x) for w, x in zip(worst, e)]
        if n % 10 == 0: print(f"step {n}: one-step max|err| vel {e[0]:.2e} dye {e[1]:.2e} pr {e[2]:.2e}  (vel scale {np.abs(gv).max():.1f})")
    print("worst one-step errors over the run: vel %.2e dye %.2e pr %.2e" % tuple(worst))

if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "onestep":
    onestep(sys.argv[1])

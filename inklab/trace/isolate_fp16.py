"""Which stage of the step carries the finale dye-mass bias under fp16?
From GPU-dumped finale states of the recorded game, run the NumPy step
with ONE stage rounded to fp16 (inputs and outputs of that stage) and the
rest in fp32, over a window of consecutive steps, and report the dye mass
drift and velocity error relative to the all-fp32 run."""
import json, sys, numpy as np
from surrogate import RefWithHook
from reference import bilinear, neighbours
d = "trace_game"; m = json.load(open(f"{d}/manifest.json")); W, H = m["W"], m["H"]
start, span = int(sys.argv[1]) if len(sys.argv) > 1 else 1800, int(sys.argv[2]) if len(sys.argv) > 2 else 100
f16 = lambda a: a.astype(np.float16).astype(np.float32)

class Stagewise(RefWithHook):
    stage = None
    def step(self, splats):
        W, H, dt = self.W, self.H, self.dt
        r = lambda name, a: f16(a) if self.stage == name else a
        v = self.vel; back = r("advect_coords", self.p - v * dt)
        adv_v = bilinear(r("advect_in", self.vel), back); adv_d = bilinear(r("advect_in", self.dye), back)
        adv_v, adv_d = r("advect_out", adv_v), r("advect_out", adv_d)
        nv = r("damp", adv_v * np.exp(-dt * self.velDamp)); nd = r("damp", adv_d * np.exp(-dt * self.dyeDamp))
        for s in splats:
            pv = np.array(s["posVel"], np.float32); col = np.array(s["color"], np.float32); pa = np.array(s["params"], np.float32)
            dd = self.p - pv[:2]; rr = max(pa[0], 0.5); g = np.exp(-(dd * dd).sum(-1) / (rr * rr)).astype(np.float32)
            nv = nv + pv[2:4] * g[..., None]; dl = np.sqrt((dd * dd).sum(-1))
            nv = nv + np.where((dl > 0.001)[..., None], dd / np.maximum(dl, 1e-9)[..., None] * (pa[2] * g)[..., None], 0)
            nd = nd + col * (pa[1] * g)[..., None]
        nv, nd = r("splat", nv), r("splat", nd)
        nv[0, :] = 0; nv[-1, :] = 0; nv[:, 0] = 0; nv[:, -1] = 0
        self.vel = nv.astype(np.float32); self.dye = r("clamp", np.minimum(nd, 1.6).astype(np.float32))
        vx, vy = self.vel[..., 0], self.vel[..., 1]
        L, R, U, D = neighbours(vy); _, _, Uu, Dd = neighbours(vx); curl = r("vort", 0.5 * ((R - L) - (Dd - Uu)))
        cL, cR, cU, cD = neighbours(curl); grad = 0.5 * np.stack([np.abs(cR) - np.abs(cL), np.abs(cD) - np.abs(cU)], -1)
        ln = np.sqrt((grad * grad).sum(-1)); n = np.where(ln[..., None] > 1e-5, grad / np.maximum(ln, 1e-12)[..., None], 0)
        force = self.vort * curl[..., None] * np.stack([n[..., 1], -n[..., 0]], -1) * dt
        self.vel = r("vort", (self.vel + np.where(ln[..., None] > 1e-5, force, 0)).astype(np.float32))
        L, R, _, _ = neighbours(self.vel[..., 0]); _, _, U, D = neighbours(self.vel[..., 1])
        div = r("div", (0.5 * ((R - L) + (D - U))).astype(np.float32))
        pr = r("jacobi", self.pr)
        for _ in range(self.iters):
            L, R, U, D = neighbours(pr); pr = r("jacobi", ((L + R + U + D - div) * 0.25).astype(np.float32))
        self.pr = pr
        L, R, U, D = neighbours(pr); nv = self.vel - 0.5 * np.stack([R - L, D - U], -1)
        nv[0, :] = 0; nv[-1, :] = 0; nv[:, 0] = 0; nv[:, -1] = 0; self.vel = r("project", nv.astype(np.float32))
        if self.stage == "state": self.vel, self.dye, self.pr = f16(self.vel), f16(self.dye), f16(self.pr)

def run(stage):
    ref = Stagewise(W, H, m["dt"], m["velDamp"], m["dyeDamp"], m["vorticity"], m["jacobiIters"]); ref.stage = stage
    a = f"{start:04d}"
    ref.vel = np.fromfile(f"{d}/vel_{a}.f32", np.float32).reshape(H, W, 2); ref.dye = np.fromfile(f"{d}/dye_{a}.f32", np.float32).reshape(H, W, 3)
    ref.pr = np.fromfile(f"{d}/pr_{a}.f32", np.float32).reshape(H, W)
    for e in m["script"][start:start + span]: ref.step(e["splats"])
    return ref
base = run(None)
print(f"window steps {start}..{start+span} (finale); stage rounded to fp16 -> dye mass ratio vs all-fp32, vel RMS rel err")
for st in ["state", "advect_coords", "advect_in", "advect_out", "damp", "splat", "clamp", "vort", "div", "jacobi", "project"]:
    r = run(st)
    print(f"  {st:14s} mass {r.dye.sum()/base.dye.sum():.4f}   vel err {np.sqrt(((r.vel-base.vel)**2).mean())/np.sqrt((base.vel**2).mean()):.3f}")

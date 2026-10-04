"""Unroll equivalence check: is one K-step submission equal to K single-step submissions? (unroll equivalence)"""
import json, numpy as np, coremltools as ct
from reference import InkRef
import unroll_graph as U
W,H,K = U.W, U.H, 8
m = json.load(open("trace_1step/manifest.json")); n = 20
ref = InkRef(m["W"],m["H"],m["dt"],m["velDamp"],m["dyeDamp"],m["vorticity"],m["jacobiIters"])
ref.vel = np.fromfile(f"trace_1step/vel_{n:04d}.f32",np.float32).reshape(H,W,2)
ref.dye = np.fromfile(f"trace_1step/dye_{n:04d}.f32",np.float32).reshape(H,W,3)
ref.pr  = np.fromfile(f"trace_1step/pr_{n:04d}.f32",np.float32).reshape(H,W)
scripts = [m["script"][n+s]["splats"] for s in range(K)]
va = np.zeros((K,2,H,W),np.float32); da = np.zeros((K,3,H,W),np.float32)
for s in range(K):
    a,b = U.splat_fields(ref, scripts[s]); va[s]=a.transpose(2,0,1); da[s]=b.transpose(2,0,1)
CU = ct.ComputeUnit.CPU_AND_NE
# A: one 8-step submission
m8 = ct.models.MLModel(f"ink_unroll_{W}x{H}_j26_fp16_u8.mlpackage", compute_units=CU)
A = m8.predict({"vel":ref.vel.transpose(2,0,1)[None],"dye":ref.dye.transpose(2,0,1)[None],
                "pr":ref.pr[None,None],"velAdd":va,"dyeAdd":da})
# B: eight 1-step submissions, chained
m1 = ct.models.MLModel(f"ink_unroll_{W}x{H}_j26_fp16_u1.mlpackage", compute_units=CU)
v,d,p = ref.vel.transpose(2,0,1)[None], ref.dye.transpose(2,0,1)[None], ref.pr[None,None]
for s in range(K):
    o = m1.predict({"vel":v,"dye":d,"pr":p,"velAdd":va[s:s+1],"dyeAdd":da[s:s+1]})
    v,d,p = o["vel_out"],o["dye_out"],o["pr_out"]
for k_,(a_,b_) in {"vel":(A["vel_out"],v),"dye":(A["dye_out"],d),"pr":(A["pr_out"],p)}.items():
    sc = max(np.abs(b_).max(),1e-9)
    print(f"{k_:4s} unrolled vs chained: max|diff| {np.abs(a_-b_).max():.3e}  ({np.abs(a_-b_).max()/sc*100:.3f}% of range {sc:.2f})")

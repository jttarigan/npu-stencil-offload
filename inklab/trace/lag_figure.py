"""Visual A/B of the splat lag: what the player sees now vs K frames stale.

Columns = chosen steps of the recorded game; rows = shipped (current field),
delay scheme at K=8 (the field 8 frames old), and |difference| x4.
Writes lag_ab.png.
"""
import sys, json, zlib, struct, numpy as np
from reference import InkRef

K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
steps = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "400,1000,1600,1900").split(",")]
W, H = 128, 256
m = json.load(open("gameplay_script.json"))
ref = InkRef(W, H, 1 / 30, 0.3, 0.03, 2.5, 26)
GAP = 3   # thin white line between panels, px (about 2 pt in the paper)
ring, cols = {}, []
for e in m["script"]:
    ref.step(e["splats"]); n = e["step"]
    ring[n] = ref.dye.copy()
    for s in steps:
        if n == s:
            cur, old = ring[s], ring.get(s - K, ring[s])
            cols.append([cur, old, np.abs(cur - old) * 4])
    for old_n in [x for x in ring if x < n - K - 2]:
        del ring[old_n]
img = np.full((3 * (H + GAP) - GAP, len(cols) * (W + GAP) - GAP, 3), 255, np.uint8)
for j, c in enumerate(cols):
    for i, p in enumerate(c):
        img[i * (H + GAP):i * (H + GAP) + H, j * (W + GAP):j * (W + GAP) + W] = (np.clip(p * 0.85, 0, 1) * 255).astype(np.uint8)
h, w, _ = img.shape
raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
def chunk(t, b): return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xffffffff)
png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
open("lag_ab.png", "wb").write(png)
print(f"wrote lag_ab.png {w}x{h}  K={K}  steps {steps}")

"""Perceptual cost of batching splats for the K-step unrolled graph.

The unrolled program runs K steps per submission, so the host must hand over
K frames of splat fields at once. It cannot know future splats, so it submits
the K frames it has just collected and the displayed field is K frames stale.
Two ways to present the result, with very different artefacts:

  DELAY  the program emits all K intermediate dye fields and the host shows
         one per frame, K frames late. Motion stays smooth; the only artefact
         is a constant K/30 s lag of the ink behind the player's head.
  HOLD   the program emits only the final field and the host shows it for K
         frames. The lag is the same on average but the background also
         updates at 30/K Hz, so the flow visibly stutters.

Both are exactly the shipped simulation (unroll_equiv.py showed one K-step
submission equals K chained steps), so the artefact is purely temporal and
can be measured from one float32 reference run.

Scale for judging the numbers: the half-precision neural engine path already
sits at 29 to 33 dB against this same float32 reference.

  .venv/bin/python lag_check.py [script.json]
"""
import sys, json, numpy as np
from reference import InkRef

SCRIPT = sys.argv[1] if len(sys.argv) > 1 else "gameplay_script.json"
KS = [2, 4, 8]
W, H = 128, 256

def psnr(a, b, peak=1.6):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse <= 0 else 10 * np.log10(peak * peak / mse)

m = json.load(open(SCRIPT))
ref = InkRef(W, H, 1 / 30, 0.3, 0.03, 2.5, 26)
hist = []                                   # dye field per step
head = []                                   # head-stir position per step (the last splat with strength 0)
for e in m["script"]:
    ref.step(e["splats"])
    hist.append(ref.dye.copy())
    st = [s for s in e["splats"] if s["params"][1] == 0]
    head.append(np.array(st[-1]["posVel"][:2], np.float32) if st else (head[-1] if head else np.zeros(2, np.float32)))
n = len(hist)
finale_start = next((i for i, e in enumerate(m["script"]) if e["state"] == "finale"), n)
print(f"{SCRIPT}: {n} steps, finale from {finale_start}")

def window(vals, lo, hi):
    v = [x for i, x in enumerate(vals) if lo <= i < hi and np.isfinite(x)]
    return (np.median(v), np.percentile(v, 5)) if v else (float("nan"),) * 2

print("\n  K   scheme  play median / 5th pct dB   finale median / 5th pct dB   head lag px")
for K in KS:
    delay, hold = [], []
    for t in range(K, n):
        delay.append(psnr(hist[t], hist[t - K]))             # shown: field K frames old
        hold.append(psnr(hist[t], hist[t - (t % K)]))        # shown: last field of the batch
    dp, dp5 = window(delay, 0, finale_start - K); df, df5 = window(delay, finale_start, n - K)
    hp, hp5 = window(hold, 0, finale_start - K); hf, hf5 = window(hold, finale_start, n - K)
    px = np.median([np.linalg.norm(head[t] - head[t - K]) for t in range(K, finale_start)])
    print(f"  {K}   delay   {dp:6.1f} / {dp5:6.1f}            {df:6.1f} / {df5:6.1f}             {px:5.2f}")
    print(f"  {K}   hold    {hp:6.1f} / {hp5:6.1f}            {hf:6.1f} / {hf5:6.1f}")

# --- smoothness: PSNR between CONSECUTIVE DISPLAYED frames ---
# Staleness PSNR above cannot see stutter (it favours hold, which is stale by
# K/2 on average rather than K). What separates the schemes is whether the
# displayed field moves every frame or jumps once per batch.
ftf = [psnr(hist[t], hist[t - 1]) for t in range(1, n)]
base = window(ftf, 0, finale_start)[0]
print(f"\nsmoothness (PSNR between consecutive displayed frames, play phase)")
print(f"  shipped / delay: {base:.1f} dB every frame, no jumps")
for K in KS:
    jumps = [psnr(hist[t - (t % K)], hist[t - 1 - ((t - 1) % K)]) for t in range(K, finale_start) if t % K == 0]
    print(f"  hold K={K}: unchanged for {K-1} of every {K} frames, then a "
          f"{np.median(jumps):.1f} dB jump {30/K:.2f} times a second")

# --- head lag in game units ---
SX = 128 / 1170.0                      # sim texels per screen pixel (iPhone 12, as recorded)
CELL = 1170 / 13.0                     # screen pixels per grid cell
print("\nhead lag of the delay scheme")
for K in KS:
    px = np.median([np.linalg.norm(head[t] - head[t - K]) for t in range(K, finale_start)])
    print(f"  K={K}: {px:5.2f} sim texels = {px/SX:6.1f} screen px = {px/SX/CELL:4.2f} grid cells "
          f"= {K/30:.3f} s")

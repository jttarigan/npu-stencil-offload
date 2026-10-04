"""Compare the game's ink backends from the per-frame CSVs one device wrote.

    python3 game_frames.py <dir with perf_*_ink-*.csv> [--md]

Each CSV is one full game (perf_<t>_fx<N>_ink-<backend>.csv; backends off,
gpu, npu, npu4, npua = neural engine off the render thread). Frames are split into PLAY and FINALE by instance count, with
the thresholds the renderer paper used per Sparks tier (x1 1000, x2 1500,
x4 2500, x8 4000 instances). For each backend and phase: frames, median /
p95 / p99 frame time, mean GPU time, mean CPU time (tick + build + encode),
mean fluid time (inkMs: the neural-engine work done in that frame, 0 on frames
without a tick; it is a PART of encodeMs, which wraps the whole render call, so
it is never added on top), and the share of frames over 1.5x the display period.
CAUTION: "CPU ms" includes the wait inside view.currentDrawable, so it is not CPU
work and must not be compared across backends (the verdict lines ignore it).
The shipped GPU prepass shows up in gpuMs instead.

The question the table answers (RQ4): relative to `gpu` (the shipped
prepass), does an NPU backend lower the GPU time by more than it raises the
CPU-side time, and does `npu4` (4 steps per submission) do so where `npu`
does not? Each backend is a SEPARATE play session, so run-to-run variation
(different pellet sequence, different finale length) sits inside every
difference; treat small nets as noise.
"""
import csv, glob, os, re, sys
from statistics import mean, median

FINALE = {1: 1000, 2: 1500, 4: 2500, 8: 4000}

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else float("nan")

def load(path):
    m = re.search(r"_fx(\d+)_ink-(\w+)\.csv$", path)
    fx, backend = int(m.group(1)), m.group(2)
    rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(path))]
    return fx, backend, rows

def summarise(rows):
    fm = [r["frameMs"] for r in rows]
    period = median(fm)            # the display period the frame loop settles on
    return {
        "n": len(rows),
        "frame_p50": median(fm), "frame_p95": pct(fm, 95), "frame_p99": pct(fm, 99),
        "gpu": mean(r["gpuMs"] for r in rows),
        "cpu": mean(r["tickMs"] + r["buildMs"] + r["encodeMs"] for r in rows),
        "ink": mean(r["inkMs"] for r in rows),
        "ink_max": max(r["inkMs"] for r in rows),
        "over": 100 * sum(f > 1.5 * period for f in fm) / len(fm),
    }

def main():
    d = sys.argv[1]; md = "--md" in sys.argv
    files = sorted(glob.glob(os.path.join(d, "**", "perf_*_ink-*.csv"), recursive=True))
    if not files:
        sys.exit(f"no perf_*_ink-*.csv under {d}")
    order = {"off": 0, "gpu": 1, "npu": 2, "npu4": 3, "npua": 4}
    results = []
    for f in files:
        fx, backend, rows = load(f)
        thr = FINALE.get(fx, 1000)
        for phase, sel in (("play", [r for r in rows if r["instances"] < thr]),
                           ("finale", [r for r in rows if r["instances"] >= thr])):
            if sel:
                results.append((phase, order.get(backend, 9), backend, fx, os.path.basename(f), summarise(sel)))
    results.sort()
    cols = ["phase", "backend", "fx", "frames", "frame p50", "p95", "p99", "GPU ms", "CPU ms",
            "ink ms (mean)", "ink ms (max)", "% frames >1.5x period"]
    if md:
        print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    else:
        print("\t".join(cols))
    base = {}
    for phase, _, backend, fx, name, s in results:
        cells = [phase, backend, f"x{fx}", str(s["n"]), f"{s['frame_p50']:.2f}", f"{s['frame_p95']:.2f}",
                 f"{s['frame_p99']:.2f}", f"{s['gpu']:.3f}", f"{s['cpu']:.3f}", f"{s['ink']:.3f}",
                 f"{s['ink_max']:.2f}", f"{s['over']:.1f}"]
        print(("| " + " | ".join(cells) + " |") if md else "\t".join(cells))
        if backend == "gpu":
            base[phase] = s
    # RQ4 in one line per NPU backend and phase, from quantities measured directly:
    # GPU time per frame against the shipped `gpu` prepass, the fluid's own
    # synchronous time on the render thread (inkMs), and the share of long frames.
    # The "CPU ms" column is NOT used: encodeMs wraps view.currentDrawable, so it
    # includes the wait for a free drawable and swings with frame pacing (with the
    # fluid off the CPU finishes early and waits most of the frame).
    print()
    for phase, _, backend, fx, name, s in results:
        if backend.startswith("npu") and phase in base:
            g = base[phase]
            saved = g["gpu"] - s["gpu"]
            print(f"{phase:6s} {backend:5s}: GPU time saved {saved:+.3f} ms/frame; fluid on the render "
                  f"thread {s['ink']:.2f} ms/frame (max {s['ink_max']:.1f}); long frames "
                  f"{g['over']:.1f}% -> {s['over']:.1f}% -> "
                  f"{'pays' if saved > s['ink'] and s['over'] <= g['over'] + 0.5 else 'does not pay'}")
    fxs = {r[3] for r in results}
    if len(fxs) > 1:
        print(f"WARNING: runs used different Sparks tiers {sorted(fxs)}; frame times are not comparable")

if __name__ == "__main__":
    main()

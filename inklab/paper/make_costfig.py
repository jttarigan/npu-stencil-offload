"""The cost-model figure, generated from the device CSVs.

    python3 make_costfig.py          # writes figures/costmodel.tex

Panel (a): one phone (A19 Pro), the three paths, t_sub against K. The fitted
lines are drawn back to K = 0 so the fixed cost a is the intercept you can read
off; the CPU control passes through the origin.
Panel (b): per-step cost t_sub/K for the accelerator route of every device
measured, log scale, so amortisation (or its absence, on the A12X) is visible.

Colours are Okabe-Ito, validated for colour-vision deficiency; every series also
carries its own marker and a legend entry, so identity never rests on hue alone.
"""
import csv, glob, os
from collections import defaultdict
from statistics import median

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
KS = [1, 2, 4, 8]
COLORS = {   # fixed order, never cycled (Okabe-Ito)
    "ink1": "0072B2", "ink2": "D55E00", "ink3": "009E73", "ink4": "E69F00", "ink5": "CC79A7",
    "ink6": "56B4E9",   # added for the A14; entities keep their earlier colours
}
MARKS = ["*", "square*", "triangle*", "diamond*", "pentagon*", "star"]


def load():
    """(device, path, K) -> median ms, from every bench CSV in results/."""
    vals = defaultdict(list)
    for p in glob.glob(f"{ROOT}/results/*/bench_*.csv") + glob.glob(f"{ROOT}/results/*/sweep*.csv"):
        with open(p) as fh:
            r = csv.reader(fh); hdr = next(r)
            for v in r:
                if len(v) == len(hdr) + 1 and hdr and hdr[0] == "device":
                    v = [v[0] + "," + v[1]] + v[2:]          # unquoted "iPad14,1"
                if len(v) == len(hdr) + 1 and hdr and hdr[0] == "device" and "soc" in hdr:
                    v = v[:7] + [v[7] + "." + v[8]] + v[9:]  # comma decimal (old Android CSVs)
                row = dict(zip(hdr, v))
                if row.get("mode") != "b2b" or not row.get("ms_per_submission"):
                    continue
                if not row["workload"].startswith("ink"):
                    continue
                vals[(row.get("device") or "M5", row["units"], int(row["K"]))].append(
                    float(row["ms_per_submission"]))
    return {k: median(v) for k, v in vals.items()}


def fit(ks, ts):
    n = len(ks); mk = sum(ks) / n; mt = sum(ts) / n
    b = sum((k - mk) * (t - mt) for k, t in zip(ks, ts)) / sum((k - mk) ** 2 for k in ks)
    return mt - b * mk, b


def series(vals, dev, path):
    ks = [k for k in KS if (dev, path, k) in vals]
    return ks, [vals[(dev, path, k)] for k in ks]


def plot(vals, dev, path, color, mark, label, per_step=False, fitline=True):
    ks, ts = series(vals, dev, path)
    if not ks: return ""
    pts = " ".join(f"({k},{(t / k if per_step else t):.4f})" for k, t in zip(ks, ts))
    out = f"\\addplot[only marks, mark={mark}, mark size=1.5pt, color={color}] coordinates {{{pts}}};\n"
    out += f"\\addlegendentry{{{label}}}\n"
    if fitline:
        a, b = fit(ks, ts)
        out += (f"\\addplot[color={color}, thick, dashed, forget plot] "
                f"coordinates {{(0,{a:.4f}) (1,{a + b:.4f})}};\n")     # extrapolated to K = 0: the intercept is a
        out += (f"\\addplot[color={color}, thick, forget plot] "
                f"coordinates {{(1,{a + b:.4f}) (8,{a + 8 * b:.4f})}};\n")
    else:                                   # per-step panel: join the measured points
        out += (f"\\addplot[color={color}, thick, forget plot] coordinates {{{pts}}};\n")
    return out


def main():
    v = load()
    defs = "\n".join(f"\\definecolor{{{n}}}{{HTML}}{{{h}}}" for n, h in COLORS.items())
    axis = ("width=\\columnwidth, height=4.4cm, xmin=0, xmax=8.6, xtick={1,2,4,8},"
            " xlabel={steps per submission $K$}, tick align=outside, tick pos=left,"
            " grid=major, grid style={gray!20, very thin}, axis line style={gray!60},"
            " label style={font=\\footnotesize}, tick label style={font=\\footnotesize},"
            " legend style={font=\\footnotesize, draw=none, fill=none, at={(0.02,0.98)},"
            " anchor=north west, row sep=-1pt}")

    # (a) the two accelerator submissions only; the CPU control sits in (b), where
    # it is a flat line, so nothing has to be clipped to keep the intercepts legible.
    a = ("\\begin{tikzpicture}\n\\begin{axis}[" + axis +
         ", ylabel={$t_\\mathrm{sub}$ (ms)}, ymin=0,"
         " xtick={0,1,2,4,8},"
         " title={\\footnotesize (a) A19 Pro, accelerator submissions}, title style={yshift=-2pt}]\n")
    a += plot(v, "iPhone18,2", "ne", "ink1", MARKS[0], "neural engine")
    a += plot(v, "iPhone18,2", "metal", "ink2", MARKS[1], "standalone Metal")
    a += "\\end{axis}\n\\end{tikzpicture}"

    axis_b = axis.replace("xmin=0, xmax=8.6, xtick={1,2,4,8},",
                          "xmode=log, log basis x=2, xmin=0.85, xmax=9.4, xtick={1,2,4,8},"
                          " xticklabels={1,2,4,8},")
    b = ("\\begin{tikzpicture}\n\\begin{axis}[" + axis_b +
         ", ylabel={per-step cost (ms)}, ymode=log, ytick={0.2,0.5,1,2,4},"
         " yticklabels={0.2,0.5,1,2,4}, ymax=5, log ticks with fixed point,"
         " title={\\footnotesize (b) accelerator route per device},"
         " title style={yshift=-2pt}, legend style={font=\\footnotesize, draw=none,"
         " fill=none, at={(0.5,-0.32)}, anchor=north, legend columns=3, row sep=-1pt,"
         " legend cell align=left,"
         " /tikz/every even column/.append style={column sep=6pt}}]\n")
    for dev, path, c, m, lab in [
            ("M5", "ne", "ink1", MARKS[0], "M5"),
            ("iPhone18,2", "ne", "ink2", MARKS[1], "A19 Pro"),
            ("iPad14,1", "ne", "ink3", MARKS[2], "A15"),
            ("iPhone13,2", "ne", "ink6", MARKS[5], "A14"),
            ("iPad8,5", "ne", "ink4", MARKS[3], "A12X, on CPU"),
            ("SM-A566B", "gles", "ink5", MARKS[4], "Exynos 1580, GLES")]:
        b += plot(v, dev, path, c, m, lab, per_step=True, fitline=False)
    ks, ts = series(v, "iPhone18,2", "cpu")      # the control: flat, nothing to amortise
    pts = " ".join(f"({k},{t / k:.4f})" for k, t in zip(ks, ts))
    b += f"\\addplot[gray, thick, densely dotted, mark=o, mark size=1.3pt] coordinates {{{pts}}};\n"
    b += "\\addlegendentry{CPU control (A19 Pro)}\n"
    b += "\\end{axis}\n\\end{tikzpicture}"

    out = (f"% GENERATED by make_costfig.py from the CSVs in inklab/results. Do not edit.\n"
           f"{defs}\n\n% (a) one phone, three paths\n{a}\n\n% (b) accelerator route per device\n{b}\n")
    os.makedirs(f"{os.path.dirname(os.path.abspath(__file__))}/figures", exist_ok=True)
    with open(f"{os.path.dirname(os.path.abspath(__file__))}/figures/costmodel.tex", "w") as fh:
        fh.write(out)
    print("wrote figures/costmodel.tex")
    for dev in ["M5", "iPhone18,2", "iPad14,1", "iPhone13,2", "iPad8,5", "SM-A566B"]:
        for path in ["ne", "metal", "cpu", "gles"]:
            ks, ts = series(v, dev, path)
            if ks:
                a_, b_ = fit(ks, ts)
                print(f"  {dev:11s} {path:6s} a={a_:7.3f} b={b_:7.3f}")


if __name__ == "__main__":
    main()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

NAVY = "#1F3864"
GREY = "#ECECEC"
YELLOW = "#FCF3CF"
EDGE = "#333333"

fig, ax = plt.subplots(figsize=(8.5, 11.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, text, fill=GREY, fc="black", fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=1.1",
                                facecolor=fill, edgecolor=EDGE, linewidth=1.0, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=fc,
            fontweight="bold" if bold else "normal", zorder=4, linespacing=1.6)
    return dict(x=x, y=y, w=w, h=h)


def arrow(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=0.9, color=EDGE, shrinkA=0, shrinkB=0, zorder=2))


def line(p0, p1):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=EDGE, linewidth=0.9, zorder=2,
            solid_capstyle="round")


def bot(b):
    return (b["x"], b["y"] - b["h"] / 2)


def top(b):
    return (b["x"], b["y"] + b["h"] / 2)


def vdown(b, dst):
    arrow(bot(b), top(dst))


data = box(50, 95, 68, 6.5, "UrbanSound8K\n8,732 clips  |  10 classes  |  10 official folds",
           fill=NAVY, fc="white", fs=10, bold=True)
spec = box(50, 85, 46, 5.5, "Log-mel spectrogram (64 × 128)", fill=NAVY, fc="white", fs=10, bold=True)
vdown(data, spec)

arch_y = 74
arch_xs = [17, 50, 83]
archs = [box(x, arch_y, 26, 6.5, t, fs=9.5, bold=True) for x, t in
         zip(arch_xs, ["Simple CNN\n4.29M params", "CRNN\n2.61M params", "VGGish\n71.66M params"])]

bus_a = 81.0
line(bot(spec), (50, bus_a))
line((arch_xs[0], bus_a), (arch_xs[2], bus_a))
for a in archs:
    arrow((a["x"], bus_a), top(a))

bus_b = 67.0
for a in archs:
    line(bot(a), (a["x"], bus_b))
line((arch_xs[0], bus_b), (arch_xs[2], bus_b))

base = box(17, 59, 26, 7, "Baseline\n(no adversarial content)", fs=9)
at1 = box(50, 60.5, 20, 5, "AT-FGSM", fs=9.5)
at2 = box(50, 53.5, 20, 5, "AT-BIM", fs=9.5)
aft1 = box(83, 60.5, 20, 5, "AFT-FGSM", fs=9.5)
aft2 = box(83, 53.5, 20, 5, "AFT-BIM", fs=9.5)


def cluster(cx, cy, cw, ch, label):
    ax.add_patch(FancyBboxPatch((cx - cw / 2, cy - ch / 2), cw, ch,
                                boxstyle="round,pad=0,rounding_size=1.1",
                                facecolor="none", edgecolor=NAVY, linewidth=1.1,
                                linestyle=(0, (4, 3)), zorder=1))
    ax.text(cx, cy + ch / 2 + 0.9, label, ha="center", va="bottom",
            fontsize=9, color=NAVY, fontweight="bold")
    return dict(x=cx, y=cy, w=cw, h=ch)


at_cl = cluster(50, 57, 25, 15, "Adversarial Training (AT)")
aft_cl = cluster(83, 57, 25, 15, "Adversarial Fine-Tuning (AFT)")

arrow((base["x"], bus_b), top(base))
arrow((at_cl["x"], bus_b), (at_cl["x"], at_cl["y"] + at_cl["h"] / 2))

detour = 47.0
line(bot(base), (base["x"], detour))
line((base["x"], detour), (aft_cl["x"], detour))
arrow((aft_cl["x"], detour), (aft_cl["x"], aft_cl["y"] - aft_cl["h"] / 2))

bus_c = 42.0
line((base["x"] - 5, base["y"] - base["h"] / 2), (base["x"] - 5, bus_c))
for b in [at_cl, aft_cl]:
    line((b["x"], b["y"] - b["h"] / 2), (b["x"], bus_c))
line((base["x"] - 5, bus_c), (aft_cl["x"], bus_c))

rob = box(27, 33, 46, 7, "Robustness evaluation\nFGSM & BIM at ε = 0.01, 0.03, 0.10",
          fill=NAVY, fc="white", fs=9.5, bold=True)
eff = box(75, 33, 46, 7, "Efficiency evaluation\nFLOPs  |  Parameters  |  Training epochs",
          fill=NAVY, fc="white", fs=9.5, bold=True)
arrow((rob["x"], bus_c), top(rob))
arrow((eff["x"], bus_c), top(eff))

rq1 = box(17, 14, 27, 6.5, "RQ1\nRobustness gain", fill=YELLOW, fs=9.5)
rq2 = box(50, 14, 27, 6.5, "RQ2\nEfficiency cost", fill=YELLOW, fs=9.5)
rq3 = box(83, 14, 27, 6.5, "RQ3\nAT vs AFT trade-off", fill=YELLOW, fs=9.5)

def elbow(x0, y0, ymid, x1, y1):
    line((x0, y0), (x0, ymid))
    line((x0, ymid), (x1, ymid))
    arrow((x1, ymid), (x1, y1))


rob_b = rob["y"] - rob["h"] / 2
eff_b = eff["y"] - eff["h"] / 2
elbow(20, rob_b, 25.0, rq1["x"], rq1["y"] + rq1["h"] / 2)
elbow(70, eff_b, 21.5, rq2["x"], rq2["y"] + rq2["h"] / 2)
line((34, rob_b), (34, 27.5))
line((34, 27.5), (rq3["x"], 27.5))
line((rq3["x"], eff_b), (rq3["x"], 27.5))
arrow((rq3["x"], 27.5), (rq3["x"], rq3["y"] + rq3["h"] / 2))

fig.savefig("figures/fig1_1_research_framework.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("figures/fig1_1_research_framework.svg", bbox_inches="tight", facecolor="white")

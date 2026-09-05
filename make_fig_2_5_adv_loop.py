import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

NAVY = "#1F3864"
ORANGE = "#C55A11"
EDGE = "#333333"

fig, ax = plt.subplots(figsize=(11, 10))
ax.set_xlim(-2.3, 2.3)
ax.set_ylim(-2.1, 1.9)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

labels = [
    "Sample a clean batch\n(from the training data)",
    "Generate adversarial examples\nfrom the current model\n(FGSM or BIM, ε = 0.03)",
    "Mix clean and adversarial\nexamples (50% clean /\n50% adversarial)",
    "Forward pass — compute loss\n(single loss on the mixed batch)",
    "Update model weights\n(backpropagation —\none step per batch)",
]

n = len(labels)
R = 1.35
angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
centers = [(R * np.cos(a), R * np.sin(a)) for a in angles]

box_w, box_h = 1.55, 0.62


def box(xy, text, fs=9.3):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                                boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor=NAVY, edgecolor=EDGE, linewidth=1.1, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="white",
            fontweight="normal", zorder=4, linespacing=1.35)


for c, t in zip(centers, labels):
    box(c, t)


def edge_point(c, target, pad=0.03):
    x0, y0 = c
    x1, y1 = target
    dx, dy = x1 - x0, y1 - y0
    dist = np.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    hw, hh = box_w / 2, box_h / 2
    if abs(ux) > 1e-9 and abs(uy) > 1e-9:
        t = min(hw / abs(ux), hh / abs(uy))
    elif abs(ux) > 1e-9:
        t = hw / abs(ux)
    else:
        t = hh / abs(uy)
    return (x0 + ux * (t + pad), y0 + uy * (t + pad))


def arrow(i, j, color=EDGE, lw=1.1, mscale=13, connectionstyle="arc3,rad=0.08"):
    src, dst = centers[i], centers[j]
    p0 = edge_point(src, dst)
    p1 = edge_point(dst, src)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=mscale,
                                 linewidth=lw, color=color, shrinkA=0, shrinkB=0,
                                 connectionstyle=connectionstyle, zorder=2))
    return p0, p1


for i in range(n):
    j = (i + 1) % n
    color = ORANGE if j in (1, 4) else EDGE
    lw = 1.7 if j in (1, 4) else 1.1
    arrow(i, j, color=color, lw=lw)

mid = [((centers[i][0] + centers[(i + 1) % n][0]) / 2,
        (centers[i][1] + centers[(i + 1) % n][1]) / 2) for i in range(n)]

ax.annotate(
    "Inner maximisation — find the\nworst-case perturbation δ\nwithin the budget ε",
    xy=mid[0], xytext=(1.95, 0.55),
    fontsize=8.6, color=ORANGE, ha="left", va="center", style="italic",
)
ax.annotate(
    "Outer minimisation — update θ\nto reduce loss on the\nperturbed inputs",
    xy=mid[3], xytext=(-2.28, -1.65),
    fontsize=8.6, color=ORANGE, ha="left", va="center", style="italic",
)

arrow(4, 0, color=EDGE, lw=1.1, connectionstyle="arc3,rad=0.35")
loop_mid = ((centers[4][0] + centers[0][0]) / 2, (centers[4][1] + centers[0][1]) / 2)
ax.text(loop_mid[0] - 0.05, loop_mid[1] + 0.95, "next batch", fontsize=9.5, color=EDGE,
        ha="center", va="center")

ax.text(0, -2.0,
        r"$\min_{\theta}\ E\left[\ \max_{\Vert\delta\Vert\leq\epsilon}\ L(\theta,\,x+\delta,\,y)\ \right]$",
        ha="center", va="center", fontsize=13, style="italic", color="#222222")

fig.savefig("figures/fig_2_5_adversarial_training_loop.png", dpi=300, facecolor="white",
            bbox_inches="tight", pad_inches=0.25)
fig.savefig("figures/fig_2_5_adversarial_training_loop.svg", facecolor="white",
            bbox_inches="tight", pad_inches=0.25)

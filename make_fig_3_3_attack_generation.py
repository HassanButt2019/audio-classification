import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"

NAVY = "#1F3864"
ORANGE = "#C55A11"
EDGE = "#333333"

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(cx, cy, w, h, text, color=NAVY, fs=10, fc="white"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.7",
                                facecolor=color, edgecolor=EDGE, linewidth=1.1, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=fc,
            zorder=4, linespacing=1.35)


def hline_arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.0, color=EDGE, shrinkA=0, shrinkB=0, zorder=2))


row1_y = 78
row2_y = 30

ax.text(4, row1_y + 15, "FGSM", fontsize=13, color=NAVY, fontweight="bold", va="center")
ax.text(4, row2_y + 20, "BIM", fontsize=13, color=NAVY, fontweight="bold", va="center")

w1 = 15.5
xs1 = [15, 35, 55, 78]
box(xs1[0], row1_y, w1, 11, "Clean input x", fs=10)
box(xs1[1], row1_y, w1, 11, "Compute gradient\n∇ₓ L(θ, x, y)", fs=9.5)
box(xs1[2], row1_y, w1, 11, "Take sign, scale by ε", color=ORANGE, fs=9.5)
box(xs1[3], row1_y, 21, 13, "x_adv = x + ε·sign(∇ₓL)\nclip to [0, 1]", fs=9.3)

for i in range(3):
    gap = (w1 / 2 + 1.2) if i < 2 else (w1 / 2 + 1.2)
    hline_arrow(xs1[i] + w1 / 2 + 0.6, xs1[i + 1] - (21 / 2 if i == 2 else w1 / 2) - 0.6, row1_y)

xs2 = [12, 48, 86]
box(xs2[0], row2_y, 16, 11, "Clean input\nx₀ = x", fs=9.5)

loop_cx, loop_w, loop_h = xs2[1], 40, 22
ax.add_patch(FancyBboxPatch((loop_cx - loop_w / 2, row2_y - loop_h / 2), loop_w, loop_h,
                            boxstyle="round,pad=0,rounding_size=1.0",
                            facecolor="none", edgecolor=NAVY, linewidth=1.1,
                            linestyle=(0, (4, 3)), zorder=1))
sub_w = 16
box(loop_cx - loop_w / 2 + sub_w / 2 + 2.0, row2_y, sub_w, 10, "Compute gradient\nat current xₜ", fs=8.8)
box(loop_cx + loop_w / 2 - sub_w / 2 - 2.0, row2_y, sub_w, 10,
    "Step by α·sign(∇)\nclip to ε-ball, [0,1]", color=ORANGE, fs=8.5)
hline_arrow(loop_cx - loop_w / 2 + 2.0 + sub_w + 0.4, loop_cx + loop_w / 2 - 2.0 - sub_w - 0.4, row2_y)

loop_arrow = FancyArrowPatch((loop_cx + 6, row2_y - loop_h / 2 - 0.3),
                             (loop_cx - 6, row2_y - loop_h / 2 - 0.3),
                             connectionstyle="arc3,rad=-0.45", arrowstyle="-|>",
                             mutation_scale=13, linewidth=1.0, color=NAVY, zorder=1)
ax.add_patch(loop_arrow)
ax.text(loop_cx, row2_y - loop_h / 2 - 8.8, "× 10 steps (evaluation)   α = ε / 10",
        ha="center", va="center", fontsize=9, color=NAVY, style="italic")

box(xs2[2], row2_y, 18, 11, "Final adversarial\ninput x_adv", fs=9.5)

hline_arrow(xs2[0] + 8 + 0.6, loop_cx - loop_w / 2 - 0.6, row2_y)
hline_arrow(loop_cx + loop_w / 2 + 0.6, xs2[2] - 9 - 0.6, row2_y)

ax.plot([5, 95], [52, 52], color="#cccccc", linewidth=0.8, zorder=0)

ax.text(50, 1, "Evaluated at ε = 0.01, 0.03, 0.10  (L∞ norm)",
        ha="center", va="center", fontsize=10.5, style="italic", color="#222222")

fig.savefig("figures/fig_3_3_attack_generation.png", dpi=300, facecolor="white",
            bbox_inches="tight", pad_inches=0.25)
fig.savefig("figures/fig_3_3_attack_generation.svg", facecolor="white",
            bbox_inches="tight", pad_inches=0.25)

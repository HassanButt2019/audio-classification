import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "DejaVu Sans"

NAVY = "#1F3864"
ORANGE = "#C55A11"
EDGE = "#333333"
GREY = "#9A9A9A"
GREY_FILL = "#D9D9D9"

MAX_EPOCHS = 50
CNN_EPOCHS = 30
AFT_EPOCHS = 10
BASELINE_SCHEMATIC_W = 14.0

LEFT, RIGHT = 15.0, 95.0
EPOCH_UNIT = (RIGHT - LEFT) / MAX_EPOCHS
AT_Y, H = 66.0, 13.0
AFT_Y = 32.0

fig, ax = plt.subplots(figsize=(13, 6.8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")


def bar(x0, x1, y, color, text=None, fs=9.6, edge=EDGE, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y - H / 2), x1 - x0, H,
                                boxstyle="round,pad=0,rounding_size=0.6",
                                facecolor=color, edgecolor=edge, linewidth=1.1,
                                linestyle=ls, zorder=3))
    if text:
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=fs,
                color="white", zorder=4, linespacing=1.35)


def circle(x, y, filled=False, color=EDGE):
    ax.plot(x, y, marker="o", markersize=7.5,
            markerfacecolor=(color if filled else "white"),
            markeredgecolor=EDGE, markeredgewidth=1.3, zorder=6)


def square(x, y, color=GREY_FILL):
    ax.plot(x, y, marker="s", markersize=8.5, markerfacecolor=color,
            markeredgecolor=EDGE, markeredgewidth=1.3, zorder=6)


bar(LEFT, RIGHT, AT_Y, NAVY,
    "Full adversarial training — 50 epochs (CRNN, VGGish), lr = 0.001")
circle(LEFT, AT_Y)
ax.text(LEFT, AT_Y + H / 2 + 5, "Random\ninitialisation", ha="center", va="bottom",
        fontsize=8.3, color=EDGE, linespacing=1.2)
ax.text(RIGHT, AT_Y - H / 2 - 5, "Trained AT model", ha="center", va="top",
        fontsize=9.5, color=NAVY, fontweight="bold")

cnn_tick_x = LEFT + CNN_EPOCHS * EPOCH_UNIT
ax.plot([cnn_tick_x, cnn_tick_x], [AT_Y - H / 2, AT_Y + H / 2],
        linestyle=(0, (3, 2)), color="white", linewidth=1.6, zorder=5)
ax.text(cnn_tick_x, AT_Y - H / 2 - 5, "CNN AT ends here\n30 epochs, lr = 0.001",
        ha="center", va="top", fontsize=8.2, color=NAVY, linespacing=1.25)

ckpt_x = LEFT + BASELINE_SCHEMATIC_W
aft_x1 = ckpt_x + AFT_EPOCHS * EPOCH_UNIT

bar(LEFT, ckpt_x, AFT_Y, GREY_FILL, edge=GREY, ls=(0, (3, 2)))
ax.text((LEFT + ckpt_x) / 2, AFT_Y, "Baseline\ntraining", ha="center", va="center",
        fontsize=7.6, color="#444444", linespacing=1.15, zorder=4)
circle(LEFT, AFT_Y)

bar(ckpt_x, aft_x1, AFT_Y, ORANGE, "Adversarial fine-tuning\n10 epochs, lr = 0.0001", fs=8.8)
square(ckpt_x, AFT_Y)
ax.text(ckpt_x, AFT_Y + H / 2 + 5, "Baseline\ncheckpoint", ha="center", va="bottom",
        fontsize=8.3, color="#444444", linespacing=1.2)
ax.text(aft_x1, AFT_Y - H / 2 - 5, "Trained AFT model", ha="center", va="top",
        fontsize=9.5, color=ORANGE, fontweight="bold")

ax.text(50, 8,
        "AFT trains for a fraction of the epochs used by full AT (10 vs. 30 CNN / 50 CRNN & VGGish),\n"
        "starting from an already-trained baseline checkpoint rather than from scratch.",
        ha="center", va="center", fontsize=9.7, style="italic", color="#222222", linespacing=1.5)

fig.savefig("figures/fig_3_4_at_vs_aft.png", dpi=300, facecolor="white",
            bbox_inches="tight", pad_inches=0.2)
fig.savefig("figures/fig_3_4_at_vs_aft.svg", facecolor="white",
            bbox_inches="tight", pad_inches=0.2)

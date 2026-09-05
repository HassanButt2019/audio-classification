import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

NAVY = "#1F3864"
GREY = "#ECECEC"
YELLOW = "#FCF3CF"
EDGE = "#333333"
PANEL_TOP = 96
PANEL_BOTTOM = 8
GAP = 2.2


def draw_panel(ax, x0, x1, boxes, caption, note=None):
    cx = (x0 + x1) / 2
    w = (x1 - x0) * 0.94
    n = len(boxes)
    avail = PANEL_TOP - PANEL_BOTTOM - (n - 1) * GAP
    heights = [b.get("h", 1.0) for b in boxes]
    unit = avail / sum(heights)

    centers = []
    y = PANEL_TOP
    for b, hh in zip(boxes, heights):
        h = hh * unit
        y_c = y - h / 2
        centers.append((y_c, h))
        y -= h + GAP

    for b, (y_c, h) in zip(boxes, centers):
        fill = b.get("fill", GREY if b["kind"] == "op" else NAVY)
        fc = "black" if fill != NAVY else "white"
        ax.add_patch(FancyBboxPatch((cx - w / 2, y_c - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.9",
                                    facecolor=fill, edgecolor=EDGE, linewidth=1.0, zorder=3))
        ax.text(cx, y_c, b["text"], ha="center", va="center", fontsize=b.get("fs", 9),
                color=fc, fontweight=b.get("weight", "normal"), zorder=4, linespacing=1.4)
        if b.get("anno"):
            ax.text(cx, y_c - h / 2 - 1.0, b["anno"], ha="center", va="top",
                    fontsize=7.8, color=NAVY, style="italic", zorder=4)

    for (y0, h0), (y1, h1) in zip(centers[:-1], centers[1:]):
        top_of_next = y1 + h1 / 2
        bot_of_prev = y0 - h0 / 2
        extra = 2.6 if boxes[centers.index((y0, h0))].get("anno") else 0.0
        ax.add_patch(FancyArrowPatch((cx, bot_of_prev - extra), (cx, top_of_next),
                                     arrowstyle="-|>", mutation_scale=9, linewidth=0.9,
                                     color=EDGE, shrinkA=0, shrinkB=0, zorder=2))

    ax.text(cx, PANEL_BOTTOM - 5.5, caption, ha="center", va="top", fontsize=10,
            color=NAVY, fontweight="bold")
    if note:
        ax.text(cx, PANEL_BOTTOM - 9.5, note, ha="center", va="top", fontsize=8,
                color="#555555", style="italic", wrap=True)


fig, axes = plt.subplots(1, 3, figsize=(16, 11))
for ax in axes:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
fig.patch.set_facecolor("white")

cnn_boxes = [
    dict(kind="io", text="64×128 log-mel spectrogram", h=1.0),
    dict(kind="op", text="Conv 3×3 (32 filters) → ReLU → MaxPool 2×2", h=1.0),
    dict(kind="op", text="Conv 3×3 (64 filters) → ReLU → MaxPool 2×2", h=1.0),
    dict(kind="op", text="Conv 3×3 (128 filters) → ReLU → MaxPool 2×2", h=1.0),
    dict(kind="op", text="Flatten (16,384)", h=0.8),
    dict(kind="io", text="Dense 256 → ReLU → Dropout (p=0.5)", h=1.0),
    dict(kind="io", text="Dense 10 (logits)", h=0.8),
]
draw_panel(axes[0], 0, 100, cnn_boxes, "Simple CNN — 4,289,802 params")

crnn_boxes = [
    dict(kind="io", text="64×128 log-mel spectrogram", h=1.0),
    dict(kind="op", text="Conv 3×3 (32) → BatchNorm → ReLU → MaxPool 2×2", h=1.0),
    dict(kind="op", text="Conv 3×3 (64) → BatchNorm → ReLU → MaxPool 2×2", h=1.0),
    dict(kind="op", text="Conv 3×3 (128) → BatchNorm → ReLU → MaxPool 2×4", h=1.0),
    dict(kind="op", text="Conv 3×3 (128) → BatchNorm → ReLU → MaxPool 2×4", h=1.0),
    dict(kind="op", text="Reshape to sequence", fill=YELLOW, h=0.9,
         anno="verified: seq. len T = 2, feature size C×F = 512"),
    dict(kind="io", text="Bidirectional GRU\nhidden = 256, layers = 2", h=1.15),
    dict(kind="op", text="Take final time step", h=0.8),
    dict(kind="io", text="Dropout (p=0.5) → Dense 10 (logits)", h=0.9),
]
draw_panel(axes[1], 0, 100, crnn_boxes, "CRNN — 2,611,530 params")

vggish_boxes = [
    dict(kind="io", text="64×96 log-mel spectrogram", h=1.0),
    dict(kind="op", text="Conv 3×3 (1→64) → ReLU → MaxPool 2×2", h=1.0),
    dict(kind="op", text="Conv 3×3 (64→128) → ReLU → MaxPool 2×2", h=1.0),
    dict(kind="op", text="Conv 3×3 (128→256) → ReLU\nConv 3×3 (256→256) → ReLU → MaxPool 2×2", h=1.3),
    dict(kind="op", text="Conv 3×3 (256→512) → ReLU\nConv 3×3 (512→512) → ReLU → MaxPool 2×2", h=1.3),
    dict(kind="op", text="Flatten (12,288)", h=0.8),
    dict(kind="io", text="Dense 4096 → ReLU → Dropout (p=0.5)", h=1.0),
    dict(kind="io", text="Dense 4096 → ReLU → Dropout (p=0.5)", h=1.0),
    dict(kind="io", text="Dense 10 (logits)", h=0.8),
]
draw_panel(axes[2], 0, 100, vggish_boxes, "VGGish — 71,657,738 params",
           note="Trained from scratch — no pretrained weights are loaded\n(Xavier-uniform init; only own checkpoints reloaded for eval/fine-tuning).")

fig.tight_layout()
fig.savefig("figures/fig_2_3_architectures.png", dpi=300, facecolor="white")
fig.savefig("figures/fig_2_3_architectures.svg", facecolor="white")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.image as mpimg
from pathlib import Path

plt.rcParams["font.family"] = "DejaVu Sans"

NAVY = "#1F3864"
EDGE = "#333333"
FIGDIR = Path(__file__).resolve().parent / "figures"
INSET_PATH = FIGDIR / "fig_2_2_example_spectrogram.png"

stages = [
    "Raw waveform\n22,050 Hz, 66,048 samples",
    "Framing\noverlapping windows",
    "FFT per frame\nsize 1,024, hop 512",
    "Stack into STFT\ntime × frequency grid",
    "Mel filterbank\n64 mel bands",
    "Log scaling\n10·log10(power), floor at −80 dB",
    "Per-clip normalisation\nscale to [0, 1]",
    "Output\n64 × 128 log-mel spectrogram",
]

n = len(stages)
w = 78
h_normal = 9.0
gap = 1.6
top = 97

has_inset = INSET_PATH.is_file()
inset_extra = 0.0
inset_iw = inset_ih = 0.0
if has_inset:
    img = mpimg.imread(str(INSET_PATH))
    inset_iw = w * 0.58
    inset_ih = inset_iw * img.shape[0] / img.shape[1]
    inset_extra = (h_normal * 0.85) + inset_ih + 2.3 - h_normal

heights = [h_normal] * (n - 1) + [h_normal + (inset_extra if has_inset else 0.0)]

centers = []
y = top
for hh in heights:
    y_c = y - hh / 2
    centers.append(y_c)
    y -= hh + gap
bottom = y + gap

fig_h = 15.5 * (top - bottom + 6) / 100
fig, ax = plt.subplots(figsize=(9, max(11, fig_h)))
ax.set_xlim(0, 100)
ax.set_ylim(bottom - 3, 100)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(y_c, hh, text, fs=10.3, image=False):
    ax.add_patch(FancyBboxPatch((50 - w / 2, y_c - hh / 2), w, hh,
                                boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor=NAVY, edgecolor=EDGE, linewidth=1.1, zorder=3))
    if image:
        ax.text(50, y_c + hh / 2 - 2.0, text, ha="center", va="top", fontsize=fs,
                color="white", zorder=4, linespacing=1.4)
        icx = 50
        icy = y_c - hh / 2 + 1.5 + inset_ih / 2
        ax.imshow(img, extent=[icx - inset_iw / 2, icx + inset_iw / 2,
                               icy - inset_ih / 2, icy + inset_ih / 2],
                  zorder=4, aspect="auto")
        ax.add_patch(Rectangle((icx - inset_iw / 2, icy - inset_ih / 2), inset_iw, inset_ih,
                               fill=False, edgecolor="white", linewidth=1.2, zorder=5))
    else:
        ax.text(50, y_c, text, ha="center", va="center", fontsize=fs, color="white",
                zorder=4, linespacing=1.4)


for i, (y_c, hh, text) in enumerate(zip(centers, heights, stages)):
    box(y_c, hh, text, image=(i == n - 1 and has_inset))

for i in range(n - 1):
    y0 = centers[i] - heights[i] / 2
    y1 = centers[i + 1] + heights[i + 1] / 2
    ax.add_patch(FancyArrowPatch((50, y0), (50, y1), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.0, color=EDGE, shrinkA=0, shrinkB=0, zorder=2))

fig.savefig(FIGDIR / "fig_2_1_waveform_to_logmel.png", dpi=300, facecolor="white",
            bbox_inches="tight", pad_inches=0.25)
fig.savefig(FIGDIR / "fig_2_1_waveform_to_logmel.svg", facecolor="white",
            bbox_inches="tight", pad_inches=0.25)

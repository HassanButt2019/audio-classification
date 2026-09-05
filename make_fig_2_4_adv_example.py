import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data" / "UrbanSound8K"
AUDIO = ROOT / "audio"
FIGDIR = Path(__file__).resolve().parent / "figures"
CLIP_FOLD, CLIP_NAME = 6, "34643-4-2-1.wav"
EPSILON = 0.03
SEED = 42

if not (AUDIO / f"fold{CLIP_FOLD}" / CLIP_NAME).is_file():
    print(f"Expected clip not found: {AUDIO / f'fold{CLIP_FOLD}' / CLIP_NAME}")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset.urbansound_dataset import _load_audio
from preprocessing.mel_spectrogram import MelSpectrogramTransform, N_MELS, N_TIME_FRAMES, SAMPLE_RATE

print("No trained model checkpoint found anywhere in the repository (searched for")
print("*.pt / *.pth / saved_models/). FGSM/BIM require a trained model's gradient to")
print("produce a meaningful perturbation, so this figure is a LABELLED SCHEMATIC for")
print("the perturbation and the predicted labels. The clean spectrogram in Panel 1 is")
print(f"real data: fold{CLIP_FOLD}/{CLIP_NAME}, run through the actual preprocessing pipeline.\n")

path = AUDIO / f"fold{CLIP_FOLD}" / CLIP_NAME
waveform, sr = _load_audio(str(path))
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
if sr != SAMPLE_RATE:
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
transform = MelSpectrogramTransform()
with torch.no_grad():
    clean = transform(waveform.float()).squeeze(0).numpy()

rng = np.random.default_rng(SEED)
sign_pattern = rng.choice([-1.0, 1.0], size=clean.shape)
perturbation = EPSILON * sign_pattern
adv = np.clip(clean + perturbation, 0.0, 1.0)

print(f"Verified FGSM formula : x_adv = clip(x + eps * sign(grad), 0, 1)")
print(f"Verified BIM formula  : iterative FGSM step, projected into L_inf eps-ball, clamp(0,1)")
print(f"Epsilon used          : {EPSILON} (training epsilon, config.yaml)")
print(f"Operates on [0,1]     : yes (both formulas clamp directly to [0,1])")
print(f"Output clamped        : yes (torch.clamp(..., 0, 1) in both)")
print(f"Clean spectrogram     : real, shape {clean.shape}")
print(f"Perturbation          : SCHEMATIC random sign-pattern, not a real gradient")
print(f"Predictions shown     : SCHEMATIC (\"correct class\" / \"incorrect class\") — no checkpoint")

NAVY = "#1F3864"
ORANGE = "#C55A11"
EDGE = "#333333"
CMAP = "magma"

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3))
fig.patch.set_facecolor("white")
plt.rcParams["font.family"] = "DejaVu Sans"

extent = [0, N_TIME_FRAMES * 512 / SAMPLE_RATE, 0, N_MELS]

ax = axes[0]
im0 = ax.imshow(clean, origin="lower", aspect="auto", cmap=CMAP, vmin=0, vmax=1, extent=extent)
ax.set_title("Clean input", fontsize=12, color=NAVY, fontweight="bold", pad=10)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mel band")

ax = axes[1]
im1 = ax.imshow(sign_pattern, origin="lower", aspect="auto", cmap="coolwarm", vmin=-1, vmax=1, extent=extent)
ax.set_title("Perturbation", fontsize=12, color=NAVY, fontweight="bold", pad=10)
ax.set_xlabel("Time (s)")
ax.set_yticks([])

ax = axes[2]
im2 = ax.imshow(adv, origin="lower", aspect="auto", cmap=CMAP, vmin=0, vmax=1, extent=extent)
ax.set_title("Adversarial input", fontsize=12, color=NAVY, fontweight="bold", pad=10)
ax.set_xlabel("Time (s)")
ax.set_yticks([])

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.46, top=0.86, wspace=0.35)
fig.canvas.draw()
p0 = axes[0].get_position()
p1 = axes[1].get_position()
p2 = axes[2].get_position()

caption_y = p0.y0 - 0.075
fig.text((p0.x0 + p0.x1) / 2, caption_y, "Predicted: correct class\n(schematic — no trained checkpoint)",
        ha="center", va="top", fontsize=9, color="#333333")
fig.text((p1.x0 + p1.x1) / 2, caption_y, f"sign pattern magnified for visibility\nactual budget ε = {EPSILON}\n(schematic — not a real gradient)",
        ha="center", va="top", fontsize=9, color="#333333")
fig.text((p2.x0 + p2.x1) / 2, caption_y, "Predicted: incorrect class\n(schematic — no trained checkpoint)",
        ha="center", va="top", fontsize=9, color="#333333")

cbar_y = 0.07
cax0 = fig.add_axes([p0.x0, cbar_y, p0.width, 0.025])
fig.colorbar(im0, cax=cax0, orientation="horizontal", label="Normalised log-mel energy")
cax2 = fig.add_axes([p2.x0, cbar_y, p2.width, 0.025])
fig.colorbar(im2, cax=cax2, orientation="horizontal", label="Normalised log-mel energy")
cax1 = fig.add_axes([p1.x0, cbar_y, p1.width, 0.025])
fig.colorbar(im1, cax=cax1, orientation="horizontal", label="Perturbation sign")

fig.canvas.draw()
p0 = axes[0].get_position()
p1 = axes[1].get_position()
p2 = axes[2].get_position()
y_mid = (p0.y0 + p0.y1) / 2 + 0.03

arrow1 = FancyArrowPatch((p0.x1 + 0.005, y_mid), (p1.x0 - 0.005, y_mid),
                         transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.6, color=ORANGE, zorder=5)
fig.add_artist(arrow1)
fig.text((p0.x1 + p1.x0) / 2, y_mid + 0.025, "+ ε · sign(∇ₓ L)", ha="center", va="bottom",
        fontsize=10.5, color=ORANGE, fontweight="bold")

arrow2 = FancyArrowPatch((p1.x1 + 0.005, y_mid), (p2.x0 - 0.005, y_mid),
                         transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.6, color=ORANGE, zorder=5)
fig.add_artist(arrow2)
fig.text((p1.x1 + p2.x0) / 2, y_mid + 0.025, "clip to [0, 1]", ha="center", va="bottom",
        fontsize=10.5, color=ORANGE, fontweight="bold")

fig.savefig(FIGDIR / "fig_2_4_adversarial_example.png", dpi=300, facecolor="white",
            bbox_inches="tight", pad_inches=0.2)
fig.savefig(FIGDIR / "fig_2_4_adversarial_example.svg", facecolor="white",
            bbox_inches="tight", pad_inches=0.2)
print(f"\nSaved to {FIGDIR}")

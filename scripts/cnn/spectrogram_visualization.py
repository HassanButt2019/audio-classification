"""
Spectrogram Visualization — Clean vs FGSM vs BIM adversarial examples.

Produces a publication-ready 2×3 figure:
  Row 0  —  Actual spectrograms   : Clean | FGSM (ε=0.03) | BIM (ε=0.03)
  Row 1  —  Perturbation maps ×10 : (none) | FGSM Δ×10    | BIM Δ×10

Output: images/cnn/{class_name}/spectrogram_visualization.png

Usage
-----
  # defaults: fold=1, sample idx=0, best_fold1.pt checkpoint
  python scripts/cnn/spectrogram_visualization.py

  # pick a specific sample
  python scripts/cnn/spectrogram_visualization.py --fold 5 --sample-idx 3

  # use a different checkpoint
  python scripts/cnn/spectrogram_visualization.py --checkpoint saved_models/cnn/normal/best_fold3.pt
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless / server runs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchattacks

# ── project root (two levels above scripts/cnn/) ──────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from dataset.urbansound_dataset import UrbanSoundDataset, CLASS_NAMES
from models.cnn import UrbanSoundCNN

# ── constants ─────────────────────────────────────────────────────────────────
EPSILON = 0.03          # middle epsilon — visible but not extreme
AMP     = 10            # perturbation amplification factor for display
DPI     = 300           # high-res PNG for thesis

BIM_STEPS = 10          # BIM iteration count (matches eval setting)
BIM_ALPHA = EPSILON / BIM_STEPS


# ── helpers ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sample(data_root: str, fold: int, sample_idx: int):
    """Load one spectrogram from UrbanSound8K using the existing preprocessing pipeline.

    Returns:
        x_clean:    Float tensor [1, 1, 64, 128]  (batch × channel × mel × time)
        label:      Integer class ID
        class_name: Human-readable class string
    """
    ds = UrbanSoundDataset(root_dir=data_root, folds=[fold])
    if sample_idx >= len(ds):
        raise IndexError(
            f"sample_idx={sample_idx} out of range for fold {fold} "
            f"({len(ds)} samples available)"
        )
    spec, label = ds[sample_idx]          # spec: (1, 64, 128)
    x_clean    = spec.unsqueeze(0)        # → (1, 1, 64, 128)
    class_name = CLASS_NAMES[label]
    return x_clean, label, class_name


def load_baseline_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load the baseline CNN from a normal-training checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run normal training first:  python run_experiments.py --model cnn --mode normal"
        )
    model = UrbanSoundCNN(num_classes=10).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint val_acc={ckpt.get('val_acc', 0):.2f}%  "
          f"epoch={ckpt.get('epoch', '?')}")
    return model


def generate_adversarials(model, x_clean, label, device):
    """Generate FGSM and BIM adversarial examples and their perturbation deltas.

    torchattacks requires model.eval() — we set it here as a guard.

    Returns:
        x_fgsm, x_bim, delta_fgsm, delta_bim — all on the same device as x_clean
    """
    x = x_clean.to(device)
    y = torch.tensor([label], dtype=torch.long).to(device)

    fgsm_attack = torchattacks.FGSM(model, eps=EPSILON)
    bim_attack  = torchattacks.BIM(model, eps=EPSILON, alpha=BIM_ALPHA, steps=BIM_STEPS)

    model.eval()
    x_fgsm = fgsm_attack(x, y)
    x_bim  = bim_attack(x, y)

    delta_fgsm = x_fgsm - x
    delta_bim  = x_bim  - x

    return x_fgsm, x_bim, delta_fgsm, delta_bim


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach, move to CPU, and squeeze batch + channel dims → (64, 128)."""
    return tensor.detach().cpu().squeeze().numpy()


# ── plotting ──────────────────────────────────────────────────────────────────

def _add_subplot(ax, data, title, cmap, vmin, vmax, ylabel=None):
    """Render one spectrogram or perturbation map into an Axes."""
    im = ax.imshow(
        data,
        origin="lower",     # low-frequency mel bins at the bottom
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    ax.set_xlabel("Time frames", fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    else:
        ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def plot_and_save(
    clean_np, adv_fgsm_np, adv_bim_np,
    delta_fgsm_np, delta_bim_np,
    class_name: str,
    save_path: str,
) -> None:
    """Build the 2×3 figure and save it as a high-res PNG."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    # ── Row 0: actual spectrograms ────────────────────────────────────────────
    _add_subplot(axes[0, 0], clean_np,    "Clean",
                 cmap="magma", vmin=0, vmax=1,
                 ylabel="Spectrogram\n\nMel bins")
    _add_subplot(axes[0, 1], adv_fgsm_np, f"FGSM  (ε = {EPSILON})",
                 cmap="magma", vmin=0, vmax=1)
    _add_subplot(axes[0, 2], adv_bim_np,  f"BIM  (ε = {EPSILON},  steps = {BIM_STEPS})",
                 cmap="magma", vmin=0, vmax=1)

    # ── Row 1: perturbation maps ──────────────────────────────────────────────
    # Use a symmetric diverging range so red = positive, blue = negative noise
    amp_max = float(max(
        np.abs(delta_fgsm_np * AMP).max(),
        np.abs(delta_bim_np  * AMP).max(),
        1e-6,
    ))

    zeros = np.zeros_like(clean_np)

    # [1,0] — blank (no perturbation on clean)
    _add_subplot(axes[1, 0], zeros, "No Perturbation",
                 cmap="magma", vmin=0, vmax=1,
                 ylabel=f"Perturbation (×{AMP})\n\nMel bins")

    # [1,1] — FGSM delta amplified
    _add_subplot(axes[1, 1], delta_fgsm_np * AMP,
                 f"FGSM  Δ  (×{AMP})",
                 cmap="seismic", vmin=-amp_max, vmax=amp_max)

    # [1,2] — BIM delta amplified
    _add_subplot(axes[1, 2], delta_bim_np * AMP,
                 f"BIM  Δ  (×{AMP})",
                 cmap="seismic", vmin=-amp_max, vmax=amp_max)

    # ── overall title ─────────────────────────────────────────────────────────
    fig.suptitle(
        f"Adversarial Perturbation Visualisation\n"
        f"Class: {class_name.replace('_', ' ').title()}   "
        f"|   Baseline CNN   "
        f"|   ε = {EPSILON}",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {save_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate clean vs FGSM vs BIM spectrogram comparison figure"
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(PROJECT_ROOT, "saved_models", "cnn", "normal", "best_fold1.pt"),
        help="Path to baseline CNN checkpoint (default: saved_models/cnn/normal/best_fold1.pt)",
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join(PROJECT_ROOT, "data", "UrbanSound8K"),
        help="Path to UrbanSound8K root directory",
    )
    parser.add_argument(
        "--fold", type=int, default=1,
        help="Fold to pick the sample from (1–10, default: 1)",
    )
    parser.add_argument(
        "--sample-idx", type=int, default=0,
        help="Index of the sample within the fold (default: 0)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(PROJECT_ROOT, "images", "cnn"),
        help="Base output directory; class name is appended automatically",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print("=" * 60)
    print(" Spectrogram Visualization")
    print("=" * 60)
    print(f" Device     : {device}")
    print(f" Checkpoint : {args.checkpoint}")
    print(f" Data root  : {args.data_root}")
    print(f" Fold       : {args.fold}   sample idx: {args.sample_idx}")
    print(f" ε          : {EPSILON}   BIM steps: {BIM_STEPS}   amplify: ×{AMP}")
    print("=" * 60)

    # ── Step 1: load and preprocess one sample ────────────────────────────────
    print("\n[1] Loading sample...")
    x_clean, label, class_name = load_sample(args.data_root, args.fold, args.sample_idx)
    print(f"    Class : {label}  →  {class_name}")
    print(f"    Shape : {tuple(x_clean.shape)}   range [{x_clean.min():.3f}, {x_clean.max():.3f}]")

    # ── Step 2: load model and generate perturbations ─────────────────────────
    print("\n[2] Loading baseline model...")
    model = load_baseline_model(args.checkpoint, device)

    print(f"\n[3] Generating adversarial examples (ε={EPSILON})...")
    x_fgsm, x_bim, delta_fgsm, delta_bim = generate_adversarials(
        model, x_clean, label, device
    )

    # ── Step 3: convert to numpy ──────────────────────────────────────────────
    print("\n[4] Converting tensors to numpy...")
    clean_np      = to_numpy(x_clean)
    adv_fgsm_np   = to_numpy(x_fgsm)
    adv_bim_np    = to_numpy(x_bim)
    delta_fgsm_np = to_numpy(delta_fgsm)
    delta_bim_np  = to_numpy(delta_bim)

    print(f"    Clean   range : [{clean_np.min():.4f},  {clean_np.max():.4f}]")
    print(f"    FGSM    range : [{adv_fgsm_np.min():.4f},  {adv_fgsm_np.max():.4f}]")
    print(f"    BIM     range : [{adv_bim_np.min():.4f},  {adv_bim_np.max():.4f}]")
    print(f"    |Δ_FGSM| max : {np.abs(delta_fgsm_np).max():.5f}")
    print(f"    |Δ_BIM|  max : {np.abs(delta_bim_np).max():.5f}")

    # ── Step 4: plot and save ─────────────────────────────────────────────────
    save_path = os.path.join(args.out_dir, class_name, "spectrogram_visualization.png")
    print(f"\n[5] Plotting 2×3 grid → {save_path}")
    plot_and_save(
        clean_np, adv_fgsm_np, adv_bim_np,
        delta_fgsm_np, delta_bim_np,
        class_name, save_path,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

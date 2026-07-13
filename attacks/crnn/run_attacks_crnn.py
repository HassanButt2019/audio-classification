"""
CRNN-aware FGSM and BIM attack evaluation.

Why this module exists (not the shared attacks/run_attacks.py)
--------------------------------------------------------------
PyTorch's cuDNN RNN backend (which backs all GRU/LSTM layers on NVIDIA GPUs)
only supports backpropagation when the model is in TRAINING mode.

The generic evaluate_fgsm / evaluate_bim in attacks/fgsm.py and attacks/bim.py
assert model.eval() — correct for CNN and VGGish (conv layers have no such
constraint) but fatal for any model with recurrent layers on CUDA:

    RuntimeError: cudnn RNN backward can only be called in training mode

Fix applied here:
    torch.backends.cudnn.flags(enabled=False)

This context manager disables the optimised cuDNN RNN kernel for the duration
of the attack-generation call, falling back to PyTorch's own CUDA
implementation which supports backward in eval mode.

Effect on evaluation correctness:
  - BatchNorm still uses running statistics (model is in .eval()).
  - Dropout is still disabled (model is in .eval()).
  - Only the RNN backward kernel changes — the forward pass is identical.
  - Speed: ~20-30 % slower per batch for the GRU backward.  Acceptable for
    evaluation (not training).

This module has NO effect on the shared CNN/VGGish attack pipeline.
"""

import os
import sys

import torch
import torchattacks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dataset.urbansound_dataset import get_fold_dataloaders
from preprocessing.mel_spectrogram import MelSpectrogramTransform


# ── per-batch attack evaluators ───────────────────────────────────────────────

def evaluate_fgsm_crnn(model, test_loader, epsilon, device):
    """Evaluate CRNN accuracy under FGSM with cuDNN-safe backward.

    Args:
        model:       CRNN model — must already be on `device`.
        test_loader: DataLoader for the test fold.
        epsilon:     L∞ perturbation budget.
        device:      torch.device.

    Returns:
        Adversarial accuracy (0–100).
    """
    model.eval()
    attack = torchattacks.FGSM(model, eps=epsilon)

    correct = 0
    total   = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.backends.cudnn.flags(enabled=False):
            adv_images = attack(images, labels)

        with torch.no_grad():
            outputs   = model(adv_images)
            predicted = outputs.argmax(dim=1)

        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


def evaluate_bim_crnn(model, test_loader, epsilon, alpha, steps, device):
    """Evaluate CRNN accuracy under BIM with cuDNN-safe backward.

    Args:
        model:       CRNN model — must already be on `device`.
        test_loader: DataLoader for the test fold.
        epsilon:     Total L∞ perturbation budget.
        alpha:       Per-step perturbation size (typically epsilon / steps).
        steps:       Number of iterative steps.
        device:      torch.device.

    Returns:
        Adversarial accuracy (0–100).
    """
    model.eval()
    attack = torchattacks.BIM(model, eps=epsilon, alpha=alpha, steps=steps)

    correct = 0
    total   = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.backends.cudnn.flags(enabled=False):
            adv_images = attack(images, labels)

        with torch.no_grad():
            outputs   = model(adv_images)
            predicted = outputs.argmax(dim=1)

        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


# ── full 10-fold runners ──────────────────────────────────────────────────────

def run_fgsm_all_folds_crnn(model_class, saved_models_dir, data_root,
                             device, epsilons, batch_size=128, num_workers=4):
    """Run FGSM attack on all 10 CRNN folds with cuDNN-safe backward.

    Args:
        model_class:      CRNN class (used to instantiate model per fold).
        saved_models_dir: Directory containing best_fold{k}.pt checkpoints.
        data_root:        Path to UrbanSound8K root.
        device:           torch.device.
        epsilons:         List of epsilon values, e.g. [0.01, 0.03, 0.1].
        batch_size:       DataLoader batch size.
        num_workers:      DataLoader workers.

    Returns:
        dict  fold_k -> {eps_e: adversarial_accuracy}
    """
    missing = [
        f"best_fold{f}.pt"
        for f in range(1, 11)
        if not os.path.exists(os.path.join(saved_models_dir, f"best_fold{f}.pt"))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints in '{saved_models_dir}': {missing}\n"
            "Train all 10 folds first."
        )

    transform = MelSpectrogramTransform()
    results   = {}

    for fold in range(1, 11):
        print(f"\n[CRNN FGSM] Attacking Fold {fold}...")

        ckpt_path  = os.path.join(saved_models_dir, f"best_fold{fold}.pt")
        model      = model_class().to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        _, test_loader = get_fold_dataloaders(
            root_dir    = data_root,
            test_fold   = fold,
            batch_size  = batch_size,
            num_workers = num_workers,
            transform   = transform,
        )

        fold_results = {}
        for eps in epsilons:
            adv_acc = evaluate_fgsm_crnn(model, test_loader, eps, device)
            fold_results[f"eps_{eps}"] = adv_acc
            print(f"  Fold {fold} | ε={eps} | Adv Acc: {adv_acc:.2f}%")

        results[f"fold_{fold}"] = fold_results

    return results


def run_bim_all_folds_crnn(model_class, saved_models_dir, data_root,
                            device, epsilons, steps=10, batch_size=128, num_workers=4):
    """Run BIM attack on all 10 CRNN folds with cuDNN-safe backward.

    Args:
        model_class:      CRNN class.
        saved_models_dir: Directory containing best_fold{k}.pt checkpoints.
        data_root:        Path to UrbanSound8K root.
        device:           torch.device.
        epsilons:         List of epsilon values.
        steps:            Number of iterative BIM steps (default: 10).
        batch_size:       DataLoader batch size.
        num_workers:      DataLoader workers.

    Returns:
        dict  fold_k -> {eps_e: adversarial_accuracy}
    """
    missing = [
        f"best_fold{f}.pt"
        for f in range(1, 11)
        if not os.path.exists(os.path.join(saved_models_dir, f"best_fold{f}.pt"))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints in '{saved_models_dir}': {missing}\n"
            "Train all 10 folds first."
        )

    transform = MelSpectrogramTransform()
    results   = {}

    for fold in range(1, 11):
        print(f"\n[CRNN BIM] Attacking Fold {fold}...")

        ckpt_path  = os.path.join(saved_models_dir, f"best_fold{fold}.pt")
        model      = model_class().to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        _, test_loader = get_fold_dataloaders(
            root_dir    = data_root,
            test_fold   = fold,
            batch_size  = batch_size,
            num_workers = num_workers,
            transform   = transform,
        )

        fold_results = {}
        for eps in epsilons:
            alpha   = eps / steps
            adv_acc = evaluate_bim_crnn(model, test_loader, eps, alpha, steps, device)
            fold_results[f"eps_{eps}"] = adv_acc
            print(f"  Fold {fold} | ε={eps}  α={alpha:.4f}  steps={steps} | Adv Acc: {adv_acc:.2f}%")

        results[f"fold_{fold}"] = fold_results

    return results

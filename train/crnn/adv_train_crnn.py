"""
CRNN adversarial training — 10-fold cross-validation on UrbanSound8K.

What differs from CNN adversarial training (train/adversarial_train.py)
------------------------------------------------------------------------
  1. Gradient clipping — clip_grad_norm_(model.parameters(), max_norm=1.0)
     applied before every optimizer.step().

     Reason: BPTT through the GRU during the backward pass of adversarial
     training can produce large gradients, especially in early epochs when
     the model is simultaneously adapting to clean and adversarial distributions.
     This is an architecture necessity, not a hyperparameter choice — it does
     not affect comparison validity with CNN/VGGish AT.

  2. Proper val split — get_fold_dataloaders_with_val (10 % of train folds)
     is used so the test fold is not seen during training decisions.

  3. Checkpoint criterion — saved by adversarial val accuracy (not clean val
     accuracy as in CNN/VGGish AT).  This matches the CRNN AT algorithm spec.

  4. 50 epochs (CNN AT uses 30).

  5. Fresh Orthogonal (GRU) + Xavier (Conv/Linear) weight init per fold.

Usage
-----
  python train/crnn/adv_train_crnn.py --attack fgsm
  python train/crnn/adv_train_crnn.py --attack bim
  python train/crnn/adv_train_crnn.py --attack fgsm --quick 100   # smoke-test
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.crnn import CRNN, crnn_weight_init
from preprocessing.mel_spectrogram import MelSpectrogramTransform
from dataset.urbansound_dataset import get_fold_dataloaders_with_val
from attacks.fgsm import get_fgsm_attack
from attacks.bim  import get_bim_attack
from train.train import get_device, evaluate
from config_loader import get_crnn_adv_fgsm_config, get_crnn_adv_bim_config


# ── attack helper ──────────────────────────────────────────────────────────────

def _build_attack(model, cfg):
    attack_type = cfg["attack_type"]
    epsilon     = cfg["adv_epsilon"]
    if attack_type == "fgsm":
        return get_fgsm_attack(model, epsilon)
    elif attack_type == "bim":
        steps = cfg.get("bim_steps", 7)
        return get_bim_attack(model, epsilon, epsilon / steps, steps)
    raise ValueError(f"Unknown attack_type: {attack_type!r}")


# ── adversarial val evaluation ─────────────────────────────────────────────────

def _evaluate_adversarial(model, loader, criterion, attack, device):
    """Evaluate on attack-perturbed inputs.  Model stays in eval mode throughout."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    for specs, labels in loader:
        specs  = specs.to(device)
        labels = labels.to(device)
        with torch.backends.cudnn.flags(enabled=False):
            x_adv = attack(specs, labels).detach()
        with torch.no_grad():
            logits = model(x_adv)
            loss   = criterion(logits, labels)
        total_loss += loss.item() * specs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += specs.size(0)
    return total_loss / total, 100.0 * correct / total


# ── one AT epoch with gradient clipping ───────────────────────────────────────

def _adv_train_epoch(model, loader, criterion, optimizer, device, cfg):
    """One epoch of adversarial training with gradient clipping.

    For each batch:
      1. Switch to eval for deterministic attack generation (BN uses running stats).
      2. Generate adversarial examples for the first n_adv samples.
      3. Switch back to train; forward on the mixed batch; backward.
      4. Clip gradients (max_norm=1.0) — prevents BPTT gradient explosions.
      5. Optimizer step.

    Returns:
        (average_loss, accuracy_percent)
    """
    total_loss = 0.0
    correct    = 0
    total      = 0

    attack = _build_attack(model, cfg)

    for specs, labels in loader:
        specs  = specs.to(device)
        labels = labels.to(device)

        batch_size = specs.size(0)
        n_adv      = max(1, int(batch_size * cfg["adv_ratio"]))

        # torchattacks requires eval mode so BN uses running stats.
        # cudnn.flags(enabled=False) lets GRU backward run in eval mode
        # (cuDNN's optimised RNN kernel blocks backward in eval mode on GPU).
        model.eval()
        with torch.backends.cudnn.flags(enabled=False):
            adv_specs = attack(specs[:n_adv], labels[:n_adv]).detach()
        model.train()

        mixed  = torch.cat([adv_specs, specs[n_adv:]], dim=0)
        logits = model(mixed)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        # Clip gradients before optimizer step — essential for GRU BPTT stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += batch_size

    return total_loss / total, 100.0 * correct / total


# ── fold runner ────────────────────────────────────────────────────────────────

def adv_train_fold(fold: int, cfg: dict) -> dict:
    """Adversarially train CRNN from scratch for one fold.

    Checkpoint saved by best adversarial val accuracy (not clean val accuracy).
    Gradient clipping (max_norm=1.0) applied every step.

    Args:
        fold: Test fold (1–10).
        cfg:  Config from get_crnn_adv_fgsm_config() or get_crnn_adv_bim_config().
              Required extra keys vs baseline: attack_type, adv_epsilon, adv_ratio.
              BIM also needs bim_steps.

    Returns:
        {best_val_acc, history, checkpoint_path}
        best_val_acc here is the best adversarial val accuracy.
    """
    device      = get_device()
    attack_type = cfg["attack_type"]
    epochs      = cfg["epochs"]

    print(f"\n{'='*66}")
    print(f" [CRNN AT / {attack_type.upper()}]  Fold {fold}/10  |  device: {device}")
    print(f" ε={cfg['adv_epsilon']}  ratio={cfg['adv_ratio']}  epochs={epochs}", end="")
    if attack_type == "bim":
        print(f"  steps={cfg.get('bim_steps', 7)}", end="")
    print()
    print(f"{'='*66}")

    transform = MelSpectrogramTransform()
    train_loader, val_loader, _ = get_fold_dataloaders_with_val(
        root_dir             = cfg["data_root"],
        test_fold            = fold,
        transform            = transform,
        val_fraction         = cfg["val_fraction"],
        batch_size           = cfg["batch_size"],
        num_workers          = cfg["num_workers"],
        seed                 = 42,
        max_samples_per_fold = cfg.get("max_samples_per_fold"),
    )

    model = CRNN(num_classes=cfg["num_classes"], dropout=cfg["dropout"]).to(device)
    model.apply(crnn_weight_init)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    os.makedirs(cfg["save_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["save_dir"], f"best_fold{fold}.pt")

    # Build a dedicated val attack for checkpoint criterion evaluation
    val_attack = _build_attack(model, cfg)

    best_adv_val_acc = -1.0
    history = {
        "train_loss":   [], "train_acc":   [],
        "val_loss":     [], "val_acc":     [],
        "adv_val_loss": [], "adv_val_acc": [],
    }

    fold_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = _adv_train_epoch(
            model, train_loader, criterion, optimizer, device, cfg
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        adv_val_loss, adv_val_acc = _evaluate_adversarial(
            model, val_loader, criterion, val_attack, device
        )

        elapsed = time.time() - t0
        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc,  4))
        history["val_loss"].append(round(val_loss,    6))
        history["val_acc"].append(round(val_acc,      4))
        history["adv_val_loss"].append(round(adv_val_loss, 6))
        history["adv_val_acc"].append(round(adv_val_acc,  4))

        improved = adv_val_acc > best_adv_val_acc
        if improved:
            best_adv_val_acc = adv_val_acc
            torch.save(
                {
                    "epoch":            epoch,
                    "fold":             fold,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          val_acc,
                    "adv_val_acc":      adv_val_acc,
                    "val_loss":         val_loss,
                    "config":           cfg,
                },
                ckpt_path,
            )

        marker = "  * saved" if improved else ""
        print(
            f" Epoch {epoch:>3}/{epochs} | "
            f"train {train_loss:.4f}/{train_acc:5.1f}% | "
            f"clean {val_loss:.4f}/{val_acc:5.1f}% | "
            f"adv {adv_val_loss:.4f}/{adv_val_acc:5.1f}%  "
            f"[{elapsed:.1f}s]{marker}"
        )

    fold_wall = time.time() - fold_start
    print(f"\n Best adv val acc for fold {fold}: {best_adv_val_acc:.2f}%  "
          f"wall: {fold_wall/3600:.2f}h")

    return {
        "best_val_acc":     best_adv_val_acc,   # key expected by run_experiments.py
        "best_adv_val_acc": best_adv_val_acc,
        "history":          history,
        "checkpoint_path":  ckpt_path,
        "wall_s":           round(fold_wall, 1),
    }


# ── full 10-fold run ───────────────────────────────────────────────────────────

def run_all_folds(cfg: dict, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    attack_type = cfg["attack_type"]

    config_snapshot = {k: v for k, v in cfg.items() if k != "data_root"}
    config_snapshot["model"] = "crnn"
    config_snapshot["mode"]  = f"adv_train_{attack_type}"
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    total_start  = time.time()
    fold_results = []
    adv_accs     = []

    for fold in range(1, 11):
        result = adv_train_fold(fold, cfg)
        fold_results.append(result)
        adv_accs.append(result["best_adv_val_acc"])
        with open(os.path.join(results_dir, f"fold_{fold}_results.json"), "w") as f:
            json.dump(result, f, indent=2)

    total_wall = time.time() - total_start
    mean_acc   = sum(adv_accs) / len(adv_accs)
    std_acc    = math.sqrt(sum((a - mean_acc) ** 2 for a in adv_accs) / len(adv_accs))

    h, rem   = divmod(int(total_wall), 3600)
    mn, sc   = divmod(rem, 60)
    wall_str = f"{h:02d}h {mn:02d}m {sc:02d}s"

    cv_summary = {
        "model":     "crnn",
        "mode":      f"adv_train_{attack_type}",
        "wall_time": wall_str,
        "config": {"epochs": cfg["epochs"], "batch_size": cfg["batch_size"],
                   "lr": cfg["lr"], "attack_type": attack_type,
                   "adv_epsilon": cfg["adv_epsilon"], "adv_ratio": cfg["adv_ratio"]},
        "per_fold": [
            {"fold": r["best_val_acc"], "adv_val_acc": round(r["best_adv_val_acc"] / 100, 4)}
            for r in fold_results
        ],
        "mean_adv_val_acc": round(mean_acc / 100, 4),
        "std_adv_val_acc":  round(std_acc  / 100, 4),
    }
    with open(os.path.join(results_dir, "cv_summary.json"), "w") as f:
        json.dump(cv_summary, f, indent=2)

    print(f"\n CRNN AT-{attack_type.upper()} complete")
    print(f" Mean adv val acc : {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f" Wall time        : {wall_str}")
    print(f" Results saved    : {results_dir}")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    parser = argparse.ArgumentParser(description="CRNN adversarial training — 10-fold CV")
    parser.add_argument("--attack", choices=["fgsm", "bim"], default="fgsm",
                        help="Attack type for adversarial training (default: fgsm)")
    parser.add_argument("--quick", type=int, default=0, metavar="N",
                        help="Smoke-test: limit each fold to N samples (0 = full dataset)")
    args = parser.parse_args()

    if args.attack == "fgsm":
        cfg = get_crnn_adv_fgsm_config()
    else:
        cfg = get_crnn_adv_bim_config()

    if args.quick:
        cfg["max_samples_per_fold"] = args.quick

    results_dir = os.path.join(_ROOT, "results", "crnn", f"adv_train_{args.attack}")
    run_all_folds(cfg, results_dir)

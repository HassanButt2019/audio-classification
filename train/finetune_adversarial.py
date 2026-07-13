import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.urbansound_dataset import get_fold_dataloaders
from models.cnn import UrbanSoundCNN
from attacks.fgsm import get_fgsm_attack
from train.train import get_device, evaluate
from train.adversarial_train import adv_train_one_epoch
from config_loader import get_finetune_fgsm_config, get_finetune_bim_config


# ── configs ───────────────────────────────────────────────────────────────────

FINETUNE_FGSM_CONFIG = get_finetune_fgsm_config()
FINETUNE_BIM_CONFIG  = get_finetune_bim_config()

# backward-compat alias
FINETUNE_CONFIG = FINETUNE_FGSM_CONFIG


# ── adversarial validation ────────────────────────────────────────────────────

def _evaluate_adversarial(model, loader, criterion, attack, device):
    """Evaluate model on FGSM-perturbed val set.

    Attack generation requires a gradient pass; the final model forward uses
    no_grad to avoid building an unnecessary computation graph.

    Returns:
        (average_loss, accuracy_percent)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for specs, labels in loader:
        specs  = specs.to(device)
        labels = labels.to(device)

        # torchattacks computes ∇_x loss internally — must NOT be inside no_grad
        x_adv = attack(specs, labels).detach()

        with torch.no_grad():
            logits = model(x_adv)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * specs.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += specs.size(0)

    return total_loss / total, 100.0 * correct / total


# ── fold runner ───────────────────────────────────────────────────────────────

def finetune_fold(
    fold:            int,
    pretrained_ckpt: str,
    cfg:             dict = FINETUNE_FGSM_CONFIG,
    model_class            = UrbanSoundCNN,
    transform              = None,
) -> dict:
    """Fine-tune a normally-trained checkpoint with adversarial examples.

    FGSM algorithm:
        1. Load baseline checkpoint — clean features are already learned.
        2. Adam optimizer at finetune_lr (10x lower than AT).
        3. For each batch: mix clean + FGSM adversarial at adv_ratio fraction
           (same strategy as adv_train_one_epoch — batch size stays fixed).
        4. After each epoch evaluate BOTH clean and adversarial val accuracy.
        5. Save checkpoint when adversarial val accuracy improves.

    BIM falls back to the original strategy (save by clean val accuracy).

    Args:
        fold:            Test fold (1-10).
        pretrained_ckpt: Path to the .pt checkpoint from normal training.
        cfg:             Must include attack_type, adv_epsilon, adv_ratio,
                         finetune_epochs, finetune_lr, save_dir.
                         BIM also needs bim_steps.
        model_class:     Model class matching the pretrained checkpoint.

    Returns:
        {"best_val_acc", "best_adv_val_acc", "history", "checkpoint_path"}
        For FGSM, best_val_acc reflects the best adversarial val accuracy.
    """
    device      = get_device()
    epochs      = cfg.get("finetune_epochs", 15)
    attack_type = cfg.get("attack_type", "fgsm")
    epsilon     = cfg["adv_epsilon"]

    print(f"\n{'='*60}")
    print(f" [FINETUNE / {attack_type.upper()}] Fold {fold}/10  |  device: {device}")
    print(f" Pretrained : {pretrained_ckpt}")
    print(
        f" ε={epsilon}  ratio={cfg['adv_ratio']}"
        f"  epochs={epochs}  lr={cfg['finetune_lr']}",
        end="",
    )
    if attack_type == "bim":
        print(f"  steps={cfg.get('bim_steps', 7)}", end="")
    print()
    print(f"{'='*60}")

    train_loader, val_loader = get_fold_dataloaders(
        root_dir             = cfg["data_root"],
        test_fold            = fold,
        batch_size           = cfg["batch_size"],
        num_workers          = cfg["num_workers"],
        max_samples_per_fold = cfg.get("max_samples_per_fold"),
        transform            = transform,
    )

    # ── load pretrained baseline ──────────────────────────────────────────────
    model      = model_class(num_classes=cfg["num_classes"], dropout=cfg["dropout"]).to(device)
    checkpoint = torch.load(pretrained_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f" Loaded pretrained weights (val acc: {checkpoint.get('val_acc', 0):.2f}%)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("finetune_lr", 0.0001))

    # FGSM attack for adversarial val evaluation.
    # torchattacks holds a live reference to the model object, so it always
    # uses the current epoch's weights without being rebuilt each epoch.
    fgsm_val_attack = get_fgsm_attack(model, epsilon) if attack_type == "fgsm" else None

    os.makedirs(cfg["save_dir"], exist_ok=True)
    checkpoint_path = os.path.join(cfg["save_dir"], f"best_fold{fold}.pt")

    best_adv_val_acc   = -1.0   # FGSM: checkpoint criterion
    best_clean_val_acc = -1.0   # BIM:  checkpoint criterion
    history = {
        "train_loss":   [], "train_acc":   [],
        "val_loss":     [], "val_acc":     [],
        "adv_val_loss": [], "adv_val_acc": [],   # populated for FGSM only
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── train: same mix strategy as AT (adv_ratio, fixed batch size) ─────
        train_loss, train_acc = adv_train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg
        )

        # ── evaluate clean val ────────────────────────────────────────────────
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # ── evaluate adversarial val (FGSM path only) ─────────────────────────
        if attack_type == "fgsm":
            adv_val_loss, adv_val_acc = _evaluate_adversarial(
                model, val_loader, criterion, fgsm_val_attack, device
            )
            history["adv_val_loss"].append(adv_val_loss)
            history["adv_val_acc"].append(adv_val_acc)
            improved = adv_val_acc > best_adv_val_acc
        else:
            adv_val_loss = adv_val_acc = float("nan")
            improved = val_acc > best_clean_val_acc

        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── checkpoint: save when the criterion metric improves ───────────────
        if improved:
            if attack_type == "fgsm":
                best_adv_val_acc = adv_val_acc
            else:
                best_clean_val_acc = val_acc
            torch.save(
                {
                    "epoch":            epoch,
                    "fold":             fold,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          val_acc,
                    "adv_val_acc":      adv_val_acc,
                    "val_loss":         val_loss,
                    "adv_val_loss":     adv_val_loss,
                    "config":           cfg,
                    "pretrained_ckpt":  pretrained_ckpt,
                },
                checkpoint_path,
            )

        marker = "  ✓ saved" if improved else ""

        if attack_type == "fgsm":
            print(
                f" Epoch {epoch:>3}/{epochs} | "
                f"train {train_loss:.4f}/{train_acc:5.1f}% | "
                f"clean {val_loss:.4f}/{val_acc:5.1f}% | "
                f"adv {adv_val_loss:.4f}/{adv_val_acc:5.1f}%  "
                f"[{elapsed:.1f}s]{marker}"
            )
        else:
            print(
                f" Epoch {epoch:>3}/{epochs} | "
                f"train loss {train_loss:.4f}  acc {train_acc:5.1f}% | "
                f"val loss {val_loss:.4f}  acc {val_acc:5.1f}%  "
                f"[{elapsed:.1f}s]{marker}"
            )

    best_for_return = best_adv_val_acc if attack_type == "fgsm" else best_clean_val_acc
    print(f"\n Best {'adv ' if attack_type == 'fgsm' else ''}val acc for fold {fold}: {best_for_return:.2f}%")

    return {
        "best_val_acc":     best_for_return,   # key expected by run_experiments.py
        "best_adv_val_acc": best_adv_val_acc,
        "history":          history,
        "checkpoint_path":  checkpoint_path,
    }

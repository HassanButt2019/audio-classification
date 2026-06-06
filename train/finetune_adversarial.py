import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.urbansound_dataset import get_fold_dataloaders
from models.cnn import UrbanSoundCNN
from train.train import CONFIG, get_device, evaluate
from train.adversarial_train import adv_train_one_epoch


# ── configs ───────────────────────────────────────────────────────────────────

FINETUNE_FGSM_CONFIG = {
    **CONFIG,
    "attack_type":    "fgsm",
    "finetune_epochs": 15,
    "finetune_lr":    0.0001,
    "adv_epsilon":    0.03,
    "adv_ratio":      0.5,
}

FINETUNE_BIM_CONFIG = {
    **CONFIG,
    "attack_type":    "bim",
    "finetune_epochs": 15,
    "finetune_lr":    0.0001,
    "adv_epsilon":    0.03,
    "adv_ratio":      0.5,
    "bim_steps":      7,
}

# backward-compat alias
FINETUNE_CONFIG = FINETUNE_FGSM_CONFIG


# ── fold runner ───────────────────────────────────────────────────────────────

def finetune_fold(
    fold:            int,
    pretrained_ckpt: str,
    cfg:             dict = FINETUNE_FGSM_CONFIG,
    model_class            = UrbanSoundCNN,
) -> dict:
    """Fine-tune a normally-trained checkpoint with adversarial examples.

    Strategy:
        1. Load weights from the normal-training checkpoint — clean feature
           representations are already learned, we just harden them.
        2. Run adversarial training (FGSM or BIM) for fewer epochs at a lower
           LR so pretrained weights are not destroyed.
        3. Save the best checkpoint by validation accuracy.

    Args:
        fold:            Test fold (1–10).
        pretrained_ckpt: Path to the .pt checkpoint from normal training.
        cfg:             Must include attack_type, finetune_epochs, finetune_lr,
                         adv_epsilon, adv_ratio, save_dir.
                         If attack_type == "bim" also needs bim_steps.
        model_class:     Model class matching the pretrained checkpoint.

    Returns:
        {"best_val_acc", "history", "checkpoint_path"}
    """
    device      = get_device()
    epochs      = cfg["finetune_epochs"]
    attack_type = cfg.get("attack_type", "fgsm")

    print(f"\n{'='*60}")
    print(f" [FINETUNE / {attack_type.upper()}] Fold {fold}/10  |  device: {device}")
    print(f" Pretrained : {pretrained_ckpt}")
    print(f" ε={cfg['adv_epsilon']}  ratio={cfg['adv_ratio']}  epochs={epochs}  lr={cfg['finetune_lr']}", end="")
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
    )

    # ── load pretrained weights ───────────────────────────────────────────────
    model      = model_class(num_classes=cfg["num_classes"], dropout=cfg["dropout"]).to(device)
    checkpoint = torch.load(pretrained_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f" Loaded pretrained weights (val acc: {checkpoint.get('val_acc', 0):.2f}%)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["finetune_lr"])

    os.makedirs(cfg["save_dir"], exist_ok=True)
    checkpoint_path = os.path.join(cfg["save_dir"], f"best_fold{fold}.pt")

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = adv_train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":            epoch,
                    "fold":             fold,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          val_acc,
                    "val_loss":         val_loss,
                    "config":           cfg,
                    "pretrained_ckpt":  pretrained_ckpt,
                },
                checkpoint_path,
            )

        marker = "  ✓ saved" if improved else ""
        print(
            f" Epoch {epoch:>3}/{epochs} | "
            f"train loss {train_loss:.4f}  acc {train_acc:5.1f}% | "
            f"val loss {val_loss:.4f}  acc {val_acc:5.1f}%  "
            f"[{elapsed:.1f}s]{marker}"
        )

    print(f"\n Best val acc for fold {fold}: {best_val_acc:.2f}%")
    return {"best_val_acc": best_val_acc, "history": history, "checkpoint_path": checkpoint_path}

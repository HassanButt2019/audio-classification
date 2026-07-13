"""
CRNN baseline training — 10-fold cross-validation on UrbanSound8K.

Differences from CNN training (train/train.py)
-----------------------------------------------
  - Architecture  : CRNN (4-block CNN front-end + Bidirectional GRU + FC head)
  - Weight init   : Xavier for Conv2d/Linear; Orthogonal for GRU (fresh each fold)
  - Val split     : 10 % of training folds held out for early stopping
  - LR scheduler  : ReduceLROnPlateau(mode='min', factor=0.5, patience=5)
  - Early stopping: patience=15 epochs on val_acc
  - Epochs        : 50 (not 30)
  - No weight_decay (matches CNN; VGGish uses 1e-4)

Differences from VGGish training (train/train_vggish.py)
---------------------------------------------------------
  - Transform     : MelSpectrogramTransform  →  input [1, 64, 128]
                    (VGGish uses VGGishMelSpectrogramTransform → [1, 64, 96])
  - No weight_decay
  - GRU layers require Orthogonal init (not Xavier)

CRITICAL — run this before adv_train_crnn.py or finetune_crnn.py.
           The 10 checkpoints in saved_models/crnn/normal/ are the required
           starting point for adversarial fine-tuning (AFT).
           If they are overwritten you must retrain from scratch.

Usage
-----
  python train/crnn/train_crnn.py
  python train/crnn/train_crnn.py --quick 100    # smoke-test, 100 samples/fold
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
from train.train import get_device
from config_loader import get_crnn_config


# ── one epoch ──────────────────────────────────────────────────────────────────

def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, 100.0 * correct / total


def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += x.size(0)
    return total_loss / total, 100.0 * correct / total


# ── one fold ───────────────────────────────────────────────────────────────────

def train_fold(fold: int, cfg: dict) -> dict:
    """Train CRNN baseline for one fold of 10-fold cross-validation.

    Args:
        fold: Test fold (1–10).
        cfg:  Config dict from get_crnn_config() or a compatible dict.
              Required keys: data_root, save_dir, batch_size, num_workers, lr,
              epochs, dropout, num_classes, val_fraction, early_stop_patience,
              lr_scheduler_factor, lr_scheduler_patience.

    Returns:
        {fold, test_acc, best_val_acc, wall_s, history, checkpoint_path}
    """
    device    = get_device()
    transform = MelSpectrogramTransform()

    print(f"\n{'='*66}")
    print(f" CRNN  |  Fold {fold}/10  |  device: {device}  |  epochs: {cfg['epochs']}")
    print(f"{'='*66}")

    # ── data — proper 3-way split: train / val / test ─────────────────────────
    train_loader, val_loader, test_loader = get_fold_dataloaders_with_val(
        root_dir             = cfg["data_root"],
        test_fold            = fold,
        transform            = transform,
        val_fraction         = cfg["val_fraction"],
        batch_size           = cfg["batch_size"],
        num_workers          = cfg["num_workers"],
        seed                 = 42,
        max_samples_per_fold = cfg.get("max_samples_per_fold"),
    )
    print(f" Train : {len(train_loader.dataset):>5}  |  "
          f"Val : {len(val_loader.dataset):>4}  |  "
          f"Test : {len(test_loader.dataset):>4}")

    # ── verify input shape before any training ─────────────────────────────────
    sample_x, _ = next(iter(train_loader))
    assert sample_x.shape[1:] == (1, 64, 128), (
        f"[CRITICAL] Input shape is {tuple(sample_x.shape[1:])} — expected (1, 64, 128). "
        "CRNN uses the same 128-frame spectrogram as CNN; check MelSpectrogramTransform."
    )
    print(f" Input shape confirmed : {tuple(sample_x.shape[1:])}  [OK]")

    # ── model — fresh Orthogonal (GRU) + Xavier (Conv/Linear) init per fold ───
    model = CRNN(
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
    ).to(device)
    model.apply(crnn_weight_init)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["lr_scheduler_factor"],
        patience=cfg["lr_scheduler_patience"],
    )

    # ── checkpoint setup ──────────────────────────────────────────────────────
    os.makedirs(cfg["save_dir"], exist_ok=True)
    # Named best_fold{N}.pt so the attack pipeline (run_attacks.py) finds it
    # without modification.
    ckpt_path = os.path.join(cfg["save_dir"], f"best_fold{fold}.pt")

    best_val_acc     = -1.0
    patience_counter = 0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr":         [],
    }

    fold_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = _eval_epoch(model, val_loader,   criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc,  4))
        history["val_loss"].append(round(val_loss,    6))
        history["val_acc"].append(round(val_acc,      4))
        history["lr"].append(current_lr)

        print(
            f"  Epoch {epoch:>3}/{cfg['epochs']}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  "
            f"lr={current_lr:.2e}"
        )

        # ── checkpoint on best val_acc ─────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":                epoch,
                    "fold":                 fold,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc":              val_acc,
                    "val_loss":             val_loss,
                },
                ckpt_path,
            )
            patience_counter = 0
        else:
            patience_counter += 1

        # ── early stopping ─────────────────────────────────────────────────────
        if patience_counter >= cfg["early_stop_patience"]:
            print(f"  [Early stop] No val_acc improvement for {cfg['early_stop_patience']} epochs.")
            break

    fold_wall = time.time() - fold_start

    # ── final test evaluation using best checkpoint ───────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_acc = _eval_epoch(model, test_loader, criterion, device)

    print(f"\n  Fold {fold} — test_acc: {test_acc:.4f}%  |  "
          f"wall: {fold_wall/3600:.2f}h  |  checkpoint: {ckpt_path}")

    return {
        "fold":            fold,
        "test_acc":        round(test_acc, 4),
        "best_val_acc":    round(best_val_acc, 4),
        "wall_s":          round(fold_wall, 1),
        "history":         history,
        "checkpoint_path": ckpt_path,
    }


# ── full 10-fold run ───────────────────────────────────────────────────────────

def run_all_folds(cfg: dict) -> None:
    os.makedirs(cfg["results_dir"], exist_ok=True)

    config_snapshot = {k: v for k, v in cfg.items() if k != "data_root"}
    config_snapshot["model"] = "crnn"
    config_snapshot["mode"]  = "normal"
    with open(os.path.join(cfg["results_dir"], "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    total_start  = time.time()
    fold_results = []
    accuracies   = []

    for fold in range(1, 11):
        result = train_fold(fold, cfg)
        fold_results.append(result)
        accuracies.append(result["test_acc"])

        with open(os.path.join(cfg["results_dir"], f"fold_{fold}_results.json"), "w") as f:
            json.dump(result, f, indent=2)

    total_wall = time.time() - total_start
    mean_acc   = sum(accuracies) / len(accuracies)
    std_acc    = math.sqrt(sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies))

    h  = int(total_wall // 3600)
    m  = int((total_wall % 3600) // 60)
    s  = int(total_wall % 60)
    wall_str = f"{h:02d}h {m:02d}m {s:02d}s"

    cv_summary = {
        "model":     "crnn",
        "mode":      "normal",
        "wall_time": wall_str,
        "config": {
            "epochs":     cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "lr":         cfg["lr"],
        },
        "per_fold": [
            {"fold": r["fold"], "accuracy": round(r["test_acc"] / 100, 4)}
            for r in fold_results
        ],
        "mean": {"accuracy": round(mean_acc / 100, 4)},
        "std":  {"accuracy": round(std_acc  / 100, 4)},
    }

    with open(os.path.join(cfg["results_dir"], "cv_summary.json"), "w") as f:
        json.dump(cv_summary, f, indent=2)

    print(f"\n{'='*66}")
    print(f" CRNN 10-fold CV complete")
    print(f" Mean accuracy : {mean_acc:.2f}% +/- {std_acc:.2f}%")
    if mean_acc < 65.0:
        print(" [WARN] Accuracy below 65% — check GRU init or learning rate.")
    elif mean_acc > 80.0:
        print(" [WARN] Accuracy above 80% — verify no data leakage between folds.")
    else:
        print(" [OK]  Accuracy in expected range 65–80%.")
    print(f" Wall time     : {wall_str}")
    print(f" Results saved : {cfg['results_dir']}")
    print(f"{'='*66}")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRNN baseline training — 10-fold CV")
    parser.add_argument("--quick", type=int, default=0, metavar="N",
                        help="Smoke-test: limit each fold to N samples (0 = full dataset)")
    args = parser.parse_args()

    cfg = get_crnn_config()
    if args.quick:
        cfg["max_samples_per_fold"] = args.quick
    cfg["results_dir"] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "results", "crnn", "normal")
    )
    run_all_folds(cfg)

import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.urbansound_dataset import get_fold_dataloaders
from models.cnn import UrbanSoundCNN


# ── training configuration ────────────────────────────────────────────────────

CONFIG = {
    "data_root":   os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "UrbanSound8K")),
    "save_dir":    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models")),
    "batch_size":  32,
    "lr":          0.001,
    "epochs":      30,
    "num_workers": 4,
    "dropout":     0.5,
    "num_classes": 10,
}


# ── device selection ──────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── training — one full epoch ─────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> tuple[float, float]:
    """Iterate over every batch in the training DataLoader for one epoch.

    For each batch:
        1. Load mel spectrogram and label onto the device.
        2. Forward pass through the CNN.
        3. Compute CrossEntropy loss.
        4. Backpropagate — compute gradients for every parameter.
        5. Update weights — Adam adjusts each parameter using its gradient.

    Args:
        model:     UrbanSoundCNN in training mode.
        loader:    Training DataLoader (shuffled, batch_size=32).
        criterion: nn.CrossEntropyLoss instance.
        optimizer: Adam optimizer.
        device:    CPU / CUDA / MPS.

    Returns:
        (average_loss_over_epoch, accuracy_percent_over_epoch)
    """
    model.train()   # enables Dropout; tells BatchNorm to use batch statistics

    total_loss = 0.0
    correct    = 0
    total      = 0

    for specs, labels in loader:

        # ── Step 1: load mel spectrogram and label ────────────────────────────
        # specs:  (32, 1, 64, 128) float32 — one log-Mel spectrogram per sample
        # labels: (32,) int64        — class ID 0-9 for each sample
        specs  = specs.to(device)
        labels = labels.to(device)

        # ── Step 2: forward pass through CNN ─────────────────────────────────
        # logits: (32, 10) — one raw score per class, no softmax yet
        logits = model(specs)

        # ── Step 3: compute loss ──────────────────────────────────────────────
        # CrossEntropyLoss = log_softmax(logits) + NLLLoss
        # A perfect prediction gives loss ≈ 0; random guess gives ≈ ln(10) ≈ 2.3
        loss = criterion(logits, labels)

        # ── Step 4: backpropagate ─────────────────────────────────────────────
        # zero_grad() must be called first so gradients from the previous batch
        # do not accumulate into this batch's gradients.
        optimizer.zero_grad()
        loss.backward()     # compute ∂loss/∂w for every parameter w in the CNN

        # ── Step 5: update weights ────────────────────────────────────────────
        # Adam uses each parameter's gradient (and its running mean/variance)
        # to compute an adaptive step size, then subtracts it from the weight.
        optimizer.step()

        # ── accumulate metrics ────────────────────────────────────────────────
        total_loss += loss.item() * specs.size(0)   # weight by batch size
        preds       = logits.argmax(dim=1)           # predicted class per sample
        correct    += (preds == labels).sum().item()
        total      += specs.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ── evaluation — one full epoch, no gradient updates ─────────────────────────

def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a validation or test fold.

    No gradients are computed and Dropout is disabled.  The model weights
    are not changed.

    Args:
        model:     UrbanSoundCNN.
        loader:    Validation or test DataLoader (not shuffled).
        criterion: nn.CrossEntropyLoss instance.
        device:    CPU / CUDA / MPS.

    Returns:
        (average_loss, accuracy_percent)
    """
    model.eval()    # disables Dropout; BatchNorm uses running statistics

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():   # saves memory — no computation graph is built
        for specs, labels in loader:
            specs  = specs.to(device)
            labels = labels.to(device)

            logits = model(specs)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * specs.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += specs.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ── training loop for one fold ────────────────────────────────────────────────

def train_fold(
    fold:        int,
    epochs:      int   = CONFIG["epochs"],
    cfg:         dict  = CONFIG,
    model_class        = UrbanSoundCNN,
) -> dict:
    """Train the CNN for one fold of the 10-fold cross-validation protocol.

    Outer loop — for each epoch:
        1. Train on the 9 training folds (train_one_epoch).
        2. Evaluate on the held-out validation fold (evaluate).
        3. If validation accuracy improved → save checkpoint to disk.

    The saved checkpoint always contains the weights that achieved the highest
    validation accuracy across all epochs, not simply the last epoch's weights.

    Args:
        fold:   Test fold (1–10).  The other 9 folds form the training set.
        epochs: Total number of training epochs.
        cfg:    Configuration dict (see CONFIG at top of file).

    Returns:
        {
            "best_val_acc":    float,           # highest val acc seen
            "history":         dict,            # loss/acc lists per epoch
            "checkpoint_path": str,             # path to saved .pt file
        }
    """
    device = get_device()
    print(f"\n{'='*60}")
    print(f" Fold {fold}/10  |  device: {device}  |  epochs: {epochs}")
    print(f"{'='*60}")

    # ── data loaders ──────────────────────────────────────────────────────────
    train_loader, val_loader = get_fold_dataloaders(
        root_dir             = cfg["data_root"],
        test_fold            = fold,
        batch_size           = cfg["batch_size"],
        num_workers          = cfg["num_workers"],
        max_samples_per_fold = cfg.get("max_samples_per_fold"),   # None → full data
    )
    print(f" Train samples : {len(train_loader.dataset)}")
    print(f" Val   samples : {len(val_loader.dataset)}")

    # ── model / loss / optimiser ──────────────────────────────────────────────
    model     = model_class(num_classes=cfg["num_classes"], dropout=cfg["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # ── checkpoint setup ──────────────────────────────────────────────────────
    os.makedirs(cfg["save_dir"], exist_ok=True)
    checkpoint_path = os.path.join(cfg["save_dir"], f"best_fold{fold}.pt")

    best_val_acc = 0.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── for each batch: forward → loss → backprop → update ────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # ── evaluate on validation fold ────────────────────────────────────────
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── save model if best accuracy ────────────────────────────────────────
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":             epoch,
                    "fold":              fold,
                    "model_state_dict":  model.state_dict(),
                    "val_acc":           val_acc,
                    "val_loss":          val_loss,
                    "config":            cfg,
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
    print(f" Checkpoint : {checkpoint_path}")

    return {
        "best_val_acc":    best_val_acc,
        "history":         history,
        "checkpoint_path": checkpoint_path,
    }


# ── full 10-fold cross-validation ─────────────────────────────────────────────

def train_all_folds(epochs: int = CONFIG["epochs"], cfg: dict = CONFIG, model_class=UrbanSoundCNN) -> None:
    """Run the complete 10-fold CV and print a summary table."""
    results = {}
    for fold in range(1, 11):
        results[fold] = train_fold(fold=fold, epochs=epochs, cfg=cfg, model_class=model_class)

    print(f"\n{'='*60}")
    print(" 10-Fold Cross-Validation Summary")
    print(f"{'='*60}")
    accs = []
    for fold, r in results.items():
        acc = r["best_val_acc"]
        accs.append(acc)
        print(f"  Fold {fold:>2}  best val acc: {acc:.2f}%")
    mean = sum(accs) / len(accs)
    print(f"{'─'*60}")
    print(f"  Mean : {mean:.2f}%")
    print(f"{'='*60}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train UrbanSoundCNN")
    parser.add_argument("--fold",   type=int, default=None,
                        help="Single fold to train (1–10). Omit to run all 10 folds.")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"],
                        help=f"Number of epochs (default: {CONFIG['epochs']})")
    args = parser.parse_args()

    if args.fold is not None:
        train_fold(fold=args.fold, epochs=args.epochs)
    else:
        train_all_folds(epochs=args.epochs)

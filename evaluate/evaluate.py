import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.urbansound_dataset import get_fold_dataloaders, CLASS_NAMES
from models.cnn import UrbanSoundCNN


# ── device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── inference — collect all predictions for a fold ────────────────────────────

def get_predictions(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Run the model over every sample in a DataLoader and collect results.

    Args:
        model:  Trained UrbanSoundCNN in eval mode.
        loader: DataLoader for the test fold (not shuffled).
        device: CPU / CUDA / MPS.

    Returns:
        (all_labels, all_preds) — flat Python lists of integer class IDs.
    """
    model.eval()

    all_labels: list[int] = []
    all_preds:  list[int] = []

    with torch.no_grad():
        for specs, labels in loader:
            specs  = specs.to(device)
            logits = model(specs)                       # (B, 10) raw scores
            preds  = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_labels, all_preds


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    labels: list[int],
    preds:  list[int],
) -> dict:
    """Compute Accuracy, Precision, Recall, and F1 Score.

    Precision, Recall, and F1 are computed in two ways:
      - macro   : unweighted mean across all 10 classes
                  (treats every class equally regardless of how many samples it has)
      - weighted: mean weighted by the number of true instances per class
                  (accounts for class imbalance — closer to overall accuracy)

    Args:
        labels: Ground-truth class IDs.
        preds:  Predicted class IDs.

    Returns:
        Dictionary containing every metric.
    """
    accuracy          = accuracy_score(labels, preds)

    precision_macro   = precision_score(labels, preds, average="macro",    zero_division=0)
    precision_weighted = precision_score(labels, preds, average="weighted", zero_division=0)

    recall_macro      = recall_score(labels, preds, average="macro",    zero_division=0)
    recall_weighted   = recall_score(labels, preds, average="weighted", zero_division=0)

    f1_macro          = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_weighted       = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy":           accuracy,
        "precision_macro":    precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro":       recall_macro,
        "recall_weighted":    recall_weighted,
        "f1_macro":           f1_macro,
        "f1_weighted":        f1_weighted,
    }


# ── pretty printer ────────────────────────────────────────────────────────────

def print_metrics(metrics: dict, fold: int | None = None) -> None:
    """Print a formatted metrics table to stdout."""
    header = f" Evaluation Results — Fold {fold}" if fold else " Evaluation Results"
    print(f"\n{'='*52}")
    print(header)
    print(f"{'='*52}")
    print(f"  {'Metric':<28} {'Macro':>8}  {'Weighted':>8}")
    print(f"  {'─'*28} {'─'*8}  {'─'*8}")
    print(f"  {'Accuracy':<28} {metrics['accuracy']*100:>7.2f}%")
    print(f"  {'Precision':<28} {metrics['precision_macro']*100:>7.2f}%  {metrics['precision_weighted']*100:>7.2f}%")
    print(f"  {'Recall':<28} {metrics['recall_macro']*100:>7.2f}%  {metrics['recall_weighted']*100:>7.2f}%")
    print(f"  {'F1 Score':<28} {metrics['f1_macro']*100:>7.2f}%  {metrics['f1_weighted']*100:>7.2f}%")
    print(f"{'='*52}")


def print_per_class_report(labels: list[int], preds: list[int]) -> None:
    """Print sklearn's full per-class precision / recall / F1 table."""
    print("\n Per-class breakdown:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=3))


def print_confusion_matrix(labels: list[int], preds: list[int]) -> None:
    """Print the 10×10 confusion matrix."""
    cm = confusion_matrix(labels, preds)
    print(" Confusion Matrix (rows=true, cols=predicted):")
    print(f"  {'':>18}", end="")
    for name in CLASS_NAMES:
        print(f"  {name[:6]:>6}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>18}", end="")
        for val in row:
            print(f"  {val:>6}", end="")
        print()
    print()


# ── evaluate one saved checkpoint ────────────────────────────────────────────

def evaluate_checkpoint(
    checkpoint_path: str,
    data_root:       str,
    fold:            int,
    batch_size:      int = 32,
    num_workers:     int = 4,
    verbose:         bool = True,
    model_class            = UrbanSoundCNN,
) -> dict:
    """Load a saved checkpoint and evaluate it on its test fold.

    Args:
        checkpoint_path: Path to the .pt file saved by train.py.
        data_root:       Path to the UrbanSound8K root directory.
        fold:            Test fold the checkpoint was trained with.
        batch_size:      Batch size for inference.
        num_workers:     DataLoader workers.
        verbose:         Print metrics table and per-class report.
        model_class:     Model class to instantiate (default: UrbanSoundCNN).

    Returns:
        Metrics dictionary from compute_metrics().
    """
    device = get_device()

    # ── load model ────────────────────────────────────────────────────────────
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model      = model_class(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ── load test data ────────────────────────────────────────────────────────
    _, test_loader = get_fold_dataloaders(
        root_dir    = data_root,
        test_fold   = fold,
        batch_size  = batch_size,
        num_workers = num_workers,
    )

    # ── collect predictions ───────────────────────────────────────────────────
    labels, preds = get_predictions(model, test_loader, device)

    # ── compute and display metrics ───────────────────────────────────────────
    metrics = compute_metrics(labels, preds)

    if verbose:
        print_metrics(metrics, fold=fold)
        print_per_class_report(labels, preds)
        print_confusion_matrix(labels, preds)

    return metrics


# ── evaluate all 10 folds and aggregate ───────────────────────────────────────

def evaluate_all_folds(
    save_dir:    str,
    data_root:   str,
    batch_size:  int = 32,
    num_workers: int = 4,
    model_class        = UrbanSoundCNN,
) -> dict:
    """Evaluate every saved fold checkpoint and compute mean metrics across all 10.

    Args:
        save_dir:    Directory containing best_fold1.pt … best_fold10.pt.
        data_root:   Path to the UrbanSound8K root directory.
        batch_size:  Batch size for inference.
        num_workers: DataLoader workers.

    Returns:
        Dictionary of mean metrics across all 10 folds.
    """
    all_metrics: list[dict] = []
    all_labels:  list[int]  = []
    all_preds:   list[int]  = []

    device = get_device()
    print(f"\n{'='*52}")
    print(" 10-Fold Cross-Validation Evaluation")
    print(f"{'='*52}")

    for fold in range(1, 11):
        ckpt_path = os.path.join(save_dir, f"best_fold{fold}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Fold {fold}: checkpoint not found — skipping.")
            continue

        checkpoint = torch.load(ckpt_path, map_location=device)
        model      = model_class(num_classes=10).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        _, test_loader = get_fold_dataloaders(
            root_dir    = data_root,
            test_fold   = fold,
            batch_size  = batch_size,
            num_workers = num_workers,
        )

        labels, preds = get_predictions(model, test_loader, device)
        metrics       = compute_metrics(labels, preds)
        all_metrics.append(metrics)
        all_labels.extend(labels)
        all_preds.extend(preds)

        print(
            f"  Fold {fold:>2} | "
            f"acc {metrics['accuracy']*100:5.2f}%  "
            f"precision {metrics['precision_weighted']*100:5.2f}%  "
            f"recall {metrics['recall_weighted']*100:5.2f}%  "
            f"F1 {metrics['f1_weighted']*100:5.2f}%"
        )

    # ── mean across folds ──────────────────────────────────────────────────────
    if not all_metrics:
        raise RuntimeError(
            f"No checkpoints found in '{save_dir}'. "
            "Run training first before calling evaluate_all_folds()."
        )
    keys = all_metrics[0].keys()
    mean_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}

    print(f"{'─'*52}")
    print_metrics(mean_metrics, fold=None)

    # ── aggregate report over all 8732 samples ─────────────────────────────────
    print("\n Aggregate per-class report (all folds combined):")
    print_per_class_report(all_labels, all_preds)

    return mean_metrics


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "UrbanSound8K"))
    SAVE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models"))

    parser = argparse.ArgumentParser(description="Evaluate UrbanSoundCNN checkpoints")
    parser.add_argument("--fold", type=int, default=None,
                        help="Evaluate a single fold (1–10). Omit to evaluate all 10.")
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if args.fold is not None:
        ckpt = os.path.join(SAVE_DIR, f"best_fold{args.fold}.pt")
        evaluate_checkpoint(
            checkpoint_path = ckpt,
            data_root       = DATA_ROOT,
            fold            = args.fold,
            batch_size      = args.batch_size,
            num_workers     = args.num_workers,
        )
    else:
        evaluate_all_folds(
            save_dir    = SAVE_DIR,
            data_root   = DATA_ROOT,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
        )

"""
10-Fold Cross-Validation — UrbanSound8K CNN Classifier
=======================================================

Rotation protocol
-----------------
  Fold 1  as test → train on folds 2–10, evaluate on fold 1
  Fold 2  as test → train on folds 1, 3–10, evaluate on fold 2
  ...
  Fold 10 as test → train on folds 1–9, evaluate on fold 10

Each audio clip appears in the test set exactly once across the 10 runs.
Final reported numbers are the mean (± std) of per-fold metrics.

Saved artefacts
---------------
  results/cnn/normal/
    config.json              — hyperparameters, preprocessing params, model architecture
    fold_1_results.json  ... fold_10_results.json
    cv_summary.json          — per-fold table + mean ± std across all 10 folds

  saved_models/cnn/normal/
    best_fold1.pt  ...  best_fold10.pt
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from train.train              import train_fold, CONFIG
from evaluate.evaluate        import evaluate_checkpoint
from models.cnn               import UrbanSoundCNN
from attacks.run_attacks      import run_fgsm_all_folds, print_fgsm_results, \
                                     run_bim_all_folds,  print_bim_results
from preprocessing.mel_spectrogram import (
    SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, TARGET_SAMPLES, N_TIME_FRAMES
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values)

def _std(values: list[float]) -> float:
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5

def _save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")


# ── build the config snapshot ─────────────────────────────────────────────────

def build_config_snapshot(epochs: int, batch_size: int) -> dict:
    """Collect every fixed parameter into one dict that goes into config.json."""
    total_params = sum(p.numel() for p in UrbanSoundCNN().parameters())

    return {
        "hyperparameters": {
            "optimizer":       "Adam",
            "learning_rate":   CONFIG["lr"],
            "batch_size":      batch_size,
            "epochs_per_fold": epochs,
            "dropout":         CONFIG["dropout"],
            "loss_function":   "CrossEntropyLoss",
        },
        "preprocessing": {
            "sample_rate":     SAMPLE_RATE,
            "n_fft":           N_FFT,
            "hop_length":      HOP_LENGTH,
            "n_mels":          N_MELS,
            "target_samples":  TARGET_SAMPLES,
            "n_time_frames":   N_TIME_FRAMES,
            "log_scale":       "10 * log10(power + 1e-9)  clamped at max-80dB",
            "normalisation":   "per-clip min-max → [0, 1]",
        },
        "model": {
            "architecture":    "UrbanSoundCNN",
            "input_shape":     [1, N_MELS, N_TIME_FRAMES],
            "conv_filters":    [32, 64, 128],
            "kernel_size":     3,
            "pooling":         "MaxPool2d(2×2) after each conv block",
            "fc_units":        256,
            "num_classes":     CONFIG["num_classes"],
            "total_parameters": total_params,
        },
        "dataset": {
            "name":            "UrbanSound8K",
            "num_classes":     10,
            "class_names":     [
                "air_conditioner", "car_horn", "children_playing", "dog_bark",
                "drilling", "engine_idling", "gun_shot", "jackhammer",
                "siren", "street_music",
            ],
            "total_samples":   8732,
            "num_folds":       10,
            "cross_validation": "10-fold, fold-stratified (official splits)",
        },
    }


# ── 10-fold cross-validation ──────────────────────────────────────────────────

_BASE_DIR = os.path.dirname(__file__)

CNN_NORMAL_RESULTS_DIR     = os.path.join(_BASE_DIR, "results",      "cnn", "normal")
CNN_NORMAL_SAVED_MODELS_DIR = os.path.join(_BASE_DIR, "saved_models", "cnn", "normal")


def cross_validate(
    epochs:               int       = CONFIG["epochs"],
    batch_size:           int       = CONFIG["batch_size"],
    num_workers:          int       = CONFIG["num_workers"],
    data_root:            str       = CONFIG["data_root"],
    save_dir:             str       = CNN_NORMAL_SAVED_MODELS_DIR,
    max_samples_per_fold: int | None = None,
) -> dict:
    """Run the complete 10-fold cross-validation pipeline and save all results.

    For each fold k  (k = 1 … 10):
        1. Train on the 9 remaining folds  → saves best_fold{k}.pt
        2. Evaluate best_fold{k}.pt on fold k  → records all metrics
        3. Save fold_k_results.json

    After all 10 folds:
        4. Save config.json  (hyperparameters + preprocessing + model info)
        5. Save cv_summary.json  (per-fold table + mean ± std)
        6. Print summary table to stdout

    Results are saved under results/cnn/normal/ and checkpoints under
    saved_models/cnn/normal/ to match the unified experiment structure.

    Returns:
        Dictionary of mean metric values across all 10 folds.
    """
    wall_start  = time.time()
    results_dir = CNN_NORMAL_RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(save_dir,    exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"#  Model  : CNN / normal")
    print(f"#  Results: {results_dir}")
    print(f"#  Models : {save_dir}")
    print(f"{'#'*60}")

    # ── save config immediately so it exists even if training is interrupted ──
    config_snapshot = build_config_snapshot(epochs, batch_size)
    _save_json(os.path.join(results_dir, "config.json"), config_snapshot)

    fold_results: list[dict] = []
    fold_cfg = {**CONFIG, "batch_size": batch_size,
                "num_workers":          num_workers,
                "data_root":            data_root,
                "save_dir":             save_dir,
                "max_samples_per_fold": max_samples_per_fold}

    if max_samples_per_fold is not None:
        print(f"  Quick mode: {max_samples_per_fold} random samples per fold")

    for fold in range(1, 11):

        print(f"\n{'#'*60}")
        print(f"#  FOLD {fold}/10")
        print(f"{'#'*60}")

        # ── Step 1: train ─────────────────────────────────────────────────────
        train_result = train_fold(fold=fold, epochs=epochs, cfg=fold_cfg)

        # ── Step 2: evaluate ──────────────────────────────────────────────────
        metrics = evaluate_checkpoint(
            checkpoint_path = train_result["checkpoint_path"],
            data_root       = data_root,
            fold            = fold,
            batch_size      = batch_size,
            num_workers     = num_workers,
            verbose         = True,
        )

        # ── Step 3: save per-fold JSON ────────────────────────────────────────
        fold_record = {
            "fold":             fold,
            "checkpoint":       train_result["checkpoint_path"],
            "best_val_acc":     train_result["best_val_acc"],
            "training_history": train_result["history"],
            "test_metrics": {
                "accuracy":           round(metrics["accuracy"],           4),
                "precision_macro":    round(metrics["precision_macro"],    4),
                "precision_weighted": round(metrics["precision_weighted"], 4),
                "recall_macro":       round(metrics["recall_macro"],       4),
                "recall_weighted":    round(metrics["recall_weighted"],    4),
                "f1_macro":           round(metrics["f1_macro"],           4),
                "f1_weighted":        round(metrics["f1_weighted"],        4),
            },
        }
        _save_json(os.path.join(results_dir, f"fold_{fold}_results.json"), fold_record)

        fold_results.append({
            "fold": fold,
            **metrics,
            "best_val_acc": train_result["best_val_acc"],
        })

    # ── Step 4: print per-fold summary table ──────────────────────────────────
    print(f"\n{'='*78}")
    print("  Per-Fold Results")
    print(f"{'='*78}")
    print(f"  {'Fold':>4}  {'Accuracy':>9}  {'Precision (W)':>13}  {'Recall (W)':>10}  {'F1 (W)':>8}")
    print(f"  {'─'*4}  {'─'*9}  {'─'*13}  {'─'*10}  {'─'*8}")
    for r in fold_results:
        print(
            f"  {r['fold']:>4}  "
            f"{r['accuracy']*100:>8.2f}%  "
            f"{r['precision_weighted']*100:>12.2f}%  "
            f"{r['recall_weighted']*100:>9.2f}%  "
            f"{r['f1_weighted']*100:>7.2f}%"
        )

    # ── Step 5: compute mean ± std ────────────────────────────────────────────
    metric_keys = [
        ("Accuracy",          "accuracy"),
        ("Precision (macro)", "precision_macro"),
        ("Precision (wtd)",   "precision_weighted"),
        ("Recall (macro)",    "recall_macro"),
        ("Recall (wtd)",      "recall_weighted"),
        ("F1 Score (macro)",  "f1_macro"),
        ("F1 Score (wtd)",    "f1_weighted"),
    ]

    mean_metrics: dict = {}
    std_metrics:  dict = {}

    print(f"\n{'='*52}")
    print("  Mean ± Std  (across 10 folds)")
    print(f"{'='*52}")
    for label, key in metric_keys:
        values = [r[key] for r in fold_results]
        m, s   = _mean(values), _std(values)
        mean_metrics[key] = m
        std_metrics[key]  = s
        print(f"  {label:<24}  {m*100:6.2f}%  ±  {s*100:.2f}%")
    print(f"{'='*52}")

    elapsed = time.time() - wall_start
    h, rem  = divmod(int(elapsed), 3600)
    mn, sc  = divmod(rem, 60)
    wall_time_str = f"{h:02d}h {mn:02d}m {sc:02d}s"
    print(f"\n  Total wall time: {wall_time_str}")

    # ── Step 6: save cv_summary.json ──────────────────────────────────────────
    cv_summary = {
        "model":      "cnn",
        "mode":       "normal",
        "wall_time":  wall_time_str,
        "config": {
            "epochs":     epochs,
            "batch_size": batch_size,
            "lr":         CONFIG["lr"],
        },
        "per_fold": [
            {
                "fold":               r["fold"],
                "accuracy":           round(r["accuracy"],           4),
                "precision_macro":    round(r["precision_macro"],    4),
                "precision_weighted": round(r["precision_weighted"], 4),
                "recall_macro":       round(r["recall_macro"],       4),
                "recall_weighted":    round(r["recall_weighted"],    4),
                "f1_macro":           round(r["f1_macro"],           4),
                "f1_weighted":        round(r["f1_weighted"],        4),
            }
            for r in fold_results
        ],
        "mean": {k: round(v, 4) for k, v in mean_metrics.items()},
        "std":  {k: round(v, 4) for k, v in std_metrics.items()},
    }
    _save_json(os.path.join(results_dir, "cv_summary.json"), cv_summary)

    print(f"\n  All results saved to: {results_dir}/")

    # per-fold clean accuracy in % — used by print_fgsm_results
    clean_accuracies = [r["accuracy"] * 100 for r in fold_results]
    return mean_metrics, clean_accuracies, results_dir, cv_summary


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="10-fold cross-validation for UrbanSoundCNN"
    )
    parser.add_argument(
        "--epochs", type=int, default=CONFIG["epochs"],
        help=f"Training epochs per fold (default: {CONFIG['epochs']})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=CONFIG["batch_size"],
        help=f"Batch size (default: {CONFIG['batch_size']})"
    )
    parser.add_argument(
        "--num-workers", type=int, default=CONFIG["num_workers"],
        help="DataLoader workers (default: 4; set 0 on Windows)"
    )
    parser.add_argument(
        "--quick", type=int, default=None, metavar="N",
        help="Use N random samples per fold instead of all data (e.g. --quick 100)"
    )
    args = parser.parse_args()

    import torch
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    _, clean_accuracies, results_dir, cv_summary = cross_validate(
        epochs               = args.epochs,
        batch_size           = args.batch_size,
        num_workers          = args.num_workers,
        max_samples_per_fold = args.quick,
    )

    epsilons = [0.01, 0.03, 0.1]

    # ── FGSM Attack Evaluation ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" FGSM Attack Evaluation")
    print("=" * 60)

    fgsm_results = run_fgsm_all_folds(
        model_class      = UrbanSoundCNN,
        saved_models_dir = CNN_NORMAL_SAVED_MODELS_DIR,
        data_root        = CONFIG["data_root"],
        device           = device,
        epsilons         = epsilons,
        batch_size       = args.batch_size,
        num_workers      = args.num_workers,
    )
    print_fgsm_results(clean_accuracies, fgsm_results, epsilons)

    # ── BIM Attack Evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" BIM Attack Evaluation  (steps=10)")
    print("=" * 60)

    bim_results = run_bim_all_folds(
        model_class      = UrbanSoundCNN,
        saved_models_dir = CNN_NORMAL_SAVED_MODELS_DIR,
        data_root        = CONFIG["data_root"],
        device           = device,
        epsilons         = epsilons,
        steps            = 10,
        batch_size       = args.batch_size,
        num_workers      = args.num_workers,
    )
    print_bim_results(clean_accuracies, bim_results, epsilons)

    # ── update cv_summary.json with attack results ────────────────────────────
    cv_summary["fgsm"] = {
        f"fold_{f}": fgsm_results[f"fold_{f}"]
        for f in range(1, 11)
        if f"fold_{f}" in fgsm_results
    }
    cv_summary["bim"] = {
        f"fold_{f}": bim_results[f"fold_{f}"]
        for f in range(1, 11)
        if f"fold_{f}" in bim_results
    }
    _save_json(os.path.join(results_dir, "cv_summary.json"), cv_summary)
    print("\n  cv_summary.json updated with FGSM and BIM results.")

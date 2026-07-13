"""
Patch script — generate cv_summary.json for crnn/normal without retraining.

Background
----------
The crnn/normal experiment completed all 10 training folds successfully and
wrote fold_{1..10}_results.json to results/crnn/normal/.  The run then crashed
during FGSM attack evaluation because the generic evaluate_fgsm asserts
model.eval(), which triggers:

    RuntimeError: cudnn RNN backward can only be called in training mode

This script:
  1. Reads the 10 existing fold_k_results.json files (clean metrics).
  2. Runs FGSM and BIM using attacks/crnn/run_attacks_crnn.py (cuDNN-safe).
  3. Writes a complete cv_summary.json that matches the format produced by
     run_experiments.py for CNN and VGGish.

Usage
-----
  python attacks/crnn/patch_normal_summary.py

No arguments needed — all paths are derived from the project root.
"""

import json
import math
import os
import sys
import time

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from models.crnn import CRNN
from attacks.crnn.run_attacks_crnn import (
    run_fgsm_all_folds_crnn,
    run_bim_all_folds_crnn,
)
from config_loader import get_eval_attack_config, get_crnn_config


# ── paths ─────────────────────────────────────────────────────────────────────

RESULTS_DIR      = os.path.join(_ROOT, "results",      "crnn", "normal")
SAVED_MODELS_DIR = os.path.join(_ROOT, "saved_models", "crnn", "normal")


# ── helpers ───────────────────────────────────────────────────────────────────

def _mean(values):
    return sum(values) / len(values)

def _std(values):
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))

def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = _device()
    print(f"\n[patch_normal_summary]  device: {device}")
    print(f" Results dir      : {RESULTS_DIR}")
    print(f" Saved models dir : {SAVED_MODELS_DIR}")

    # ── 1. Load the 10 fold results written during training ───────────────────
    metric_keys = [
        "accuracy",
        "precision_macro", "precision_weighted",
        "recall_macro",    "recall_weighted",
        "f1_macro",        "f1_weighted",
    ]

    fold_records   = []
    clean_accs_pct = []   # accuracy * 100 for attack result tables

    for fold in range(1, 11):
        path = os.path.join(RESULTS_DIR, f"fold_{fold}_results.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing fold result: {path}\n"
                "Run 'python run_experiments.py --model crnn --mode normal' first."
            )
        with open(path) as f:
            data = json.load(f)
        fold_records.append(data)
        clean_accs_pct.append(data["test_metrics"]["accuracy"] * 100)

    print(f"\n Loaded {len(fold_records)} fold results  [OK]")
    for r in fold_records:
        print(f"  Fold {r['fold']:>2}  accuracy: {r['test_metrics']['accuracy']*100:.2f}%")

    # ── 2. Compute mean ± std across folds ───────────────────────────────────
    metric_values = {k: [r["test_metrics"][k] for r in fold_records] for k in metric_keys}
    mean_metrics  = {k: round(_mean(v), 4) for k, v in metric_values.items()}
    std_metrics   = {k: round(_std(v),  4) for k, v in metric_values.items()}

    # ── 3. Run FGSM and BIM attacks ───────────────────────────────────────────
    eval_cfg  = get_eval_attack_config()
    epsilons  = eval_cfg["eval_epsilons"]
    bim_steps = eval_cfg["bim_eval_steps"]
    data_root = get_crnn_config()["data_root"]

    print(f"\n{'='*60}")
    print(f" CRNN / normal — FGSM Attack Evaluation")
    print(f" epsilons: {epsilons}")
    print(f"{'='*60}")
    t0 = time.time()
    fgsm_results = run_fgsm_all_folds_crnn(
        model_class      = CRNN,
        saved_models_dir = SAVED_MODELS_DIR,
        data_root        = data_root,
        device           = device,
        epsilons         = epsilons,
        batch_size       = 128,
        num_workers      = 4,
    )
    print(f" FGSM done in {(time.time()-t0)/60:.1f} min")

    print(f"\n{'='*60}")
    print(f" CRNN / normal — BIM Attack Evaluation  (steps={bim_steps})")
    print(f"{'='*60}")
    t0 = time.time()
    bim_results = run_bim_all_folds_crnn(
        model_class      = CRNN,
        saved_models_dir = SAVED_MODELS_DIR,
        data_root        = data_root,
        device           = device,
        epsilons         = epsilons,
        steps            = bim_steps,
        batch_size       = 128,
        num_workers      = 4,
    )
    print(f" BIM done in {(time.time()-t0)/60:.1f} min")

    # ── 4. Print summary tables ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(" FGSM Attack Results — CRNN / normal")
    print(f"{'='*70}")
    header = f"{'Fold':<8} {'Clean':>8}"
    for eps in epsilons:
        header += f" {'ε='+str(eps):>10}"
    print(header)
    print("-" * 70)
    for fold in range(1, 11):
        row = f"Fold {fold:<3} {clean_accs_pct[fold-1]:>7.2f}%"
        for eps in epsilons:
            row += f" {fgsm_results[f'fold_{fold}'][f'eps_{eps}']:>9.2f}%"
        print(row)
    print("-" * 70)
    row = f"{'Mean':<8} {_mean(clean_accs_pct):>7.2f}%"
    for eps in epsilons:
        vals = [fgsm_results[f"fold_{f}"][f"eps_{eps}"] for f in range(1, 11)]
        row += f" {_mean(vals):>9.2f}%"
    print(row)
    print(f"{'='*70}")

    print(f"\n{'='*70}")
    print(f" BIM Attack Results — CRNN / normal  (steps={bim_steps})")
    print(f"{'='*70}")
    print(header)
    print("-" * 70)
    for fold in range(1, 11):
        row = f"Fold {fold:<3} {clean_accs_pct[fold-1]:>7.2f}%"
        for eps in epsilons:
            row += f" {bim_results[f'fold_{fold}'][f'eps_{eps}']:>9.2f}%"
        print(row)
    print("-" * 70)
    row = f"{'Mean':<8} {_mean(clean_accs_pct):>7.2f}%"
    for eps in epsilons:
        vals = [bim_results[f"fold_{f}"][f"eps_{eps}"] for f in range(1, 11)]
        row += f" {_mean(vals):>9.2f}%"
    print(row)
    print(f"{'='*70}")

    # ── 5. Write cv_summary.json ──────────────────────────────────────────────
    cv_summary = {
        "model":     "crnn",
        "mode":      "normal",
        "wall_time": "N/A (assembled from fold results)",
        "config": {
            "model":      "crnn",
            "mode":       "normal",
            "epochs":     50,
            "batch_size": 128,
            "lr":         0.001,
        },
        "per_fold": [
            {"fold": r["fold"], **{k: r["test_metrics"][k] for k in metric_keys}}
            for r in fold_records
        ],
        "mean": mean_metrics,
        "std":  std_metrics,
        "fgsm": {
            f"fold_{f}": fgsm_results[f"fold_{f}"]
            for f in range(1, 11)
        },
        "bim": {
            f"fold_{f}": bim_results[f"fold_{f}"]
            for f in range(1, 11)
        },
    }

    out_path = os.path.join(RESULTS_DIR, "cv_summary.json")
    with open(out_path, "w") as f:
        json.dump(cv_summary, f, indent=2)

    print(f"\n cv_summary.json written → {out_path}")
    print(f" Mean clean accuracy : {mean_metrics['accuracy']*100:.2f}% ± {std_metrics['accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()

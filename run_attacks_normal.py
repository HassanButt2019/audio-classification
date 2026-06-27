"""
Run FGSM & BIM attacks on the already-trained CNN Normal model
and save the results into the existing cv_summary.json.

Usage
-----
    python run_attacks_normal.py
"""

import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from attacks.run_attacks import run_fgsm_all_folds, run_bim_all_folds, \
                                  print_fgsm_results, print_bim_results
from models.cnn import UrbanSoundCNN
from config_loader import get_base_config, get_eval_attack_config

# ── config ────────────────────────────────────────────────────────────────────

BASE_DIR         = os.path.dirname(__file__)
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models", "cnn", "normal")
RESULTS_DIR      = os.path.join(BASE_DIR, "results", "cnn", "normal")

_base        = get_base_config()
_eval        = get_eval_attack_config()
EPSILONS     = _eval["eval_epsilons"]
BIM_STEPS    = _eval["bim_eval_steps"]

# ── device ────────────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# ── locate cv_summary.json ────────────────────────────────────────────────────

# support both flat layout and timestamped subdirectory layout
candidates = glob.glob(os.path.join(RESULTS_DIR, "cv_summary.json")) + \
             glob.glob(os.path.join(RESULTS_DIR, "*", "cv_summary.json"))

if not candidates:
    print(f"ERROR: No cv_summary.json found under {RESULTS_DIR}")
    sys.exit(1)

# pick most recently modified
cv_summary_path = max(candidates, key=os.path.getmtime)
print(f"cv_summary.json → {cv_summary_path}")

with open(cv_summary_path) as f:
    cv_summary = json.load(f)

# pull clean per-fold accuracies from the existing summary
clean_accuracies = [pf["accuracy"] * 100 for pf in cv_summary["per_fold"]]

# ── FGSM attack ───────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  FGSM Attack  [CNN / normal]")
print(f"{'='*60}")
fgsm_results = run_fgsm_all_folds(
    model_class      = UrbanSoundCNN,
    saved_models_dir = SAVED_MODELS_DIR,
    data_root        = _base["data_root"],
    device           = device,
    epsilons         = EPSILONS,
    batch_size       = _base["batch_size"],
    num_workers      = _base["num_workers"],
)
print_fgsm_results(clean_accuracies, fgsm_results, EPSILONS)

# ── BIM attack ────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  BIM Attack  [CNN / normal]  steps={BIM_STEPS}")
print(f"{'='*60}")
bim_results = run_bim_all_folds(
    model_class      = UrbanSoundCNN,
    saved_models_dir = SAVED_MODELS_DIR,
    data_root        = _base["data_root"],
    device           = device,
    epsilons         = EPSILONS,
    steps            = BIM_STEPS,
    batch_size       = _base["batch_size"],
    num_workers      = _base["num_workers"],
)
print_bim_results(clean_accuracies, bim_results, EPSILONS)

# ── patch cv_summary.json ─────────────────────────────────────────────────────

cv_summary["fgsm"] = {
    f"fold_{f}": fgsm_results[f"fold_{f}"]
    for f in range(1, 11) if f"fold_{f}" in fgsm_results
}
cv_summary["bim"] = {
    f"fold_{f}": bim_results[f"fold_{f}"]
    for f in range(1, 11) if f"fold_{f}" in bim_results
}

with open(cv_summary_path, "w") as f:
    json.dump(cv_summary, f, indent=2)

print(f"\n  Attack results saved → {cv_summary_path}")

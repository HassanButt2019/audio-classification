"""
Experiment Runner — 15 experiments (3 models × 5 training modes)
=================================================================

Training modes
--------------
  normal            — standard training on clean data
  adv_train_fgsm    — train from scratch mixing clean + FGSM adversarial examples
  adv_train_bim     — train from scratch mixing clean + BIM adversarial examples
  adv_finetune_fgsm — load normal checkpoint, fine-tune with FGSM adversarial examples
  adv_finetune_bim  — load normal checkpoint, fine-tune with BIM adversarial examples

Directory layout
----------------
  results/
    cnn/
      normal/  adv_train_fgsm/  adv_train_bim/  adv_finetune_fgsm/  adv_finetune_bim/
    vggish/  (same 5 modes)
    crnn/    (same 5 modes)

  saved_models/
    cnn/
      normal/  adv_train_fgsm/  adv_train_bim/  adv_finetune_fgsm/  adv_finetune_bim/
    vggish/  (same 5 modes)
    crnn/    (same 5 modes)

Each leaf directory contains:
  best_fold{1..10}.pt   — model checkpoints
  config.json           — hyperparameters saved before training starts
  fold_{1..10}_results.json
  cv_summary.json       — mean ± std + FGSM & BIM attack results

Dependency order
----------------
  adv_finetune_fgsm and adv_finetune_bim both require 'normal' to have run first
  for the same model (they load its best_fold{k}.pt as the pretrained base).

Usage
-----
  python run_experiments.py                          # all 15 experiments
  python run_experiments.py --model cnn              # all 5 modes for CNN
  python run_experiments.py --model cnn --mode normal
  python run_experiments.py --model cnn --mode adv_train_fgsm
  python run_experiments.py --model cnn --mode adv_finetune_bim
  python run_experiments.py --model cnn --mode normal --quick 100  # smoke-test
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))

from models.cnn    import UrbanSoundCNN
from models.vggish import VGGish
from models.crnn   import CRNN

from train.train               import train_fold, CONFIG
from train.adversarial_train   import adv_train_fold, ADV_FGSM_CONFIG, ADV_BIM_CONFIG
from train.finetune_adversarial import finetune_fold, FINETUNE_FGSM_CONFIG, FINETUNE_BIM_CONFIG

# CRNN requires dedicated training functions:
#   - Orthogonal GRU init + val split + early stopping (normal)
#   - Gradient clipping to stabilise BPTT in AT and AFT
#   - Checkpoint saved by adversarial val accuracy in AT (not clean val acc)
from train.crnn.train_crnn     import train_fold      as crnn_train_fold
from train.crnn.adv_train_crnn import adv_train_fold  as crnn_adv_train_fold
from train.crnn.finetune_crnn  import finetune_fold   as crnn_finetune_fold

# VGGish baseline uses a dedicated train_fold that applies VGGishMelSpectrogramTransform
# internally (96 time-frames); AT and AFT use the generic functions with transform=.
from train.train_vggish        import train_fold      as vggish_train_fold

from evaluate.evaluate         import evaluate_checkpoint
from attacks.run_attacks       import run_fgsm_all_folds, print_fgsm_results, \
                                      run_bim_all_folds,  print_bim_results
# CRNN attack evaluation uses cudnn.flags(enabled=False) so GRU backward works
# in eval mode — the generic evaluate_fgsm/bim assert model.eval() which
# triggers "cudnn RNN backward can only be called in training mode" on GPU.
from attacks.crnn.run_attacks_crnn import run_fgsm_all_folds_crnn, \
                                          run_bim_all_folds_crnn
from preprocessing.mel_spectrogram import (
    SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, TARGET_SAMPLES, N_TIME_FRAMES,
    VGGishMelSpectrogramTransform,
)
from config_loader import (
    get_eval_attack_config,
    get_crnn_config,
    get_crnn_adv_fgsm_config,
    get_crnn_adv_bim_config,
    get_crnn_finetune_fgsm_config,
    get_crnn_finetune_bim_config,
    get_vggish_config,
    get_vggish_adv_fgsm_config,
    get_vggish_adv_bim_config,
    get_vggish_finetune_fgsm_config,
    get_vggish_finetune_bim_config,
)

_eval_cfg = get_eval_attack_config()


# ── paths ─────────────────────────────────────────────────────────────────────

BASE_DIR          = os.path.dirname(__file__)
RESULTS_ROOT      = os.path.join(BASE_DIR, "results")
SAVED_MODELS_ROOT = os.path.join(BASE_DIR, "saved_models")
DATA_ROOT         = CONFIG["data_root"]


# ── registries ────────────────────────────────────────────────────────────────

MODELS = {
    "cnn":    UrbanSoundCNN,
    "vggish": VGGish,
    "crnn":   CRNN,
}

MODES = [
    "normal",
    "adv_train_fgsm",
    "adv_train_bim",
    "adv_finetune_fgsm",
    "adv_finetune_bim",
]

MODE_CONFIGS = {
    "normal":            CONFIG,
    "adv_train_fgsm":    ADV_FGSM_CONFIG,
    "adv_train_bim":     ADV_BIM_CONFIG,
    "adv_finetune_fgsm": FINETUNE_FGSM_CONFIG,
    "adv_finetune_bim":  FINETUNE_BIM_CONFIG,
}

# CRNN uses its own configs because it needs:
#   epochs=50 (not 30), val_fraction=0.1, early_stop_patience, lr_scheduler_*
CRNN_MODE_CONFIGS = {
    "normal":            get_crnn_config(),
    "adv_train_fgsm":    get_crnn_adv_fgsm_config(),
    "adv_train_bim":     get_crnn_adv_bim_config(),
    "adv_finetune_fgsm": get_crnn_finetune_fgsm_config(),
    "adv_finetune_bim":  get_crnn_finetune_bim_config(),
}

# VGGish uses its own configs because it needs:
#   weight_decay=1e-4, epochs=50, val_fraction=0.1, early_stop_patience, lr_scheduler_*
VGGISH_MODE_CONFIGS = {
    "normal":            get_vggish_config(),
    "adv_train_fgsm":    get_vggish_adv_fgsm_config(),
    "adv_train_bim":     get_vggish_adv_bim_config(),
    "adv_finetune_fgsm": get_vggish_finetune_fgsm_config(),
    "adv_finetune_bim":  get_vggish_finetune_bim_config(),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _mean(values):
    return sum(values) / len(values)

def _std(values):
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5

def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")

def _experiment_dirs(model_name, mode):
    results_dir      = os.path.join(RESULTS_ROOT,      model_name, mode)
    saved_models_dir = os.path.join(SAVED_MODELS_ROOT, model_name, mode)
    os.makedirs(results_dir,      exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)
    return results_dir, saved_models_dir

def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── per-fold runner ───────────────────────────────────────────────────────────

def run_fold(fold, model_name, mode, model_class, results_dir, saved_models_dir,
             cfg, max_samples_per_fold=None):
    """Train + evaluate one fold for the given model and mode."""

    fold_cfg = {
        **cfg,
        "data_root":            DATA_ROOT,
        "save_dir":             saved_models_dir,
        "max_samples_per_fold": max_samples_per_fold,
    }

    if model_name == "crnn":
        # CRNN must use dedicated training functions to apply:
        #   - Orthogonal GRU weight init (normal)
        #   - Val split + early stopping (normal)
        #   - Gradient clipping for BPTT stability (adv_train + adv_finetune)
        #   - Checkpoint by adversarial val accuracy in AT
        if mode == "normal":
            train_result = crnn_train_fold(fold=fold, cfg=fold_cfg)

        elif mode in ("adv_train_fgsm", "adv_train_bim"):
            train_result = crnn_adv_train_fold(fold=fold, cfg=fold_cfg)

        elif mode in ("adv_finetune_fgsm", "adv_finetune_bim"):
            normal_ckpt = os.path.join(
                SAVED_MODELS_ROOT, model_name, "normal", f"best_fold{fold}.pt"
            )
            if not os.path.exists(normal_ckpt):
                raise FileNotFoundError(
                    f"CRNN baseline checkpoint not found: {normal_ckpt}\n"
                    "Run 'python train/crnn/train_crnn.py' or "
                    "'python run_experiments.py --model crnn --mode normal' first."
                )
            train_result = crnn_finetune_fold(
                fold=fold, pretrained_ckpt=normal_ckpt, cfg=fold_cfg,
            )

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    elif model_name == "vggish":
        # VGGish — baseline uses dedicated train_fold (handles transform internally);
        # AT and AFT use generic functions with VGGishMelSpectrogramTransform passed in.
        vggish_transform = VGGishMelSpectrogramTransform()

        if mode == "normal":
            train_result = vggish_train_fold(fold=fold, cfg=fold_cfg)

        elif mode in ("adv_train_fgsm", "adv_train_bim"):
            train_result = adv_train_fold(
                fold=fold, cfg=fold_cfg, model_class=model_class,
                transform=vggish_transform,
            )

        elif mode in ("adv_finetune_fgsm", "adv_finetune_bim"):
            normal_ckpt = os.path.join(
                SAVED_MODELS_ROOT, model_name, "normal", f"best_fold{fold}.pt"
            )
            if not os.path.exists(normal_ckpt):
                raise FileNotFoundError(
                    f"VGGish baseline checkpoint not found: {normal_ckpt}\n"
                    "Run 'python run_experiments.py --model vggish --mode normal' first."
                )
            train_result = finetune_fold(
                fold=fold, pretrained_ckpt=normal_ckpt,
                cfg=fold_cfg, model_class=model_class,
                transform=vggish_transform,
            )

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    else:
        # CNN — generic training functions
        if mode == "normal":
            train_result = train_fold(
                fold=fold, epochs=fold_cfg["epochs"],
                cfg=fold_cfg, model_class=model_class,
            )

        elif mode in ("adv_train_fgsm", "adv_train_bim"):
            train_result = adv_train_fold(
                fold=fold, cfg=fold_cfg, model_class=model_class,
            )

        elif mode in ("adv_finetune_fgsm", "adv_finetune_bim"):
            normal_ckpt = os.path.join(
                SAVED_MODELS_ROOT, model_name, "normal", f"best_fold{fold}.pt"
            )
            if not os.path.exists(normal_ckpt):
                raise FileNotFoundError(
                    f"Normal-training checkpoint not found: {normal_ckpt}\n"
                    f"Run 'normal' mode for {model_name} first."
                )
            train_result = finetune_fold(
                fold=fold, pretrained_ckpt=normal_ckpt,
                cfg=fold_cfg, model_class=model_class,
            )

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    eval_transform = VGGishMelSpectrogramTransform() if model_name == "vggish" else None
    metrics = evaluate_checkpoint(
        checkpoint_path = train_result["checkpoint_path"],
        data_root       = DATA_ROOT,
        fold            = fold,
        batch_size      = fold_cfg["batch_size"],
        num_workers     = fold_cfg["num_workers"],
        verbose         = True,
        model_class     = model_class,
        transform       = eval_transform,
    )

    fold_record = {
        "fold":             fold,
        "checkpoint":       train_result["checkpoint_path"],
        "best_val_acc":     train_result["best_val_acc"],
        "training_history": train_result["history"],
        "test_metrics":     {k: round(v, 4) for k, v in metrics.items()},
    }
    _save_json(os.path.join(results_dir, f"fold_{fold}_results.json"), fold_record)
    return {**metrics, "best_val_acc": train_result["best_val_acc"], "fold": fold}


# ── single experiment (one model + one mode, all 10 folds) ───────────────────

def run_experiment(model_name, mode, max_samples_per_fold=None, epochs=None):
    """Run the full 10-fold CV for one model/mode combination."""
    model_class = MODELS[model_name]
    if model_name == "crnn":
        cfg = CRNN_MODE_CONFIGS[mode]
    elif model_name == "vggish":
        cfg = VGGISH_MODE_CONFIGS[mode]
    else:
        cfg = MODE_CONFIGS[mode]
    results_dir, saved_models_dir = _experiment_dirs(model_name, mode)

    print(f"\n{'#'*60}")
    print(f"#  Model : {model_name.upper()}   Mode: {mode}")
    print(f"#  Results    : {results_dir}")
    print(f"#  Checkpoints: {saved_models_dir}")
    print(f"{'#'*60}")

    # save config immediately so it persists even if training is interrupted
    attack_info = {}
    if "attack_type" in cfg:
        attack_info = {
            "attack_type": cfg["attack_type"],
            "adv_epsilon": cfg["adv_epsilon"],
            "adv_ratio":   cfg["adv_ratio"],
        }
        if cfg["attack_type"] == "bim":
            attack_info["bim_steps"] = cfg.get("bim_steps", 7)

    # --epochs flag overrides the default from config
    if epochs is not None:
        cfg = dict(cfg)   # shallow copy — don't mutate the module-level config
        if mode.startswith("adv_finetune"):
            cfg["finetune_epochs"] = epochs
        else:
            cfg["epochs"] = epochs

    effective_epochs = cfg.get("finetune_epochs", cfg["epochs"])

    config_snapshot = {
        "model":      model_name,
        "mode":       mode,
        "epochs":     effective_epochs,
        "batch_size": cfg["batch_size"],
        "lr":         cfg.get("finetune_lr", cfg["lr"]),
        **attack_info,
    }
    _save_json(os.path.join(results_dir, "config.json"), config_snapshot)

    wall_start   = time.time()
    fold_results = []

    for fold in range(1, 11):
        print(f"\n{'#'*60}")
        print(f"#  FOLD {fold}/10")
        print(f"{'#'*60}")
        fold_results.append(
            run_fold(fold, model_name, mode, model_class,
                     results_dir, saved_models_dir, cfg, max_samples_per_fold)
        )

    # ── per-fold summary table ────────────────────────────────────────────────
    metric_keys = [
        ("Accuracy",          "accuracy"),
        ("Precision (macro)", "precision_macro"),
        ("Precision (wtd)",   "precision_weighted"),
        ("Recall (macro)",    "recall_macro"),
        ("Recall (wtd)",      "recall_weighted"),
        ("F1 Score (macro)",  "f1_macro"),
        ("F1 Score (wtd)",    "f1_weighted"),
    ]

    print(f"\n{'='*78}")
    print(f"  Per-Fold Results  [{model_name.upper()} / {mode}]")
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

    mean_metrics, std_metrics = {}, {}
    print(f"\n{'='*52}")
    print(f"  Mean ± Std  [{model_name.upper()} / {mode}]")
    print(f"{'='*52}")
    for label, key in metric_keys:
        vals = [r[key] for r in fold_results]
        m, s = _mean(vals), _std(vals)
        mean_metrics[key] = m
        std_metrics[key]  = s
        print(f"  {label:<24}  {m*100:6.2f}%  ±  {s*100:.2f}%")
    print(f"{'='*52}")

    elapsed   = time.time() - wall_start
    h, rem    = divmod(int(elapsed), 3600)
    mn, sc    = divmod(rem, 60)
    wall_time = f"{h:02d}h {mn:02d}m {sc:02d}s"

    # ── attack evaluation ─────────────────────────────────────────────────────
    epsilons  = _eval_cfg["eval_epsilons"]
    bim_steps = _eval_cfg["bim_eval_steps"]
    device    = _device()

    clean_accuracies = [r["accuracy"] * 100 for r in fold_results]

    print(f"\n{'='*60}")
    print(f" FGSM Attack Evaluation  [{model_name.upper()} / {mode}]")
    print(f"{'='*60}")
    attack_transform = VGGishMelSpectrogramTransform() if model_name == "vggish" else None

    if model_name == "crnn":
        fgsm_results = run_fgsm_all_folds_crnn(
            model_class      = model_class,
            saved_models_dir = saved_models_dir,
            data_root        = DATA_ROOT,
            device           = device,
            epsilons         = epsilons,
            batch_size       = cfg["batch_size"],
            num_workers      = cfg["num_workers"],
        )
    else:
        fgsm_results = run_fgsm_all_folds(
            model_class      = model_class,
            saved_models_dir = saved_models_dir,
            data_root        = DATA_ROOT,
            device           = device,
            epsilons         = epsilons,
            batch_size       = cfg["batch_size"],
            num_workers      = cfg["num_workers"],
            transform        = attack_transform,
        )
    print_fgsm_results(clean_accuracies, fgsm_results, epsilons)

    print(f"\n{'='*60}")
    print(f" BIM Attack Evaluation  [{model_name.upper()} / {mode}]  steps={bim_steps}")
    print(f"{'='*60}")
    if model_name == "crnn":
        bim_results = run_bim_all_folds_crnn(
            model_class      = model_class,
            saved_models_dir = saved_models_dir,
            data_root        = DATA_ROOT,
            device           = device,
            epsilons         = epsilons,
            steps            = bim_steps,
            batch_size       = cfg["batch_size"],
            num_workers      = cfg["num_workers"],
        )
    else:
        bim_results = run_bim_all_folds(
            model_class      = model_class,
            saved_models_dir = saved_models_dir,
            data_root        = DATA_ROOT,
            device           = device,
            epsilons         = epsilons,
            steps            = bim_steps,
            batch_size       = cfg["batch_size"],
            num_workers      = cfg["num_workers"],
            transform        = attack_transform,
        )
    print_bim_results(clean_accuracies, bim_results, epsilons)

    # ── cv_summary.json ───────────────────────────────────────────────────────
    cv_summary = {
        "model":     model_name,
        "mode":      mode,
        "wall_time": wall_time,
        "config":    config_snapshot,
        "per_fold": [
            {"fold": r["fold"], **{key: round(r[key], 4) for _, key in metric_keys}}
            for r in fold_results
        ],
        "mean": {k: round(v, 4) for k, v in mean_metrics.items()},
        "std":  {k: round(v, 4) for k, v in std_metrics.items()},
        "fgsm": {
            f"fold_{f}": fgsm_results[f"fold_{f}"]
            for f in range(1, 11) if f"fold_{f}" in fgsm_results
        },
        "bim": {
            f"fold_{f}": bim_results[f"fold_{f}"]
            for f in range(1, 11) if f"fold_{f}" in bim_results
        },
    }
    _save_json(os.path.join(results_dir, "cv_summary.json"), cv_summary)
    print(f"\n  Total wall time: {wall_time}")

    return mean_metrics


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run thesis experiments (15 total)")
    parser.add_argument(
        "--model", choices=list(MODELS.keys()) + ["all"], default="all",
        help="Model to run (default: all)"
    )
    parser.add_argument(
        "--mode", choices=MODES + ["all"], default="all",
        help="Training mode to run (default: all)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override training epochs (default: 30 for normal/adv_train, 15 for adv_finetune)"
    )
    parser.add_argument(
        "--quick", type=int, default=None, metavar="N",
        help="Smoke-test: use N random samples per fold"
    )
    args = parser.parse_args()

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    modes_to_run  = MODES               if args.mode  == "all" else [args.mode]

    # preserve dependency order: normal must run before adv_finetune_*
    ordered_modes = [m for m in MODES if m in modes_to_run]

    for model_name in models_to_run:
        for mode in ordered_modes:
            run_experiment(
                model_name           = model_name,
                mode                 = mode,
                max_samples_per_fold = args.quick,
                epochs               = args.epochs,
            )

    print("\nAll experiments complete.")

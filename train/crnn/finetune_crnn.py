"""
CRNN adversarial fine-tuning (AFT) — 10-fold cross-validation on UrbanSound8K.

What differs from CNN adversarial fine-tuning (train/finetune_adversarial.py)
-------------------------------------------------------------------------------
  1. Gradient clipping — clip_grad_norm_(model.parameters(), max_norm=1.0)
     applied before every optimizer.step().

     Reason: BPTT through the GRU can cause large gradients during fine-tuning,
     especially in early epochs when clean temporal patterns and adversarial
     robustness are simultaneously being adjusted.

  2. Proper val split — get_fold_dataloaders_with_val (10 % of train folds)
     consistent with CRNN baseline and AT training.

  3. Clean accuracy monitoring per epoch.
     The GRU layers are more susceptible to catastrophic forgetting than CNN
     layers because they encode temporal dependencies.  If clean val_acc drops
     more than 3 pp in the first 3 epochs, this is reported as a finding —
     it indicates the GRU is losing clean temporal patterns too quickly.

  4. Checkpoint criterion:
     - FGSM: saved by best adversarial val accuracy (same as CRNN AT)
     - BIM:  saved by best clean val accuracy (same as CNN finetune BIM)

  5. Lower LR (0.0001 vs 0.001) — preserves pretrained features (Jeddi et al. 2020).

Usage
-----
  python train/crnn/finetune_crnn.py --attack fgsm
  python train/crnn/finetune_crnn.py --attack bim
  python train/crnn/finetune_crnn.py --attack fgsm --quick 100   # smoke-test

  Requires: saved_models/crnn/normal/best_fold{1..10}.pt
  Run train/crnn/train_crnn.py first.
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

from models.crnn import CRNN
from preprocessing.mel_spectrogram import MelSpectrogramTransform
from dataset.urbansound_dataset import get_fold_dataloaders_with_val
from attacks.fgsm import get_fgsm_attack
from attacks.bim  import get_bim_attack
from train.train import get_device, evaluate
from config_loader import get_crnn_finetune_fgsm_config, get_crnn_finetune_bim_config


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
    """Evaluate on attack-perturbed inputs.  Model in eval mode throughout."""
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


# ── one AFT epoch with gradient clipping ──────────────────────────────────────

def _finetune_epoch(model, loader, criterion, optimizer, device, cfg):
    """One fine-tuning epoch: mixed clean + adversarial batch with gradient clipping.

    Identical mixing strategy to AT (adv_ratio, fixed batch size).
    clip_grad_norm_ applied before optimizer.step() for GRU BPTT stability.

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

        model.eval()
        with torch.backends.cudnn.flags(enabled=False):
            adv_specs = attack(specs[:n_adv], labels[:n_adv]).detach()
        model.train()

        mixed  = torch.cat([adv_specs, specs[n_adv:]], dim=0)
        logits = model(mixed)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += batch_size

    return total_loss / total, 100.0 * correct / total


# ── fold runner ────────────────────────────────────────────────────────────────

def finetune_fold(fold: int, pretrained_ckpt: str, cfg: dict) -> dict:
    """Fine-tune CRNN baseline checkpoint with adversarial examples.

    Algorithm (FGSM path):
        1. Load crnn_baseline_fold{N}.pt — clean features already learned.
        2. Adam at lr=0.0001 (10× lower than AT to preserve clean knowledge).
        3. Each batch: mix clean + adversarial at adv_ratio fraction.
        4. After each epoch: evaluate both clean and adversarial val accuracy.
        5. Save checkpoint when adversarial val accuracy improves (FGSM).
           Save when clean val accuracy improves (BIM).

    CRNN-specific monitoring:
        Prints a warning if clean val_acc drops > 3 pp in the first 3 epochs.
        GRU temporal features are more susceptible to catastrophic forgetting
        than CNN spatial features.  A large early drop is a thesis finding.

    Args:
        fold:            Test fold (1–10).
        pretrained_ckpt: Path to crnn normal checkpoint (best_fold{N}.pt).
        cfg:             Config from get_crnn_finetune_fgsm/bim_config().

    Returns:
        {best_val_acc, best_adv_val_acc, history, checkpoint_path, wall_s}
        best_val_acc is the checkpoint criterion metric (adv for FGSM, clean for BIM).
    """
    device      = get_device()
    attack_type = cfg["attack_type"]
    epochs      = cfg.get("finetune_epochs", 10)
    epsilon     = cfg["adv_epsilon"]

    print(f"\n{'='*66}")
    print(f" [CRNN AFT / {attack_type.upper()}]  Fold {fold}/10  |  device: {device}")
    print(f" Pretrained : {pretrained_ckpt}")
    print(f" ε={epsilon}  ratio={cfg['adv_ratio']}  epochs={epochs}  "
          f"lr={cfg['finetune_lr']}", end="")
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

    # ── load pretrained baseline weights ──────────────────────────────────────
    model      = CRNN(num_classes=cfg["num_classes"], dropout=cfg["dropout"]).to(device)
    checkpoint = torch.load(pretrained_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    baseline_clean_acc = checkpoint.get("val_acc", float("nan"))
    print(f" Loaded pretrained weights (val_acc: {baseline_clean_acc:.2f}%)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["finetune_lr"])

    # val attack for checkpoint criterion evaluation (FGSM path only)
    fgsm_val_attack = get_fgsm_attack(model, epsilon) if attack_type == "fgsm" else None

    os.makedirs(cfg["save_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["save_dir"], f"best_fold{fold}.pt")

    best_adv_val_acc   = -1.0
    best_clean_val_acc = -1.0
    history = {
        "train_loss":   [], "train_acc":   [],
        "val_loss":     [], "val_acc":     [],
        "adv_val_loss": [], "adv_val_acc": [],
    }

    # clean accuracy monitoring — track first 3 epochs for GRU forgetting
    initial_clean_acc = None

    fold_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = _finetune_epoch(
            model, train_loader, criterion, optimizer, device, cfg
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if attack_type == "fgsm":
            adv_val_loss, adv_val_acc = _evaluate_adversarial(
                model, val_loader, criterion, fgsm_val_attack, device
            )
            improved = adv_val_acc > best_adv_val_acc
        else:
            adv_val_loss = adv_val_acc = float("nan")
            improved = val_acc > best_clean_val_acc

        elapsed = time.time() - t0
        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc,  4))
        history["val_loss"].append(round(val_loss,    6))
        history["val_acc"].append(round(val_acc,      4))
        history["adv_val_loss"].append(adv_val_loss if not math.isnan(adv_val_loss) else None)
        history["adv_val_acc"].append(adv_val_acc   if not math.isnan(adv_val_acc)  else None)

        # ── GRU catastrophic forgetting monitor ───────────────────────────────
        if epoch == 1:
            initial_clean_acc = val_acc
        if epoch <= 3 and initial_clean_acc is not None:
            clean_drop = initial_clean_acc - val_acc
            if clean_drop > 3.0:
                print(
                    f"  [GRU FORGETTING WARNING] Clean val_acc dropped {clean_drop:.2f} pp "
                    f"by epoch {epoch} (initial: {initial_clean_acc:.2f}%, "
                    f"current: {val_acc:.2f}%).  "
                    "GRU temporal features may be forgetting too fast — record as thesis finding."
                )

        if improved:
            if attack_type == "fgsm":
                best_adv_val_acc = adv_val_acc
            else:
                best_clean_val_acc = val_acc
            torch.save(
                {
                    "epoch":           epoch,
                    "fold":            fold,
                    "model_state_dict": model.state_dict(),
                    "val_acc":         val_acc,
                    "adv_val_acc":     adv_val_acc,
                    "val_loss":        val_loss,
                    "adv_val_loss":    adv_val_loss,
                    "config":          cfg,
                    "pretrained_ckpt": pretrained_ckpt,
                },
                ckpt_path,
            )

        marker = "  * saved" if improved else ""

        if attack_type == "fgsm":
            print(
                f" Epoch {epoch:>2}/{epochs} | "
                f"train {train_loss:.4f}/{train_acc:5.1f}% | "
                f"clean {val_loss:.4f}/{val_acc:5.1f}% | "
                f"adv {adv_val_loss:.4f}/{adv_val_acc:5.1f}%  "
                f"[{elapsed:.1f}s]{marker}"
            )
        else:
            print(
                f" Epoch {epoch:>2}/{epochs} | "
                f"train {train_loss:.4f}/{train_acc:5.1f}% | "
                f"clean {val_loss:.4f}/{val_acc:5.1f}%  "
                f"[{elapsed:.1f}s]{marker}"
            )

    fold_wall = time.time() - fold_start
    best_for_return = best_adv_val_acc if attack_type == "fgsm" else best_clean_val_acc

    # Final clean accuracy drop vs baseline
    final_clean_acc = history["val_acc"][-1] if history["val_acc"] else float("nan")
    clean_drop_total = baseline_clean_acc - final_clean_acc
    print(f"\n Best {'adv ' if attack_type == 'fgsm' else ''}val acc fold {fold}: "
          f"{best_for_return:.2f}%  |  clean acc drop: {clean_drop_total:.2f} pp  |  "
          f"wall: {fold_wall/3600:.2f}h")

    return {
        "best_val_acc":     best_for_return,   # key expected by run_experiments.py
        "best_adv_val_acc": best_adv_val_acc,
        "history":          history,
        "checkpoint_path":  ckpt_path,
        "wall_s":           round(fold_wall, 1),
        "clean_acc_drop_pp": round(clean_drop_total, 4),
    }


# ── full 10-fold run ───────────────────────────────────────────────────────────

def run_all_folds(cfg: dict, results_dir: str, baseline_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    attack_type = cfg["attack_type"]

    config_snapshot = {k: v for k, v in cfg.items() if k != "data_root"}
    config_snapshot["model"] = "crnn"
    config_snapshot["mode"]  = f"adv_finetune_{attack_type}"
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    total_start  = time.time()
    fold_results = []

    for fold in range(1, 11):
        baseline_ckpt = os.path.join(baseline_dir, f"best_fold{fold}.pt")
        if not os.path.exists(baseline_ckpt):
            raise FileNotFoundError(
                f"CRNN baseline checkpoint not found: {baseline_ckpt}\n"
                "Run 'python train/crnn/train_crnn.py' first."
            )
        result = finetune_fold(fold, baseline_ckpt, cfg)
        fold_results.append(result)
        with open(os.path.join(results_dir, f"fold_{fold}_results.json"), "w") as f:
            json.dump(result, f, indent=2)

    total_wall = time.time() - total_start
    h, rem   = divmod(int(total_wall), 3600)
    mn, sc   = divmod(rem, 60)
    wall_str = f"{h:02d}h {mn:02d}m {sc:02d}s"

    best_accs = [r["best_val_acc"] for r in fold_results]
    mean_acc  = sum(best_accs) / len(best_accs)
    std_acc   = math.sqrt(sum((a - mean_acc) ** 2 for a in best_accs) / len(best_accs))

    # Clean accuracy drop summary across folds
    clean_drops = [r.get("clean_acc_drop_pp", float("nan")) for r in fold_results]
    valid_drops = [d for d in clean_drops if not math.isnan(d)]
    mean_drop   = sum(valid_drops) / len(valid_drops) if valid_drops else float("nan")

    print(f"\n CRNN AFT-{attack_type.upper()} complete")
    print(f" Mean best val acc   : {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f" Mean clean acc drop : {mean_drop:.2f} pp  (CNN AFT reference: 2.54 pp)")
    print(f" Wall time           : {wall_str}")

    cv_summary = {
        "model":     "crnn",
        "mode":      f"adv_finetune_{attack_type}",
        "wall_time": wall_str,
        "mean_best_val_acc":    round(mean_acc / 100, 4),
        "std_best_val_acc":     round(std_acc  / 100, 4),
        "mean_clean_acc_drop_pp": round(mean_drop, 4),
        "per_fold": [
            {"fold": r["checkpoint_path"].split("fold")[-1].replace(".pt", ""),
             "best_val_acc": round(r["best_val_acc"] / 100, 4),
             "clean_acc_drop_pp": r.get("clean_acc_drop_pp", None)}
            for r in fold_results
        ],
    }
    with open(os.path.join(results_dir, "cv_summary.json"), "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f" Results saved : {results_dir}")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    parser = argparse.ArgumentParser(description="CRNN adversarial fine-tuning — 10-fold CV")
    parser.add_argument("--attack", choices=["fgsm", "bim"], default="fgsm",
                        help="Attack type for adversarial fine-tuning (default: fgsm)")
    parser.add_argument("--quick", type=int, default=0, metavar="N",
                        help="Smoke-test: limit each fold to N samples (0 = full dataset)")
    args = parser.parse_args()

    if args.attack == "fgsm":
        cfg = get_crnn_finetune_fgsm_config()
    else:
        cfg = get_crnn_finetune_bim_config()

    if args.quick:
        cfg["max_samples_per_fold"] = args.quick

    baseline_dir = os.path.join(_ROOT, "saved_models", "crnn", "normal")
    results_dir  = os.path.join(_ROOT, "results",      "crnn", f"adv_finetune_{args.attack}")
    run_all_folds(cfg, results_dir, baseline_dir)

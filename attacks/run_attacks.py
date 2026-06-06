import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from attacks.fgsm import evaluate_fgsm
from attacks.bim  import evaluate_bim
from dataset.urbansound_dataset import get_fold_dataloaders


# ── FGSM ─────────────────────────────────────────────────────────────────────

def print_fgsm_results(clean_accuracies, fgsm_results, epsilons):
    """Print a side-by-side table of clean vs FGSM adversarial accuracy per fold."""
    print("\n" + "=" * 70)
    print(" FGSM Attack Results — All Folds")
    print("=" * 70)

    header = f"{'Fold':<8} {'Clean':>8}"
    for eps in epsilons:
        header += f" {'ε='+str(eps):>10}"
    print(header)
    print("-" * 70)

    for fold in range(1, 11):
        fold_key = f"fold_{fold}"
        if fold_key not in fgsm_results:
            continue
        row = f"Fold {fold:<3} {clean_accuracies[fold-1]:>7.2f}%"
        for eps in epsilons:
            adv_acc = fgsm_results[fold_key][f"eps_{eps}"]
            row += f" {adv_acc:>9.2f}%"
        print(row)

    print("-" * 70)

    completed_folds = [f for f in range(1, 11) if f"fold_{f}" in fgsm_results]
    avg_clean = sum(clean_accuracies[f-1] for f in completed_folds) / len(completed_folds)
    avg_row = f"{'Mean':<8} {avg_clean:>7.2f}%"
    for eps in epsilons:
        key = f"eps_{eps}"
        avg_adv = sum(
            fgsm_results[f"fold_{f}"][key] for f in completed_folds
        ) / len(completed_folds)
        avg_row += f" {avg_adv:>9.2f}%"
    print(avg_row)
    print("=" * 70)


def run_fgsm_all_folds(model_class, saved_models_dir, data_root,
                        device, epsilons, batch_size=32, num_workers=4):
    """Run FGSM attack on all 10 folds.

    Args:
        model_class:      Model class (e.g. UrbanSoundCNN).
        saved_models_dir: Directory containing best_fold{k}.pt checkpoints.
        data_root:        Path to UrbanSound8K root directory.
        device:           torch.device.
        epsilons:         List of epsilon values to test.
        batch_size:       DataLoader batch size.
        num_workers:      DataLoader workers.

    Returns:
        dict mapping fold_k -> {eps_e: adversarial_accuracy}
    """
    missing = [
        f"best_fold{f}.pt"
        for f in range(1, 11)
        if not os.path.exists(os.path.join(saved_models_dir, f"best_fold{f}.pt"))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints in '{saved_models_dir}': {missing}\n"
            "Train all 10 folds first."
        )

    results = {}

    for fold in range(1, 11):
        print(f"\n[FGSM] Attacking Fold {fold}...")

        checkpoint_path = os.path.join(saved_models_dir, f"best_fold{fold}.pt")
        model = model_class().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        _, test_loader = get_fold_dataloaders(
            root_dir    = data_root,
            test_fold   = fold,
            batch_size  = batch_size,
            num_workers = num_workers,
        )

        fold_results = {}
        for eps in epsilons:
            adv_acc = evaluate_fgsm(model, test_loader, eps, device)
            fold_results[f"eps_{eps}"] = adv_acc
            print(f"  Fold {fold} | ε={eps} | Adv Acc: {adv_acc:.2f}%")

        results[f"fold_{fold}"] = fold_results

    return results


# ── BIM ──────────────────────────────────────────────────────────────────────

def print_bim_results(clean_accuracies, bim_results, epsilons):
    """Print a side-by-side table of clean vs BIM adversarial accuracy per fold."""
    print("\n" + "=" * 70)
    print(" BIM Attack Results — All Folds")
    print("=" * 70)

    header = f"{'Fold':<8} {'Clean':>8}"
    for eps in epsilons:
        header += f" {'ε='+str(eps):>10}"
    print(header)
    print("-" * 70)

    for fold in range(1, 11):
        fold_key = f"fold_{fold}"
        if fold_key not in bim_results:
            continue
        row = f"Fold {fold:<3} {clean_accuracies[fold-1]:>7.2f}%"
        for eps in epsilons:
            adv_acc = bim_results[fold_key][f"eps_{eps}"]
            row += f" {adv_acc:>9.2f}%"
        print(row)

    print("-" * 70)

    completed_folds = [f for f in range(1, 11) if f"fold_{f}" in bim_results]
    avg_clean = sum(clean_accuracies[f-1] for f in completed_folds) / len(completed_folds)
    avg_row = f"{'Mean':<8} {avg_clean:>7.2f}%"
    for eps in epsilons:
        key = f"eps_{eps}"
        avg_adv = sum(
            bim_results[f"fold_{f}"][key] for f in completed_folds
        ) / len(completed_folds)
        avg_row += f" {avg_adv:>9.2f}%"
    print(avg_row)
    print("=" * 70)


def run_bim_all_folds(model_class, saved_models_dir, data_root,
                      device, epsilons, steps=10, batch_size=32, num_workers=4):
    """Run BIM attack on all 10 folds.

    Alpha (per-step size) is set to epsilon / steps for each epsilon value,
    which ensures the full budget is reachable within the step count.

    Args:
        model_class:      Model class (e.g. UrbanSoundCNN).
        saved_models_dir: Directory containing best_fold{k}.pt checkpoints.
        data_root:        Path to UrbanSound8K root directory.
        device:           torch.device.
        epsilons:         List of epsilon values to test.
        steps:            Number of iterative steps (default: 10).
        batch_size:       DataLoader batch size.
        num_workers:      DataLoader workers.

    Returns:
        dict mapping fold_k -> {eps_e: adversarial_accuracy}
    """
    missing = [
        f"best_fold{f}.pt"
        for f in range(1, 11)
        if not os.path.exists(os.path.join(saved_models_dir, f"best_fold{f}.pt"))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints in '{saved_models_dir}': {missing}\n"
            "Train all 10 folds first."
        )

    results = {}

    for fold in range(1, 11):
        print(f"\n[BIM] Attacking Fold {fold}...")

        checkpoint_path = os.path.join(saved_models_dir, f"best_fold{fold}.pt")
        model = model_class().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        _, test_loader = get_fold_dataloaders(
            root_dir    = data_root,
            test_fold   = fold,
            batch_size  = batch_size,
            num_workers = num_workers,
        )

        fold_results = {}
        for eps in epsilons:
            alpha   = eps / steps   # per-step budget
            adv_acc = evaluate_bim(model, test_loader, eps, alpha, steps, device)
            fold_results[f"eps_{eps}"] = adv_acc
            print(f"  Fold {fold} | ε={eps}  α={alpha:.4f}  steps={steps} | Adv Acc: {adv_acc:.2f}%")

        results[f"fold_{fold}"] = fold_results

    return results


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from models.cnn import UrbanSoundCNN
    from train.train import CONFIG

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    saved_models_dir = os.path.join(
        os.path.dirname(__file__), "..", "saved_models", "cnn", "normal"
    )
    epsilons = [0.01, 0.03, 0.1]

    # ── clean accuracy ────────────────────────────────────────────────────────
    clean_accuracies = []
    for fold in range(1, 11):
        ckpt_path = os.path.join(saved_models_dir, f"best_fold{fold}.pt")
        model = UrbanSoundCNN().to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        _, test_loader = get_fold_dataloaders(
            root_dir    = CONFIG["data_root"],
            test_fold   = fold,
            batch_size  = 32,
            num_workers = CONFIG["num_workers"],
        )

        correct = total = 0
        with torch.no_grad():
            for specs, labels in test_loader:
                specs, labels = specs.to(device), labels.to(device)
                preds  = model(specs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        acc = 100 * correct / total
        clean_accuracies.append(acc)
        print(f" Fold {fold:>2} clean accuracy: {acc:.2f}%")

    # ── FGSM ──────────────────────────────────────────────────────────────────
    fgsm_results = run_fgsm_all_folds(
        model_class      = UrbanSoundCNN,
        saved_models_dir = saved_models_dir,
        data_root        = CONFIG["data_root"],
        device           = device,
        epsilons         = epsilons,
    )
    print_fgsm_results(clean_accuracies, fgsm_results, epsilons)

    # ── BIM ───────────────────────────────────────────────────────────────────
    bim_results = run_bim_all_folds(
        model_class      = UrbanSoundCNN,
        saved_models_dir = saved_models_dir,
        data_root        = CONFIG["data_root"],
        device           = device,
        epsilons         = epsilons,
        steps            = 10,
    )
    print_bim_results(clean_accuracies, bim_results, epsilons)

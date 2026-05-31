import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from attacks.fgsm import evaluate_fgsm
from dataset.urbansound_dataset import get_fold_dataloaders


def print_fgsm_results(clean_accuracies, fgsm_results, epsilons):
    """Print a side-by-side table of clean vs adversarial accuracy per fold.

    clean_accuracies: list of 10 floats in % (e.g. 71.23)
    fgsm_results:     dict from run_fgsm_all_folds
    epsilons:         list of epsilon values used
    """
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
    """
    Run FGSM attack on all 10 folds.

    model_class:      CNN class (e.g. UrbanSoundCNN)
    saved_models_dir: directory containing best_fold{k}.pt checkpoints
    data_root:        path to UrbanSound8K root directory
    device:           torch.device (cuda / mps / cpu)
    epsilons:         list of epsilon values to test
    batch_size:       DataLoader batch size
    num_workers:      DataLoader workers

    returns: dict mapping fold_k -> {eps_e: adversarial_accuracy}
    """
    # Check 4 — verify all 10 checkpoints exist before starting so we fail
    # fast instead of discovering a missing file mid-run after waiting hours.
    missing = [
        f"best_fold{f}.pt"
        for f in range(1, 11)
        if not os.path.exists(os.path.join(saved_models_dir, f"best_fold{f}.pt"))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints in '{saved_models_dir}': {missing}\n"
            "Run main.py to train all 10 folds first."
        )

    # Check 3 — the test_loader built here uses the same MelSpectrogramTransform
    # and min-max normalisation as training, so adversarial perturbations operate
    # on the same [0, 1] normalised scale the model was trained on. Do not swap
    # in a different loader or bypass preprocessing between training and this call.

    results = {}

    for fold in range(1, 11):

        print(f"\nAttacking Fold {fold}...")

        # ── load checkpoint ───────────────────────────────────────────────────
        checkpoint_path = os.path.join(saved_models_dir, f"best_fold{fold}.pt")

        model = model_class().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # ── build test loader for this fold ───────────────────────────────────
        _, test_loader = get_fold_dataloaders(
            root_dir    = data_root,
            test_fold   = fold,
            batch_size  = batch_size,
            num_workers = num_workers,
        )

        fold_results = {}

        # ── test each epsilon value ───────────────────────────────────────────
        for eps in epsilons:
            adv_acc = evaluate_fgsm(model, test_loader, eps, device)
            fold_results[f"eps_{eps}"] = adv_acc
            print(f"  Fold {fold} | ε={eps} | Adversarial Accuracy: {adv_acc:.2f}%")

        results[f"fold_{fold}"] = fold_results

    return results


if __name__ == "__main__":
    import torch
    from models.cnn import UrbanSoundCNN
    from train.train import CONFIG

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    epsilons = [0.01, 0.03, 0.1]

    print(f"\n{'='*60}")
    print(" FGSM Attack Evaluation")
    print(f"{'='*60}")
    print(f" Device     : {device}")
    print(f" Epsilons   : {epsilons}")
    print(f" Models dir : {CONFIG['save_dir']}")

    # ── compute clean accuracy per fold before attacking ──────────────────────
    clean_accuracies = []

    for fold in range(1, 11):
        ckpt_path = os.path.join(CONFIG["save_dir"], f"best_fold{fold}.pt")
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
                preds = model(specs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        acc = 100 * correct / total
        clean_accuracies.append(acc)
        print(f" Fold {fold:>2} clean accuracy: {acc:.2f}%")

    # ── run FGSM ──────────────────────────────────────────────────────────────
    fgsm_results = run_fgsm_all_folds(
        model_class      = UrbanSoundCNN,
        saved_models_dir = CONFIG["save_dir"],
        data_root        = CONFIG["data_root"],
        device           = device,
        epsilons         = epsilons,
    )

    print_fgsm_results(clean_accuracies, fgsm_results, epsilons)
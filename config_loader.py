import os
import yaml

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
_CONFIG_PATH  = os.path.join(_PROJECT_ROOT, "config.yaml")


def _load_yaml() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_base_config() -> dict:
    raw = _load_yaml()
    t   = raw["training"]
    return {
        "data_root":   os.path.join(_PROJECT_ROOT, t["data_root"]),
        "save_dir":    os.path.join(_PROJECT_ROOT, t["save_dir"]),
        "batch_size":  t["batch_size"],
        "lr":          t["lr"],
        "epochs":      t["epochs"],
        "num_workers": t["num_workers"],
        "dropout":     t["dropout"],
        "num_classes": t["num_classes"],
    }


def get_adv_fgsm_config() -> dict:
    raw = _load_yaml()
    cfg = get_base_config()
    atk = raw["attacks"]["fgsm"]
    cfg.update({
        "attack_type": "fgsm",
        "adv_epsilon": atk["epsilon"],
        "adv_ratio":   atk["adv_ratio"],
    })
    return cfg


def get_adv_bim_config() -> dict:
    raw = _load_yaml()
    cfg = get_base_config()
    atk = raw["attacks"]["bim"]
    cfg.update({
        "attack_type": "bim",
        "adv_epsilon": atk["epsilon"],
        "adv_ratio":   atk["adv_ratio"],
        "bim_steps":   atk["train_steps"],
    })
    return cfg


def get_eval_attack_config() -> dict:
    """Return epsilon sweep and BIM eval steps used during robustness evaluation."""
    raw = _load_yaml()
    atk = raw["attacks"]
    return {
        "eval_epsilons": atk["eval_epsilons"],
        "bim_eval_steps": atk["bim"]["eval_steps"],
    }


def get_finetune_fgsm_config() -> dict:
    raw = _load_yaml()
    cfg = get_adv_fgsm_config()
    ft  = raw["finetune"]
    cfg.update({
        "finetune_epochs": ft["epochs"],
        "finetune_lr":     ft["lr"],
    })
    return cfg


def get_finetune_bim_config() -> dict:
    raw = _load_yaml()
    cfg = get_adv_bim_config()
    ft  = raw["finetune"]
    cfg.update({
        "finetune_epochs": ft["epochs"],
        "finetune_lr":     ft["lr"],
    })
    return cfg


# ── CRNN configs ──────────────────────────────────────────────────────────────
# CRNN requires dedicated configs because it differs from CNN/VGGish in three ways:
#   1. 50 epochs (not 30) for baseline and AT
#   2. Validation split (val_fraction=0.1) for proper early stopping
#   3. LR scheduler settings carried through to AT/AFT scripts

def get_crnn_config() -> dict:
    """CRNN baseline training config.

    50 epochs with ReduceLROnPlateau scheduler and early stopping (patience=15).
    10 % of train folds held out as validation for early stopping criterion.
    No weight_decay — matches CNN; VGGish uses 1e-4.
    """
    raw = _load_yaml()
    t   = raw["training"]
    c   = raw["crnn"]
    return {
        "data_root":             os.path.join(_PROJECT_ROOT, t["data_root"]),
        "save_dir":              os.path.join(_PROJECT_ROOT, c["save_dir"]),
        "batch_size":            c["batch_size"],
        "num_workers":           c["num_workers"],
        "lr":                    c["lr"],
        "epochs":                c["epochs"],
        "dropout":               c["dropout"],
        "num_classes":           c["num_classes"],
        "val_fraction":          c["val_fraction"],
        "early_stop_patience":   c["early_stop_patience"],
        "lr_scheduler_factor":   c["lr_scheduler_factor"],
        "lr_scheduler_patience": c["lr_scheduler_patience"],
    }


def get_crnn_adv_fgsm_config() -> dict:
    """CRNN adversarial training with FGSM.

    Inherits baseline hyperparameters (50 epochs, val_fraction, scheduler).
    Checkpoint saved by adversarial val accuracy (not clean val acc).
    """
    raw = _load_yaml()
    cfg = get_crnn_config()
    atk = raw["attacks"]["fgsm"]
    cfg.update({
        "attack_type": "fgsm",
        "adv_epsilon": atk["epsilon"],
        "adv_ratio":   atk["adv_ratio"],
        "save_dir":    os.path.join(_PROJECT_ROOT, "saved_models", "crnn", "adv_train_fgsm"),
    })
    return cfg


def get_crnn_adv_bim_config() -> dict:
    """CRNN adversarial training with BIM.

    Inherits baseline hyperparameters (50 epochs, val_fraction, scheduler).
    Checkpoint saved by adversarial val accuracy (not clean val acc).
    """
    raw = _load_yaml()
    cfg = get_crnn_config()
    atk = raw["attacks"]["bim"]
    cfg.update({
        "attack_type": "bim",
        "adv_epsilon": atk["epsilon"],
        "adv_ratio":   atk["adv_ratio"],
        "bim_steps":   atk["train_steps"],
        "save_dir":    os.path.join(_PROJECT_ROOT, "saved_models", "crnn", "adv_train_bim"),
    })
    return cfg


def get_crnn_finetune_fgsm_config() -> dict:
    """CRNN adversarial fine-tuning with FGSM.

    Loads crnn baseline checkpoint; fine-tunes at lr=0.0001 for 10 epochs.
    Checkpoint saved by adversarial val accuracy.
    """
    raw = _load_yaml()
    cfg = get_crnn_adv_fgsm_config()
    ft  = raw["finetune"]
    cfg.update({
        "finetune_epochs": ft["epochs"],
        "finetune_lr":     ft["lr"],
        "save_dir":        os.path.join(_PROJECT_ROOT, "saved_models", "crnn", "adv_finetune_fgsm"),
    })
    return cfg


def get_crnn_finetune_bim_config() -> dict:
    """CRNN adversarial fine-tuning with BIM.

    Loads crnn baseline checkpoint; fine-tunes at lr=0.0001 for 10 epochs.
    Checkpoint saved by clean val accuracy.
    """
    raw = _load_yaml()
    cfg = get_crnn_adv_bim_config()
    ft  = raw["finetune"]
    cfg.update({
        "finetune_epochs": ft["epochs"],
        "finetune_lr":     ft["lr"],
        "save_dir":        os.path.join(_PROJECT_ROOT, "saved_models", "crnn", "adv_finetune_bim"),
    })
    return cfg

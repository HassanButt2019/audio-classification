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

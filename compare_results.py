"""
Compare Results — unified table across all 15 experiments.

Reads every cv_summary.json found under results/<model>/<mode>/ and
produces three tables:

  Table 1 — Full comparison: clean acc + F1 + FGSM×3 + BIM×3
  Table 2 — Clean accuracy / chosen metric per model × mode
  Table 3 — FGSM robustness (adversarial accuracy) per epsilon
  Table 4 — BIM robustness (adversarial accuracy) per epsilon

Usage
-----
  python compare_results.py
  python compare_results.py --metric f1_weighted
"""

import argparse
import json
import os

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")

MODELS = ["cnn", "vggish", "crnn"]

MODES = [
    "normal",
    "adv_train_fgsm",
    "adv_train_bim",
    "adv_finetune_fgsm",
    "adv_finetune_bim",
]

EPSILONS = [0.01, 0.03, 0.1]

MODE_LABELS = {
    "normal":           "Normal",
    "adv_train_fgsm":   "Adv Train FGSM",
    "adv_train_bim":    "Adv Train BIM",
    "adv_finetune_fgsm": "Adv Finetune FGSM",
    "adv_finetune_bim":  "Adv Finetune BIM",
}


# ── loader ────────────────────────────────────────────────────────────────────

def load_summary(model_name, mode):
    path = os.path.join(RESULTS_ROOT, model_name, mode, "cv_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _attack_mean(summary, attack_key, eps):
    """Average adversarial accuracy over all 10 folds for the given attack/epsilon."""
    if summary is None:
        return None
    data = summary.get(attack_key, {})
    key  = f"eps_{eps}"
    vals = [
        data[f"fold_{f}"][key]
        for f in range(1, 11)
        if f"fold_{f}" in data and key in data[f"fold_{f}"]
    ]
    return sum(vals) / len(vals) if vals else None


# ── table 1: full comparison ──────────────────────────────────────────────────

def print_full_table():
    """Clean acc + F1 + FGSM ×3 + BIM ×3 for all 15 experiments."""
    col   = 9
    head  = 26
    sep   = f"  {'─'*head}" + f"  {'─'*col}" * 8

    print(f"\n{'='*119}")
    print(f"  Full Comparison Table  (15 experiments)")
    print(f"{'='*119}")
    print(
        f"  {'Model / Mode':<{head}}"
        f"  {'Clean':^{col}}"
        f"  {'F1 (wtd)':^{col}}"
        f"  {'FGSM .01':^{col}}"
        f"  {'FGSM .03':^{col}}"
        f"  {'FGSM .10':^{col}}"
        f"  {'BIM .01':^{col}}"
        f"  {'BIM .03':^{col}}"
        f"  {'BIM .10':^{col}}"
    )
    print(sep)

    def fp(v, scale=100):
        return f"{v*scale:6.2f}%" if v is not None else "  --- "

    def fa(v):
        return f"{v:6.2f}%" if v is not None else "  --- "

    for model_name in MODELS:
        for mode in MODES:
            summary = load_summary(model_name, mode)
            label   = f"{model_name.upper()} / {MODE_LABELS[mode]}"

            if summary:
                clean = summary["mean"].get("accuracy")
                f1    = summary["mean"].get("f1_weighted")
                f01   = _attack_mean(summary, "fgsm", 0.01)
                f03   = _attack_mean(summary, "fgsm", 0.03)
                f10   = _attack_mean(summary, "fgsm", 0.1)
                b01   = _attack_mean(summary, "bim",  0.01)
                b03   = _attack_mean(summary, "bim",  0.03)
                b10   = _attack_mean(summary, "bim",  0.1)
                print(
                    f"  {label:<{head}}"
                    f"  {fp(clean):^{col}}"
                    f"  {fp(f1):^{col}}"
                    f"  {fa(f01):^{col}}"
                    f"  {fa(f03):^{col}}"
                    f"  {fa(f10):^{col}}"
                    f"  {fa(b01):^{col}}"
                    f"  {fa(b03):^{col}}"
                    f"  {fa(b10):^{col}}"
                )
            else:
                print(f"  {label:<{head}}  (no results yet)")

        print(sep)

    print(f"{'='*119}\n")


# ── table 2: clean metric per model × mode ────────────────────────────────────

def print_clean_table(metric_key):
    col_w      = 20
    header_col = 12

    print(f"\n{'='*120}")
    print(f"  Clean Performance  |  metric: {metric_key}")
    print(f"{'='*120}")
    print(f"  {'Model':<{header_col}}", end="")
    for mode in MODES:
        print(f"  {MODE_LABELS[mode]:^{col_w}}", end="")
    print()
    print(f"  {'─'*header_col}", end="")
    for _ in MODES:
        print(f"  {'─'*col_w}", end="")
    print()

    for model_name in MODELS:
        print(f"  {model_name.upper():<{header_col}}", end="")
        for mode in MODES:
            summary = load_summary(model_name, mode)
            value   = summary["mean"].get(metric_key) if summary else None
            display = f"{value*100:6.2f}%" if value is not None else "  ---  "
            print(f"  {display:^{col_w}}", end="")
        print()

    print(f"{'='*120}")


# ── table 3: FGSM robustness ──────────────────────────────────────────────────

def print_fgsm_table():
    col_w      = 10
    header_col = 26

    print(f"\n{'='*80}")
    print(f"  FGSM Robustness  (mean adversarial accuracy over 10 folds)")
    print(f"{'='*80}")
    print(f"  {'Model / Mode':<{header_col}}", end="")
    for eps in EPSILONS:
        print(f"  {'ε='+str(eps):^{col_w}}", end="")
    print()
    print(f"  {'─'*header_col}", end="")
    for _ in EPSILONS:
        print(f"  {'─'*col_w}", end="")
    print()

    for model_name in MODELS:
        for mode in MODES:
            summary = load_summary(model_name, mode)
            label   = f"{model_name.upper()} / {MODE_LABELS[mode]}"
            print(f"  {label:<{header_col}}", end="")
            for eps in EPSILONS:
                mean    = _attack_mean(summary, "fgsm", eps) if summary else None
                display = f"{mean:6.2f}%" if mean is not None else "  ---  "
                print(f"  {display:^{col_w}}", end="")
            print()

        print(f"  {'─'*header_col}", end="")
        for _ in EPSILONS:
            print(f"  {'─'*col_w}", end="")
        print()

    print(f"{'='*80}")


# ── table 4: BIM robustness ───────────────────────────────────────────────────

def print_bim_table():
    col_w      = 10
    header_col = 26

    print(f"\n{'='*80}")
    print(f"  BIM Robustness  (mean adversarial accuracy over 10 folds, steps=10)")
    print(f"{'='*80}")
    print(f"  {'Model / Mode':<{header_col}}", end="")
    for eps in EPSILONS:
        print(f"  {'ε='+str(eps):^{col_w}}", end="")
    print()
    print(f"  {'─'*header_col}", end="")
    for _ in EPSILONS:
        print(f"  {'─'*col_w}", end="")
    print()

    for model_name in MODELS:
        for mode in MODES:
            summary = load_summary(model_name, mode)
            label   = f"{model_name.upper()} / {MODE_LABELS[mode]}"
            print(f"  {label:<{header_col}}", end="")
            for eps in EPSILONS:
                mean    = _attack_mean(summary, "bim", eps) if summary else None
                display = f"{mean:6.2f}%" if mean is not None else "  ---  "
                print(f"  {display:^{col_w}}", end="")
            print()

        print(f"  {'─'*header_col}", end="")
        for _ in EPSILONS:
            print(f"  {'─'*col_w}", end="")
        print()

    print(f"{'='*80}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all 15 experiment results")
    parser.add_argument(
        "--metric", default="accuracy",
        help="Primary metric for Table 2 (default: accuracy)"
    )
    args = parser.parse_args()

    print_full_table()
    print_clean_table(args.metric)
    print_fgsm_table()
    print_bim_table()

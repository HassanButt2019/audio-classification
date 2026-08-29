"""
RQ1 — Adversarial Robustness of VGGish Training Strategies.

Reads pre-computed per-fold attack results from cv_summary.json files
(produced by the training + evaluation pipeline) and aggregates them
across all 10 folds into three output tables:

  1. rq1_robustness_summary.csv
       Strategy | Clean | FGSM eps=0.01 | FGSM eps=0.03 | FGSM eps=0.10
                        | BIM  eps=0.01 | BIM  eps=0.03 | BIM  eps=0.10

  2. rq1_robustness_drop.csv
       Strategy | drop in accuracy (pp) from clean, per attack and epsilon

  3. rq1_cross_attack_robustness.csv
       Mean FGSM robustness vs mean BIM robustness per strategy — reveals
       whether adversarial training generalises across attack types.

Outputs saved to:
  research_answers/vggish/AtTraining_RQ1/

Usage
-----
  python scripts/vggish/rq1_adversarial_robustness.py
"""

import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODES = [
    ("normal",            "Baseline"),
    ("adv_train_fgsm",    "AT-FGSM"),
    ("adv_train_bim",     "AT-BIM"),
    ("adv_finetune_fgsm", "AFT-FGSM"),
    ("adv_finetune_bim",  "AFT-BIM"),
]

EPSILONS   = [0.01, 0.03, 0.1]
EPS_LABELS = ["eps=0.01", "eps=0.03", "eps=0.10"]


# ── helpers ────────────────────────────────────────────────────────────────────

def _find_cv_summary(mode):
    base = os.path.join(ROOT, "results", "vggish", mode)
    direct = os.path.join(base, "cv_summary.json")
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(base, "**", "cv_summary.json"), recursive=True)
    return matches[0] if matches else None


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _fold_mean(attack_dict, eps_key):
    vals = [
        attack_dict[f"fold_{f}"][eps_key]
        for f in range(1, 11)
        if f"fold_{f}" in attack_dict and eps_key in attack_dict[f"fold_{f}"]
    ]
    return _mean(vals)


def _print_table(headers, rows):
    col_w = [
        max(len(str(h)), max(len(str(r[i])) for r in rows))
        for i, h in enumerate(headers)
    ]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)


# ── load and aggregate ─────────────────────────────────────────────────────────

def load_strategy_stats():
    stats = []
    for mode, strategy in MODES:
        cv_path = _find_cv_summary(mode)
        if cv_path is None:
            print(f"[WARN] cv_summary not found for '{mode}' — skipping")
            continue

        with open(cv_path) as f:
            summary = json.load(f)

        clean_acc = _mean([entry["accuracy"] * 100
                           for entry in summary.get("per_fold", [])])

        fgsm_dict = summary.get("fgsm", {})
        bim_dict  = summary.get("bim", {})

        fgsm_accs = {}
        bim_accs  = {}
        for eps in EPSILONS:
            key = f"eps_{eps}"
            fgsm_accs[eps] = _fold_mean(fgsm_dict, key)
            bim_accs[eps]  = _fold_mean(bim_dict,  key)

        stats.append({
            "strategy":  strategy,
            "clean_acc": clean_acc,
            "fgsm":      fgsm_accs,
            "bim":       bim_accs,
        })

    return stats


# ── writers ────────────────────────────────────────────────────────────────────

def write_summary(stats, out_dir):
    headers = (
        ["Strategy", "Clean (%)"]
        + [f"FGSM {lbl} (%)" for lbl in EPS_LABELS]
        + [f"BIM  {lbl} (%)" for lbl in EPS_LABELS]
    )
    rows = []
    for s in stats:
        row = [s["strategy"], f"{s['clean_acc']:.2f}"]
        for eps in EPSILONS:
            row.append(f"{s['fgsm'][eps]:.2f}")
        for eps in EPSILONS:
            row.append(f"{s['bim'][eps]:.2f}")
        rows.append(row)

    print("\n--- RQ1 : Robustness Summary ---")
    _print_table(headers, rows)

    path = os.path.join(out_dir, "rq1_robustness_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  -> {path}")
    return path


def write_drop(stats, out_dir):
    headers = (
        ["Strategy", "Clean (%)"]
        + [f"FGSM Drop {lbl} (pp)" for lbl in EPS_LABELS]
        + [f"BIM  Drop {lbl} (pp)" for lbl in EPS_LABELS]
    )
    rows = []
    for s in stats:
        row = [s["strategy"], f"{s['clean_acc']:.2f}"]
        for eps in EPSILONS:
            row.append(f"{s['clean_acc'] - s['fgsm'][eps]:.2f}")
        for eps in EPSILONS:
            row.append(f"{s['clean_acc'] - s['bim'][eps]:.2f}")
        rows.append(row)

    print("\n--- RQ1 : Robustness Drop per Epsilon ---")
    _print_table(headers, rows)

    path = os.path.join(out_dir, "rq1_robustness_drop.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  -> {path}")
    return path


def write_cross_attack(stats, out_dir):
    headers = [
        "Strategy",
        "Mean FGSM Acc (%)",
        "Mean BIM Acc (%)",
        "FGSM-BIM Gap (pp)",
        "Mean Robustness (%)",
    ]
    rows = []
    json_rows = []
    for s in stats:
        mean_fgsm = _mean([s["fgsm"][eps] for eps in EPSILONS])
        mean_bim  = _mean([s["bim"][eps]  for eps in EPSILONS])
        gap       = mean_fgsm - mean_bim
        mean_all  = _mean([s["fgsm"][eps] for eps in EPSILONS] +
                          [s["bim"][eps]  for eps in EPSILONS])
        rows.append([
            s["strategy"],
            f"{mean_fgsm:.2f}",
            f"{mean_bim:.2f}",
            f"{gap:+.2f}",
            f"{mean_all:.2f}",
        ])
        json_rows.append({
            "strategy":        s["strategy"],
            "mean_fgsm_acc":   round(mean_fgsm, 4),
            "mean_bim_acc":    round(mean_bim,  4),
            "fgsm_bim_gap_pp": round(gap, 4),
            "mean_robustness": round(mean_all, 4),
        })

    print("\n--- RQ1 : Cross-Attack Robustness ---")
    _print_table(headers, rows)

    path = os.path.join(out_dir, "rq1_cross_attack_robustness.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  -> {path}")
    return path, json_rows


def write_json(stats, cross_json, out_dir):
    payload = []
    for s in stats:
        payload.append({
            "strategy":  s["strategy"],
            "clean_acc": round(s["clean_acc"], 4),
            "fgsm": {f"eps_{eps}": round(s["fgsm"][eps], 4) for eps in EPSILONS},
            "bim":  {f"eps_{eps}": round(s["bim"][eps],  4) for eps in EPSILONS},
            "fgsm_drop": {
                f"eps_{eps}": round(s["clean_acc"] - s["fgsm"][eps], 4)
                for eps in EPSILONS
            },
            "bim_drop": {
                f"eps_{eps}": round(s["clean_acc"] - s["bim"][eps], 4)
                for eps in EPSILONS
            },
        })

    path = os.path.join(out_dir, "rq1_full_results.json")
    with open(path, "w") as f:
        json.dump({"robustness": payload, "cross_attack": cross_json}, f, indent=2)
    print(f"  -> {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("RQ1 — VGGish Adversarial Robustness Analysis")
    print("Strategies: Baseline | AT-FGSM | AT-BIM | AFT-FGSM | AFT-BIM")
    print("Attacks:    FGSM + BIM  |  eps = 0.01, 0.03, 0.10")
    print("=" * 60)

    out_dir = os.path.join(ROOT, "research_answers", "vggish", "AtTraining_RQ1")
    os.makedirs(out_dir, exist_ok=True)

    stats = load_strategy_stats()
    if not stats:
        print("[ERROR] No cv_summary files found. Run training pipeline first.")
        sys.exit(1)

    write_summary(stats, out_dir)
    write_drop(stats, out_dir)
    _, cross_json = write_cross_attack(stats, out_dir)
    write_json(stats, cross_json, out_dir)

    print(f"\nAll outputs saved to: {out_dir}\n")


if __name__ == "__main__":
    main()

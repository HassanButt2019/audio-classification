"""
Efficiency measurement for CRNN — RQ2 (Inference) and RQ3 (Training).

Outputs
-------
research_answers/crnn/Inference Efficiency (RQ2)/rq2_inference_efficiency.csv
research_answers/crnn/Inference Efficiency (RQ2)/rq2_inference_efficiency.json
research_answers/crnn/Training Efficiency (RQ3)/rq3_training_efficiency.csv
research_answers/crnn/Training Efficiency (RQ3)/rq3_training_efficiency.json

Usage
-----
  python scripts/crnn/efficiency_measurement.py
"""

import csv
import glob
import json
import os
import re
import sys

import torch
from ptflops import get_model_complexity_info

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from models.crnn import CRNN

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODES = [
    ("normal",            "Baseline"),
    ("adv_train_fgsm",    "AT-FGSM"),
    ("adv_train_bim",     "AT-BIM"),
    ("adv_finetune_fgsm", "AFT-FGSM"),
    ("adv_finetune_bim",  "AFT-BIM"),
]

# CRNN input: (channels=1, mel_bins=64, time_frames=128) — same as CNN
INPUT_RES = (1, 64, 128)


# ── helpers ────────────────────────────────────────────────────────────────────

def _parse_wall_time(s):
    """'07h 17m 59s' → total seconds (int)."""
    m = re.fullmatch(r"(\d+)h\s+(\d+)m\s+(\d+)s", s.strip())
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def _find_cv_summary(mode):
    base = os.path.join(ROOT, "results", "crnn", mode)
    direct = os.path.join(base, "cv_summary.json")
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(base, "**", "cv_summary.json"), recursive=True)
    return matches[0] if matches else None


def _measure_flops():
    """Run ptflops on a fresh CRNN instance (architecture-level measurement).

    ptflops counts MACs for Conv layers normally; GRU MACs are approximated
    using the standard formula: 3 * (input_size + hidden_size) * hidden_size
    per direction per time step.
    """
    model = CRNN(num_classes=10, dropout=0.5)
    model.eval()
    macs, params = get_model_complexity_info(
        model,
        INPUT_RES,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    return macs, params


def _print_table(headers, rows):
    col_w = [max(len(str(h)), max(len(str(r[i])) for r in rows))
             for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CRNN Efficiency Measurement")
    print(f"Input shape : {INPUT_RES}  (channels, mel_bins, time_frames)")
    print("=" * 60)

    # ── ptflops ────────────────────────────────────────────────────────────────
    print("\nRunning ptflops ...")
    macs, params = _measure_flops()
    gmac     = macs   / 1e9
    m_params = params / 1e6
    print(f"  Computational complexity : {gmac:.4f} GMac")
    print(f"  Number of parameters     : {m_params:.4f} M")

    # ── collect training stats from cv_summary files ───────────────────────────
    rq3_data = []
    baseline_secs = None
    for mode, strategy in MODES:
        cv_path = _find_cv_summary(mode)
        if cv_path is None:
            print(f"\n[WARN] cv_summary not found for mode '{mode}' — skipping")
            continue
        with open(cv_path) as f:
            summary = json.load(f)
        wall_str = summary.get("wall_time", "")
        epochs   = summary.get("config", {}).get("epochs", "?")
        secs     = _parse_wall_time(wall_str)
        rq3_data.append({
            "strategy":  strategy,
            "epochs":    epochs,
            "wall_time": wall_str,
            "secs":      secs,
        })
        if strategy == "Baseline":
            baseline_secs = secs

    # ── RQ2: Inference Efficiency ──────────────────────────────────────────────
    rq2_dir = os.path.join(ROOT, "research_answers", "crnn", "Inference Efficiency (RQ2)")
    os.makedirs(rq2_dir, exist_ok=True)

    rq2_rows = [
        [strategy, f"{gmac:.4f}", f"{m_params:.4f}"]
        for _, strategy in MODES
    ]
    rq2_headers = ["Strategy", "FLOPs (GMac)", "Parameters (M)"]

    csv_path = os.path.join(rq2_dir, "rq2_inference_efficiency.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(rq2_headers)
        w.writerows(rq2_rows)

    json_path = os.path.join(rq2_dir, "rq2_inference_efficiency.json")
    rq2_json = [
        {"strategy": strategy, "flops_gmac": round(gmac, 4), "parameters_m": round(m_params, 4)}
        for _, strategy in MODES
    ]
    with open(json_path, "w") as f:
        json.dump(rq2_json, f, indent=2)

    print("\n--- RQ2 : Inference Efficiency ---")
    _print_table(rq2_headers, rq2_rows)
    print(f"\n  CSV  -> {csv_path}")
    print(f"  JSON -> {json_path}")

    # ── RQ3: Training Efficiency ───────────────────────────────────────────────
    rq3_dir = os.path.join(ROOT, "research_answers", "crnn", "Training Efficiency (RQ3)")
    os.makedirs(rq3_dir, exist_ok=True)

    rq3_rows = []
    rq3_json = []
    for entry in rq3_data:
        secs = entry["secs"]
        if baseline_secs and secs:
            rel_cost = f"{secs / baseline_secs:.2f}x"
            rel_val  = round(secs / baseline_secs, 2)
        else:
            rel_cost = "N/A"
            rel_val  = None
        rq3_rows.append([entry["strategy"], entry["epochs"], entry["wall_time"], rel_cost])
        rq3_json.append({
            "strategy":      entry["strategy"],
            "epochs":        entry["epochs"],
            "wall_time":     entry["wall_time"],
            "relative_cost": rel_val,
        })

    rq3_headers = ["Strategy", "Epochs", "Wall Time", "Relative Cost"]

    csv_path = os.path.join(rq3_dir, "rq3_training_efficiency.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(rq3_headers)
        w.writerows(rq3_rows)

    json_path = os.path.join(rq3_dir, "rq3_training_efficiency.json")
    with open(json_path, "w") as f:
        json.dump(rq3_json, f, indent=2)

    print("\n--- RQ3 : Training Efficiency ---")
    _print_table(rq3_headers, rq3_rows)
    print(f"\n  CSV  -> {csv_path}")
    print(f"  JSON -> {json_path}")
    print()


if __name__ == "__main__":
    main()

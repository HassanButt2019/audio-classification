# Audio Classification — Adversarial Robustness Study

Urban sound classification on UrbanSound8K using CNN, VGGish, and CRNN models,
evaluated under 5 training modes across 15 total experiments.

---

## Project Structure

```
thesis_code/
├── models/
│   ├── cnn.py                  — CNN model
│   ├── vggish.py               — VGGish model (stub)
│   └── crnn.py                 — CRNN model (stub)
├── train/
│   ├── train.py                — Normal training (10-fold CV)
│   ├── adversarial_train.py    — Adversarial training from scratch (FGSM or BIM)
│   └── finetune_adversarial.py — Adversarial fine-tuning from pretrained weights (FGSM or BIM)
├── attacks/
│   ├── fgsm.py                 — FGSM attack
│   ├── bim.py                  — BIM attack (iterative FGSM)
│   └── run_attacks.py          — Run FGSM + BIM across all folds
├── evaluate/
│   └── evaluate.py             — Evaluation metrics (accuracy, F1, etc.)
├── dataset/
│   └── urbansound_dataset.py   — UrbanSound8K DataLoader
├── preprocessing/
│   └── mel_spectrogram.py      — Log-Mel spectrogram preprocessing
├── main.py                     — CNN normal training entry point
├── run_experiments.py          — Unified runner for all 15 experiments
└── compare_results.py          — Print comparison tables across all experiments
```

---

## Training Modes

| Mode | Description |
|---|---|
| `normal` | Standard training on clean data |
| `adv_train_fgsm` | Train from scratch with FGSM adversarial examples mixed in |
| `adv_train_bim` | Train from scratch with BIM adversarial examples mixed in |
| `adv_finetune_fgsm` | Load `normal` checkpoint, fine-tune with FGSM adversarial examples |
| `adv_finetune_bim` | Load `normal` checkpoint, fine-tune with BIM adversarial examples |

> `adv_finetune_*` modes require `normal` to have run first for the same model.

---

## Results & Checkpoints Structure

```
results/
  cnn/
    normal/
    adv_train_fgsm/
    adv_train_bim/
    adv_finetune_fgsm/
    adv_finetune_bim/
  vggish/  (same 5 modes)
  crnn/    (same 5 modes)

saved_models/
  cnn/
    normal/          — best_fold1.pt … best_fold10.pt
    adv_train_fgsm/
    adv_train_bim/
    adv_finetune_fgsm/
    adv_finetune_bim/
  vggish/  ...
  crnn/    ...
```

Each leaf folder contains:
- `config.json` — hyperparameters saved before training starts
- `fold_1_results.json` … `fold_10_results.json` — per-fold training history and test metrics
- `cv_summary.json` — mean ± std + FGSM & BIM attack results across all 10 folds

---

## Running Experiments

### CNN normal training (shortcut via main.py)

```bash
python main.py

# Options
python main.py --epochs 30
python main.py --batch-size 64
python main.py --quick 100    # smoke-test with 100 samples per fold
```

### Single model, single mode

```bash
python run_experiments.py --model cnn --mode normal
python run_experiments.py --model cnn --mode adv_train_fgsm
python run_experiments.py --model cnn --mode adv_train_bim
python run_experiments.py --model cnn --mode adv_finetune_fgsm   # requires normal first
python run_experiments.py --model cnn --mode adv_finetune_bim    # requires normal first

python run_experiments.py --model vggish --mode normal
python run_experiments.py --model vggish --mode adv_train_fgsm
python run_experiments.py --model vggish --mode adv_train_bim
python run_experiments.py --model vggish --mode adv_finetune_fgsm
python run_experiments.py --model vggish --mode adv_finetune_bim

python run_experiments.py --model crnn --mode normal
python run_experiments.py --model crnn --mode adv_train_fgsm
python run_experiments.py --model crnn --mode adv_train_bim
python run_experiments.py --model crnn --mode adv_finetune_fgsm
python run_experiments.py --model crnn --mode adv_finetune_bim
```

### With custom epochs

```bash
# default epochs: 30 for normal / adv_train_*, 15 for adv_finetune_*
python run_experiments.py --model cnn --mode normal            --epochs 50
python run_experiments.py --model cnn --mode adv_train_fgsm   --epochs 50
python run_experiments.py --model cnn --mode adv_train_bim    --epochs 50
python run_experiments.py --model cnn --mode adv_finetune_fgsm --epochs 20
python run_experiments.py --model cnn --mode adv_finetune_bim  --epochs 20
```

> `--epochs` works for any model and mode combination. For `adv_finetune_*` it sets the fine-tuning epochs; for all other modes it sets the full training epochs.

### Smoke-test with --quick

```bash
# use N random samples per fold to verify the pipeline runs end-to-end
python run_experiments.py --model cnn --mode normal            --quick 100
python run_experiments.py --model cnn --mode adv_train_fgsm   --quick 100
python run_experiments.py --model cnn --mode adv_finetune_fgsm --quick 100

# combine with --epochs for a very fast smoke-test
python run_experiments.py --model cnn --mode normal --epochs 2 --quick 100
```

> `--quick N` limits each fold to N random samples. Results will not be meaningful — use only to verify the code runs without errors.

### All 5 modes for one model

```bash
python run_experiments.py --model cnn
```

### All 15 experiments at once

```bash
python run_experiments.py
```

> Modes run in order: `normal → adv_train_fgsm → adv_train_bim → adv_finetune_fgsm → adv_finetune_bim`.
> Fine-tuning modes automatically use the `normal` checkpoint as their starting point.

### Quick smoke-test (N samples per fold)

```bash
python run_experiments.py --model cnn --mode normal --quick 100
```

---

## Comparing Results

After experiments complete, print unified comparison tables:

```bash
python compare_results.py

# Use a different primary metric for Table 2
python compare_results.py --metric f1_weighted
```

**Four tables are printed:**
1. Full table — clean acc + F1 + FGSM ×3 + BIM ×3 for all 15 experiments
2. Clean metric per model × mode
3. FGSM robustness (adversarial accuracy at ε = 0.01, 0.03, 0.1)
4. BIM robustness (adversarial accuracy at ε = 0.01, 0.03, 0.1, steps = 10)

---

## Dataset

[UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) — 8732 labeled
audio clips across 10 urban sound classes, pre-split into 10 folds.

Place the dataset at:
```
data/UrbanSound8K/
  metadata/
    UrbanSound8K.csv
  audio/
    fold1/  fold2/  ...  fold10/
```

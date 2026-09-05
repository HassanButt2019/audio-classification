from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data" / "UrbanSound8K"
CSV = ROOT / "metadata" / "UrbanSound8K.csv"
AUDIO = ROOT / "audio"
FIGDIR = Path(__file__).resolve().parent / "figures"
CMAP = "magma"
N_CANDIDATES = 15
SEED = 42

def check_inputs():
    missing = []
    if not CSV.is_file():
        missing.append(f"metadata CSV : {CSV}")
    folds = [AUDIO / f"fold{i}" for i in range(1, 11)]
    absent = [f for f in folds if not f.is_dir()]
    if absent:
        missing.append(f"audio folds  : {AUDIO}/fold1 ... fold10 ({len(absent)} of 10 missing)")
    if missing:
        print("UrbanSound8K data not found. Expected:")
        for m in missing:
            print("  " + m)
        print("\nPlace the dataset so that these paths exist, then re-run.")
        sys.exit(1)


def spectrogram(path: Path) -> np.ndarray:
    waveform, sr = _load_audio(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
    with torch.no_grad():
        spec = TRANSFORM(waveform.float())
    return spec.squeeze(0).numpy()


def energy(path: Path) -> float:
    waveform, sr = _load_audio(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
    clip = waveform[..., :TARGET_SAMPLES].float()
    return float(torch.sqrt(torch.mean(clip ** 2)))


def path_of(row) -> Path:
    return AUDIO / f"fold{int(row['fold'])}" / row["slice_file_name"]


def pick_representative(df: pd.DataFrame, class_name: str) -> pd.Series:
    pool = df[df["class"] == class_name]
    full = pool[(pool["end"] - pool["start"]) >= TARGET_SAMPLES / SAMPLE_RATE]
    salient = full[full["salience"] == 1]
    for candidates in (salient, full, pool):
        if len(candidates):
            break
    sample = candidates.sample(n=min(N_CANDIDATES, len(candidates)), random_state=SEED)
    rms = sample.apply(lambda r: energy(path_of(r)), axis=1)
    return sample.loc[rms.idxmax()]


def time_axis():
    return N_TIME_FRAMES * HOP_LENGTH / SAMPLE_RATE


def fig_example(spec: np.ndarray, name: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(spec, origin="lower", aspect="auto", cmap=CMAP, vmin=0.0, vmax=1.0,
                   extent=[0, time_axis(), 0, N_MELS])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel band")
    fig.colorbar(im, ax=ax, label="Normalised log-mel energy")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_2_2_example_spectrogram.png", dpi=300)
    fig.savefig(FIGDIR / "fig_2_2_example_spectrogram.svg")
    plt.close(fig)


def fig_grid(specs: dict):
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    for ax, cls in zip(axes.ravel(), CLASS_NAMES):
        im = ax.imshow(specs[cls], origin="lower", aspect="auto", cmap=CMAP, vmin=0.0, vmax=1.0,
                       extent=[0, time_axis(), 0, N_MELS])
        ax.set_title(cls.replace("_", " "), fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mel band")
    fig.tight_layout(rect=[0, 0, 0.93, 1])
    cax = fig.add_axes([0.945, 0.12, 0.012, 0.76])
    fig.colorbar(im, cax=cax, label="Normalised log-mel energy")
    fig.savefig(FIGDIR / "fig_3_1_spectrograms_by_class.png", dpi=300)
    fig.savefig(FIGDIR / "fig_3_1_spectrograms_by_class.svg")
    plt.close(fig)


def fig_distribution(counts: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = [c.replace("_", " ") for c in counts.index]
    bars = ax.barh(labels, counts.values, color="#1F3864")
    ax.invert_yaxis()
    ax.set_xlabel("Number of clips")
    ax.set_xlim(0, counts.max() * 1.12)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_width() + counts.max() * 0.015, bar.get_y() + bar.get_height() / 2,
                str(int(value)), va="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_3_2_class_distribution.png", dpi=300)
    fig.savefig(FIGDIR / "fig_3_2_class_distribution.svg")
    plt.close(fig)


check_inputs()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset.urbansound_dataset import CLASS_NAMES, _load_audio
from preprocessing.mel_spectrogram import (
    HOP_LENGTH,
    MelSpectrogramTransform,
    N_MELS,
    N_TIME_FRAMES,
    SAMPLE_RATE,
    TARGET_SAMPLES,
)

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

TRANSFORM = MelSpectrogramTransform()
FIGDIR.mkdir(exist_ok=True)
meta = pd.read_csv(CSV)

picks = {cls: pick_representative(meta, cls) for cls in CLASS_NAMES}
specs = {cls: spectrogram(path_of(row)) for cls, row in picks.items()}
for cls, row in picks.items():
    print(f"{cls:18s} fold{int(row['fold'])}/{row['slice_file_name']}")

loudest = max(picks, key=lambda c: energy(path_of(picks[c])))
example = picks[loudest]
print(f"\nFigure 2.2 clip: fold{int(example['fold'])}/{example['slice_file_name']}  ({loudest})")
fig_example(specs[loudest], example["slice_file_name"])
fig_grid(specs)

counts = meta["class"].value_counts().reindex(CLASS_NAMES)
fig_distribution(counts)

print("\nClips per class (all 10 folds):")
for cls, n in counts.items():
    print(f"  {cls:18s} {int(n)}")
print(f"  {'TOTAL':18s} {int(counts.sum())}")
print(f"\nSaved to {FIGDIR}")

import os
import sys
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader

# Import preprocessing pipeline so __getitem__ owns the full audio → tensor flow.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocessing.mel_spectrogram import MelSpectrogramTransform, SAMPLE_RATE


# ── dataset constants ─────────────────────────────────────────────────────────

CLASS_NAMES = [
    "air_conditioner",   # 0
    "car_horn",          # 1
    "children_playing",  # 2
    "dog_bark",          # 3
    "drilling",          # 4
    "engine_idling",     # 5
    "gun_shot",          # 6
    "jackhammer",        # 7
    "siren",             # 8
    "street_music",      # 9
]

NUM_CLASSES = len(CLASS_NAMES)   # 10


# ── dataset class ─────────────────────────────────────────────────────────────

class UrbanSoundDataset(Dataset):
    """PyTorch Dataset for the UrbanSound8K urban audio classification benchmark.

    Implements the three methods required by torch.utils.data.Dataset so that
    PyTorch's DataLoader can handle batching, shuffling, and parallel loading
    automatically.

    Each call to __getitem__(idx) executes the full preprocessing pipeline:
        1. Look up the file path and label for sample idx from the CSV metadata.
        2. Load the raw audio waveform from disk (lazy — one file at a time).
        3. Convert stereo to mono (channel average).
        4. Resample to 22 050 Hz if the file's native rate differs.
        5. Convert to a log-scaled, normalised Mel spectrogram (64 × 128).
        6. Return (spectrogram_tensor, integer_label).

    Fold-based splitting
    --------------------
    UrbanSound8K ships with 10 pre-defined folds designed for 10-fold cross-
    validation.  Pass the desired fold numbers at construction time:

        train_ds = UrbanSoundDataset(root, folds=[1,2,3,4,5,6,7,8,9])
        test_ds  = UrbanSoundDataset(root, folds=[10])

    No audio clip ever appears in both a training fold and the test fold, which
    prevents data leakage — a hard methodological requirement for this thesis.

    Args:
        root_dir: Path to the UrbanSound8K root (the directory that contains
                  the ``audio/`` and ``metadata/`` sub-directories).
        folds:    List of fold numbers (1–10) to include.  ``None`` loads all
                  10 folds (useful for inference on the full dataset).
    """

    def __init__(
        self,
        root_dir:             str,
        folds:                list[int] | None = None,
        max_samples_per_fold: int | None       = None,
        seed:                 int              = 42,
    ):
        self.audio_dir = os.path.join(root_dir, "audio")

        # ── Step 1 of __getitem__: load the CSV index ─────────────────────────
        # UrbanSound8K.csv has one row per audio clip with columns:
        #   slice_file_name, fsID, start, end, salience, fold, classID, class
        metadata_path = os.path.join(root_dir, "metadata", "UrbanSound8K.csv")
        df = pd.read_csv(metadata_path)

        if folds is not None:
            df = df[df["fold"].isin(folds)].reset_index(drop=True)

        # Optional: draw a fixed random subset from each fold independently.
        # This keeps class distribution representative within each fold slice.
        # Existing callers that do not pass max_samples_per_fold are unaffected.
        if max_samples_per_fold is not None:
            rng = np.random.default_rng(seed)
            groups = []
            for fold_id, group in df.groupby("fold"):
                n = min(max_samples_per_fold, len(group))
                groups.append(group.sample(n=n, random_state=int(rng.integers(1 << 31))))
            df = pd.concat(groups).reset_index(drop=True)

        # Store as a plain list of dicts so iloc is never called inside the
        # hot path; dict access is faster than pandas row access at scale.
        self.samples = df[["slice_file_name", "fold", "classID"]].to_dict("records")

        # Single shared transform instance — stateless, so safe across workers.
        self._transform = MelSpectrogramTransform()

    # ── Contract method 1 ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return total number of audio clips in the selected folds.

        PyTorch calls this to know when one full epoch of training is complete.
        """
        return len(self.samples)

    # ── Contract method 2 ─────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load, preprocess, and return one sample.

        PyTorch calls this repeatedly during training — once per sample per
        epoch.  The full 6-step pipeline runs here so nothing is held in memory
        between calls (lazy loading).

        Args:
            idx: Integer index in [0, len(self)).

        Returns:
            spectrogram: Float32 tensor of shape (1, 64, 128), values in [0, 1].
            label:       Integer class ID in [0, 9].
        """
        sample = self.samples[idx]

        # Step 1 — resolve file path from the metadata index
        audio_path = os.path.join(
            self.audio_dir,
            f"fold{sample['fold']}",
            sample["slice_file_name"],
        )

        # Step 2 — load raw waveform from disk
        waveform, sample_rate = _load_audio(audio_path)   # (C, T), int

        # Step 3 — convert stereo to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # (1, T)

        # Step 4 — resample to 22 050 Hz if the file's native rate differs
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=SAMPLE_RATE
            )(waveform)

        # Steps 5 & 6 — Mel spectrogram + log scale + [0,1] normalisation
        # MelSpectrogramTransform also pads/trims the waveform to TARGET_SAMPLES
        # (66 048 samples) so the output is always exactly (1, 64, 128).
        spectrogram = self._transform(waveform)            # (1, 64, 128)

        label = int(sample["classID"])

        return spectrogram, label

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_class_name(self, class_id: int) -> str:
        """Map an integer class ID to its human-readable name."""
        return CLASS_NAMES[class_id]


# ── audio loader (module-level so it can be reused by attack scripts) ─────────

def _load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load an audio file and return (waveform_tensor, sample_rate).

    Tries torchaudio first; falls back to librosa when the torchaudio backend
    (torchcodec) is not installed in this environment.
    """
    try:
        return torchaudio.load(path)
    except ImportError:
        y, sr = librosa.load(path, sr=None, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]            # (N,) → (1, N)
        return torch.from_numpy(np.ascontiguousarray(y)), sr


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_fold_dataloaders(
    root_dir:             str,
    test_fold:            int,
    batch_size:           int       = 32,
    num_workers:          int       = 4,
    max_samples_per_fold: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders for one fold of 10-fold cross-validation.

    The fold structure is:

        test_fold = 10  →  train on folds 1–9,   test on fold 10
        test_fold = 9   →  train on folds 1–8,10, test on fold 9
        ...

    No overlap between train and test folds — data leakage is impossible by
    construction.

    Args:
        root_dir:    Path to the UrbanSound8K root directory.
        test_fold:   Fold number (1–10) held out for testing.
        batch_size:  Samples per batch for both loaders.
        num_workers: CPU workers for parallel data loading.

    Returns:
        (train_loader, test_loader)
    """
    train_folds = [f for f in range(1, 11) if f != test_fold]

    train_dataset = UrbanSoundDataset(root_dir=root_dir, folds=train_folds,
                                      max_samples_per_fold=max_samples_per_fold)
    test_dataset  = UrbanSoundDataset(root_dir=root_dir, folds=[test_fold],
                                      max_samples_per_fold=max_samples_per_fold)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # randomise order each epoch
        num_workers=num_workers,
        pin_memory=True,        # faster CPU→GPU transfer
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # deterministic order for evaluation
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


# ── Three-way DataLoader factory ─────────────────────────────────────────────

def get_fold_dataloaders_3way(
    root_dir:             str,
    test_fold:            int,
    batch_size:           int       = 32,
    num_workers:          int       = 4,
    max_samples_per_fold: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders for one fold of 10-fold CV.

    To avoid leaking the test fold into checkpoint selection, a separate
    validation fold is carved out of the training folds using a deterministic
    rotation:

        val_fold  = (test_fold % 10) + 1   →  cycles 2,3,...,10,1
        test_fold = held out entirely for final reporting
        train     = remaining 8 folds

    This means checkpoint selection (early stopping) never sees the test fold.

    Args:
        root_dir:    Path to the UrbanSound8K root directory.
        test_fold:   Fold number (1–10) held out for final testing.
        batch_size:  Samples per batch for all three loaders.
        num_workers: CPU workers for parallel data loading.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    val_fold    = (test_fold % 10) + 1
    train_folds = [f for f in range(1, 11) if f != test_fold and f != val_fold]

    train_dataset = UrbanSoundDataset(root_dir=root_dir, folds=train_folds,
                                      max_samples_per_fold=max_samples_per_fold)
    val_dataset   = UrbanSoundDataset(root_dir=root_dir, folds=[val_fold],
                                      max_samples_per_fold=max_samples_per_fold)
    test_dataset  = UrbanSoundDataset(root_dir=root_dir, folds=[test_fold],
                                      max_samples_per_fold=max_samples_per_fold)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ── smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "UrbanSound8K"))

    # --- single sample ---
    ds = UrbanSoundDataset(root_dir=ROOT, folds=[1])
    print(f"Samples in fold 1 : {len(ds)}")

    spec, label = ds[0]
    print(f"Spectrogram shape : {tuple(spec.shape)}   (expect (1, 64, 128))")
    print(f"Value range       : [{spec.min():.4f}, {spec.max():.4f}]  (expect [0, 1])")
    print(f"Label             : {label}  →  {ds.get_class_name(label)}")

    # --- fold split integrity ---
    train_loader, test_loader = get_fold_dataloaders(ROOT, test_fold=10, batch_size=16, num_workers=0)
    total = len(train_loader.dataset) + len(test_loader.dataset)
    print(f"\nFold 10 as test:")
    print(f"  Train samples : {len(train_loader.dataset)}")
    print(f"  Test  samples : {len(test_loader.dataset)}")
    print(f"  Total         : {total}  (expect 8732)")

    # --- batch shape ---
    batch_specs, batch_labels = next(iter(train_loader))
    print(f"\nBatch specs  : {tuple(batch_specs.shape)}  (expect (16, 1, 64, 128))")
    print(f"Batch labels : {tuple(batch_labels.shape)}  (expect (16,))")
    print(f"Dtype        : {batch_specs.dtype}  (expect torch.float32)")

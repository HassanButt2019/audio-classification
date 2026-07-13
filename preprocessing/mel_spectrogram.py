import torch
import torch.nn as nn
import torchaudio.transforms as T


# ── shared constants (identical across all models — thesis requirement) ────────
SAMPLE_RATE = 22050
N_FFT       = 1024
HOP_LENGTH  = 512
N_MELS      = 64

# Frame count formula (center=False):  frames = 1 + (N - n_fft) // hop
#   N = (frames - 1) * hop + n_fft

# CNN  — 128 frames  →  N = (128-1)*512 + 1024 = 66 048  (~3.0 s)
TARGET_SAMPLES   = (128 - 1) * HOP_LENGTH + N_FFT   # 66048
N_TIME_FRAMES    = 128

# VGGish — 96 frames  →  N = (96-1)*512 + 1024 = 49 664  (~2.25 s)
# NOTE: The algorithm spec says "4 seconds = 88 200 samples", but
# 88 200 samples with hop=512/n_fft=1024/center=False produces 171 frames,
# NOT 96. The CRITICAL CHECK in the spec (output must be [1,64,96]) is the
# binding constraint. 49 664 samples is the only value consistent with both
# the preprocessing parameters and the required output shape.
VGGISH_TARGET_SAMPLES = (96 - 1) * HOP_LENGTH + N_FFT   # 49664
VGGISH_N_TIME_FRAMES  = 96
# ──────────────────────────────────────────────────────────────────────────────


class MelSpectrogramTransform(nn.Module):
    """Convert a raw waveform tensor to a normalised log-Mel spectrogram.

    Input : (1, T)  float32 waveform, already resampled to SAMPLE_RATE
    Output: (1, 64, 128) float32 in [0, 1]

    Pipeline
    --------
    1. Pad or trim to TARGET_SAMPLES so the output always has 128 time frames.
    2. Compute power Mel spectrogram (n_mels=64, n_fft=1024, hop=512).
    3. Convert to log scale:  10 * log10(S + 1e-9)  (dB, clamped to ≥ -80 dB).
    4. Per-clip min-max normalisation to [0, 1].
    """

    def __init__(self):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
            # center=False so frames = 1 + (N - n_fft) // hop = 128 exactly
            center=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Tensor of shape (1, T).
        Returns:
            Tensor of shape (1, 64, 128), values in [0, 1].
        """
        waveform = _fix_length(waveform, TARGET_SAMPLES)   # (1, 66048)
        spec = self.mel(waveform)                          # (1, 64, 128)
        log_spec = _to_log_db(spec)                        # (1, 64, 128)
        norm_spec = _min_max_normalise(log_spec)           # (1, 64, 128)
        return norm_spec


# ── helpers ───────────────────────────────────────────────────────────────────

def _fix_length(waveform: torch.Tensor, target: int) -> torch.Tensor:
    """Pad (right, zero) or trim a waveform to exactly `target` samples."""
    n = waveform.shape[-1]
    if n < target:
        waveform = torch.nn.functional.pad(waveform, (0, target - n))
    elif n > target:
        waveform = waveform[..., :target]
    return waveform


def _to_log_db(spec: torch.Tensor, amin: float = 1e-9, top_db: float = 80.0) -> torch.Tensor:
    """Power spectrogram → dB, floored at (max - top_db)."""
    log_spec = 10.0 * torch.log10(spec.clamp(min=amin))
    # floor at max - top_db so silence doesn't go to -inf
    log_spec = torch.max(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - top_db)
    return log_spec


def _min_max_normalise(x: torch.Tensor) -> torch.Tensor:
    """Per-clip min-max normalisation to [0, 1]."""
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    denom = (x_max - x_min).clamp(min=1e-8)
    return (x - x_min) / denom


# ── VGGish transform ─────────────────────────────────────────────────────────

class VGGishMelSpectrogramTransform(nn.Module):
    """Identical pipeline to MelSpectrogramTransform — only target length differs.

    Input : (1, T)  float32 waveform, already resampled to SAMPLE_RATE
    Output: (1, 64, 96) float32 in [0, 1]

    The four pipeline steps are bit-for-bit identical to the CNN transform:
      1. Pad/trim to VGGISH_TARGET_SAMPLES (49 664 samples, ~2.25 s).
      2. Power Mel spectrogram  (n_mels=64, n_fft=1024, hop=512, center=False).
      3. Log-dB scale           (10·log10(S + 1e-9), clamped ≥ max − 80 dB).
      4. Per-clip min-max norm  → [0, 1].

    The ONLY difference vs MelSpectrogramTransform is VGGISH_TARGET_SAMPLES,
    which is derived from the required 96 time-frame output:
        target = (96 − 1) × hop + n_fft = 49 664
    """

    def __init__(self):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
            center=False,   # frames = 1 + (N - n_fft) // hop = 96 exactly
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Tensor of shape (1, T).
        Returns:
            Tensor of shape (1, 64, 96), values in [0, 1].
        """
        waveform = _fix_length(waveform, VGGISH_TARGET_SAMPLES)  # (1, 49664)
        spec     = self.mel(waveform)                             # (1, 64, 96)
        log_spec = _to_log_db(spec)                               # (1, 64, 96)
        norm     = _min_max_normalise(log_spec)                   # (1, 64, 96)
        assert norm.shape[-1] == VGGISH_N_TIME_FRAMES, (
            f"[CRITICAL] VGGish spectrogram has {norm.shape[-1]} time frames, "
            f"expected {VGGISH_N_TIME_FRAMES}. Fix VGGISH_TARGET_SAMPLES."
        )
        return norm


# ── quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dataset.urbansound_dataset import UrbanSoundDataset

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "UrbanSound8K"))

    dataset = UrbanSoundDataset(
        root_dir=root,
        folds=[1],
    )

    spec, label = dataset[0]
    print(f"Output shape : {tuple(spec.shape)}")   # expect (1, 64, 128)
    print(f"Value range  : [{spec.min():.4f}, {spec.max():.4f}]")  # expect [0, 1]
    print(f"Label        : {label} ({dataset.get_class_name(label)})")
    assert spec.shape == (1, N_MELS, N_TIME_FRAMES), f"unexpected shape {spec.shape}"
    assert 0.0 <= spec.min() and spec.max() <= 1.0, "normalisation out of [0,1]"
    print("All assertions passed.")

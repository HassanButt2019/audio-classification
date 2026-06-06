import torch
import torch.nn as nn


class VGGish(nn.Module):
    """VGGish-style CNN for urban sound classification on UrbanSound8K.

    Input:  (batch, 1, 64, 128)  — log-Mel spectrogram
    Output: (batch, 10)          — raw logits

    Architecture (planned)
    ----------------------
    Stacked 3×3 Conv blocks with BatchNorm, two conv layers per block,
    MaxPool after each block (VGG-style depth), then a global average pool
    or flattened FC classifier head.

    NOT YET IMPLEMENTED — skeleton only.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        raise NotImplementedError("VGGish model is not yet implemented.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

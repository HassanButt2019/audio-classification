import torch
import torch.nn as nn


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for urban sound classification.

    Input:  (batch, 1, 64, 128)  — log-Mel spectrogram
    Output: (batch, 10)          — raw logits

    Architecture (planned)
    ----------------------
    2 conv blocks extract local spectro-temporal features, output reshaped to
    (batch, time_steps, features), fed into a 2-layer bidirectional GRU.
    Final hidden state projected to class logits via a linear layer.

    NOT YET IMPLEMENTED — skeleton only.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout:     float = 0.5,
        hidden_size: int = 128,
        num_layers:  int = 2,
    ):
        super().__init__()
        raise NotImplementedError("CRNN model is not yet implemented.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

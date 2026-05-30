import torch
import torch.nn as nn


class UrbanSoundCNN(nn.Module):
    """CNN for urban sound classification on UrbanSound8K.

    Input:  (batch, 1, 64, 128)  — log-Mel spectrogram
    Output: (batch, 10)          — raw logits, one per class

    Architecture
    ------------
    Three convolutional blocks (conv → ReLU → max-pool) extract increasingly
    abstract spectro-temporal features, then two fully-connected layers map
    the flattened feature volume to class scores.

    Do NOT apply Softmax here.  nn.CrossEntropyLoss fuses log-softmax and
    NLLLoss for numerical stability.  At inference time call
    torch.softmax(logits, dim=1) to get probabilities.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()

        # ── Convolutional blocks ──────────────────────────────────────────────
        # padding=1 keeps spatial dims unchanged after each 3×3 conv.
        # Max-pool(2,2) halves height and width after each block.

        # Block 1:  (B,  1, 64, 128) → (B, 32, 32, 64)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2:  (B, 32, 32, 64) → (B, 64, 16, 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3:  (B, 64, 16, 32) → (B, 128, 8, 16)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Classifier head ───────────────────────────────────────────────────
        # Flatten: 128 × 8 × 16 = 16 384 features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 16, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float32 tensor of shape (batch, 1, 64, 128).

        Returns:
            Logits tensor of shape (batch, 10).
        """
        x = self.block1(x)   # (B,  1, 64, 128) → (B, 32, 32, 64)
        x = self.block2(x)   # (B, 32, 32,  64) → (B, 64, 16, 32)
        x = self.block3(x)   # (B, 64, 16,  32) → (B,128,  8, 16)
        x = self.classifier(x)  # (B, 16384) → (B, 10)
        return x


if __name__ == "__main__":
    model = UrbanSoundCNN()
    model.eval()

    dummy = torch.zeros(8, 1, 64, 128)   # batch of 8 spectrograms
    logits = model(dummy)

    # ── shape checks ─────────────────────────────────────────────────────────
    assert logits.shape == (8, 10), f"unexpected output shape: {logits.shape}"
    print(f"Input  shape : {tuple(dummy.shape)}")
    print(f"Output shape : {tuple(logits.shape)}")

    # ── probabilities at inference ────────────────────────────────────────────
    probs = torch.softmax(logits, dim=1)
    print(f"Prob   sum   : {probs[0].sum().item():.6f}  (expect 1.0)")

    # ── parameter count ───────────────────────────────────────────────────────
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")

    # ── layer-by-layer breakdown ──────────────────────────────────────────────
    print("\nLayer breakdown:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name:<30} {params:>10,} params")

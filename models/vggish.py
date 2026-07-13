import torch
import torch.nn as nn


class VGGish(nn.Module):
    """VGGish-style CNN for urban sound classification on UrbanSound8K.

    Input:  (batch, 1, 64, 96)  — log-Mel spectrogram (n_mels=64, t=96 frames)
    Output: (batch, 10)          — raw logits, one per class

    Architecture
    ------------
    Block 1 : Conv(1→64,   3×3, pad=1) → ReLU → MaxPool(2,2) → [B,  64, 32, 48]
    Block 2 : Conv(64→128, 3×3, pad=1) → ReLU → MaxPool(2,2) → [B, 128, 16, 24]
    Block 3 : Conv(128→256,3×3, pad=1) → ReLU
              Conv(256→256,3×3, pad=1) → ReLU → MaxPool(2,2) → [B, 256,  8, 12]
    Block 4 : Conv(256→512,3×3, pad=1) → ReLU
              Conv(512→512,3×3, pad=1) → ReLU → MaxPool(2,2) → [B, 512,  4,  6]
    Flatten : 512 × 4 × 6 = 12 288 features
    FC head : Linear(12288→4096) → ReLU → Dropout(p)
              Linear(4096 →4096) → ReLU → Dropout(p)
              Linear(4096 →10)

    Do NOT apply Softmax here. nn.CrossEntropyLoss fuses log-softmax and NLLLoss
    for numerical stability. Call torch.softmax(logits, dim=1) only at inference.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()

        # ── Convolutional blocks ──────────────────────────────────────────────
        # Block 1: (B,   1, 64, 96) → (B,  64, 32, 48)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1,   out_channels=64,  kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: (B,  64, 32, 48) → (B, 128, 16, 24)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: (B, 128, 16, 24) → (B, 256, 8, 12)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: (B, 256, 8, 12) → (B, 512, 4, 6)
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Classifier head ───────────────────────────────────────────────────
        # Flatten: 512 × 4 × 6 = 12 288
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float32 tensor of shape (batch, 1, 64, 96).

        Returns:
            Logits tensor of shape (batch, 10).
        """
        x = self.block1(x)      # (B,  64, 32, 48)
        x = self.block2(x)      # (B, 128, 16, 24)
        x = self.block3(x)      # (B, 256,  8, 12)
        x = self.block4(x)      # (B, 512,  4,  6)
        x = self.classifier(x)  # (B, 10)
        return x


def xavier_uniform_init(module: nn.Module) -> None:
    """Apply Xavier uniform initialisation to Conv2d and Linear layers.

    Called via model.apply(xavier_uniform_init) before the first training epoch.
    Biases are zero-initialised.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


if __name__ == "__main__":
    model = VGGish()
    model.apply(xavier_uniform_init)
    model.eval()

    dummy = torch.zeros(4, 1, 64, 96)
    logits = model(dummy)

    assert logits.shape == (4, 10), f"unexpected output shape: {logits.shape}"
    print(f"Input  shape : {tuple(dummy.shape)}")
    print(f"Output shape : {tuple(logits.shape)}")

    probs = torch.softmax(logits, dim=1)
    print(f"Prob   sum   : {probs[0].sum().item():.6f}  (expect 1.0)")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")

    print("\nLayer breakdown:")
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            params = sum(p.numel() for p in m.parameters())
            print(f"  {name:<35} {params:>12,} params")

    # Trace intermediate shapes
    print("\nShape trace:")
    x = dummy
    for i, block in enumerate([model.block1, model.block2, model.block3, model.block4], 1):
        x = block(x)
        print(f"  After block{i}: {tuple(x.shape)}")
    print(f"  After flatten+FC: {tuple(model.classifier(x).shape)}")

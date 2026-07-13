import torch
import torch.nn as nn


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for urban sound classification.

    Input:  (batch, 1, 64, 128)  — log-Mel spectrogram (identical shape to CNN)
    Output: (batch, 10)          — raw logits, no softmax

    Architecture
    ------------
    CNN front-end — 4 blocks, each: Conv2d → BatchNorm → ReLU → MaxPool

      Block 1: Conv2d(1→32,   3×3, pad=1) → BN → ReLU → MaxPool(2,2)  → [B,  32, 32, 64]
      Block 2: Conv2d(32→64,  3×3, pad=1) → BN → ReLU → MaxPool(2,2)  → [B,  64, 16, 32]
      Block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2,4)  → [B, 128,  8,  8]
      Block 4: Conv2d(128→128,3×3, pad=1) → BN → ReLU → MaxPool(2,4)  → [B, 128,  4,  2]

    Reshape:  [B, C=128, F=4, T=2] → permute(0,3,1,2) → view → [B, T=2, C×F=512]
              The time axis T feeds the GRU as the sequence dimension.
              Channels × freq bins (128×4 = 512) become the GRU input feature size.

    BiGRU:    GRU(input=512, hidden=256, layers=2, bidirectional=True, batch_first=True)
              Output: [B, 2, 512]  (256 forward + 256 backward concatenated)
              Take last time step only: output[:, -1, :] → [B, 512]

    Head:     Dropout(0.5) → Linear(512, 10)

    Do NOT apply Softmax here.  nn.CrossEntropyLoss fuses log-softmax and NLLLoss
    for numerical stability.  Call torch.softmax(logits, dim=1) only at inference.

    Weight initialisation — apply once before training via model.apply(crnn_weight_init):
      Conv2d / Linear : Xavier uniform  (standard for feedforward layers)
      GRU weights     : Orthogonal      (standard for RNNs; prevents gradient issues
                                         caused by BPTT through recurrent connections)
      All biases      : zeros
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout:     float = 0.5,
        hidden_size: int = 256,
        num_layers:  int = 2,
    ):
        super().__init__()

        # Block 1: (B,  1, 64, 128) → (B, 32, 32, 64)
        self.block1 = nn.Sequential(
            nn.Conv2d(1,   32,  kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # Block 2: (B, 32, 32, 64) → (B, 64, 16, 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(32,  64,  kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # Block 3: (B, 64, 16, 32) → (B, 128, 8, 8)
        # MaxPool(2,4): freq 16→8, time 32→8
        self.block3 = nn.Sequential(
            nn.Conv2d(64,  128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4)),
        )

        # Block 4: (B, 128, 8, 8) → (B, 128, 4, 2)
        # MaxPool(2,4): freq 8→4, time 8→2
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4)),
        )

        # After CNN: [B, 128, 4, 2] → reshaped to [B, 2, 512]
        # GRU input_size = 128 channels × 4 freq bins = 512
        self.gru = nn.GRU(
            input_size=128 * 4,      # 512
            hidden_size=hidden_size,  # 256
            num_layers=num_layers,    # 2
            batch_first=True,
            bidirectional=True,
            dropout=0.0,  # no inter-layer GRU dropout; handled by head Dropout
        )

        # Head: 512 (256 fwd + 256 bwd) → num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float32 tensor of shape (batch, 1, 64, 128).
        Returns:
            Logits tensor of shape (batch, 10).
        """
        # ── CNN front-end ─────────────────────────────────────────────────────
        x = self.block1(x)    # (B,   1, 64, 128) → (B,  32, 32, 64)
        x = self.block2(x)    # (B,  32, 32,  64) → (B,  64, 16, 32)
        x = self.block3(x)    # (B,  64, 16,  32) → (B, 128,  8,  8)
        x = self.block4(x)    # (B, 128,  8,   8) → (B, 128,  4,  2)

        # ── reshape [B, C, F, T] → [B, T, C×F] ───────────────────────────────
        # dim layout after CNN: channels=128, freq=4, time=2
        # GRU needs: batch × sequence × features
        B, C, F, T = x.shape                       # B, 128, 4, 2
        x = x.permute(0, 3, 1, 2).contiguous()    # [B, T=2, C=128, F=4]
        x = x.view(B, T, C * F)                   # [B, 2, 512]

        # ── bidirectional GRU ─────────────────────────────────────────────────
        x, _ = self.gru(x)    # [B, 2, 512]  (256 fwd + 256 bwd at each step)

        # take last time step — carries maximal temporal context
        x = x[:, -1, :]       # [B, 512]

        # ── classifier head ───────────────────────────────────────────────────
        return self.classifier(x)   # [B, 10]


def crnn_weight_init(module: nn.Module) -> None:
    """Apply Xavier uniform to Conv2d/Linear; Orthogonal to GRU weight matrices.

    Called once before the first training epoch via model.apply(crnn_weight_init).

    GRU weight matrices (weight_ih, weight_hh) get Orthogonal initialisation to
    prevent vanishing/exploding gradients through recurrent connections at the
    start of training.  Xavier initialisation is for feedforward layers only.
    All biases are zero-initialised.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


if __name__ == "__main__":
    model = CRNN()
    model.apply(crnn_weight_init)
    model.eval()

    dummy = torch.zeros(8, 1, 64, 128)
    logits = model(dummy)

    assert logits.shape == (8, 10), f"unexpected output shape: {logits.shape}"
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
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.GRU)):
            params = sum(p.numel() for p in m.parameters())
            print(f"  {name:<40} {params:>12,} params")

    print("\nShape trace:")
    x = dummy
    for i, block in enumerate([model.block1, model.block2, model.block3, model.block4], 1):
        x = block(x)
        print(f"  After block{i}: {tuple(x.shape)}")
    B, C, F, T = x.shape
    x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
    print(f"  After reshape:  {tuple(x.shape)}  (should be [8, 2, 512])")
    x, _ = model.gru(x)
    print(f"  After BiGRU:    {tuple(x.shape)}  (should be [8, 2, 512])")
    x = x[:, -1, :]
    print(f"  Last time step: {tuple(x.shape)}  (should be [8, 512])")
    x = model.classifier(x)
    print(f"  After head:     {tuple(x.shape)}  (should be [8, 10])")

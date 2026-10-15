"""
src/model.py
============
PyTorch 1D-Convolutional Neural Network (1D-CNN) baseline model
for AAMI EC57 5-class ECG arrhythmia anomaly detection.

Input Shape:  (Batch_Size, 1, 200)  [Single-lead ECG, 200 samples ~555 ms]
Output Shape: (Batch_Size, 5)       [Logits for classes: N, S, V, F, Q]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """
    Standard 1D Convolutional Block consisting of:
    Conv1D -> BatchNorm1D -> ReLU -> MaxPool1D -> Dropout
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool_size: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class ECGCNN1D(nn.Module):
    """
    Baseline 1D-CNN architecture for ECG heartbeat classification.

    Parameters
    ----------
    in_channels : int
        Number of ECG input channels (default: 1 for single-lead MLII).
    num_classes : int
        Number of output heartbeat categories (default: 5 for AAMI EC57).
    base_filters : int
        Number of feature maps in the first convolutional layer (default: 32).
    dropout : float
        Dropout probability for regularization (default: 0.25).
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        base_filters: int = 32,
        dropout: float = 0.25
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature Extraction Backbone
        self.block1 = ConvBlock1D(in_channels, base_filters, kernel_size=7, pool_size=2, dropout=dropout)
        self.block2 = ConvBlock1D(base_filters, base_filters * 2, kernel_size=5, pool_size=2, dropout=dropout)
        self.block3 = ConvBlock1D(base_filters * 2, base_filters * 4, kernel_size=5, pool_size=2, dropout=dropout)
        self.block4 = ConvBlock1D(base_filters * 4, base_filters * 4, kernel_size=3, pool_size=2, dropout=dropout)

        # Global Average Pooling (aggregates temporal dimension)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(base_filters * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (Batch_Size, 1, 200).

        Returns
        -------
        torch.Tensor
            Unnormalized logits of shape (Batch_Size, num_classes).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, L) -> (B, 1, L)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.global_pool(x)       # (B, channels, 1)
        x = x.flatten(start_dim=1)    # (B, channels)
        logits = self.classifier(x)   # (B, num_classes)
        return logits


def get_model_summary(model: nn.Module) -> str:
    """Return model parameter count and structure summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        f"ECGCNN1D Architecture:\n"
        f"  - Total Parameters:     {total_params:,}\n"
        f"  - Trainable Parameters: {trainable_params:,}\n"
    )


if __name__ == "__main__":
    # Smoke test architecture
    model = ECGCNN1D(in_channels=1, num_classes=5)
    dummy_input = torch.randn(16, 1, 200)
    output = model(dummy_input)
    print(get_model_summary(model))
    print(f"Output shape for batch size 16: {output.shape} (Expected: [16, 5])")
    assert output.shape == (16, 5), "Output shape mismatch!"
    print("Model architecture verification passed!")

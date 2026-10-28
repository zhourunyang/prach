"""One-dimensional ResNet backbone with pluggable classification heads."""

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.skip = (
            nn.Sequential(nn.Conv1d(in_channels, channels, 1, stride=stride, bias=False),
                          nn.BatchNorm1d(channels))
            if stride != 1 or in_channels != channels else nn.Identity()
        )

    def forward(self, x):
        shortcut = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + shortcut, inplace=True)


class PRACHResNet(nn.Module):
    """[B, 2, 8192] IQ -> 512-d pooled -> embedding -> logits.

    head="linear"  : plain affine classifier (cross-entropy baseline).
    head="cosface" : normalised prototype classifier with additive angular margin,
                     which is the geometry the hypersphere/energy scorers assume.
    """

    def __init__(self, embedding_dim=128, num_classes=7, head="linear",
                 scale=16.0, margin=0.2):
        super().__init__()
        if head not in ("linear", "cosface"):
            raise ValueError(f"unknown head {head}")
        self.head = head
        self.scale = nn.Parameter(torch.tensor(float(scale))) if head == "cosface" else None
        self.margin = float(margin)
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        layers, incoming = [], 64
        for group, channels in enumerate((64, 128, 256, 512)):
            stride = 1 if group == 0 else 2
            layers.extend((BasicBlock1D(incoming, channels, stride),
                           BasicBlock1D(channels, channels)))
            incoming = channels
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.projection = nn.Linear(512, embedding_dim)
        if head == "linear":
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
            nn.init.xavier_uniform_(self.weight)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def embed(self, x):
        """Return (unnormalised features for the CE head, unit embedding for scoring)."""
        hidden = self.pool(self.blocks(self.stem(x))).squeeze(-1)
        features = self.projection(self.dropout(hidden))
        return features, F.normalize(features.float(), dim=1)

    def forward(self, x, targets=None):
        features, embedding = self.embed(x)
        if self.head == "linear":
            return self.classifier(features), embedding
        cosine = embedding @ F.normalize(self.weight, dim=1).t()
        logits = self.scale.float() * cosine
        if targets is not None and self.margin > 0:
            onehot = F.one_hot(targets, cosine.shape[1]).to(cosine.dtype)
            logits = self.scale.float() * (cosine - self.margin * onehot)
        return logits, embedding


def build(embedding_dim=128, num_classes=7, head="linear", **kwargs):
    return PRACHResNet(embedding_dim=embedding_dim, num_classes=num_classes,
                       head=head, **kwargs)

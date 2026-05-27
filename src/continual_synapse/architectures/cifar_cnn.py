"""CIFAR CNN architectures for the Split-CIFAR-100 CI cross-benchmark.

Two models, mirroring the Phase 4 / Phase 5.5 hippocampe ↔ neocortex
dual system but scaled up for 3-channel 32x32 imagery:

- :class:`CIFARHippocampus` — three conv-BN-ReLU-pool blocks (32 →
  64 → 128 channels) ending in a global-average-pooled linear
  classifier. ~107K params, designed to be the volatile fast
  learner.
- :class:`CIFARNeocortex` — Reduced ResNet-18, the standard CIFAR
  variant: 3x3 stem with no maxpool, four stages of two basic
  blocks each (64 → 128 → 256 → 512 channels, strides 1-2-2-2),
  GAP + linear classifier. ~11M params, designed for slow-learner
  capacity.

Both models expose a ``features(x)`` method that returns a
``{"low", "mid", "high"}`` dict of intermediate feature maps. These
hooks let the Phase 5.6.2 memory adapter snapshot multi-level
activations exactly like the Phase 4 MNIST recipe — just over
spatial conv maps instead of flat MLP activations.

The ``low / mid / high`` choice was made so the spatial resolution
matches across the two models (16x16 → 8x8 → 4x4), giving the
upcoming memory adapter a chance to compare or composite
representations across the dual-system pair.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- small CNN hippocampe ----------


class CIFARHippocampus(nn.Module):
    """Small fast CNN for CIFAR-100 class-incremental.

    Designed for the hippocampe role: volatile fast learner with
    multi-level features extractable for memory storage. Layout
    mirrors the typical "lightweight CIFAR baseline" used in
    continual-learning papers — three conv-BN-ReLU-pool blocks
    growing channel count 32 → 64 → 128, ending in GAP +
    classifier. ~107K params at ``num_classes=100``.
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        # Block 1: 32x32x3 → (conv + bn + relu) → 32x32x32 → pool → 16x16x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        # Block 2: 16x16x32 → 16x16x64 → pool → 8x8x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        # Block 3: 8x8x64 → 8x8x128 → pool → 4x4x128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2)
        self.classifier = nn.Linear(128, num_classes)

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return multi-level feature maps for memory storage.

        Shapes (batch dim B implicit):
            low:  (B, 32, 16, 16)
            mid:  (B, 64, 8, 8)
            high: (B, 128, 4, 4)
        """
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.pool(h)
        low = h
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool(h)
        mid = h
        h = F.relu(self.bn3(self.conv3(h)))
        h = self.pool(h)
        high = h
        return {"low": low, "mid": mid, "high": high}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        gap = F.adaptive_avg_pool2d(feats["high"], 1).flatten(1)
        return self.classifier(gap)


# ---------- ResNet-18 (CIFAR variant) neocortex ----------


class BasicBlock(nn.Module):
    """Standard ResNet basic block: two 3x3 convs + residual.

    Uses ``bias=False`` on convs preceding BatchNorm (the BN's
    affine term subsumes the bias). When the block changes spatial
    resolution (``stride != 1``) or channel count, the residual
    branch goes through a 1x1 conv + BN to align shapes; otherwise
    the identity shortcut is used.
    """

    expansion: int = 1

    def __init__(
        self, in_planes: int, planes: int, stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample: nn.Sequential | None = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity)


class CIFARNeocortex(nn.Module):
    """Reduced ResNet-18 for CIFAR-100 class-incremental.

    Standard CIFAR variant: 3x3 stem (no initial maxpool), four
    stages of two ``BasicBlock``s each. Channel widths
    64 → 128 → 256 → 512 with stride-2 downsampling at the start
    of stages 2, 3, 4. Ends in global average pool + linear
    classifier. ~11M params at ``num_classes=100``.
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        # CIFAR-style stem — no initial 7x7 / maxpool combo so the
        # 32x32 input doesn't get downsampled before any blocks
        # see it.
        self.stem_conv = nn.Conv2d(
            3, 64, kernel_size=3, padding=1, bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(64)

        # Stage 1: 32x32x64 (no downsample on entry).
        self.stage1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )
        # Stage 2: 16x16x128 (stride 2 in first block).
        self.stage2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
        )
        # Stage 3: 8x8x256.
        self.stage3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),
        )
        # Stage 4: 4x4x512.
        self.stage4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512),
        )
        self.classifier = nn.Linear(512, num_classes)

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return multi-level feature maps for memory storage.

        Shapes (batch dim B implicit):
            low:  (B, 128, 16, 16)  — output of stage 2
            mid:  (B, 256,  8,  8)  — output of stage 3
            high: (B, 512,  4,  4)  — output of stage 4
        """
        h = F.relu(self.stem_bn(self.stem_conv(x)))
        h = self.stage1(h)
        h = self.stage2(h)
        low = h
        h = self.stage3(h)
        mid = h
        h = self.stage4(h)
        high = h
        return {"low": low, "mid": mid, "high": high}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        gap = F.adaptive_avg_pool2d(feats["high"], 1).flatten(1)
        return self.classifier(gap)

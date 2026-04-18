"""
Dense-FPN: Dense Cross-Stage Feature Pyramid Network

A neck module that enriches multi-scale FPN outputs through dense connectivity:
  Phase 1: Top-down dense intermediate maps (M2-M5) where each lower level
           directly receives signals from ALL higher stages (not just adjacent).
  Phase 2: Bottom-up aggregation (P2-P5) via stride-2 learned downsampling.

Input : OrderedDict of FPN features {'0': P2, '1': P3, '2': P4, '3': P5}
Output: Enriched dict with the same keys (plus 'pool' passthrough if present).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseFPN(nn.Module):
    """
    Dense Feature Pyramid Network neck.

    Replaces (or augments) standard FPN with dense cross-stage connectivity:

    Phase 1 — top-down dense intermediate maps:
        M5 = L5
        M4 = L4 + Up(M5)
        M3 = L3 + Up(M4) + Up(M5)          <- receives M5 directly
        M2 = L2 + Up(M3) + Up(M4) + Up(M5) <- receives all higher stages

    Phase 2 — bottom-up aggregation with learned stride-2 conv:
        P2 = M2
        P3 = M3 + Down(P2)
        P4 = M4 + Down(P3)
        P5 = M5 + Down(P4)

    Args:
        channels (int): Channel width for all pyramid levels (default: 256).
    """

    def __init__(self, channels: int = 256):
        super().__init__()

        # Lateral refinement: learnable per-level 1×1 projection
        self.lateral = nn.ModuleDict({
            '0': nn.Conv2d(channels, channels, 1),
            '1': nn.Conv2d(channels, channels, 1),
            '2': nn.Conv2d(channels, channels, 1),
            '3': nn.Conv2d(channels, channels, 1),
        })

        # Learned stride-2 downsampling for bottom-up phase (Phase 2)
        self.downsample = nn.ModuleDict({
            '01': nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            '12': nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            '23': nn.Conv2d(channels, channels, 3, stride=2, padding=1),
        })

        # Per-level output refinement after fusion
        self.output_conv = nn.ModuleDict({
            k: nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for k in ('0', '1', '2', '3')
        })

    @staticmethod
    def _up(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Upsample src to match target's spatial dimensions."""
        return F.interpolate(src, size=target.shape[-2:], mode='nearest')

    def forward(self, features: dict) -> dict:
        """
        Args:
            features: dict{'0': P2, '1': P3, '2': P4, '3': P5, ['pool': P6]}

        Returns:
            Enriched dict with the same key structure.
        """
        # --- Lateral projection ------------------------------------------
        L2 = self.lateral['0'](features['0'])
        L3 = self.lateral['1'](features['1'])
        L4 = self.lateral['2'](features['2'])
        L5 = self.lateral['3'](features['3'])

        # --- Phase 1: top-down dense intermediate maps -------------------
        M5 = L5
        M4 = L4 + self._up(M5, L4)
        M3 = L3 + self._up(M4, L3) + self._up(M5, L3)
        M2 = L2 + self._up(M3, L2) + self._up(M4, L2) + self._up(M5, L2)

        # --- Phase 2: bottom-up aggregation ------------------------------
        P2 = M2
        P3 = M3 + self.downsample['01'](P2)
        P4 = M4 + self.downsample['12'](P3)
        P5 = M5 + self.downsample['23'](P4)

        # --- Output refinement -------------------------------------------
        out = {
            '0': self.output_conv['0'](P2),
            '1': self.output_conv['1'](P3),
            '2': self.output_conv['2'](P4),
            '3': self.output_conv['3'](P5),
        }

        # Passthrough the pooled level used by RPN for large anchors
        if 'pool' in features:
            out['pool'] = features['pool']

        return out


def test_dense_fpn():
    print("Testing DenseFPN...")
    print("=" * 70)

    fpn = DenseFPN(channels=256)
    fpn.eval()

    # Simulate FPN outputs for a 640×640 image
    features = {
        '0': torch.randn(2, 256, 160, 160),  # P2  stride-4
        '1': torch.randn(2, 256,  80,  80),  # P3  stride-8
        '2': torch.randn(2, 256,  40,  40),  # P4  stride-16
        '3': torch.randn(2, 256,  20,  20),  # P5  stride-32
        'pool': torch.randn(2, 256, 10,  10), # P6  stride-64
    }

    with torch.no_grad():
        out = fpn(features)

    for k, v in out.items():
        assert v.shape == features[k].shape, f"Shape mismatch at key '{k}'"
        print(f"  [{k}] {tuple(features[k].shape)} -> {tuple(v.shape)} ✓")

    params = sum(p.numel() for p in fpn.parameters())
    print(f"\nParameters: {params / 1e6:.2f}M")
    print("✓ DenseFPN test passed!")
    print("=" * 70)


if __name__ == '__main__':
    test_dense_fpn()

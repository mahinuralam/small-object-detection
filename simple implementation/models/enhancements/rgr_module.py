"""
Reconstruction-Guided Refinement (RGR) Module — poster Fig. 2(b) right half.

Architecture:
    Input: error map Δ (B, 3, H, W)  +  P2 features (B, 256, H/4, W/4)

    1. Resize Δ to P2 spatial dims  → (B, 3, H/4, W/4)

    2. Dual depthwise-conv paths on resized Δ:

         Path 1 (3×3):
             DepthwiseConv(3→3, k=3)  →  GlobalAvgPool  →  Flatten
             FC(3 → 256)  →  ReLU  →  FC(256 → 256)

         Path 2 (5×5):
             DepthwiseConv(3→3, k=5)  →  GlobalAvgPool  →  Flatten
             FC(3 → 256)  →  ReLU  →  FC(256 → 256)

    3. Add paths  →  Sigmoid  →  channel attention  (B, 256, 1, 1)

    4. P2_enhanced = P2  ×  channel_attention

    Output: enhanced P2 (B, 256, H/4, W/4)

Interpretation note (see REFACTOR_NOTES.md §RGR-1)
----------------------------------------------------
The poster shows "Flatten → FC" after the depthwise conv.  A literal spatial
flatten on a 160×160 feature map would produce a 76 800-element FC input —
impractical.  We interpret this as: depthwise conv → GlobalAvgPool → flatten
the C-element vector → FC.  This is functionally equivalent to SE-style
channel excitation and matches the described "channel attention" semantics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DualPathAttention(nn.Module):
    """Single depthwise-conv path: DW → GAP → FC → ReLU → FC."""

    def __init__(self, kernel_size: int, error_channels: int, feat_channels: int):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(
            error_channels, error_channels, kernel_size,
            padding=padding, groups=error_channels,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)          # → (B, error_channels, 1, 1)
        self.fc1 = nn.Linear(error_channels, feat_channels)
        self.fc2 = nn.Linear(feat_channels, feat_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, error_channels, H, W) → (B, feat_channels)"""
        x = self.dw(x)                              # (B, C_err, H, W)
        x = self.gap(x).flatten(1)                  # (B, C_err)  ← "Flatten"
        x = F.relu(self.fc1(x), inplace=True)       # (B, 256)    ← "FC → ReLU"
        x = self.fc2(x)                             # (B, 256)    ← "FC"
        return x


class RGRModule(nn.Module):
    """
    Reconstruction-Guided Refinement Module (poster Fig. 2(b) right).

    Takes the reconstruction error map Δ and uses it to generate channel-wise
    attention that selectively amplifies P2 features in hard-to-detect regions.

    Args:
        error_channels (int): Channels in the error map (default 3 = RGB diff).
        feat_channels  (int): P2 channel width (default 256).
    """

    def __init__(self, error_channels: int = 3, feat_channels: int = 256):
        super().__init__()
        self.path3 = _DualPathAttention(3, error_channels, feat_channels)
        self.path5 = _DualPathAttention(5, error_channels, feat_channels)

    def forward(
        self,
        error_map: torch.Tensor,
        p2_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            error_map:   Δ = |r_img − img_inputs|  (B, 3, H, W)
            p2_features: P2 from CD-DPA             (B, 256, H/4, W/4)

        Returns:
            P2_enhanced: (B, 256, H/4, W/4)
        """
        # Resize error map to P2 spatial dimensions
        err = F.interpolate(
            error_map, size=p2_features.shape[-2:],
            mode='bilinear', align_corners=False,
        )                                           # (B, 3, H/4, W/4)

        a3 = self.path3(err)                        # (B, 256)
        a5 = self.path5(err)                        # (B, 256)

        # Sum paths → sigmoid → channel gate  (B, 256, 1, 1)
        attn = torch.sigmoid(a3 + a5).unsqueeze(-1).unsqueeze(-1)

        return p2_features * attn


# ── Self-test ──────────────────────────────────────────────────────────────
def _test():
    print("Testing RGRModule (poster Fig. 2b right)...")
    rgr = RGRModule(error_channels=3, feat_channels=256)
    rgr.eval()

    error_map  = torch.rand(2, 3, 640, 640)         # Δ at full resolution
    p2         = torch.randn(2, 256, 160, 160)       # P2 after CD-DPA

    with torch.no_grad():
        p2_enhanced = rgr(error_map, p2)

    assert p2_enhanced.shape == p2.shape, f"Shape mismatch: {p2_enhanced.shape}"

    # Gradient test
    p2.requires_grad_(True)
    error_map.requires_grad_(True)
    rgr.train()
    out = rgr(error_map, p2)
    out.sum().backward()
    assert p2.grad is not None
    assert error_map.grad is not None

    params = sum(p.numel() for p in rgr.parameters()) / 1e6
    print(f"  Output shape: {p2_enhanced.shape}")
    print(f"  Params:       {params:.3f}M")
    print(f"  Grad flow:    ✓")
    print("✓ RGRModule test passed!")


if __name__ == '__main__':
    _test()

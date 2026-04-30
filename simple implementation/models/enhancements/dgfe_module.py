"""
Difference map Guided Feature Enhancement (DGFE) Module.

Previously named RGRModule. Same architecture and weights — renamed to match
the paper terminology.

Architecture:
    Input: difference map Δ (B, 1, H, W)  +  P2 features (B, 256, H/4, W/4)

    1. Resize Δ to P2 spatial dims  → (B, 1, H/4, W/4)

    2. Channel attention — dual depthwise-conv paths on resized Δ:
         Path 1 (3×3): DW → GAP → FC(1→256) → ReLU → FC(256→256)
         Path 2 (5×5): DW → GAP → FC(1→256) → ReLU → FC(256→256)
         Sum → Sigmoid → (B, 256, 1, 1)

    3. Spatial attention — lightweight conv stack on resized Δ:
         Conv(1→16, 3×3) → ReLU → Conv(16→8, 3×3) → ReLU → Conv(8→1, 1×1) → Sigmoid
         Output: (B, 1, H/4, W/4)  — high where Δ is high (hard-to-reconstruct regions)

    4. P2_enhanced = P2 × channel_attention × (1 + spatial_attention)
       Residual form: spatial gate starts near 0 → (1+0)=1 → no disruption at init
       Grows during training to amplify at high-Δ locations only (never suppresses)

    Output: enhanced P2 (B, 256, H/4, W/4)
    - Channel gate:  rescales feature channels globally based on Δ magnitude
    - Spatial gate:  amplifies features WHERE reconstruction error is high
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


class DGFEModule(nn.Module):
    """
    Difference map Guided Feature Enhancement (DGFE) Module.

    Dual-gate attention driven by reconstruction error map Δ:
      - Channel gate  (B, 256, 1, 1): WHAT channels to amplify globally
      - Spatial gate  (B,   1, H, W): WHERE to amplify (at high-Δ locations)

    Args:
        error_channels (int): Channels in the error map (1 = grayscale Δ).
        feat_channels  (int): P2 channel width (default 256).
    """

    def __init__(self, error_channels: int = 1, feat_channels: int = 256):
        super().__init__()
        # Channel attention paths (unchanged from original)
        self.path3 = _DualPathAttention(3, error_channels, feat_channels)
        self.path5 = _DualPathAttention(5, error_channels, feat_channels)

        # Spatial attention: lightweight conv stack preserving H×W structure of Δ
        # Output is used as residual: P2 × ch_attn × (1 + sp_attn)
        # Initialised near zero so training starts without disrupting existing weights
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(error_channels, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )
        # Init last conv bias to -6 so sigmoid ≈ 0 at start → (1 + 0) = 1 → no disruption
        nn.init.zeros_(self.spatial_attn[-2].weight)
        nn.init.constant_(self.spatial_attn[-2].bias, -6.0)

    def forward(
        self,
        error_map: torch.Tensor,
        p2_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            error_map:   Δ = |r_img − img_inputs|  (B, 1, H, W)
            p2_features: P2 from CD-DPA             (B, 256, H/4, W/4)

        Returns:
            P2_enhanced: (B, 256, H/4, W/4)
        """
        # Resize Δ to P2 spatial dimensions
        err = F.interpolate(
            error_map, size=p2_features.shape[-2:],
            mode='bilinear', align_corners=False,
        )                                               # (B, 1, H/4, W/4)

        # Channel gate: WHAT channels matter (global, per-channel scalar)
        a3 = self.path3(err)                            # (B, 256)
        a5 = self.path5(err)                            # (B, 256)
        ch_attn = torch.sigmoid(a3 + a5).unsqueeze(-1).unsqueeze(-1)  # (B, 256, 1, 1)

        # Spatial gate: WHERE to amplify (preserves H×W structure of Δ)
        # Residual form: (1 + sp_attn) starts at 1.0 → no disruption to existing weights
        # As training progresses sp_attn grows at high-Δ locations → additive spatial boost
        sp_attn = self.spatial_attn(err)                # (B, 1, H/4, W/4)  values in [0,1]

        return p2_features * ch_attn * (1.0 + sp_attn)


# ── Self-test ──────────────────────────────────────────────────────────────
def _test():
    print("Testing DGFEModule (spatial + channel attention)...")
    dgfe = DGFEModule(error_channels=1, feat_channels=256)
    dgfe.eval()

    error_map = torch.rand(2, 1, 512, 512)       # Δ at full tile resolution
    p2        = torch.randn(2, 256, 128, 128)     # P2 crop (128×128)

    with torch.no_grad():
        p2_enhanced = dgfe(error_map, p2)

    assert p2_enhanced.shape == p2.shape, f"Shape mismatch: {p2_enhanced.shape}"

    # Gradient test
    p2.requires_grad_(True)
    error_map.requires_grad_(True)
    dgfe.train()
    out = dgfe(error_map, p2)
    out.sum().backward()
    assert p2.grad is not None
    assert error_map.grad is not None

    params = sum(p.numel() for p in dgfe.parameters()) / 1e6
    print(f"  Output shape : {p2_enhanced.shape}")
    print(f"  Params       : {params:.4f}M")
    print(f"  Grad flow    : ok")
    print("ok DGFEModule test passed!")


# Backward compatibility alias
DGFE = DGFEModule

if __name__ == '__main__':
    _test()

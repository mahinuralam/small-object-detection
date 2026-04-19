"""
Reconstruction Module — poster Fig. 2(b) left half.

Architecture (exactly as shown in Fig. 2(b)):
    P2  (B, 256, H/4, W/4)
        ConvTranspose2d(3×3, 128) + ReLU   →  (B, 128, H/2, W/2)
        ResidualBlock(128)
        ConvTranspose2d(3×3, 64)  + ReLU   →  (B, 64,  H,   W)
        ResidualBlock(64)
        Conv2d(3×3, 3) + Sigmoid            →  r_img (B, 3, H, W)  ∈ [0, 1]

The reconstruction error map is computed externally:
    Δ = |r_img − img_inputs|      (B, 3, H, W)

Training loss:
    L_rec = L1(r_img, img_inputs)

Relation to existing reconstruction_head.py
--------------------------------------------
reconstruction_head.py follows the SR-TOD paper (Cao et al., ECCV 2024) and
uses DoubleConv blocks.  This module follows the poster (Fig. 2(b)) and uses
explicit ResidualBlocks with BN.  Both are kept; this file is used by the full
framework, reconstruction_head.py is used by FasterRCNN_SRTOD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard residual block: Conv-BN-ReLU-Conv-BN + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block(x) + x, inplace=True)


class ReconstructionModule(nn.Module):
    """
    Poster Fig. 2(b) left: Reconstruction Module.

    Reconstructs the input image from P2 FPN features so that the pixel-level
    reconstruction error Δ = |r_img − I| can be used by RGRModule to guide
    feature enhancement.

    Args:
        in_channels (int): P2 channel width.  Default: 256.
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()

        # ConvTranspose2d(256→128, 3×3, stride=2) + ReLU
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, 128, kernel_size=3, stride=2,
                padding=1, output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.rb1 = ResidualBlock(128)

        # ConvTranspose2d(128→64, 3×3, stride=2) + ReLU
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2,
                padding=1, output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.rb2 = ResidualBlock(64)

        # Conv2d(64→3, 3×3) + Sigmoid
        self.out = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, p2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p2: P2 features (B, 256, H/4, W/4).
        Returns:
            r_img: Reconstructed image (B, 3, H, W) in [0, 1].
        """
        x = self.up1(p2)   # (B, 128, H/2, W/2)
        x = self.rb1(x)
        x = self.up2(x)    # (B, 64, H, W)
        x = self.rb2(x)
        return self.out(x)  # (B, 3, H, W)

    # ------------------------------------------------------------------
    @staticmethod
    def error_map(r_img: torch.Tensor, img_inputs: torch.Tensor) -> torch.Tensor:
        """
        Pixel-level reconstruction error map.

        Returns |r_img − img_inputs| keeping all 3 channels so RGRModule
        can process multi-channel error patterns.

        Args:
            r_img:      (B, 3, H, W) reconstructed, in [0, 1]
            img_inputs: (B, 3, H, W) original,      in [0, 1]  (no mean/std)
        Returns:
            Δ: (B, 3, H, W) ≥ 0
        """
        return torch.abs(r_img - img_inputs)

    @staticmethod
    def reconstruction_loss(
        r_img: torch.Tensor, img_inputs: torch.Tensor
    ) -> torch.Tensor:
        """L1 reconstruction loss (same formula as SR-TOD)."""
        return F.l1_loss(r_img, img_inputs)


# ── Self-test ──────────────────────────────────────────────────────────────
def _test():
    print("Testing ReconstructionModule (poster Fig. 2b left)...")
    rm = ReconstructionModule(in_channels=256)
    rm.eval()

    p2 = torch.randn(2, 256, 160, 160)          # 640×640 input → P2
    img = torch.rand(2, 3, 640, 640)             # original [0,1]

    with torch.no_grad():
        r_img = rm(p2)
        assert r_img.shape == (2, 3, 640, 640),  f"Bad shape: {r_img.shape}"
        assert r_img.min() >= 0 and r_img.max() <= 1, "Out of [0,1]"

        delta = ReconstructionModule.error_map(r_img, img)
        assert delta.shape == (2, 3, 640, 640)
        assert delta.min() >= 0

        loss = ReconstructionModule.reconstruction_loss(r_img, img)
        assert loss.item() > 0

    params = sum(p.numel() for p in rm.parameters()) / 1e6
    print(f"  r_img:  {r_img.shape}  range [{r_img.min():.3f}, {r_img.max():.3f}]")
    print(f"  Δ:      {delta.shape}")
    print(f"  L_rec:  {loss.item():.4f}")
    print(f"  Params: {params:.2f}M")
    print("✓ ReconstructionModule test passed!")


if __name__ == '__main__':
    _test()

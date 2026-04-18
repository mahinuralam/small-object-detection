"""
Reconstruction Head (RH) module for SR-TOD
Reconstructs input image from P2 FPN features to generate difference maps.

Based on: "Visible and Clear: Finding Tiny Objects in Difference Map"
           (Cao et al., ECCV 2024)
           https://github.com/Hiyuur/SR-TOD

Design (from the paper)
=======================
    P2 features  (N, 256, H/4, W/4)
        ↓  Up_direct(256 → 128)   →  (N, 128, H/2, W/2)
        ↓  Up_direct(128 → 64)    →  (N, 64,  H,   W)
        ↓  OutConv(64 → 3) + Sigmoid → r_img  (N, 3, H, W)  in [0, 1]

    Difference map:
        Δ(x,y) = (1/3) Σ_c |r_img(x,y,c) - I(x,y,c)|    (N, 1, H, W)

    The map highlights tiny objects because the reconstruction from P2
    (which is at 1/4 resolution) loses fine detail of small objects, so
    Δ is large there.

Training
--------
The RH is trained **end-to-end** with the detector:
    L = L_det + λ · L1(r_img, I)
where I is the original image normalised to [0, 1] (no mean/std).

The difference map is **not** an explicit loss target — it emerges
naturally from the L1 reconstruction objective.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [ReLU]) * 2
    Two sequential 3x3 convolutions with ReLU activation.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up_direct(nn.Module):
    """Upscaling then double conv.
    Uses ConvTranspose2d for 2× upsampling followed by DoubleConv.
    """

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Default: ConvTranspose2d (matches SR-TOD official code)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1):
        x = self.up(x1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """Output convolution with Sigmoid activation.
    Maps to RGB channels (3) with values in [0, 1].
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class ReconstructionHead(nn.Module):
    """
    Reconstruction Head (RH) for SR-TOD.

    Reconstructs the input image from P2 FPN features and computes the
    difference map Δ between reconstructed and original images.

    Architecture:
        P2 (256ch, H/4, W/4)
        → Up1 (128ch, H/2, W/2)
        → Up2 (64ch, H, W)
        → OutConv (3ch RGB, H, W) with Sigmoid

    Args:
        in_channels (int):  Input channels from P2 features.  Default: 256
        out_channels (int): Output RGB channels.               Default: 3
        bilinear (bool):    Use bilinear upsampling instead of ConvTranspose2d.

    Example::

        rh = ReconstructionHead()
        p2 = torch.randn(4, 256, 160, 160)          # batch of 4, 640×640 input
        r_img = rh(p2)                                # (4, 3, 640, 640)
        diff  = rh.difference_map(r_img, original)    # (4, 1, 640, 640)
    """

    def __init__(self, in_channels=256, out_channels=3, bilinear=False):
        super(ReconstructionHead, self).__init__()
        self.up1 = Up_direct(in_channels, 128, bilinear=bilinear)
        self.up2 = Up_direct(128, 64, bilinear=bilinear)
        self.out_conv = OutConv(64, out_channels)

    def forward(self, x):
        """
        Reconstruct from P2 features.

        Args:
            x: P2 features (N, 256, H/4, W/4)

        Returns:
            r_img: Reconstructed image (N, 3, H, W) with values in [0, 1]
        """
        P0 = self.up1(x)                # (N, 128, H/2, W/2)
        P0 = self.up2(P0)               # (N, 64,  H,   W)
        r_img = self.out_conv(P0)        # (N, 3,   H,   W)
        return r_img

    # ------------------------------------------------------------------
    # Difference map (SR-TOD §3.2)
    # ------------------------------------------------------------------

    @staticmethod
    def difference_map(
        r_img: torch.Tensor,
        img_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the single-channel difference map Δ.

        Exact formula from the SR-TOD official implementation:
            Δ = sum(|r_img − I|, dim=channel) / 3

        Args:
            r_img:      Reconstructed image (N, 3, H, W) in [0, 1]
            img_inputs: Original image      (N, 3, H, W) in [0, 1]
                        Must NOT be mean/std normalised.

        Returns:
            Δ: Difference map (N, 1, H, W), non-negative float.
               High values indicate hard-to-reconstruct regions
               (tiny objects / fine detail).
        """
        return torch.sum(
            torch.abs(r_img - img_inputs),
            dim=1,
            keepdim=True,
        ) / 3.0

    # ------------------------------------------------------------------
    # Reconstruction loss (SR-TOD §3.3)
    # ------------------------------------------------------------------

    @staticmethod
    def reconstruction_loss(
        r_img: torch.Tensor,
        img_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        L1 reconstruction loss  (matches SR-TOD official code).

        L_res = mean( |r_img − I| )

        Args:
            r_img:      (N, 3, H, W) in [0, 1]
            img_inputs: (N, 3, H, W) in [0, 1], un-normalised original
        """
        return torch.nn.functional.l1_loss(r_img, img_inputs)

    # ------------------------------------------------------------------
    # Convenience: forward + diff in one call
    # ------------------------------------------------------------------

    def forward_with_diff(
        self,
        p2_features: torch.Tensor,
        img_inputs: torch.Tensor,
    ):
        """
        End-to-end convenience used by the detector's loss() and predict():
            r_img, diff_map = rh.forward_with_diff(P2, img_inputs)

        Args:
            p2_features: (N, 256, H/4, W/4)
            img_inputs:  (N, 3, H, W) in [0, 1]

        Returns:
            r_img:    (N, 3, H, W)    reconstructed image
            diff_map: (N, 1, H, W)    difference map Δ
        """
        r_img    = self.forward(p2_features)
        diff_map = self.difference_map(r_img, img_inputs)
        return r_img, diff_map


if __name__ == "__main__":
    # ── Self-test ──
    print("Testing ReconstructionHead (SR-TOD)...")

    rh = ReconstructionHead(in_channels=256, out_channels=3)

    # P2 features for a 640×640 input → 160×160
    p2_features = torch.randn(2, 256, 160, 160)
    img_inputs  = torch.rand(2, 3, 640, 640)           # simulated [0,1] original

    # Basic forward
    reconstructed = rh(p2_features)
    assert reconstructed.shape == (2, 3, 640, 640), "Output shape mismatch!"
    assert reconstructed.min() >= 0 and reconstructed.max() <= 1, "Out of [0,1]!"
    print(f"✓ forward: {p2_features.shape} → {reconstructed.shape}")

    # Difference map
    diff = rh.difference_map(reconstructed, img_inputs)
    assert diff.shape == (2, 1, 640, 640), "Diff map shape mismatch!"
    assert diff.min() >= 0, "Diff map should be non-negative!"
    print(f"✓ difference_map: {diff.shape}, range [{diff.min():.4f}, {diff.max():.4f}]")

    # Reconstruction loss
    loss = rh.reconstruction_loss(reconstructed, img_inputs)
    print(f"✓ reconstruction_loss: {loss.item():.4f}")

    # Combined forward
    r_img, d_map = rh.forward_with_diff(p2_features, img_inputs)
    assert r_img.shape == (2, 3, 640, 640)
    assert d_map.shape == (2, 1, 640, 640)
    print(f"✓ forward_with_diff: r_img={r_img.shape}, diff_map={d_map.shape}")

    print("\n✓ ReconstructionHead test passed!")

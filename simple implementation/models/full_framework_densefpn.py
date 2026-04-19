"""
Full Framework with DenseFPN neck — poster Fig. 3 (DenseFPN variant).

Pipeline:
    Image → ResNet50 → Standard FPN → DenseFPN → {P2, P3, P4, P5, pool}
                                                    ↓ CD-DPA on P2, P3, P4
                                                P2_cddpa → ReconstructionModule → r_img
                                                           error_map = |r_img − img_inputs|
                                                P2_cddpa + error_map → RGR → P2_final
                                                {P2_final, P3_cddpa, P4_cddpa, P5, pool}
                                                    ↓ Faster R-CNN RPN + ROI Head
                                                Detections

Difference from full_framework.py (IFPN variant):
    IFPN bakes dense cross-stage connectivity into the backbone neck, replacing
    the standard FPN entirely.  Here we keep torchvision's standard FPN
    (pretrained on COCO) and stack DenseFPN on top as a post-FPN enrichment
    module.  This gives the same dense-connectivity benefit while retaining
    COCO-pretrained lateral weights.

Joint training loss:
    L = L_det  +  λ_rec × L1(r_img, img_inputs)
    λ_rec = 0.1  (default)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from models.enhancements.dense_fpn import DenseFPN
from models.enhancements.cddpa_module import CDDPA
from models.enhancements.reconstruction_module import ReconstructionModule
from models.enhancements.rgr_module import RGRModule


class FasterRCNN_FullFramework_DenseFPN(nn.Module):
    """
    Full poster framework with DenseFPN neck.

    Components:
        backbone   : ResNet50 + torchvision Standard FPN  (COCO pretrained)
        dense_fpn  : DenseFPN applied to P2–P5 after standard FPN
        cd_dpa     : CDDPA on P2 ('0'), P3 ('1'), P4 ('2')
        recon      : ReconstructionModule  (P2 → r_img)
        rgr        : RGRModule  (error_map + P2 → P2_final)
        detector   : Faster R-CNN RPN + ROI heads (unchanged)

    Args:
        num_classes (int): Foreground classes + 1 background. Default: 11.
        fpn_channels (int): Channel width (DenseFPN, CD-DPA). Default: 256.
        lambda_rec (float): Weight of reconstruction loss. Default: 0.1.
        pretrained_backbone (bool): COCO-pretrained FPN backbone. Default: True.
        trainable_backbone_layers (int): ResNet layer groups to train. Default: 3.
        use_checkpoint (bool): Gradient checkpoint inside CDDPA. Default: False.
    """

    def __init__(
        self,
        num_classes: int = 11,
        fpn_channels: int = 256,
        lambda_rec: float = 0.1,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        # ── Standard FPN backbone (torchvision) ──────────────────────────
        if pretrained_backbone:
            base = fasterrcnn_resnet50_fpn(
                weights='DEFAULT',
                trainable_backbone_layers=trainable_backbone_layers,
            )
        else:
            base = fasterrcnn_resnet50_fpn(
                weights=None,
                trainable_backbone_layers=5,
            )

        in_features = base.roi_heads.box_predictor.cls_score.in_features
        base.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.base_model = base

        # ── DenseFPN neck (post-FPN enrichment on P2–P5) ─────────────────
        self.dense_fpn = DenseFPN(channels=fpn_channels)

        # ── CD-DPA on P2 ('0'), P3 ('1'), P4 ('2') ───────────────────────
        self.cddpa_levels = ['0', '1', '2']
        self.cd_dpa = nn.ModuleDict({
            lvl: CDDPA(channels=fpn_channels, use_checkpoint=use_checkpoint)
            for lvl in self.cddpa_levels
        })

        # ── Reconstruction + RGR ──────────────────────────────────────────
        self.recon = ReconstructionModule(in_channels=fpn_channels)
        self.rgr   = RGRModule(error_channels=3, feat_channels=fpn_channels)

        # ── Loss ──────────────────────────────────────────────────────────
        self.lambda_rec = lambda_rec
        self._loss_rec_fn = nn.L1Loss(reduction='mean')

        # ── Diagnostics ───────────────────────────────────────────────────
        p_dfpn  = sum(p.numel() for p in self.dense_fpn.parameters()) / 1e6
        p_cddpa = sum(p.numel() for m in self.cd_dpa.values()
                      for p in m.parameters()) / 1e6
        p_recon = sum(p.numel() for p in self.recon.parameters()) / 1e6
        p_rgr   = sum(p.numel() for p in self.rgr.parameters()) / 1e6
        p_total = sum(p.numel() for p in self.parameters()) / 1e6
        print("✓ FasterRCNN_FullFramework_DenseFPN initialised")
        print(f"  Backbone (ResNet50+Standard FPN): pretrained={pretrained_backbone}")
        print(f"  DenseFPN neck:                   {p_dfpn:.2f}M")
        print(f"  CD-DPA (P2,P3,P4):               {p_cddpa:.2f}M")
        print(f"  ReconstructionModule:            {p_recon:.2f}M")
        print(f"  RGRModule:                       {p_rgr:.2f}M")
        print(f"  Total:                           {p_total:.2f}M")
        print(f"  λ_rec = {lambda_rec}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_img_inputs(original_images: list, image_list) -> torch.Tensor:
        """Un-normalised [0,1] reconstruction target — same logic as IFPN variant."""
        batch_h, batch_w = image_list.tensors.shape[2:]
        resized_sizes = list(image_list.image_sizes)
        batch = []
        for idx, img in enumerate(original_images):
            img = img.float()
            if img.max() > 1.0:
                img = img / 255.0
            if idx < len(resized_sizes):
                tgt_h, tgt_w = resized_sizes[idx]
                if img.shape[1] != tgt_h or img.shape[2] != tgt_w:
                    img = F.interpolate(
                        img.unsqueeze(0), size=(tgt_h, tgt_w),
                        mode='bilinear', align_corners=False,
                    ).squeeze(0)
            _, h, w = img.shape
            pad_b, pad_r = batch_h - h, batch_w - w
            if pad_b > 0 or pad_r > 0:
                img = F.pad(img, (0, pad_r, 0, pad_b), value=0)
            batch.append(img)
        return torch.stack(batch).to(image_list.tensors.device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: list, targets: list = None):
        """
        Args:
            images:  list[Tensor(3,H,W)] — raw RGB, [0,1] float or [0,255] uint8.
            targets: list[dict] with 'boxes'(N,4) and 'labels'(N,) — train only.

        Returns:
            Training:  dict of scalar losses including 'loss_reconstruction'.
            Inference: list[dict] with 'boxes', 'labels', 'scores'.
        """
        original_images = [img.clone() for img in images]

        # 1. Torchvision transform (resize + normalise + batch)
        images_t, targets_p = self.base_model.transform(images, targets)

        # 2. Un-normalised [0,1] target for reconstruction loss
        img_inputs = self._build_img_inputs(original_images, images_t)

        # 3. Standard FPN backbone → {P2, P3, P4, P5, pool}
        features = self.base_model.backbone(images_t.tensors)

        # 4. DenseFPN enrichment (dense cross-stage on P2–P5; pool passthrough)
        features = self.dense_fpn(features)

        # 5. CD-DPA: enhance P2, P3, P4
        for lvl in self.cddpa_levels:
            features[lvl] = self.cd_dpa[lvl](features[lvl])

        # 6. Reconstruction stream on CD-DPA P2
        p2_for_recon = features['0'].clone()
        r_img = self.recon(p2_for_recon)                    # (B, 3, H, W)

        # 7. Error map Δ = |r_img − img_inputs|
        error_map = ReconstructionModule.error_map(r_img, img_inputs)

        # 8. RGR: channel-guided refinement of P2
        features['0'] = self.rgr(error_map, features['0'])  # P2_final

        # 9. Detection
        if self.training:
            if targets is None:
                raise ValueError("targets required during training")

            proposals, proposal_losses = self.base_model.rpn(
                images_t, features, targets_p
            )
            _, detector_losses = self.base_model.roi_heads(
                features, proposals, images_t.image_sizes, targets_p
            )

            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)
            losses['loss_reconstruction'] = (
                self._loss_rec_fn(r_img, img_inputs) * self.lambda_rec
            )
            return losses

        else:
            proposals, _ = self.base_model.rpn(images_t, features, None)
            detections, _ = self.base_model.roi_heads(
                features, proposals, images_t.image_sizes, None
            )
            detections = self.base_model.transform.postprocess(
                detections,
                images_t.image_sizes,
                [(img.shape[1], img.shape[2]) for img in original_images],
            )
            return detections

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.cd_dpa.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)


# ── Self-test ──────────────────────────────────────────────────────────────
def _test():
    print("\nTesting FasterRCNN_FullFramework_DenseFPN (no pretrained)...")
    print("=" * 70)

    model = FasterRCNN_FullFramework_DenseFPN(
        num_classes=11,
        lambda_rec=0.1,
        pretrained_backbone=False,
        use_checkpoint=False,
    )

    images = [torch.rand(3, 640, 640) for _ in range(2)]
    targets = [
        {'boxes':  torch.tensor([[10., 20., 80., 90.], [200., 150., 300., 280.]]),
         'labels': torch.tensor([1, 3])},
        {'boxes':  torch.tensor([[50., 60., 120., 130.]]),
         'labels': torch.tensor([2])},
    ]

    # Training mode
    print("\n1. Training mode")
    model.train()
    loss_dict = model(images, targets)
    print(f"   Loss keys: {list(loss_dict.keys())}")
    for k, v in loss_dict.items():
        assert torch.isfinite(v), f"Non-finite loss: {k}={v}"
        print(f"   {k}: {v.item():.4f}")
    print(f"   Total: {sum(loss_dict.values()).item():.4f}")
    assert 'loss_reconstruction' in loss_dict, "Missing reconstruction loss"

    # Inference mode
    print("\n2. Inference mode")
    model.eval()
    with torch.no_grad():
        dets = model(images)
    print(f"   Images: {len(dets)}")
    print(f"   Boxes shape: {dets[0]['boxes'].shape}")

    print("\n✓ FasterRCNN_FullFramework_DenseFPN test passed!")
    print("=" * 70)


if __name__ == '__main__':
    _test()

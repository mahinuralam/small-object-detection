"""
SR-TOD Enhanced Faster R-CNN
============================
End-to-end integration of the Reconstruction Head (RH), Difference Map,
and DGFE module following the official SR-TOD implementation.

Paper: "Visible and Clear: Finding Tiny Objects in Difference Map"
        Cao et al., ECCV 2024  — https://arxiv.org/abs/2405.11276
Code:   https://github.com/Hiyuur/SR-TOD

Key design (from the paper)
---------------------------
1. RH takes P2 FPN features → reconstructs original image (Sigmoid → [0,1])
2. Difference map  Δ = sum(|r_img − I|, dim=1, keepdim=True) / 3
3. DGFE uses Δ + a learnable threshold to re-weight P2 features
4. Enhanced P2 replaces original P2 → rest of Faster R-CNN unchanged
5. Training loss =  L_det (standard Faster R-CNN)  +  L_res (L1 reconstruction)
   All of RH, DGFE, and learnable_thresh train jointly with the detector.

Critical detail — img_inputs
-----------------------------
The original image for the reconstruction target must be normalised to
[0, 1] **without** ImageNet mean/std.  The SR-TOD data preprocessor
passes this as a separate tensor alongside the preprocessed batch.
In this torchvision-based implementation we capture the raw [0,1] images
before the GeneralizedRCNNTransform applies mean/std normalisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.enhancements.reconstruction_head import ReconstructionHead
from models.enhancements.dgfe_module import DGFE


class FasterRCNN_SRTOD(nn.Module):
    """
    SR-TOD Enhanced Faster R-CNN.

    Architecture::

        Image (N, 3, H, W)
            │
            ├─ img_inputs = I / 255  (un-normalised [0,1] target for RH)
            │
            ▼  GeneralizedRCNNTransform (mean/std normalisation + resize)
            │
            ▼  ResNet50 → FPN → {P2, P3, P4, P5, P6}
            │
            ├─ RH(P2.clone()) → r_img  (N, 3, H, W)
            │
            ├─ Δ = sum(|r_img − img_inputs|, dim=1, keepdim=True) / 3
            │
            ├─ enhanced_P2 = DGFE(P2, Δ, learnable_thresh)
            │
            ▼  {enhanced_P2, P3, P4, P5, P6} → RPN → ROI → Detections
            │
            ▼  L = L_det + L_res

    Args:
        num_classes (int):  Number of classes including background.
        learnable_thresh (float):  Initial filtration threshold for DGFE.
            SR-TOD default = 4/255 ≈ 0.0156862.
        pretrained_backbone (bool): Use ImageNet-pretrained ResNet50.
        loss_res_weight (float):  Weight multiplier for L_res.  Default 1.0.
    """

    def __init__(
        self,
        num_classes=11,
        learnable_thresh=0.0156862,
        pretrained_backbone=True,
        loss_res_weight=1.0,
    ):
        super(FasterRCNN_SRTOD, self).__init__()

        # ── Base detector ────────────────────────────────────────────────
        backbone = resnet_fpn_backbone(
            'resnet50', pretrained=pretrained_backbone
        )
        self.base_model = FasterRCNN(backbone, num_classes=91)

        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        # ── SR-TOD modules ───────────────────────────────────────────────
        self.rh   = ReconstructionHead(in_channels=256, out_channels=3)
        self.dgfe = DGFE(gate_channels=256, reduction_ratio=16)

        # Learnable threshold  (4/255 — matches SR-TOD official init)
        self.learnable_thresh = nn.Parameter(
            torch.tensor(learnable_thresh), requires_grad=True
        )

        # L1 reconstruction loss  (SR-TOD uses nn.L1Loss, NOT MSE)
        self.loss_res_fn     = nn.L1Loss(reduction='mean')
        self.loss_res_weight = loss_res_weight

        self.num_classes = num_classes

    # ------------------------------------------------------------------
    # Feature extraction (FPN)
    # ------------------------------------------------------------------

    def extract_features(self, images):
        """Extract FPN features from the backbone.

        Args:
            images: ImageList produced by GeneralizedRCNNTransform.
        Returns:
            OrderedDict  {'0': P2, '1': P3, '2': P4, '3': P5, 'pool': P6}
        """
        return self.base_model.backbone(images.tensors)

    # ------------------------------------------------------------------
    # Build un-normalised [0,1] target image for the RH
    # ------------------------------------------------------------------

    @staticmethod
    def _build_img_inputs(images, image_list):
        """
        Build img_inputs — the un-normalised [0,1] reconstruction target.

        The SR-TOD data preprocessor passes the original image scaled to
        [0,1] (no mean/std) as a separate tensor.  In torchvision we don't
        have that luxury, so we build it here from the raw input list before
        transforms by simply cloning + batching the images.

        Args:
            images:     list[Tensor(3,H,W)]  raw RGB in [0,1]
            image_list: ImageList from transform (provides padded size)

        Returns:
            img_inputs: (N, 3, H_pad, W_pad) in [0,1]
        """
        # image_list.image_sizes stores per-image resized sizes after
        # GeneralizedRCNNTransform; image_list.tensors stores the padded batch.
        batch_h, batch_w = image_list.tensors.shape[2:]
        resized_sizes = list(image_list.image_sizes)
        batch = []
        for idx, img in enumerate(images):
            # Ensure [0,1] — handle both uint8 and float conventions
            if img.max() > 1.0:
                img = img.float() / 255.0
            else:
                img = img.float()

            # Match torchvision transform output size for this sample.
            # This keeps RH target aligned with reconstructed r_img geometry.
            if idx < len(resized_sizes):
                tgt_h, tgt_w = resized_sizes[idx]
                if img.shape[1] != tgt_h or img.shape[2] != tgt_w:
                    img = F.interpolate(
                        img.unsqueeze(0),
                        size=(tgt_h, tgt_w),
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0)

            # Pad/resize to match image_list spatial dims
            _, h, w = img.shape
            pad_bottom = batch_h - h
            pad_right = batch_w - w
            if pad_bottom > 0 or pad_right > 0:
                img = F.pad(img, (0, pad_right, 0, pad_bottom), value=0)
            batch.append(img)
        return torch.stack(batch).to(image_list.tensors.device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images, targets=None):
        """
        Forward pass with SR-TOD enhancement (end-to-end).

        Args:
            images:  list[Tensor(3,H,W)] — raw RGB, either [0,1] float or
                     [0,255] uint8.
            targets: list[dict]  (training only)
                     Each dict: 'boxes' (N,4) float, 'labels' (N,) int64.

        Returns:
            Training:  dict of losses (includes 'loss_reconstruction')
            Inference: list[dict] with 'boxes', 'labels', 'scores'
        """
        # Keep raw reference for img_inputs
        original_images = [img.clone() for img in images]

        # ---- Transform (resize, normalise, batch) ----
        images_transformed, targets_processed = self.base_model.transform(
            images, targets
        )

        # ---- Build un-normalised [0,1] target for RH ----
        img_inputs = self._build_img_inputs(original_images, images_transformed)

        # ---- Backbone + FPN ----
        features = self.extract_features(images_transformed)

        # ---- SR-TOD: Reconstruct from P2, compute difference map ----
        r_img, difference_map = self.rh.forward_with_diff(
            features['0'].clone(), img_inputs
        )

        # ---- SR-TOD: Enhance P2 with DGFE ----
        features['0'] = self.dgfe(
            features['0'], difference_map, self.learnable_thresh
        )

        # ---- Training: compute all losses ----
        if self.training:
            if targets is None:
                raise ValueError("targets required in training mode")

            # RPN
            proposals, proposal_losses = self.base_model.rpn(
                images_transformed, features, targets_processed
            )
            # ROI head
            _, detector_losses = self.base_model.roi_heads(
                features, proposals,
                images_transformed.image_sizes, targets_processed
            )

            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)

            # SR-TOD reconstruction loss (L1 — matches official code)
            loss_res = self.loss_res_fn(r_img, img_inputs)
            losses['loss_reconstruction'] = loss_res * self.loss_res_weight

            return losses

        # ---- Inference ----
        else:
            proposals, _ = self.base_model.rpn(
                images_transformed, features, None
            )
            detections, _ = self.base_model.roi_heads(
                features, proposals,
                images_transformed.image_sizes, None
            )
            detections = self.base_model.transform.postprocess(
                detections,
                images_transformed.image_sizes,
                [(img.shape[1], img.shape[2]) for img in original_images],
            )
            return detections


if __name__ == "__main__":
    print("Testing FasterRCNN_SRTOD (SR-TOD paper)...")
    print("=" * 70)

    model = FasterRCNN_SRTOD(
        num_classes=11,
        learnable_thresh=0.0156862,
        pretrained_backbone=False,
    )

    # ── Inference ──
    print("\n1. Inference mode")
    model.eval()
    images = [torch.rand(3, 640, 640) for _ in range(2)]
    with torch.no_grad():
        preds = model(images)
    print(f"   {len(preds)} images, first has {len(preds[0]['boxes'])} detections")

    # ── Training ──
    print("\n2. Training mode")
    model.train()
    targets = [
        {'boxes': torch.tensor([[10, 20, 50, 60], [100, 150, 200, 250]],
                                dtype=torch.float32),
         'labels': torch.tensor([1, 2], dtype=torch.int64)},
        {'boxes': torch.tensor([[30, 40, 80, 90]], dtype=torch.float32),
         'labels': torch.tensor([3], dtype=torch.int64)},
    ]
    losses = model(images, targets)
    print(f"   Losses: {list(losses.keys())}")
    print(f"   L_reconstruction (L1): {losses['loss_reconstruction']:.4f}")
    print(f"   L_total: {sum(losses.values()):.4f}")

    # ── Learnable threshold ──
    print(f"\n3. Learnable threshold: {model.learnable_thresh.item():.6f} "
          f"({model.learnable_thresh.item()*255:.2f}/255)")

    print("\n✓ FasterRCNN_SRTOD test passed!")
    print("=" * 70)

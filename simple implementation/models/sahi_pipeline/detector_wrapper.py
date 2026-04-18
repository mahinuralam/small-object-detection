"""
Detector Wrappers for the 4-Stage Confidence-Guided SAHI Pipeline
==================================================================
CDDPADetector      — Stage 1 base detector (CD-DPA Faster R-CNN)
SRTODTileDetector  — Stage 3 tile detector (SR-TOD: RH + DGFE + detection)
BaseDetector       — Backward-compatible plain Faster R-CNN wrapper
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Union, Tuple
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# ---------------------------------------------------------------------------
# Stage 1 — CD-DPA full-image + SAHI tile detector
# ---------------------------------------------------------------------------

class CDDPADetector(nn.Module):
    """
    Wraps FasterRCNN_CDDPA with:
        - predict()     → standard detection dict (used on full image + tiles)

    Note: extract_p2() is kept for backward compatibility but is no longer
    needed in the 4-stage pipeline (ReconstructionHead removed from Stage 1).
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        num_classes: int = 11,
        device: str = 'cuda',
        score_thresh: float = 0.05,
    ):
        super().__init__()
        self.device = device
        self.score_thresh = score_thresh
        self.num_classes = num_classes

        from models.cddpa_model import FasterRCNN_CDDPA

        self.model = FasterRCNN_CDDPA(
            num_classes=num_classes,
            enhance_levels=['0', '1', '2'],
            use_checkpoint=False,   # inference — no need
            pretrained=(checkpoint_path is None),
        )

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=device)
            state = ckpt.get('model_state_dict', ckpt)
            self.model.load_state_dict(state)
            print(f"  CDDPADetector loaded from {checkpoint_path}")
        else:
            print("  CDDPADetector initialised (no checkpoint — random CD-DPA heads)")

        self.model.to(device)
        self.model.eval()

    # ── Preprocessing ─────────────────────────────────────────────────

    def _preprocess_image(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = torch.from_numpy(image).permute(2, 0, 1)
            elif image.ndim == 2:
                image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        return image

    # ── Inference ─────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, image: Union[np.ndarray, torch.Tensor],
                score_thresh: float = None) -> Dict[str, torch.Tensor]:
        if score_thresh is None:
            score_thresh = self.score_thresh
        img = self._preprocess_image(image).to(self.device)
        self.model.eval()
        preds = self.model([img])[0]
        keep = preds['scores'] >= score_thresh
        return {
            'boxes':  preds['boxes'][keep].cpu(),
            'scores': preds['scores'][keep].cpu(),
            'labels': preds['labels'][keep].cpu(),
        }

    # ── P2 extraction (backward-compat, not used in 4-stage pipeline) ──

    @torch.no_grad()
    def extract_p2(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract P2 FPN features (with CD-DPA enhancement) from the image.

        Note: This method is kept for backward compatibility. The 4-stage
        pipeline no longer uses it (ReconstructionHead removed from Stage 1).
        SR-TOD has its own backbone for P2 extraction in Stage 3.

        Args:
            image: (3, H, W) float [0,1] or numpy (H,W,3).
        Returns:
            P2 tensor (1, 256, H/4, W/4) on self.device.
        """
        img = self._preprocess_image(image).to(self.device)
        base = self.model.base_model

        # Use model.transform to get properly padded input
        image_list, _ = base.transform([img], None)

        # Run backbone with CD-DPA enhancement (monkey-patch as in forward)
        original_fwd = base.backbone.forward

        def enhanced_fwd(x):
            feats = original_fwd(x)
            for lvl in self.model.enhance_levels:
                if lvl in feats:
                    feats[lvl] = self.model.enhancers[lvl](feats[lvl])
            return feats

        base.backbone.forward = enhanced_fwd
        features = base.backbone(image_list.tensors)
        base.backbone.forward = original_fwd

        return features['0']  # P2

    def __repr__(self):
        return f"CDDPADetector(num_classes={self.num_classes}, device={self.device})"


# ---------------------------------------------------------------------------
# Stage 3 — SR-TOD tile detector (runs on weak tiles only)
# ---------------------------------------------------------------------------

class SRTODTileDetector(nn.Module):
    """
    Wraps FasterRCNN_SRTOD with a predict() interface matching BaseDetector.
    Used by SAHIInferenceRunner for Stage 3 weak-tile detection.
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        num_classes: int = 11,
        device: str = 'cuda',
        score_thresh: float = 0.05,
    ):
        super().__init__()
        self.device = device
        self.score_thresh = score_thresh
        self.num_classes = num_classes

        from models.srtod_model import FasterRCNN_SRTOD

        self.model = FasterRCNN_SRTOD(
            num_classes=num_classes,
            learnable_thresh=0.0156862,
            pretrained_backbone=(checkpoint_path is None),
        )

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=device)
            state = ckpt.get('model_state_dict', ckpt)
            self.model.load_state_dict(state)
            print(f"  SRTODTileDetector loaded from {checkpoint_path}")
        else:
            print("  SRTODTileDetector initialised (no checkpoint)")

        self.model.to(device)
        self.model.eval()

    def _preprocess_image(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = torch.from_numpy(image).permute(2, 0, 1)
            elif image.ndim == 2:
                image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        return image

    @torch.no_grad()
    def predict(self, image: Union[np.ndarray, torch.Tensor],
                score_thresh: float = None) -> Dict[str, torch.Tensor]:
        if score_thresh is None:
            score_thresh = self.score_thresh
        img = self._preprocess_image(image).to(self.device)
        self.model.eval()
        preds = self.model([img])[0]
        keep = preds['scores'] >= score_thresh
        return {
            'boxes':  preds['boxes'][keep].cpu(),
            'scores': preds['scores'][keep].cpu(),
            'labels': preds['labels'][keep].cpu(),
        }

    def __repr__(self):
        return f"SRTODTileDetector(num_classes={self.num_classes}, device={self.device})"


# ---------------------------------------------------------------------------
# Backward-compatible — plain Faster R-CNN wrapper
# ---------------------------------------------------------------------------

class BaseDetector(nn.Module):
    """Wrapper for plain Faster R-CNN (baseline). Kept for backward
    compatibility with existing eval scripts."""

    def __init__(self, checkpoint_path=None, num_classes=11,
                 device='cuda', score_thresh=0.05):
        super().__init__()
        self.device = device
        self.score_thresh = score_thresh
        self.num_classes = num_classes

        if checkpoint_path:
            from models.baseline import get_baseline_model
            self.model = get_baseline_model(num_classes=num_classes - 1, pretrained=False)
            ckpt = torch.load(checkpoint_path, map_location=device)
            state = ckpt.get('model_state_dict', ckpt)
            self.model.load_state_dict(state)
        else:
            self.model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        self.model.to(device)
        self.model.eval()

    def _preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = torch.from_numpy(image).permute(2, 0, 1)
            elif image.ndim == 2:
                image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        return image

    @torch.no_grad()
    def predict(self, image, score_thresh=None):
        if score_thresh is None:
            score_thresh = self.score_thresh
        img = self._preprocess_image(image).to(self.device)
        self.model.eval()
        preds = self.model([img])[0]
        keep = preds['scores'] >= score_thresh
        return {
            'boxes':  preds['boxes'][keep].cpu(),
            'scores': preds['scores'][keep].cpu(),
            'labels': preds['labels'][keep].cpu(),
        }

    @torch.no_grad()
    def extract_p2(self, tile: torch.Tensor) -> torch.Tensor:
        """Extract P2 FPN features from a tile (no CD-DPA)."""
        img = self._preprocess_image(tile).to(self.device)
        image_list, _ = self.model.transform([img], None)
        features = self.model.backbone(image_list.tensors)
        return features['0']

    def __repr__(self):
        return f"BaseDetector(num_classes={self.num_classes}, device={self.device})"

"""
SAHI Pipeline Configuration — 4-Stage Confidence-Guided Architecture
=====================================================================
Stage 1: CD-DPA + SAHI (full image + all tiles) → D_base + per-tile confidence
Stage 2: Weak tile selection (bottom-K by confidence score)
Stage 3: SR-TOD (RH + DGFE + detection) on K weak tiles → D_sr
Stage 4: Final fusion (D_base + D_sr → class-wise NMS → D_final)
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class SAHIPipelineConfig:
    """Configuration for the 4-stage confidence-guided SAHI pipeline."""

    # ===== Stage 1: CD-DPA + SAHI =====
    cddpa_checkpoint: str = "results/outputs_cddpa/best_model_cddpa.pth"

    # ===== Stage 2: Weak tile selection =====
    K: int = 10                  # Number of weak tiles sent to SR-TOD
    min_expected_dets: int = 2   # Expected detections per tile (density normalization)

    # ===== Stage 3: SR-TOD on weak tiles =====
    srtod_checkpoint: str = "results/outputs_srtod/best_model_srtod.pth"

    # ===== Tiling Parameters =====
    tile_size: Tuple[int, int] = (320, 320)
    overlap_width_ratio: float = 0.25
    overlap_height_ratio: float = 0.25

    # ===== SAHI Postprocess Parameters =====
    postprocess_type: str = 'GREEDYNMM'
    postprocess_match_metric: str = 'IOS'
    postprocess_match_threshold: float = 0.5

    # ===== NMS Parameters =====
    iou_tile_merge: float = 0.6
    iou_final: float = 0.65

    # ===== Detector Parameters =====
    num_classes: int = 11        # VisDrone: 10 classes + background
    detection_score_thresh: float = 0.4
    base_score_thresh: float = 0.05  # Min score for low-confidence detections

    # ===== Device =====
    device: str = 'cuda'
    seed: int = 42

    # ===== Debug =====
    debug: bool = False
    debug_dir: str = 'results/debug'

    # ===== Backward-compat aliases (read-only) =====
    @property
    def detector_checkpoint(self):
        return self.cddpa_checkpoint

    @property
    def topN_tiles(self):
        return self.K

    @property
    def overlap(self):
        return self.overlap_width_ratio

    def __post_init__(self):
        """Validate configuration."""
        assert self.K > 0, "K must be positive"
        assert self.min_expected_dets > 0, "min_expected_dets must be positive"
        assert 0 <= self.iou_tile_merge <= 1
        assert 0 <= self.iou_final <= 1
        assert 0 < self.overlap_width_ratio < 1
        assert 0 < self.overlap_height_ratio < 1
        assert self.postprocess_type in ['GREEDYNMM', 'NMS']
        assert self.postprocess_match_metric in ['IOS', 'IOU']

    def to_dict(self):
        return {
            'K': self.K,
            'min_expected_dets': self.min_expected_dets,
            'tile_size': self.tile_size,
            'overlap_width_ratio': self.overlap_width_ratio,
            'overlap_height_ratio': self.overlap_height_ratio,
            'postprocess_type': self.postprocess_type,
            'postprocess_match_metric': self.postprocess_match_metric,
            'postprocess_match_threshold': self.postprocess_match_threshold,
            'iou_tile_merge': self.iou_tile_merge,
            'iou_final': self.iou_final,
            'cddpa_checkpoint': self.cddpa_checkpoint,
            'srtod_checkpoint': self.srtod_checkpoint,
            'num_classes': self.num_classes,
            'detection_score_thresh': self.detection_score_thresh,
            'base_score_thresh': self.base_score_thresh,
            'device': self.device,
            'seed': self.seed,
            'debug': self.debug,
            'debug_dir': self.debug_dir,
        }

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    def __repr__(self):
        return (
            f"SAHIPipelineConfig(\n"
            f"  Weak tile budget: K={self.K}, min_expected_dets={self.min_expected_dets}\n"
            f"  Tiling: size={self.tile_size}, overlap=({self.overlap_width_ratio:.2f}, {self.overlap_height_ratio:.2f})\n"
            f"  Postprocess: {self.postprocess_type}/{self.postprocess_match_metric}, thresh={self.postprocess_match_threshold}\n"
            f"  NMS: tile_merge={self.iou_tile_merge}, final={self.iou_final}\n"
            f"  Score Threshold: {self.detection_score_thresh}\n"
            f"  CD-DPA ckpt: {self.cddpa_checkpoint}\n"
            f"  SR-TOD ckpt: {self.srtod_checkpoint}\n"
            f"  Device: {self.device}\n"
            f")"
        )


# Preset configurations
PRESETS = {
    'fast': SAHIPipelineConfig(
        K=5,
        tile_size=(256, 256),
        overlap_width_ratio=0.2,
        overlap_height_ratio=0.2,
        min_expected_dets=2,
        detection_score_thresh=0.5,
    ),

    'balanced': SAHIPipelineConfig(
        K=10,
        tile_size=(320, 320),
        overlap_width_ratio=0.25,
        overlap_height_ratio=0.25,
        min_expected_dets=2,
        detection_score_thresh=0.4,
    ),

    'accurate': SAHIPipelineConfig(
        K=20,
        tile_size=(512, 512),
        overlap_width_ratio=0.3,
        overlap_height_ratio=0.3,
        min_expected_dets=3,
        detection_score_thresh=0.3,
    ),
}


def get_preset_config(preset: str = 'balanced') -> SAHIPipelineConfig:
    """Get preset configuration."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(PRESETS.keys())}")
    return PRESETS[preset]


if __name__ == "__main__":
    config = SAHIPipelineConfig()
    print(config)
    print("\n" + "="*50 + "\n")
    for name, preset in PRESETS.items():
        print(f"{name.upper()} preset:")
        print(preset)
        print()
    print("Config tests passed!")

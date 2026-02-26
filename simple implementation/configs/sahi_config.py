"""
SAHI Pipeline Configuration
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SAHIPipelineConfig:
    """
    Configuration for uncertainty-triggered SAHI pipeline
    """
    
    # ===== Uncertainty Parameters =====
    theta: float = 0.5  # Trigger threshold [0,1]
    base_score_thresh: float = 0.3  # Min score for uncertainty computation
    
    # ===== Tiling Parameters =====
    tile_size: Tuple[int, int] = (384, 384)  # (width, height) of tiles (larger to reduce splits)
    overlap_width_ratio: float = 0.25  # Overlap ratio for width (0.2-0.3 recommended)
    overlap_height_ratio: float = 0.25  # Overlap ratio for height (0.2-0.3 recommended)
    topN_tiles: int = 16  # Number of tiles to process
    
    # ===== SAHI Postprocess Parameters =====
    postprocess_type: str = 'GREEDYNMM'  # 'GREEDYNMM' or 'NMS' (GREEDYNMM recommended)
    postprocess_match_metric: str = 'IOS'  # 'IOS' or 'IOU' (IOS better for slices)
    postprocess_match_threshold: float = 0.5  # Merge threshold (increase to 0.6 if duplicates)
    
    # ===== NMS Parameters =====
    iou_tile_merge: float = 0.6  # IoU for merging tile detections (strict to avoid duplicates)
    iou_final: float = 0.65  # IoU for final fusion NMS (strict global NMS)
    
    # ===== Model Paths =====
    detector_checkpoint: str = "results/outputs/best_model.pth"  # Trained baseline Faster R-CNN (38.02% mAP)
    reconstructor_checkpoint: str = None  # Path to reconstructor checkpoint
    
    # ===== Detector Parameters =====
    num_classes: int = 11  # VisDrone: 10 classes + background
    detection_score_thresh: float = 0.4  # Min score for final detections (0.35-0.5 for SAHI)
    
    # ===== Device =====
    device: str = 'cuda'
    seed: int = 42
    
    # ===== Debug =====
    debug: bool = False  # Save debug visualizations
    debug_dir: str = 'results/debug'
    
    def __post_init__(self):
        """Validate configuration"""
        assert 0 <= self.theta <= 1, "theta must be in [0, 1]"
        assert 0 <= self.base_score_thresh <= 1, "base_score_thresh must be in [0, 1]"
        assert self.topN_tiles > 0, "topN_tiles must be positive"
        assert 0 <= self.iou_tile_merge <= 1, "iou_tile_merge must be in [0, 1]"
        assert 0 <= self.iou_final <= 1, "iou_final must be in [0, 1]"
        assert 0 <= self.overlap_width_ratio < 1, "overlap_width_ratio must be in [0, 1)"
        assert 0 <= self.overlap_height_ratio < 1, "overlap_height_ratio must be in [0, 1)"
        assert self.postprocess_type in ['GREEDYNMM', 'NMS'], "postprocess_type must be GREEDYNMM or NMS"
        assert self.postprocess_match_metric in ['IOS', 'IOU'], "postprocess_match_metric must be IOS or IOU"
        assert 0 <= self.postprocess_match_threshold <= 1, "postprocess_match_threshold must be in [0, 1]"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'theta': self.theta,
            'base_score_thresh': self.base_score_thresh,
            'tile_size': self.tile_size,
            'overlap_width_ratio': self.overlap_width_ratio,
            'overlap_height_ratio': self.overlap_height_ratio,
            'topN_tiles': self.topN_tiles,
            'postprocess_type': self.postprocess_type,
            'postprocess_match_metric': self.postprocess_match_metric,
            'postprocess_match_threshold': self.postprocess_match_threshold,
            'iou_tile_merge': self.iou_tile_merge,
            'iou_final': self.iou_final,
            'detector_checkpoint': self.detector_checkpoint,
            'reconstructor_checkpoint': self.reconstructor_checkpoint,
            'num_classes': self.num_classes,
            'detection_score_thresh': self.detection_score_thresh,
            'device': self.device,
            'seed': self.seed,
            'debug': self.debug,
            'debug_dir': self.debug_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary"""
        return cls(**config_dict)
    
    def __repr__(self):
        return (
            f"SAHIPipelineConfig(\n"
            f"  Uncertainty: theta={self.theta}, base_thresh={self.base_score_thresh}\n"
            f"  Tiling: size={self.tile_size}, overlap=({self.overlap_width_ratio:.2f}, {self.overlap_height_ratio:.2f}), topN={self.topN_tiles}\n"
            f"  Postprocess: {self.postprocess_type}/{self.postprocess_match_metric}, thresh={self.postprocess_match_threshold}\n"
            f"  NMS: tile_merge={self.iou_tile_merge}, final={self.iou_final}\n"
            f"  Score Threshold: {self.detection_score_thresh}\n"
            f"  Device: {self.device}\n"
            f")"
        )


# Preset configurations
PRESETS = {
    'fast': SAHIPipelineConfig(
        theta=0.7,  # Only trigger on very uncertain cases
        topN_tiles=8,  # Fewer tiles
        tile_size=(320, 320),  # Larger tiles
        overlap_width_ratio=0.25,  # 25% overlap
        overlap_height_ratio=0.25,
        detection_score_thresh=0.45,  # Higher threshold
        iou_tile_merge=0.6,
        iou_final=0.65
    ),
    
    'balanced': SAHIPipelineConfig(
        theta=0.5,  # Default balance
        topN_tiles=16,
        tile_size=(384, 384),  # Larger tiles
        overlap_width_ratio=0.25,
        overlap_height_ratio=0.25,
        detection_score_thresh=0.4,
        iou_tile_merge=0.6,
        iou_final=0.65
    ),
    
    'accurate': SAHIPipelineConfig(
        theta=0.3,  # Aggressive SAHI
        topN_tiles=32,  # More tiles
        tile_size=(448, 448),  # Even larger tiles
        overlap_width_ratio=0.3,  # More overlap
        overlap_height_ratio=0.3,
        detection_score_thresh=0.35,  # Lower to catch more
        iou_tile_merge=0.6,
        iou_final=0.65
    )
}


def get_preset_config(preset: str = 'balanced') -> SAHIPipelineConfig:
    """
    Get preset configuration
    
    Args:
        preset: One of 'fast', 'balanced', 'accurate'
        
    Returns:
        SAHIPipelineConfig
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(PRESETS.keys())}")
    return PRESETS[preset]


if __name__ == "__main__":
    # Test config
    config = SAHIPipelineConfig()
    print(config)
    print("\n" + "="*50 + "\n")
    
    # Test presets
    for name, preset in PRESETS.items():
        print(f"{name.upper()} preset:")
        print(preset)
        print()
    
    print("✓ Config tests passed!")

"""
Tests for SAHI Pipeline Components
"""
import torch
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sahi_pipeline.uncertainty import UncertaintyEstimator
from models.sahi_pipeline.residual import ResidualMapComputer
from models.sahi_pipeline.tiles import TileSelector
from models.sahi_pipeline.fuse import DetectionFusion
from configs.sahi_config import SAHIPipelineConfig


class TestUncertaintyEstimator:
    """Test uncertainty estimation"""
    
    def setup_method(self):
        self.estimator = UncertaintyEstimator(base_score_thresh=0.3)
    
    def test_empty_detections(self):
        """Empty detections should give maximum uncertainty"""
        dets = {
            'boxes': torch.empty(0, 4),
            'scores': torch.empty(0),
            'labels': torch.empty(0, dtype=torch.long)
        }
        U_t = self.estimator.compute_uncertainty(dets)
        assert U_t == 1.0, f"Expected U_t=1.0 for empty detections, got {U_t}"
    
    def test_high_confidence(self):
        """High confidence should give low uncertainty"""
        dets = {
            'boxes': torch.tensor([[10, 10, 50, 50]]),
            'scores': torch.tensor([0.95]),
            'labels': torch.tensor([1])
        }
        U_t = self.estimator.compute_uncertainty(dets)
        assert U_t < 0.3, f"Expected low uncertainty for high confidence, got {U_t}"
    
    def test_low_confidence(self):
        """Low confidence should give high uncertainty"""
        dets = {
            'boxes': torch.tensor([[10, 10, 50, 50]]),
            'scores': torch.tensor([0.3]),
            'labels': torch.tensor([1])
        }
        U_t = self.estimator.compute_uncertainty(dets)
        assert U_t > 0.5, f"Expected high uncertainty for low confidence, got {U_t}"
    
    def test_mixed_confidence(self):
        """Mixed confidence should give medium uncertainty"""
        dets = {
            'boxes': torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
            'scores': torch.tensor([0.9, 0.4]),
            'labels': torch.tensor([1, 2])
        }
        U_t = self.estimator.compute_uncertainty(dets)
        assert 0.2 < U_t < 0.7, f"Expected medium uncertainty, got {U_t}"
    
    def test_should_trigger(self):
        """Test SAHI triggering logic"""
        # High confidence - should not trigger
        dets_high = {
            'boxes': torch.tensor([[10, 10, 50, 50]]),
            'scores': torch.tensor([0.95]),
            'labels': torch.tensor([1])
        }
        assert not self.estimator.should_trigger_sahi(dets_high, theta=0.5)
        
        # Low confidence - should trigger
        dets_low = {
            'boxes': torch.tensor([[10, 10, 50, 50]]),
            'scores': torch.tensor([0.3]),
            'labels': torch.tensor([1])
        }
        assert self.estimator.should_trigger_sahi(dets_low, theta=0.5)


class TestResidualMapComputer:
    """Test residual map computation"""
    
    def setup_method(self):
        self.computer = ResidualMapComputer()
    
    def test_identical_images(self):
        """Identical images should give zero residual"""
        img = torch.rand(3, 100, 100)
        residual = self.computer.compute_residual_map(img, img, normalize=False)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)
    
    def test_normalization(self):
        """Normalized residual should be in [0, 1]"""
        img1 = torch.rand(3, 100, 100)
        img2 = torch.rand(3, 100, 100)
        residual = self.computer.compute_residual_map(img1, img2, normalize=True)
        assert residual.min() >= 0 and residual.max() <= 1
    
    def test_shape(self):
        """Output shape should be (H, W)"""
        img1 = torch.rand(3, 100, 100)
        img2 = torch.rand(3, 100, 100)
        residual = self.computer.compute_residual_map(img1, img2)
        assert residual.shape == (100, 100)
    
    def test_batch(self):
        """Should handle batch of images"""
        img1 = torch.rand(2, 3, 100, 100)
        img2 = torch.rand(2, 3, 100, 100)
        residual = self.computer.compute_residual_map(img1, img2)
        assert residual.shape == (2, 100, 100)


class TestTileSelector:
    """Test tile selection"""
    
    def setup_method(self):
        self.selector = TileSelector(tile_size=(160, 160), stride=(80, 80))
    
    def test_tile_generation(self):
        """Should generate tiles covering the image"""
        image_size = (640, 640)
        tiles = self.selector.generate_tiles(image_size)
        assert len(tiles) > 0, "No tiles generated"
        
        # Check all tiles are valid
        for (x0, y0, x1, y1) in tiles:
            assert 0 <= x0 < x1 <= 640
            assert 0 <= y0 < y1 <= 640
    
    def test_tile_selection_bright_region(self):
        """Selected tiles should overlap with high-residual region"""
        # Create residual map with bright patch
        residual = torch.zeros(640, 640)
        residual[200:400, 200:400] = 1.0
        
        # Select top tiles
        tiles = self.selector.select_tiles(residual, topN=5, image_size=(640, 640))
        
        # Check that at least some tiles overlap with bright region
        overlaps = []
        for (x0, y0, x1, y1) in tiles:
            overlap = (x0 < 400 and x1 > 200 and y0 < 400 and y1 > 200)
            overlaps.append(overlap)
        
        assert sum(overlaps) > 0, "No tiles selected in high-residual region"
    
    def test_tile_scoring(self):
        """Tiles with higher residual should score higher"""
        residual = torch.zeros(640, 640)
        residual[0:160, 0:160] = 0.5
        residual[200:360, 200:360] = 1.0
        
        tile1 = (0, 0, 160, 160)
        tile2 = (200, 200, 360, 360)
        
        score1 = self.selector.score_tile(residual, tile1)
        score2 = self.selector.score_tile(residual, tile2)
        
        assert score2 > score1, "Higher residual tile should score higher"


class TestDetectionFusion:
    """Test detection fusion"""
    
    def setup_method(self):
        self.fusion = DetectionFusion(iou_thresh=0.5)
    
    def test_empty_detections(self):
        """Should handle empty inputs"""
        empty = {
            'boxes': torch.empty(0, 4),
            'scores': torch.empty(0),
            'labels': torch.empty(0, dtype=torch.long)
        }
        result = self.fusion.fuse(empty, empty)
        assert len(result['boxes']) == 0
    
    def test_overlapping_detections(self):
        """Should remove overlapping detections with NMS"""
        base = {
            'boxes': torch.tensor([[10, 10, 50, 50]]),
            'scores': torch.tensor([0.9]),
            'labels': torch.tensor([1])
        }
        sahi = {
            'boxes': torch.tensor([[12, 12, 52, 52]]),  # Overlaps with base
            'scores': torch.tensor([0.85]),
            'labels': torch.tensor([1])
        }
        
        fused = self.fusion.fuse(base, sahi)
        # Should keep only one (higher score)
        assert len(fused['boxes']) == 1
        assert fused['scores'][0] == 0.9
    
    def test_non_overlapping(self):
        """Should keep non-overlapping detections"""
        base = {
            'boxes': torch.tensor([[10, 10, 50, 50]]),
            'scores': torch.tensor([0.9]),
            'labels': torch.tensor([1])
        }
        sahi = {
            'boxes': torch.tensor([[200, 200, 250, 250]]),
            'scores': torch.tensor([0.85]),
            'labels': torch.tensor([1])
        }
        
        fused = self.fusion.fuse(base, sahi)
        assert len(fused['boxes']) == 2


class TestConfig:
    """Test configuration"""
    
    def test_default_config(self):
        """Default config should be valid"""
        config = SAHIPipelineConfig()
        assert 0 <= config.theta <= 1
        assert config.topN_tiles > 0
    
    def test_presets(self):
        """All presets should be valid"""
        from configs.sahi_config import get_preset_config
        
        for preset in ['fast', 'balanced', 'accurate']:
            config = get_preset_config(preset)
            assert 0 <= config.theta <= 1
            assert config.topN_tiles > 0
    
    def test_invalid_theta(self):
        """Invalid theta should raise error"""
        with pytest.raises(AssertionError):
            SAHIPipelineConfig(theta=1.5)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])

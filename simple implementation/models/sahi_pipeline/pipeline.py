"""
Main SAHI Pipeline
Orchestrates all components for uncertainty-triggered inference
"""
import torch
import numpy as np
import time
from typing import Dict, Tuple, Union
from pathlib import Path

from .detector_wrapper import BaseDetector
from .uncertainty import UncertaintyEstimator
from .residual import ResidualMapComputer
from .tiles import TileSelector
from .sahi_runner import SAHIInferenceRunner
from .fuse import DetectionFusion


class SAHIPipeline:
    """
    Complete uncertainty-triggered SAHI pipeline
    
    Process:
        1. Run base detector on full frame
        2. Compute uncertainty from base detections
        3. If uncertain: run SAHI on residual-guided tiles
        4. Fuse base and SAHI detections
    
    Features:
        - Adaptive: Automatically adjusts processing based on uncertainty
        - Efficient: Only runs SAHI when needed
        - Smart: Selects tiles based on reconstruction residuals
    """
    
    def __init__(self, config):
        """
        Initialize pipeline
        
        Args:
            config: SAHIPipelineConfig instance
        """
        self.config = config
        self.device = config.device
        
        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize components
        print("Initializing SAHI Pipeline...")
        
        # 1. Base detector
        self.detector = BaseDetector(
            checkpoint_path=config.detector_checkpoint,
            num_classes=config.num_classes,
            device=config.device,
            score_thresh=config.detection_score_thresh
        )
        print("✓ Detector loaded")
        
        # 2. Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            base_score_thresh=config.base_score_thresh
        )
        print("✓ Uncertainty estimator initialized")
        
        # 3. Reconstructor
        if config.reconstructor_checkpoint:
            from models.enhancements.lightweight_reconstructor import LightweightReconstructor
            self.reconstructor = LightweightReconstructor()
            checkpoint = torch.load(config.reconstructor_checkpoint, map_location=config.device)
            self.reconstructor.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
            self.reconstructor.to(config.device)
            self.reconstructor.eval()
            print("✓ Reconstructor loaded")
        else:
            self.reconstructor = None
            print("⚠ No reconstructor - SAHI will use random tile selection")
        
        # 4. Residual computer
        self.residual_computer = ResidualMapComputer()
        
        # 5. Tile selector
        self.tile_selector = TileSelector(
            tile_size=config.tile_size,
            overlap_width_ratio=config.overlap_width_ratio,
            overlap_height_ratio=config.overlap_height_ratio
        )
        print("✓ Tile selector initialized")
        
        # 6. SAHI runner
        self.sahi_runner = SAHIInferenceRunner(
            detector=self.detector,
            merge_iou_thresh=config.iou_tile_merge,
            postprocess_type=config.postprocess_type,
            postprocess_match_metric=config.postprocess_match_metric,
            postprocess_match_threshold=config.postprocess_match_threshold
        )
        print("✓ SAHI runner initialized")
        
        # 7. Fusion
        self.fusion = DetectionFusion(
            iou_thresh=config.iou_final
        )
        print("✓ Fusion module initialized")
        
        # Debug mode
        if config.debug:
            Path(config.debug_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n✓ Pipeline ready! ({config.device})")
        print(f"  Uncertainty threshold: {config.theta}")
        print(f"  Tile processing: Top-{config.topN_tiles} of {config.tile_size}")
    
    @torch.no_grad()
    def process_image(
        self,
        image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Process single image
        
        Args:
            image: RGB image (H,W,3) numpy or (3,H,W) tensor
            
        Returns:
            detections: Dict with 'boxes', 'scores', 'labels'
            metadata: Dict with processing info
        """
        start_time = time.time()
        
        # Preprocess image
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")
        else:
            image_tensor = image
        
        image_tensor = image_tensor.to(self.device)
        _, H, W = image_tensor.shape
        
        # Timing breakdown
        timings = {}
        
        # 1. Base detection
        t0 = time.time()
        base_dets = self.detector.predict(image_tensor)
        timings['base_detection'] = (time.time() - t0) * 1000
        
        # 2. Uncertainty computation
        t0 = time.time()
        U_t = self.uncertainty_estimator.compute_uncertainty(base_dets)
        timings['uncertainty'] = (time.time() - t0) * 1000
        
        # Metadata
        metadata = {
            'U_t': U_t,
            'triggered': False,
            'num_base_dets': len(base_dets['boxes']),
            'num_sahi_dets': 0,
            'num_final_dets': 0,
            'num_tiles': 0,
            'timings': timings
        }
        
        # 3. Decision: Trigger SAHI?
        if U_t < self.config.theta:
            # High confidence - return base detections
            metadata['num_final_dets'] = len(base_dets['boxes'])
            metadata['latency_ms'] = (time.time() - start_time) * 1000
            return base_dets, metadata
        
        # 4. Trigger SAHI
        metadata['triggered'] = True
        
        # 5. Reconstruction (if available)
        if self.reconstructor is not None:
            t0 = time.time()
            reconstructed = self.reconstructor(image_tensor.unsqueeze(0))[0]
            timings['reconstruction'] = (time.time() - t0) * 1000
            
            # 6. Residual map
            t0 = time.time()
            residual_map = self.residual_computer.compute_residual_map(
                image_tensor, reconstructed, normalize=True
            )
            timings['residual'] = (time.time() - t0) * 1000
            
            # 7. Tile selection (residual-guided)
            t0 = time.time()
            selected_tiles = self.tile_selector.select_tiles(
                residual_map, 
                topN=self.config.topN_tiles,
                image_size=(W, H)
            )
            timings['tile_selection'] = (time.time() - t0) * 1000
        else:
            # Fallback: Random tile selection
            all_tiles = self.tile_selector.generate_tiles((W, H))
            selected_tiles = np.random.choice(
                len(all_tiles), 
                min(self.config.topN_tiles, len(all_tiles)),
                replace=False
            )
            selected_tiles = [all_tiles[i] for i in selected_tiles]
            residual_map = None
        
        metadata['num_tiles'] = len(selected_tiles)
        
        # 8. SAHI inference
        t0 = time.time()
        sahi_dets = self.sahi_runner.run_on_tiles(image_tensor, selected_tiles)
        timings['sahi_inference'] = (time.time() - t0) * 1000
        metadata['num_sahi_dets'] = len(sahi_dets['boxes'])
        
        # 9. Fusion
        t0 = time.time()
        final_dets = self.fusion.fuse(base_dets, sahi_dets)
        timings['fusion'] = (time.time() - t0) * 1000
        metadata['num_final_dets'] = len(final_dets['boxes'])
        
        # Total latency
        metadata['latency_ms'] = (time.time() - start_time) * 1000
        
        # Debug visualization
        if self.config.debug:
            self._save_debug(image_tensor, base_dets, sahi_dets, final_dets, 
                           residual_map, selected_tiles, metadata)
        
        return final_dets, metadata
    
    def _save_debug(self, image, base_dets, sahi_dets, final_dets, 
                    residual_map, tiles, metadata):
        """Save debug visualizations"""
        import cv2
        import matplotlib.pyplot as plt
        
        # Convert to numpy
        img_np = (image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # 1. Base detections
        ax = axes[0, 0]
        ax.imshow(img_np)
        for box in base_dets['boxes']:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(rect)
        ax.set_title(f"Base Detections (n={len(base_dets['boxes'])})")
        ax.axis('off')
        
        # 2. Residual map + tiles
        ax = axes[0, 1]
        if residual_map is not None:
            ax.imshow(residual_map.cpu().numpy(), cmap='hot')
            for (x0, y0, x1, y1) in tiles:
                rect = plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                    fill=False, edgecolor='cyan', linewidth=2)
                ax.add_patch(rect)
            ax.set_title(f"Residual Map + Selected Tiles (n={len(tiles)})")
        else:
            ax.imshow(img_np)
            ax.set_title("No Reconstructor")
        ax.axis('off')
        
        # 3. SAHI detections
        ax = axes[1, 0]
        ax.imshow(img_np)
        for box in sahi_dets['boxes']:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        ax.set_title(f"SAHI Detections (n={len(sahi_dets['boxes'])})")
        ax.axis('off')
        
        # 4. Final fused detections
        ax = axes[1, 1]
        ax.imshow(img_np)
        for box in final_dets['boxes']:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(rect)
        ax.set_title(f"Final Detections (n={len(final_dets['boxes'])})")
        ax.axis('off')
        
        plt.suptitle(f"U_t={metadata['U_t']:.3f}, Latency={metadata['latency_ms']:.1f}ms")
        plt.tight_layout()
        
        # Save
        debug_path = Path(self.config.debug_dir) / f"debug_{int(time.time()*1000)}.png"
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def __repr__(self):
        return f"SAHIPipeline(theta={self.config.theta}, device={self.device})"


if __name__ == "__main__":
    print("SAHIPipeline requires config - test in integration tests")
    print("✓ Pipeline module created successfully!")

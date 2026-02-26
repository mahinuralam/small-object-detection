"""
Base Detector Wrapper for Fast R-CNN
Provides unified interface for detection with consistent output format
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Union, Tuple
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class BaseDetector(nn.Module):
    """
    Wrapper for Faster R-CNN detector
    
    Provides consistent interface:
    - Input: RGB image (numpy array or tensor)
    - Output: Dict with 'boxes', 'scores', 'labels' in COCO format [x1,y1,x2,y2]
    """
    
    def __init__(
        self,
        checkpoint_path: str = None,
        num_classes: int = 11,  # VisDrone: 10 classes + background
        device: str = 'cuda',
        score_thresh: float = 0.05  # Low threshold for base detection
    ):
        """
        Initialize detector
        
        Args:
            checkpoint_path: Path to model checkpoint (if None, use pretrained COCO)
            num_classes: Number of classes including background
            device: Device to run on
            score_thresh: Minimum score threshold for detections
        """
        super().__init__()
        
        self.device = device
        self.score_thresh = score_thresh
        self.num_classes = num_classes
        
        # Load model
        if checkpoint_path:
            # Load custom checkpoint
            self.model = self._load_from_checkpoint(checkpoint_path, num_classes)
        else:
            # Use pretrained COCO model
            self.model = fasterrcnn_resnet50_fpn(
                weights='DEFAULT'
            )
        
        self.model.to(device)
        self.model.eval()
    
    def _load_from_checkpoint(self, checkpoint_path: str, num_classes: int):
        """Load model from checkpoint"""
        from models.baseline import get_baseline_model
        
        model = get_baseline_model(
            num_classes=num_classes - 1,  # Exclude background
            pretrained=False
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def _preprocess_image(
        self, 
        image: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess image to model input format
        
        Args:
            image: RGB image as numpy array (H,W,3) uint8 or tensor (3,H,W) float32
            
        Returns:
            Preprocessed tensor (3, H, W) float32 in [0,1]
        """
        if isinstance(image, np.ndarray):
            # Convert numpy (H,W,3) to tensor (3,H,W)
            if image.ndim == 3 and image.shape[2] == 3:
                image = torch.from_numpy(image).permute(2, 0, 1)
            elif image.ndim == 2:
                # Grayscale - convert to RGB
                image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")
            
            # Normalize to [0, 1]
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
        
        # Ensure float32 and proper range
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        
        return image
    
    @torch.no_grad()
    def predict(
        self, 
        image: Union[np.ndarray, torch.Tensor],
        score_thresh: float = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run detection on image
        
        Args:
            image: Input image
            score_thresh: Override score threshold (default: use self.score_thresh)
            
        Returns:
            Dict with:
                - 'boxes': (N, 4) tensor [x1, y1, x2, y2]
                - 'scores': (N,) tensor
                - 'labels': (N,) tensor (0-indexed, excluding background)
        """
        if score_thresh is None:
            score_thresh = self.score_thresh
        
        # Preprocess
        image_tensor = self._preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Run model
        self.model.eval()
        predictions = self.model([image_tensor])[0]
        
        # Filter by score threshold
        keep = predictions['scores'] >= score_thresh
        
        result = {
            'boxes': predictions['boxes'][keep].cpu(),
            'scores': predictions['scores'][keep].cpu(),
            'labels': predictions['labels'][keep].cpu()
        }
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[np.ndarray, torch.Tensor]],
        score_thresh: float = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run detection on batch of images
        
        Args:
            images: List of images
            score_thresh: Score threshold
            
        Returns:
            List of detection dicts
        """
        if score_thresh is None:
            score_thresh = self.score_thresh
        
        # Preprocess all images
        image_tensors = [self._preprocess_image(img).to(self.device) for img in images]
        
        # Run model
        self.model.eval()
        predictions = self.model(image_tensors)
        
        # Filter and format results
        results = []
        for pred in predictions:
            keep = pred['scores'] >= score_thresh
            results.append({
                'boxes': pred['boxes'][keep].cpu(),
                'scores': pred['scores'][keep].cpu(),
                'labels': pred['labels'][keep].cpu()
            })
        
        return results
    
    def __repr__(self):
        return (f"BaseDetector(num_classes={self.num_classes}, "
                f"device={self.device}, score_thresh={self.score_thresh})")

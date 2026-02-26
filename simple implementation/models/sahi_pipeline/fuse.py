"""
Detection Fusion
Fuses base detections and SAHI detections with strict final NMS
"""
import torch
from typing import Dict
from torchvision.ops import nms


class DetectionFusion:
    """
    Fuses detections from base detector and SAHI
    
    Process:
        1. Concatenate base and SAHI detections
        2. Apply strict class-wise NMS with final IoU threshold (0.6-0.65)
    
    Key improvement:
        - Strict final NMS (0.65 IoU) catches remaining duplicates after SAHI merge
        - This is mandatory even after SAHI's GREEDYNMM processing
    
    Intuition:
        - Base detector: Global context, fast
        - SAHI: Detailed tiles, slow but accurate
        - Fusion: Best of both worlds with aggressive duplicate removal
    """
    
    def __init__(self, iou_thresh: float = 0.65):
        """
        Initialize fusion module
        
        Args:
            iou_thresh: IoU threshold for final NMS (0.6-0.65 recommended for strict merging)
        """
        self.iou_thresh = iou_thresh
    
    def fuse(
        self,
        base_dets: Dict[str, torch.Tensor],
        sahi_dets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse base and SAHI detections
        
        Args:
            base_dets: Detections from base detector
            sahi_dets: Detections from SAHI
            
        Returns:
            Fused detections dict
        """
        # Handle empty cases
        if len(base_dets['boxes']) == 0 and len(sahi_dets['boxes']) == 0:
            return {
                'boxes': torch.empty(0, 4),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long)
            }
        
        if len(base_dets['boxes']) == 0:
            return sahi_dets
        
        if len(sahi_dets['boxes']) == 0:
            return base_dets
        
        # Concatenate detections
        all_boxes = torch.cat([base_dets['boxes'], sahi_dets['boxes']], dim=0)
        all_scores = torch.cat([base_dets['scores'], sahi_dets['scores']], dim=0)
        all_labels = torch.cat([base_dets['labels'], sahi_dets['labels']], dim=0)
        
        # Apply class-wise NMS
        fused = self._class_wise_nms(all_boxes, all_scores, all_labels)
        
        return fused
    
    def _class_wise_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Apply class-wise NMS
        
        Args:
            boxes: (N, 4) tensor
            scores: (N,) tensor
            labels: (N,) tensor
            
        Returns:
            Filtered detections dict
        """
        unique_labels = labels.unique()
        
        keep_boxes = []
        keep_scores = []
        keep_labels = []
        
        # NMS per class
        for label in unique_labels:
            mask = labels == label
            class_boxes = boxes[mask]
            class_scores = scores[mask]
            
            # Run NMS
            keep_indices = nms(class_boxes, class_scores, self.iou_thresh)
            
            keep_boxes.append(class_boxes[keep_indices])
            keep_scores.append(class_scores[keep_indices])
            keep_labels.append(labels[mask][keep_indices])
        
        # Concatenate
        if len(keep_boxes) == 0:
            return {
                'boxes': torch.empty(0, 4),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long)
            }
        
        return {
            'boxes': torch.cat(keep_boxes, dim=0),
            'scores': torch.cat(keep_scores, dim=0),
            'labels': torch.cat(keep_labels, dim=0)
        }


if __name__ == "__main__":
    # Test fusion
    fusion = DetectionFusion(iou_thresh=0.5)
    
    # Create dummy detections
    base_dets = {
        'boxes': torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]]),
        'scores': torch.tensor([0.9, 0.8]),
        'labels': torch.tensor([1, 2])
    }
    
    sahi_dets = {
        'boxes': torch.tensor([[12, 12, 52, 52], [200, 200, 250, 250]]),
        'scores': torch.tensor([0.85, 0.95]),
        'labels': torch.tensor([1, 3])
    }
    
    # Fuse
    fused = fusion.fuse(base_dets, sahi_dets)
    
    print(f"Base dets: {len(base_dets['boxes'])}")
    print(f"SAHI dets: {len(sahi_dets['boxes'])}")
    print(f"Fused dets: {len(fused['boxes'])}")
    
    # Should remove overlapping detection (IoU > 0.5)
    assert len(fused['boxes']) < len(base_dets['boxes']) + len(sahi_dets['boxes'])
    print("✓ Detection fusion test passed!")

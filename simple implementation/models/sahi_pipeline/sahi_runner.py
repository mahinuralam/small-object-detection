"""
SAHI Inference Runner — Stages 1 & 3
Runs any detector (CD-DPA or SR-TOD) on selected tiles and merges results.

Used twice in the 4-stage pipeline:
  Stage 1: CD-DPA on all SAHI tiles → GREEDYNMM merge for D_base
  Stage 3: SR-TOD on K weak tiles   → GREEDYNMM merge for D_sr
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Union


class SAHIInferenceRunner:
    """
    Runs SAHI (Slicing Aided Hyper Inference) on selected tiles
    
    Process:
        1. For each tile: crop image, run detector
        2. Map detected boxes back to full-image coordinates
        3. Merge all tile detections with GREEDYNMM/IOS strategy
    
    Key improvements for duplicate handling:
        - IOS (Intersection over Smaller) instead of IoU for overlap metric
        - GREEDYNMM (Greedy Non-Maximum Merging) strategy
        - Strict merge thresholds to eliminate duplicates
    """
    
    def __init__(
        self,
        detector,
        merge_iou_thresh: float = 0.6,
        postprocess_type: str = 'GREEDYNMM',
        postprocess_match_metric: str = 'IOS',
        postprocess_match_threshold: float = 0.5
    ):
        """
        Initialize SAHI runner
        
        Args:
            detector: Any object with a predict(image) -> dict method
                      (CDDPADetector, SRTODTileDetector, or BaseDetector)
            merge_iou_thresh: IoU threshold for merging tile detections
            postprocess_type: 'GREEDYNMM' or 'NMS'
            postprocess_match_metric: 'IOS' or 'IOU'
            postprocess_match_threshold: Threshold for matching duplicates
        """
        self.detector = detector
        self.merge_iou_thresh = merge_iou_thresh
        self.postprocess_type = postprocess_type
        self.postprocess_match_metric = postprocess_match_metric
        self.postprocess_match_threshold = postprocess_match_threshold
    
    def run_on_tiles(
        self,
        image: torch.Tensor,
        tiles: List[Tuple[int, int, int, int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Run detector on selected tiles and merge results
        
        Args:
            image: Full image (3, H, W) tensor in [0, 1]
            tiles: List of (x0, y0, x1, y1) tile coordinates
            
        Returns:
            Merged detections dict with 'boxes', 'scores', 'labels'
        """
        if len(tiles) == 0:
            # No tiles - return empty detections
            return {
                'boxes': torch.empty(0, 4),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long)
            }
        
        # Process each tile
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for (x0, y0, x1, y1) in tiles:
            # Crop tile
            tile_img = image[:, y0:y1, x0:x1]
            
            # Run detector on tile
            tile_dets = self.detector.predict(tile_img)
            
            # Skip if no detections
            if len(tile_dets['boxes']) == 0:
                continue
            
            # Map boxes to full-image coordinates
            boxes = tile_dets['boxes'].clone()
            boxes[:, [0, 2]] += x0  # x coordinates
            boxes[:, [1, 3]] += y0  # y coordinates
            
            all_boxes.append(boxes)
            all_scores.append(tile_dets['scores'])
            all_labels.append(tile_dets['labels'])
        
        # Concatenate all detections
        if len(all_boxes) == 0:
            return {
                'boxes': torch.empty(0, 4),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long)
            }
        
        merged_boxes = torch.cat(all_boxes, dim=0)
        merged_scores = torch.cat(all_scores, dim=0)
        merged_labels = torch.cat(all_labels, dim=0)
        
        # Apply postprocessing based on strategy
        if self.postprocess_type == 'GREEDYNMM':
            merged = self._greedy_nmm(
                merged_boxes,
                merged_scores,
                merged_labels,
                self.postprocess_match_metric,
                self.postprocess_match_threshold
            )
        else:  # NMS
            merged = self._class_wise_nms(
                merged_boxes, 
                merged_scores, 
                merged_labels,
                self.merge_iou_thresh
            )
        
        return merged
    
    def _compute_ios(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Intersection over Smaller (IOS) between two sets of boxes
        
        IOS is better than IoU for SAHI because when an object is split across tiles,
        one box might be much smaller (partial object). IOS catches these duplicates.
        
        Args:
            boxes1: (N, 4) tensor in [x0, y0, x1, y1] format
            boxes2: (M, 4) tensor in [x0, y0, x1, y1] format
            
        Returns:
            (N, M) tensor of IOS values
        """
        # Compute areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Compute intersections
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
        wh = (rb - lt).clamp(min=0)  # (N, M, 2)
        intersection = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
        
        # Compute IOS: intersection / min(area1, area2)
        smaller_area = torch.min(area1[:, None], area2[None, :])  # (N, M)
        ios = intersection / (smaller_area + 1e-6)
        
        return ios
    
    def _greedy_nmm(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        metric: str = 'IOS',
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Greedy Non-Maximum Merging (GREEDYNMM)
        
        Unlike NMS which suppresses duplicates, NMM merges them by:
        1. Sort detections by score (descending)
        2. For each detection, merge all similar detections (IOS > threshold)
        3. Keep the highest-scoring merged detection
        
        This is more aggressive at removing duplicates from overlapping tiles.
        
        Args:
            boxes: (N, 4) tensor
            scores: (N,) tensor
            labels: (N,) tensor
            metric: 'IOS' or 'IOU'
            threshold: Matching threshold
            
        Returns:
            Filtered detections dict
        """
        from torchvision.ops import box_iou
        
        unique_labels = labels.unique()
        
        keep_boxes = []
        keep_scores = []
        keep_labels = []
        
        # Process each class separately
        for label in unique_labels:
            mask = labels == label
            class_boxes = boxes[mask]
            class_scores = scores[mask]
            
            if len(class_boxes) == 0:
                continue
            
            # Sort by score (descending)
            sorted_indices = torch.argsort(class_scores, descending=True)
            sorted_boxes = class_boxes[sorted_indices]
            sorted_scores = class_scores[sorted_indices]
            
            # Track which boxes have been merged
            merged_mask = torch.zeros(len(sorted_boxes), dtype=torch.bool)
            
            for i in range(len(sorted_boxes)):
                if merged_mask[i]:
                    continue
                
                # Compute overlap with remaining boxes
                remaining_boxes = sorted_boxes[~merged_mask]
                
                if metric == 'IOS':
                    overlaps = self._compute_ios(
                        sorted_boxes[i:i+1],
                        remaining_boxes
                    )[0]
                else:  # IOU
                    overlaps = box_iou(
                        sorted_boxes[i:i+1],
                        remaining_boxes
                    )[0]
                
                # Find matches above threshold
                match_mask = overlaps > threshold
                
                # Mark matched boxes as merged (they're duplicates)
                remaining_indices = torch.where(~merged_mask)[0]
                matched_indices = remaining_indices[match_mask]
                merged_mask[matched_indices] = True
                
                # Keep the current box (highest score among matches)
                # Note: we already marked it as merged, so unmark it to keep
                merged_mask[i] = False
            
            # Keep non-merged boxes
            keep_boxes.append(sorted_boxes[~merged_mask])
            keep_scores.append(sorted_scores[~merged_mask])
            keep_labels.append(torch.full((torch.sum(~merged_mask),), label, dtype=torch.long))
        
        # Concatenate results
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
    
    def _class_wise_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        iou_thresh: float
    ) -> Dict[str, torch.Tensor]:
        """
        Apply class-wise NMS to detections
        
        Args:
            boxes: (N, 4) tensor
            scores: (N,) tensor
            labels: (N,) tensor
            iou_thresh: IoU threshold
            
        Returns:
            Filtered detections dict
        """
        from torchvision.ops import nms
        
        # Get unique classes
        unique_labels = labels.unique()
        
        keep_boxes = []
        keep_scores = []
        keep_labels = []
        
        # Apply NMS per class
        for label in unique_labels:
            mask = labels == label
            class_boxes = boxes[mask]
            class_scores = scores[mask]
            
            # Run NMS
            keep_indices = nms(class_boxes, class_scores, iou_thresh)
            
            keep_boxes.append(class_boxes[keep_indices])
            keep_scores.append(class_scores[keep_indices])
            keep_labels.append(labels[mask][keep_indices])
        
        # Concatenate results
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
    print("SAHI runner requires BaseDetector - test in integration tests")
    print("✓ SAHI runner module created successfully!")

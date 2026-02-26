"""
Uncertainty Estimator
Computes scalar uncertainty from detection confidence scores
"""
import torch
import numpy as np
from typing import Dict


class UncertaintyEstimator:
    """
    Computes uncertainty U_t from detection scores
    
    Formula:
        If no detections: U_t = 1.0 (maximum uncertainty)
        Otherwise: U_t = 1 - (0.6 * s_max + 0.4 * s_mean)
    
    Intuition:
        - High confidence detections → Low uncertainty → Skip SAHI
        - Low/mixed confidence → High uncertainty → Trigger SAHI
        - No detections → Maximum uncertainty → Trigger SAHI
    """
    
    def __init__(self, base_score_thresh: float = 0.3):
        """
        Initialize estimator
        
        Args:
            base_score_thresh: Minimum score to consider for uncertainty
                              (filters out very low confidence detections)
        """
        self.base_score_thresh = base_score_thresh
    
    def compute_uncertainty(
        self, 
        detections: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute uncertainty score from detections
        
        Args:
            detections: Dict with 'scores' key (N,) tensor
            
        Returns:
            U_t: Uncertainty in [0, 1], where 1 = maximum uncertainty
        """
        scores = detections['scores']
        
        # Filter by minimum threshold
        valid_scores = scores[scores >= self.base_score_thresh]
        
        # Case 1: No valid detections = maximum uncertainty
        if len(valid_scores) == 0:
            return 1.0
        
        # Case 2: Compute uncertainty from statistics
        s_max = valid_scores.max().item()
        s_mean = valid_scores.mean().item()
        
        # Weighted combination: max confidence matters more
        U_t = 1.0 - (0.6 * s_max + 0.4 * s_mean)
        
        # Clamp to [0, 1]
        U_t = np.clip(U_t, 0.0, 1.0)
        
        return U_t
    
    def compute_uncertainty_batch(
        self,
        detections_list: list
    ) -> np.ndarray:
        """
        Compute uncertainty for batch of detections
        
        Args:
            detections_list: List of detection dicts
            
        Returns:
            Array of uncertainties (batch_size,)
        """
        uncertainties = [
            self.compute_uncertainty(det) 
            for det in detections_list
        ]
        return np.array(uncertainties)
    
    def should_trigger_sahi(
        self,
        detections: Dict[str, torch.Tensor],
        theta: float = 0.5
    ) -> bool:
        """
        Decide whether to trigger SAHI based on uncertainty
        
        Args:
            detections: Detection dict
            theta: Uncertainty threshold
            
        Returns:
            True if U_t >= theta (should run SAHI)
        """
        U_t = self.compute_uncertainty(detections)
        return U_t >= theta
    
    def get_statistics(
        self,
        detections: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Get detailed statistics about detections
        
        Args:
            detections: Detection dict
            
        Returns:
            Dict with statistics
        """
        scores = detections['scores']
        valid_scores = scores[scores >= self.base_score_thresh]
        
        if len(valid_scores) == 0:
            return {
                'num_detections': 0,
                'num_valid': 0,
                's_max': 0.0,
                's_mean': 0.0,
                's_std': 0.0,
                'U_t': 1.0
            }
        
        return {
            'num_detections': len(scores),
            'num_valid': len(valid_scores),
            's_max': valid_scores.max().item(),
            's_mean': valid_scores.mean().item(),
            's_std': valid_scores.std().item(),
            'U_t': self.compute_uncertainty(detections)
        }
    
    def __repr__(self):
        return f"UncertaintyEstimator(base_score_thresh={self.base_score_thresh})"

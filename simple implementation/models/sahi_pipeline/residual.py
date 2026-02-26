"""
Residual Map Computer
Computes and normalizes pixel-wise residual between original and reconstructed images
"""
import torch
import numpy as np
from typing import Union


class ResidualMapComputer:
    """
    Computes residual map from original and reconstructed images
    
    Formula:
        Δ(x,y) = mean_c(|I(x,y,c) - I_hat(x,y,c)|)
        Δ_norm(x,y) = (Δ(x,y) - Δ_min) / (Δ_max - Δ_min + ε)
    
    Intuition:
        - High residual → Hard to reconstruct → Likely contains objects
        - Low residual → Easy to reconstruct → Background/uniform regions
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize computer
        
        Args:
            epsilon: Small constant to avoid division by zero
        """
        self.epsilon = epsilon
    
    def compute_residual_map(
        self,
        original: Union[torch.Tensor, np.ndarray],
        reconstructed: Union[torch.Tensor, np.ndarray],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute residual map between original and reconstructed images
        
        Args:
            original: Original image (3, H, W) or (B, 3, H, W) in [0, 1]
            reconstructed: Reconstructed image, same shape as original
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Residual map (H, W) or (B, H, W) float32
        """
        # Convert to tensors if needed
        if isinstance(original, np.ndarray):
            original = torch.from_numpy(original)
        if isinstance(reconstructed, np.ndarray):
            reconstructed = torch.from_numpy(reconstructed)
        
        # Ensure same shape
        assert original.shape == reconstructed.shape, \
            f"Shape mismatch: {original.shape} vs {reconstructed.shape}"
        
        # Compute absolute difference
        diff = torch.abs(original - reconstructed)
        
        # Average across RGB channels
        if diff.ndim == 4:  # (B, C, H, W)
            residual = diff.mean(dim=1)  # (B, H, W)
        elif diff.ndim == 3:  # (C, H, W)
            residual = diff.mean(dim=0)  # (H, W)
        else:
            raise ValueError(f"Invalid input shape: {diff.shape}")
        
        # Normalize to [0, 1] per image
        if normalize:
            residual = self._normalize(residual)
        
        return residual
    
    def _normalize(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Normalize residual map to [0, 1] using min-max normalization
        
        Args:
            residual: (H, W) or (B, H, W) tensor
            
        Returns:
            Normalized residual in [0, 1]
        """
        if residual.ndim == 2:
            # Single image
            r_min = residual.min()
            r_max = residual.max()
            normalized = (residual - r_min) / (r_max - r_min + self.epsilon)
        elif residual.ndim == 3:
            # Batch of images - normalize each independently
            B = residual.shape[0]
            normalized = torch.zeros_like(residual)
            for i in range(B):
                r_min = residual[i].min()
                r_max = residual[i].max()
                normalized[i] = (residual[i] - r_min) / (r_max - r_min + self.epsilon)
        else:
            raise ValueError(f"Invalid residual shape: {residual.shape}")
        
        return normalized
    
    def compute_statistics(
        self,
        residual_map: torch.Tensor
    ) -> dict:
        """
        Compute statistics of residual map
        
        Args:
            residual_map: (H, W) or (B, H, W) tensor
            
        Returns:
            Dict with statistics
        """
        return {
            'mean': residual_map.mean().item(),
            'std': residual_map.std().item(),
            'min': residual_map.min().item(),
            'max': residual_map.max().item(),
            'median': residual_map.median().item()
        }


if __name__ == "__main__":
    # Test residual computer
    computer = ResidualMapComputer()
    
    # Create synthetic images
    original = torch.rand(3, 640, 640)
    reconstructed = original + torch.randn(3, 640, 640) * 0.1  # Add noise
    
    # Compute residual
    residual = computer.compute_residual_map(original, reconstructed)
    
    print(f"Original shape: {original.shape}")
    print(f"Residual shape: {residual.shape}")
    print(f"Residual range: [{residual.min():.4f}, {residual.max():.4f}]")
    
    stats = computer.compute_statistics(residual)
    print(f"Statistics: {stats}")
    
    assert residual.shape == (640, 640), "Output shape mismatch"
    assert residual.min() >= 0 and residual.max() <= 1, "Output not normalized"
    print("✓ Residual computer test passed!")

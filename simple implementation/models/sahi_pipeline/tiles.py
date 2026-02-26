"""
Tile Selector
Generates and selects tiles based on residual map scores
"""
import torch
import numpy as np
from typing import List, Tuple


class TileSelector:
    """
    Generates grid tiles and selects top-N based on residual scores
    
    Strategy:
        1. Generate candidate tiles in a grid pattern
        2. Score each tile by summing residual values within it
        3. Select top-N tiles with highest scores
    
    Intuition:
        - High residual regions likely contain objects
        - Focus SAHI processing on these informative tiles
        - Skip uniform/background regions
    """
    
    def __init__(
        self,
        tile_size: Tuple[int, int] = (384, 384),
        overlap_width_ratio: float = 0.25,
        overlap_height_ratio: float = 0.25
    ):
        """
        Initialize tile selector
        
        Args:
            tile_size: (width, height) of each tile
            overlap_width_ratio: Overlap ratio for width (0.2-0.3 recommended)
            overlap_height_ratio: Overlap ratio for height (0.2-0.3 recommended)
        """
        self.tile_size = tile_size
        self.overlap_width_ratio = overlap_width_ratio
        self.overlap_height_ratio = overlap_height_ratio
        
        # Compute stride from overlap
        tile_w, tile_h = tile_size
        self.stride = (
            int(tile_w * (1 - overlap_width_ratio)),
            int(tile_h * (1 - overlap_height_ratio))
        )
    
    def generate_tiles(
        self,
        image_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate grid tiles covering the image
        
        Args:
            image_size: (width, height) of full image
            
        Returns:
            List of tiles as (x0, y0, x1, y1) coordinates
        """
        img_w, img_h = image_size
        tile_w, tile_h = self.tile_size
        stride_w, stride_h = self.stride
        
        tiles = []
        
        # Generate grid
        y = 0
        while y < img_h:
            x = 0
            while x < img_w:
                # Compute tile bounds
                x0 = x
                y0 = y
                x1 = min(x + tile_w, img_w)
                y1 = min(y + tile_h, img_h)
                
                # Adjust if tile is too small (at image boundaries)
                if x1 - x0 < tile_w // 2:
                    x0 = max(0, img_w - tile_w)
                    x1 = img_w
                if y1 - y0 < tile_h // 2:
                    y0 = max(0, img_h - tile_h)
                    y1 = img_h
                
                tiles.append((x0, y0, x1, y1))
                
                # Move to next column
                x += stride_w
                if x >= img_w:
                    break
            
            # Move to next row
            y += stride_h
            if y >= img_h:
                break
        
        # Remove duplicates (can happen at boundaries)
        tiles = list(set(tiles))
        
        return tiles
    
    def score_tile(
        self,
        residual_map: torch.Tensor,
        tile: Tuple[int, int, int, int]
    ) -> float:
        """
        Score a tile based on total residual within it
        
        Args:
            residual_map: (H, W) residual map
            tile: (x0, y0, x1, y1) tile coordinates
            
        Returns:
            Score (sum of residuals within tile)
        """
        x0, y0, x1, y1 = tile
        
        # Extract tile region
        tile_residual = residual_map[y0:y1, x0:x1]
        
        # Sum residuals (higher = more interesting)
        score = tile_residual.sum().item()
        
        return score
    
    def select_tiles(
        self,
        residual_map: torch.Tensor,
        topN: int,
        image_size: Tuple[int, int] = None
    ) -> List[Tuple[int, int, int, int]]:
        """
        Select top-N tiles with highest residual scores
        
        Args:
            residual_map: (H, W) residual map
            topN: Number of tiles to select
            image_size: (width, height) of image (if None, infer from residual_map)
            
        Returns:
            List of top-N tiles as (x0, y0, x1, y1)
        """
        # Infer image size from residual map if not provided
        if image_size is None:
            H, W = residual_map.shape
            image_size = (W, H)
        
        # Generate all candidate tiles
        tiles = self.generate_tiles(image_size)
        
        # Score all tiles
        scores = [self.score_tile(residual_map, tile) for tile in tiles]
        
        # Select top-N
        top_indices = np.argsort(scores)[-topN:]
        selected_tiles = [tiles[i] for i in top_indices]
        
        return selected_tiles
    
    def visualize_tiles(
        self,
        image_size: Tuple[int, int],
        selected_tiles: List[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Create visualization of tile grid
        
        Args:
            image_size: (width, height)
            selected_tiles: Optional list of selected tiles to highlight
            
        Returns:
            Binary mask (H, W) with tile boundaries
        """
        img_w, img_h = image_size
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        # Draw all tiles
        all_tiles = self.generate_tiles(image_size)
        for (x0, y0, x1, y1) in all_tiles:
            mask[y0, x0:x1] = 128
            mask[y1-1, x0:x1] = 128
            mask[y0:y1, x0] = 128
            mask[y0:y1, x1-1] = 128
        
        # Highlight selected tiles
        if selected_tiles:
            for (x0, y0, x1, y1) in selected_tiles:
                mask[y0:y0+5, x0:x1] = 255
                mask[y1-5:y1, x0:x1] = 255
                mask[y0:y1, x0:x0+5] = 255
                mask[y0:y1, x1-5:x1] = 255
        
        return mask


if __name__ == "__main__":
    # Test tile selector
    selector = TileSelector(tile_size=(320, 320), stride=(160, 160))
    
    # Create synthetic residual map with bright region
    residual = torch.zeros(640, 640)
    residual[200:400, 200:400] = 1.0  # Bright patch (likely objects)
    
    # Generate and score tiles
    image_size = (640, 640)
    all_tiles = selector.generate_tiles(image_size)
    print(f"Generated {len(all_tiles)} tiles")
    
    # Select top tiles
    selected = selector.select_tiles(residual, topN=8, image_size=image_size)
    print(f"Selected {len(selected)} tiles")
    
    # Check that selected tiles overlap with bright region
    overlaps = []
    for (x0, y0, x1, y1) in selected:
        overlap = (x0 < 400 and x1 > 200 and y0 < 400 and y1 > 200)
        overlaps.append(overlap)
    
    print(f"Tiles overlapping bright region: {sum(overlaps)}/{len(selected)}")
    assert sum(overlaps) > 0, "No tiles selected in high-residual region!"
    print("✓ Tile selector test passed!")

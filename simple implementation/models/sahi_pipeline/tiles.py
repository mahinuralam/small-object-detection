"""
Tile Selection — Stage 2 (4-stage architecture)
================================================

Generate a sliding-window tile grid and select BOTTOM-K tiles by
confidence-based scoring from Stage 1 CD-DPA+SAHI:

    S(i) = mean_conf(tile_i) × min(n_dets(tile_i) / n_expected, 1.0)

Weak tiles (lowest S) are sent to SR-TOD in Stage 3.

Design rationale
----------------
- Per-tile confidence from CD-DPA+SAHI is the DIRECT failure signal.
  No reconstruction error proxy, no cheap structure signals, no alpha.
- Tiles with S(i) = 0 → either no detections or all zero-confidence.
  These are the highest-priority weak tiles.
- Tiles with low S(i) → few or low-confidence detections.
  CD-DPA struggled here; SR-TOD reconstruction can help.
- Tiles with high S(i) → many confident detections.
  CD-DPA succeeded; no expensive SR-TOD needed.
"""
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict


class ConfidenceScoringTiler:
    """
    Generates a sliding-window tile grid and selects bottom-K weak tiles.

        Stage 2 — select_weak(tile_grid, confidences, tile_dets, K) → weak tiles

    Also provides backward-compatible grid generation for legacy callers.
    """

    def __init__(
        self,
        tile_size: Tuple[int, int] = (320, 320),
        overlap_width_ratio: float = 0.25,
        overlap_height_ratio: float = 0.25,
    ):
        """
        Args:
            tile_size:             (width, height) of each crop.
            overlap_width_ratio:   Horizontal overlap as a fraction of tile width.
            overlap_height_ratio:  Vertical overlap as a fraction of tile height.
        """
        self.tile_size = tile_size
        self.overlap_width_ratio  = overlap_width_ratio
        self.overlap_height_ratio = overlap_height_ratio

        tile_w, tile_h = tile_size
        self.stride = (
            int(tile_w * (1 - overlap_width_ratio)),
            int(tile_h * (1 - overlap_height_ratio)),
        )

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------

    def generate_grid(
        self,
        image_shape: Tuple[int, int],   # (H, W)  or  (W, H) if legacy
        _legacy_wh: bool = False,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate all candidate tiles for the image.

        Args:
            image_shape: (H, W) — height first, matching tensor shape convention.
                         Pass _legacy_wh=True for old (W, H) callers.

        Returns:
            List of (x0, y0, x1, y1) tile bounding boxes (pixel coords).
        """
        if _legacy_wh:
            img_w, img_h = image_shape
        else:
            img_h, img_w = image_shape

        tile_w, tile_h = self.tile_size
        stride_w, stride_h = self.stride

        tiles = set()
        y = 0
        while True:
            x = 0
            while True:
                x0 = x
                y0 = y
                x1 = min(x + tile_w, img_w)
                y1 = min(y + tile_h, img_h)

                # At right / bottom edge: shift tile left/up to fill
                if x1 - x0 < tile_w // 2:
                    x0 = max(0, img_w - tile_w)
                    x1 = img_w
                if y1 - y0 < tile_h // 2:
                    y0 = max(0, img_h - tile_h)
                    y1 = img_h

                tiles.add((x0, y0, x1, y1))

                x += stride_w
                if x >= img_w:
                    break

            y += stride_h
            if y >= img_h:
                break

        return list(tiles)

    # ------------------------------------------------------------------
    # Stage 2 — Confidence-based weak tile selection
    # ------------------------------------------------------------------

    def score_tiles(
        self,
        tile_grid: List[Tuple[int, int, int, int]],
        tile_confidences: List[float],
        tile_detections: List[Dict[str, torch.Tensor]],
        min_expected: int = 2,
    ) -> List[float]:
        """
        Score each tile using confidence × density.

        S(i) = mean_conf(tile_i) × min(n_dets(tile_i) / min_expected, 1.0)

        Args:
            tile_grid:        List of (x0, y0, x1, y1) tiles.
            tile_confidences: Per-tile mean confidence from CD-DPA (Stage 1).
            tile_detections:  Per-tile detection dicts from CD-DPA (Stage 1).
            min_expected:     Expected detections per tile for density normalization.

        Returns:
            List of float scores, one per tile. LOWER = weaker = needs SR-TOD.
        """
        scores = []
        for i in range(len(tile_grid)):
            conf = tile_confidences[i]
            n_dets = len(tile_detections[i]['boxes'])
            density_factor = min(n_dets / max(min_expected, 1), 1.0)
            s_i = conf * density_factor
            scores.append(s_i)
        return scores

    def select_weak(
        self,
        tile_grid: List[Tuple[int, int, int, int]],
        tile_confidences: List[float],
        tile_detections: List[Dict[str, torch.Tensor]],
        K: int = 10,
        min_expected: int = 2,
    ) -> Tuple[
        List[Tuple[int, int, int, int]],
        List[int],
        List[float],
    ]:
        """
        Stage 2: Select bottom-K tiles by confidence score (weakest tiles).

        Args:
            tile_grid:        All grid tiles from generate_grid().
            tile_confidences: Per-tile mean confidence from CD-DPA Stage 1.
            tile_detections:  Per-tile detection dicts from CD-DPA Stage 1.
            K:                Number of weak tiles to return.
            min_expected:     Expected detections per tile.

        Returns:
            (weak_tiles, weak_indices, all_scores)
            - weak_tiles:   List of K (x0, y0, x1, y1) tiles with lowest scores
            - weak_indices: Indices into tile_grid for the weak tiles
            - all_scores:   All tile scores (for metadata/visualization)
        """
        scores = self.score_tiles(
            tile_grid, tile_confidences, tile_detections, min_expected
        )
        K = min(K, len(tile_grid))

        # Bottom-K: sort by score ascending, take first K
        sorted_indices = np.argsort(scores)[:K]
        weak_tiles = [tile_grid[i] for i in sorted_indices]
        weak_indices = sorted_indices.tolist()

        return weak_tiles, weak_indices, scores

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------

    # Alias for old callers
    generate_tiles = generate_grid

    def select_tiles(
        self,
        residual_map: torch.Tensor,
        topN: int,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Legacy single-signal API: select top-N tiles by residual score.

        Kept for backward compatibility with scripts that call the old
        TileSelector interface.
        """
        if image_size is None:
            H, W = residual_map.shape
        else:
            if image_size[0] > image_size[1]:
                W, H = image_size
            else:
                H, W = image_size

        tiles = self.generate_grid((H, W))
        scores = [residual_map[y0:y1, x0:x1].sum().item()
                  for (x0, y0, x1, y1) in tiles]
        topN = min(topN, len(tiles))
        top_idx = np.argsort(scores)[-topN:]
        return [tiles[i] for i in top_idx]


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

DualScoringTiler = ConfidenceScoringTiler
TileSelector = ConfidenceScoringTiler


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tiler = ConfidenceScoringTiler(tile_size=(320, 320))

    H, W = 720, 1280
    all_tiles = tiler.generate_grid((H, W))
    print(f"Grid tiles: {len(all_tiles)}")

    # Simulate CD-DPA+SAHI Stage 1 results
    import random
    random.seed(42)

    tile_confs = [random.uniform(0.0, 0.9) for _ in all_tiles]
    tile_dets = []
    for c in tile_confs:
        n = int(c * 5)  # more confident → more detections
        tile_dets.append({
            'boxes': torch.rand(n, 4) if n > 0 else torch.empty(0, 4),
            'scores': torch.tensor([c] * n) if n > 0 else torch.empty(0),
            'labels': torch.ones(n, dtype=torch.long) if n > 0 else torch.empty(0, dtype=torch.long),
        })

    # Stage 2: Select weak tiles
    K = 10
    weak_tiles, weak_idx, scores = tiler.select_weak(
        all_tiles, tile_confs, tile_dets, K=K, min_expected=2,
    )
    assert len(weak_tiles) == K, f"Expected {K}, got {len(weak_tiles)}"
    assert len(weak_idx) == K
    # Verify bottom-K selection (lowest scores)
    sorted_scores = sorted(scores)
    for idx in weak_idx:
        assert scores[idx] <= sorted_scores[K], (
            f"Score {scores[idx]} should be in bottom-K (threshold={sorted_scores[K]})"
        )
    print(f"Stage 2: {K} weak tiles selected (bottom-K by confidence)")
    print(f"  Weak tile scores: {[f'{scores[i]:.3f}' for i in weak_idx]}")

    print("ConfidenceScoringTiler self-test passed")

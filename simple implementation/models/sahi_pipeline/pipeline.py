"""
Confidence-Guided SAHI Pipeline — 4-Stage Orchestrator
=======================================================

Implements the redesigned architecture:

  Stage 1 — CD-DPA + SAHI: full image + all tiles → D_base + per-tile confidence
  Stage 2 — Weak tile selection: bottom-K tiles by confidence score
  Stage 3 — SR-TOD (RH + DGFE + detection) on K weak tiles → D_sr
  Stage 4 — Final fusion: D_base + D_sr → class-wise NMS → D_final

Design notes
------------
- CD-DPA runs on both the full image AND all SAHI tiles in Stage 1.
  This serves dual purpose: (1) produce the best possible base detection,
  and (2) generate per-tile confidence scores as a direct failure signal.
- Per-tile confidence = mean score × density factor. Tiles with lowest
  confidence are the ones where CD-DPA struggled — no reconstructor proxy,
  no cheap signals, no alpha balancing.
- SR-TOD (separate model: RH + DGFE + detection) runs only on K weak
  tiles identified in Stage 2.
- Eliminates ReconstructionHead, UncertaintyMapper, and CheapSignalBuilder
  from the pipeline — simpler, faster, more direct.
"""
import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, Tuple, Union, Optional, List

from .detector_wrapper import CDDPADetector, SRTODTileDetector, BaseDetector
from .tiles import ConfidenceScoringTiler
from .sahi_runner import SAHIInferenceRunner
from .fuse import DetectionFusion


class ConfidenceGuidedSAHIPipeline:
    """
    4-stage confidence-guided SAHI pipeline.

    Config attributes used:
        K              int    Weak tiles sent to SR-TOD               [10]
        min_expected_dets int  Density normalization factor           [2]
        tile_size      tuple  (w, h) of each tile                    [(320,320)]
        iou_final      float  Final NMS IoU threshold                [0.65]
        cddpa_checkpoint  str   Path to CD-DPA checkpoint
        srtod_checkpoint  str   Path to SR-TOD checkpoint
        device         str    'cuda' or 'cpu'
        debug          bool   Save debug visualisations              [False]
    """

    def __init__(self, config):
        self.config = config
        self.device = getattr(config, 'device', 'cuda')

        print("Initialising ConfidenceGuidedSAHIPipeline (4-stage)...")

        # ── Stage 1: CD-DPA Detector (full image + SAHI tiles) ──────────
        cddpa_ckpt = getattr(config, 'cddpa_checkpoint',
                             getattr(config, 'detector_checkpoint', None))
        self.detector = CDDPADetector(
            checkpoint_path=cddpa_ckpt,
            num_classes=getattr(config, 'num_classes', 11),
            device=self.device,
            score_thresh=getattr(config, 'detection_score_thresh', 0.4),
        )
        print("  Stage 1: CD-DPA detector loaded")

        # ── Stage 1: SAHI runner for CD-DPA tile inference ──────────────
        self.cddpa_sahi_runner = SAHIInferenceRunner(
            detector=self.detector,
            merge_iou_thresh=getattr(config, 'iou_tile_merge', 0.6),
            postprocess_type=getattr(config, 'postprocess_type', 'GREEDYNMM'),
            postprocess_match_metric=getattr(config, 'postprocess_match_metric', 'IOS'),
            postprocess_match_threshold=getattr(config, 'postprocess_match_threshold', 0.5),
        )
        print("  Stage 1: SAHI runner ready (CD-DPA)")

        # ── Stage 2: Tiler + weak tile selector ────────────────────────
        tile_size = getattr(config, 'tile_size', (320, 320))
        overlap = getattr(config, 'overlap_width_ratio', 0.25)
        self.tiler = ConfidenceScoringTiler(
            tile_size=tile_size,
            overlap_width_ratio=overlap,
            overlap_height_ratio=getattr(config, 'overlap_height_ratio', overlap),
        )
        print("  Stage 2: ConfidenceScoringTiler ready")

        # ── Stage 3: SR-TOD tile detector ───────────────────────────────
        srtod_ckpt = getattr(config, 'srtod_checkpoint', None)
        self.srtod_detector = SRTODTileDetector(
            checkpoint_path=srtod_ckpt,
            num_classes=getattr(config, 'num_classes', 11),
            device=self.device,
            score_thresh=getattr(config, 'detection_score_thresh', 0.4),
        )
        print("  Stage 3: SR-TOD tile detector loaded")

        # ── Stage 3: SAHI runner for SR-TOD tile inference ──────────────
        self.srtod_runner = SAHIInferenceRunner(
            detector=self.srtod_detector,
            merge_iou_thresh=getattr(config, 'iou_tile_merge', 0.6),
            postprocess_type=getattr(config, 'postprocess_type', 'GREEDYNMM'),
            postprocess_match_metric=getattr(config, 'postprocess_match_metric', 'IOS'),
            postprocess_match_threshold=getattr(config, 'postprocess_match_threshold', 0.5),
        )
        print("  Stage 3: SAHI runner ready (SR-TOD)")

        # ── Stage 4: Fusion ─────────────────────────────────────────────
        self.fusion = DetectionFusion(
            iou_thresh=getattr(config, 'iou_final', 0.65)
        )
        print("  Stage 4: Fusion module ready")

        # Debug
        if getattr(config, 'debug', False):
            Path(getattr(config, 'debug_dir', 'results/debug')).mkdir(
                parents=True, exist_ok=True
            )

        K = getattr(config, 'K', 10)
        min_exp = getattr(config, 'min_expected_dets', 2)
        print(f"\n  Pipeline ready  |  device={self.device}  "
              f"|  K={K}  min_expected_dets={min_exp}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def process_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Run the 4-stage confidence-guided pipeline on one image.

        Args:
            image: RGB image — numpy (H, W, 3)  OR  torch (3, H, W)

        Returns:
            detections: dict  'boxes' (N,4), 'scores' (N,), 'labels' (N,)
            metadata:   processing details
        """
        t_start = time.time()
        timings: Dict[str, float] = {}

        image_tensor = self._to_tensor(image)
        _, H, W = image_tensor.shape

        # ─────────────────────────────────────────────────────────────
        # Stage 1 — CD-DPA + SAHI: full image + all tiles
        # ─────────────────────────────────────────────────────────────

        # 1a. CD-DPA on full image
        t0 = time.time()
        D_base_full = self.detector.predict(image_tensor)
        timings['cddpa_full_image'] = (time.time() - t0) * 1000

        # 1b. Generate tile grid
        t0 = time.time()
        tile_grid = self.tiler.generate_grid((H, W))
        timings['grid_generation'] = (time.time() - t0) * 1000

        # 1c. CD-DPA on each tile + collect per-tile confidence
        t0 = time.time()
        tile_detections: List[Dict[str, torch.Tensor]] = []
        tile_confidences: List[float] = []

        for (x0, y0, x1, y1) in tile_grid:
            # Crop tile from image
            tile_img = image_tensor[:, y0:y1, x0:x1]

            # Run CD-DPA on tile
            tile_dets = self.detector.predict(tile_img)

            # Remap boxes to full-image coordinates
            if len(tile_dets['boxes']) > 0:
                tile_dets_remapped = {
                    'boxes':  tile_dets['boxes'].clone(),
                    'scores': tile_dets['scores'].clone(),
                    'labels': tile_dets['labels'].clone(),
                }
                tile_dets_remapped['boxes'][:, [0, 2]] += x0
                tile_dets_remapped['boxes'][:, [1, 3]] += y0
            else:
                tile_dets_remapped = tile_dets

            tile_detections.append(tile_dets_remapped)

            # Per-tile confidence: direct failure signal
            if len(tile_dets['scores']) > 0:
                c_i = tile_dets['scores'].mean().item()
            else:
                c_i = 0.0  # no detections = likely failure
            tile_confidences.append(c_i)

        timings['cddpa_sahi_tiles'] = (time.time() - t0) * 1000

        # 1d. Merge full-image + tile detections → D_base
        t0 = time.time()
        D_base = self._merge_all_detections(D_base_full, tile_detections)
        timings['sahi_merge'] = (time.time() - t0) * 1000

        # ─────────────────────────────────────────────────────────────
        # Stage 2 — Weak tile selection (bottom-K by confidence)
        # ─────────────────────────────────────────────────────────────
        K = getattr(self.config, 'K', 10)
        min_expected = getattr(self.config, 'min_expected_dets', 2)

        t0 = time.time()
        weak_tiles, weak_indices, tile_scores = self.tiler.select_weak(
            tile_grid, tile_confidences, tile_detections,
            K=K, min_expected=min_expected,
        )
        timings['weak_selection'] = (time.time() - t0) * 1000

        # ─────────────────────────────────────────────────────────────
        # Stage 3 — SR-TOD on K weak tiles
        # ─────────────────────────────────────────────────────────────
        t0 = time.time()
        D_sr = self.srtod_runner.run_on_tiles(image_tensor, weak_tiles)
        timings['srtod_weak_tiles'] = (time.time() - t0) * 1000

        # ─────────────────────────────────────────────────────────────
        # Stage 4 — Final fusion
        # ─────────────────────────────────────────────────────────────
        t0 = time.time()
        D_final = self.fusion.fuse(D_base, D_sr)
        timings['final_fusion'] = (time.time() - t0) * 1000

        timings['total'] = (time.time() - t_start) * 1000

        metadata = self._build_metadata(
            D_base, D_sr, D_final,
            tile_grid, tile_confidences, tile_scores,
            weak_tiles, weak_indices, K,
            timings,
        )

        if getattr(self.config, 'debug', False):
            self._save_debug(
                image_tensor, D_base, D_sr, D_final,
                tile_grid, tile_confidences, weak_tiles, weak_indices,
                metadata,
            )

        return D_final, metadata

    # ------------------------------------------------------------------
    # Merge helper — combines full-image + all tile detections via GREEDYNMM
    # ------------------------------------------------------------------

    def _merge_all_detections(
        self,
        D_full: Dict[str, torch.Tensor],
        tile_dets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Merge full-image detections with all tile detections using
        the CD-DPA SAHI runner's GREEDYNMM strategy.
        """
        all_boxes  = [D_full['boxes']]
        all_scores = [D_full['scores']]
        all_labels = [D_full['labels']]

        for td in tile_dets:
            if len(td['boxes']) > 0:
                all_boxes.append(td['boxes'])
                all_scores.append(td['scores'])
                all_labels.append(td['labels'])

        if len(all_boxes) == 1 and len(all_boxes[0]) == 0:
            return {
                'boxes':  torch.empty(0, 4),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long),
            }

        merged_boxes  = torch.cat(all_boxes, dim=0)
        merged_scores = torch.cat(all_scores, dim=0)
        merged_labels = torch.cat(all_labels, dim=0)

        # Use GREEDYNMM to remove duplicates
        return self.cddpa_sahi_runner._greedy_nmm(
            merged_boxes, merged_scores, merged_labels,
            metric=getattr(self.config, 'postprocess_match_metric', 'IOS'),
            threshold=getattr(self.config, 'postprocess_match_threshold', 0.5),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            if image.ndim == 3 and image.shape[2] == 3:
                image = torch.from_numpy(image).permute(2, 0, 1)
            elif image.ndim == 3 and image.shape[0] == 3:
                image = torch.from_numpy(image)
            else:
                raise ValueError(f"Unsupported ndarray shape: {image.shape}")
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[2] == 3:
                image = image.permute(2, 0, 1)
            elif not (image.dim() == 3 and image.shape[0] == 3):
                raise ValueError(f"Unsupported tensor shape: {image.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        return image.float().to(self.device)

    @staticmethod
    def _build_metadata(
        D_base, D_sr, D_final,
        tile_grid, tile_confidences, tile_scores,
        weak_tiles, weak_indices, K,
        timings,
    ) -> Dict:
        return {
            'num_base_dets':     len(D_base['boxes']),
            'num_sr_dets':       len(D_sr['boxes']),
            'num_final_dets':    len(D_final['boxes']),
            'K_selected':        len(weak_tiles),
            'K_config':          K,
            'total_tiles':       len(tile_grid),
            'latency_ms':        timings['total'],
            'timings':           timings,
            # Per-tile confidence data for visualisation
            'tile_grid':         tile_grid,
            'tile_confidences':  tile_confidences,
            'tile_scores':       tile_scores,
            'weak_tile_indices': weak_indices,
            'weak_tiles':        weak_tiles,
        }

    def _save_debug(
        self, image, D_base, D_sr, D_final,
        tile_grid, tile_confidences, weak_tiles, weak_indices,
        metadata,
    ):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        img_np = (image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        def draw_boxes(ax, img, dets, color, title):
            ax.imshow(img)
            for box in dets['boxes']:
                x1, y1, x2, y2 = box.float().tolist()
                ax.add_patch(patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.5, edgecolor=color, facecolor='none'))
            ax.set_title(f"{title}  (n={len(dets['boxes'])})")
            ax.axis('off')

        # Stage 1: D_base (CD-DPA + SAHI merged)
        draw_boxes(axes[0, 0], img_np, D_base, 'lime',
                   'Stage 1 — D_base (CD-DPA+SAHI)')

        # Stage 1: Tile confidence heatmap
        ax = axes[0, 1]
        ax.imshow(img_np, alpha=0.5)
        for i, (x0, y0, x1, y1) in enumerate(tile_grid):
            c = tile_confidences[i]
            # Color: red=low confidence, green=high confidence
            color = (1.0 - c, c, 0.0, 0.4)
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     facecolor=color, edgecolor='gray',
                                     linewidth=0.5)
            ax.add_patch(rect)
            ax.text(x0 + 3, y0 + 12, f'{c:.2f}', fontsize=5, color='white')
        ax.set_title('Stage 1 — Tile Confidence Map')
        ax.axis('off')

        # Stage 2: Weak tiles highlighted
        ax = axes[0, 2]
        ax.imshow(img_np)
        for i, (x0, y0, x1, y1) in enumerate(tile_grid):
            color = 'red' if i in weak_indices else 'green'
            alpha_val = 0.8 if i in weak_indices else 0.2
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     fill=False, edgecolor=color,
                                     linewidth=2 if i in weak_indices else 0.5,
                                     alpha=alpha_val)
            ax.add_patch(rect)
        ax.set_title(f'Stage 2 — Weak tiles  (K={len(weak_tiles)})')
        ax.axis('off')

        # Stage 3: SR-TOD detections on weak tiles
        draw_boxes(axes[1, 0], img_np, D_sr, 'red',
                   'Stage 3 — D_sr (SR-TOD on weak tiles)')

        # Stage 4: Final fused detections
        draw_boxes(axes[1, 1], img_np, D_final, 'dodgerblue',
                   'Stage 4 — D_final (fused)')

        # Timing breakdown
        ax = axes[1, 2]
        ax.axis('off')
        timing_text = "Timing Breakdown\n" + "─" * 30 + "\n"
        for key, val in metadata['timings'].items():
            if key != 'total':
                timing_text += f"  {key}: {val:.1f} ms\n"
        timing_text += "─" * 30 + f"\n  TOTAL: {metadata['timings']['total']:.1f} ms"
        ax.text(0.1, 0.9, timing_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(
            f"4-Stage Confidence-Guided SAHI  |  K={metadata['K_config']}  "
            f"|  {metadata['latency_ms']:.0f} ms  "
            f"|  base={metadata['num_base_dets']}  sr={metadata['num_sr_dets']}  "
            f"final={metadata['num_final_dets']}",
            fontsize=12)
        plt.tight_layout()

        debug_dir = Path(getattr(self.config, 'debug_dir', 'results/debug'))
        out_path = debug_dir / f"debug_{int(time.time()*1000)}.png"
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()

    def __repr__(self):
        K = getattr(self.config, 'K', '?')
        return (f"ConfidenceGuidedSAHIPipeline(K={K}, "
                f"device={self.device})")


# Backward-compatible aliases
BudgetedSAHIPipeline = ConfidenceGuidedSAHIPipeline
SAHIPipeline = ConfidenceGuidedSAHIPipeline


if __name__ == "__main__":
    print("ConfidenceGuidedSAHIPipeline requires a config object.")
    print("Run the integration test in tests/test_sahi_pipeline.py")
    print("pipeline.py import check passed")

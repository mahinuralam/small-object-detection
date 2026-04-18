"""
Confidence-Guided SAHI Pipeline
================================
4-stage architecture:
  Stage 1 — CD-DPA + SAHI (full image + all tiles) → D_base + per-tile confidence
  Stage 2 — Weak tile selection: bottom-K tiles by confidence score
  Stage 3 — SR-TOD (RH + DGFE + detection) on K weak tiles → D_sr
  Stage 4 — Final fusion: D_base + D_sr → class-wise NMS → D_final

Primary classes
----------------
ConfidenceGuidedSAHIPipeline — main 4-stage orchestrator
CDDPADetector                — Stage 1 full-image + SAHI tile CD-DPA detector
SRTODTileDetector            — Stage 3 SR-TOD weak-tile detector
ConfidenceScoringTiler       — tile grid + confidence-based bottom-K selection (Stage 2)

Backward-compatible aliases
----------------------------
BudgetedSAHIPipeline  → ConfidenceGuidedSAHIPipeline
SAHIPipeline          → ConfidenceGuidedSAHIPipeline
DualScoringTiler      → ConfidenceScoringTiler
TileSelector          → ConfidenceScoringTiler
BaseDetector          → kept for legacy code
"""

# ── Core ──────────────────────────────────────────────────────────────
from .detector_wrapper import BaseDetector               # backward-compat
from .detector_wrapper import CDDPADetector               # Stage 1
from .detector_wrapper import SRTODTileDetector            # Stage 3
from .sahi_runner import SAHIInferenceRunner
from .fuse import DetectionFusion

# ── Stage 2: tile scoring + bottom-K selection ───────────────────────
from .tiles import ConfidenceScoringTiler
from .tiles import DualScoringTiler                     # backward-compat alias
from .tiles import TileSelector                         # backward-compat alias

# ── 4-stage pipeline orchestrator ────────────────────────────────────
from .pipeline import ConfidenceGuidedSAHIPipeline
from .pipeline import BudgetedSAHIPipeline              # backward-compat alias
from .pipeline import SAHIPipeline                      # backward-compat alias

__all__ = [
    # Primary classes (4-stage)
    'ConfidenceGuidedSAHIPipeline',
    'CDDPADetector',
    'SRTODTileDetector',
    'ConfidenceScoringTiler',

    # Unchanged modules
    'BaseDetector',
    'SAHIInferenceRunner',
    'DetectionFusion',

    # Backward-compatible aliases
    'BudgetedSAHIPipeline',
    'SAHIPipeline',
    'DualScoringTiler',
    'TileSelector',
]

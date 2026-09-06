"""
Part 3 — Preprocessing & Geospatial Pipeline.

Public interface (architecture.md Section 3.3 — the literal contract Part 2
codes against):

    validate_and_prepare(file_path, declared_modality, declared_timestamp) -> ImageMetadata
    tile_image(image_id, task, tile_size=1024, overlap_pct=0.15) -> list[Tile]
    check_coregistration(image_a_id, image_b_id) -> CoregistrationResult
    stitch_detections(tile_evidence, image_id) -> Evidence

Everything else in this package (normalize, store, utils, config, and the
pure helper functions inside tiling/coregistration/stitching) is an
implementation detail Part 2 shouldn't need to import directly.
"""

from .validation import validate_and_prepare
from .tiling import tile_image
from .coregistration import check_coregistration
from .stitching import stitch_detections

from . import config

__all__ = [
    "validate_and_prepare",
    "tile_image",
    "check_coregistration",
    "stitch_detections",
    "config",
]

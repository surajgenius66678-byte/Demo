"""
tile_image() — architecture.md Section 3.3.

Split into two layers:
  - _axis_offsets / compute_tile_windows: pure grid math, numpy-only, fully
    unit-testable without rasterio.
  - tile_image: the literal public interface Part 2 codes against. Opens the
    COG, performs windowed reads (never loads the full array), normalizes
    each tile per-modality, writes each tile to its own small GeoTIFF, and
    returns canonical Tile objects.

Design decision (architecture.md doesn't fully pin this down): `task` is
accepted for interface compliance and audit-trail logging, but the grid
itself is purely a function of (width, height, tile_size, overlap_pct) —
it does not vary by task. That's deliberate: it guarantees two images of
identical dimensions always get identical tile grids, which is what
change-detection needs to keep before/after tiles aligned tile-for-tile,
and it means tile_id is independent of task, so tiles are naturally
reusable/cached across tasks run against the same image at the same
tile_size/overlap rather than re-materialized per task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import config, normalize, store, utils

if TYPE_CHECKING:
    from backend.shared.schemas import Tile, TaskType


# --------------------------------------------------------------------------
# Pure grid math — no rasterio, no I/O
# --------------------------------------------------------------------------

def _axis_offsets(dim: int, tile_size: int, stride: int) -> list[int]:
    """
    Offsets along one axis (row or column).

    - If the whole axis fits in a single tile, returns [0] — one tile spans
      the full axis (its length will be `dim`, not `tile_size`; see
      compute_tile_windows).
    - Otherwise, marches by `stride` while a full tile still fits, then
      anchors the FINAL offset exactly at `dim - tile_size`. That guarantees
      the last tile is always a full tile_size — never a thin sliver — at
      the cost of slightly more than nominal overlap between the last two
      tiles right at the edge. This is standard practice for sliding-window
      tiling and is preferred over leaving a partial-width/height trailing
      tile that downstream models would see inconsistently sized input from.
    """
    if dim <= tile_size:
        return [0]
    offsets: list[int] = []
    pos = 0
    while pos + tile_size < dim:
        offsets.append(pos)
        pos += stride
    flush = dim - tile_size
    if not offsets or offsets[-1] != flush:
        offsets.append(flush)
    return offsets


def compute_tile_windows(width: int, height: int, tile_size: int, overlap_pct: float) -> list[dict]:
    """
    Computes a covering grid of tile windows over a width x height image.
    Returns dicts {"col_off", "row_off", "width", "height"} in row-major
    order. Every returned window is strictly within [0, width) x [0, height)
    — windows are clipped/anchored at the boundary, never padded past it.

    Invariant: whenever an axis dimension exceeds tile_size, every window's
    extent along that axis is exactly tile_size (no slivers). When an axis
    dimension is <= tile_size, there is exactly one row/column of windows
    along that axis, sized to the full dimension.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not (0.0 <= overlap_pct < 1.0):
        raise ValueError("overlap_pct must be in [0, 1)")

    stride = max(1, int(round(tile_size * (1 - overlap_pct))))

    row_offs = _axis_offsets(height, tile_size, stride)
    col_offs = _axis_offsets(width, tile_size, stride)

    windows = []
    for row_off in row_offs:
        for col_off in col_offs:
            w = min(tile_size, width - col_off)
            h = min(tile_size, height - row_off)
            windows.append({"col_off": col_off, "row_off": row_off, "width": w, "height": h})
    return windows


# --------------------------------------------------------------------------
# I/O layer — rasterio-dependent, not executable in the build sandbox
# (see PART3_BUILD_NOTES.md). Logic mirrors the pure grid math above exactly.
# --------------------------------------------------------------------------

def tile_image(
    image_id: str,
    task: "TaskType",
    tile_size: int = config.DEFAULT_TILE_SIZE,
    overlap_pct: float = config.DEFAULT_OVERLAP_PCT,
) -> "list[Tile]":
    """
    architecture.md 3.3 literal interface:
        def tile_image(image_id, task, tile_size=1024, overlap_pct=0.15) -> list[Tile]

    Reads the stored COG for image_id using rasterio, computes the tile grid
    via compute_tile_windows, and for each window: performs a windowed read
    (rasterio.windows.Window — never a full-array read), normalizes per the
    image's modality, writes the tile to its own small GeoTIFF, and records
    a Tile.
    """
    import rasterio
    from rasterio.windows import Window, transform as window_transform

    from backend.shared.schemas import Tile  # local import: only needed once pydantic is available

    meta = store.load_metadata(image_id)  # raises store.ImageNotFoundError if unknown
    if not meta.get("is_valid", False):
        raise ValueError(f"image_id={image_id!r} failed validation and cannot be tiled: "
                          f"{meta.get('validation_errors')}")

    modality = meta["modality"]
    tiles: list[Tile] = []

    with rasterio.open(meta["cog_path"]) as dataset:
        windows = compute_tile_windows(dataset.width, dataset.height, tile_size, overlap_pct)

        for w in windows:
            rio_window = Window(w["col_off"], w["row_off"], w["width"], w["height"])

            # Windowed read: GDAL decodes only this window's pixels, never
            # the full array, regardless of source image size.
            array = dataset.read(window=rio_window)
            normalized = normalize.normalize_tile(array, modality)

            tile_transform = window_transform(rio_window, dataset.transform)
            affine_values = list(tile_transform)[:6]  # [a, b, c, d, e, f]

            tile_id = utils.compute_key_hash(
                image_id, tile_size, overlap_pct, w["col_off"], w["row_off"]
            )
            array_path = store.tile_array_path(image_id, tile_id)

            profile = {
                "driver": "GTiff",
                "height": w["height"],
                "width": w["width"],
                "count": normalized.shape[0],
                "dtype": "float32",
                "crs": dataset.crs,
                "transform": tile_transform,
                "compress": "DEFLATE",
            }
            store.ensure_image_dirs(image_id)
            with rasterio.open(array_path, "w", **profile) as dst:
                dst.write(normalized)

            tile_dict = {
                "tile_id": tile_id,
                "image_id": image_id,
                "col_off": w["col_off"],
                "row_off": w["row_off"],
                "width": w["width"],
                "height": w["height"],
                "affine_transform": affine_values,
                "array_path": array_path,
            }
            store.save_tile_record(image_id, tile_id, tile_dict)
            tiles.append(Tile(**tile_dict))

    return tiles

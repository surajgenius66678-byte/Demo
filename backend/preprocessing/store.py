"""
Local storage for Part 3 artifacts, addressed by content hash — never by the
uploaded filename (cross-cutting requirement).

tile_image() and check_coregistration() only receive an `image_id: str` per
the interface contract in architecture.md Section 3.3 — this module is what
lets Part 3 resolve that id back to a COG path and metadata across separate
calls.

Layout:
    {DATA_DIR}/{image_id}/
        cog.tif                # the converted Cloud-Optimized GeoTIFF
        metadata.json          # ImageMetadata fields, as plain JSON
        tiles/
            {tile_id}.tif       # one small GeoTIFF per tile
            {tile_id}.json      # that Tile's fields, as plain JSON
        stitched/
            {key}.tif           # merged change-map mosaics from stitch_detections

This module works with plain dicts, not pydantic models — it has no
dependency on the schema library at all. Callers convert at the boundary via
`model.model_dump()` going in and `Schema(**loaded_dict)` coming out. That
keeps the storage layer testable on its own and keeps schema validation
concerns out of the persistence layer.
"""

from __future__ import annotations

import json
import os

from . import config


class ImageNotFoundError(LookupError):
    """Raised when an image_id has no corresponding entry in the store."""


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------

def image_dir(image_id: str) -> str:
    return os.path.join(config.DATA_DIR, image_id)


def cog_path_for(image_id: str) -> str:
    return os.path.join(image_dir(image_id), "cog.tif")


def metadata_path_for(image_id: str) -> str:
    return os.path.join(image_dir(image_id), "metadata.json")


def tiles_dir_for(image_id: str) -> str:
    return os.path.join(image_dir(image_id), "tiles")


def stitched_dir_for(image_id: str) -> str:
    return os.path.join(image_dir(image_id), "stitched")


def tile_array_path(image_id: str, tile_id: str) -> str:
    return os.path.join(tiles_dir_for(image_id), f"{tile_id}.tif")


def ensure_image_dirs(image_id: str) -> None:
    os.makedirs(image_dir(image_id), exist_ok=True)
    os.makedirs(tiles_dir_for(image_id), exist_ok=True)
    os.makedirs(stitched_dir_for(image_id), exist_ok=True)


def _atomic_write_json(path: str, payload: dict) -> None:
    """Write-to-temp-then-rename so a crash mid-write never leaves a
    truncated/corrupt JSON file behind (os.replace is atomic on POSIX)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


# --------------------------------------------------------------------------
# Image-level metadata
# --------------------------------------------------------------------------

def save_metadata(image_id: str, metadata: dict) -> None:
    ensure_image_dirs(image_id)
    _atomic_write_json(metadata_path_for(image_id), metadata)


def load_metadata(image_id: str) -> dict:
    path = metadata_path_for(image_id)
    if not os.path.exists(path):
        raise ImageNotFoundError(f"no stored metadata for image_id={image_id!r}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def has_image(image_id: str) -> bool:
    return os.path.exists(metadata_path_for(image_id))


# --------------------------------------------------------------------------
# Tile-level records
# --------------------------------------------------------------------------

def save_tile_record(image_id: str, tile_id: str, tile: dict) -> None:
    ensure_image_dirs(image_id)
    path = os.path.join(tiles_dir_for(image_id), f"{tile_id}.json")
    _atomic_write_json(path, tile)


def load_tile_record(image_id: str, tile_id: str) -> dict:
    path = os.path.join(tiles_dir_for(image_id), f"{tile_id}.json")
    if not os.path.exists(path):
        raise ImageNotFoundError(f"no stored tile record for image_id={image_id!r} tile_id={tile_id!r}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

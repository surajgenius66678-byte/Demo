"""
Tunable constants for the Preprocessing & Geospatial part (Part 3).

Kept in one place so the numbers referenced by architecture.md's hardening
list (pixel-count cap, parse timeout, alignment thresholds, NMS threshold)
are easy to find and tune without hunting through the module.

All of these can be overridden with environment variables so a demo box
and a CI box can use different limits without editing code.
"""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# --- Storage -----------------------------------------------------------
# Content-hash-addressed store: {DATA_DIR}/{image_id}/... — never filename-keyed.
DATA_DIR = os.environ.get("SATQUERY_DATA_DIR", os.path.join(os.getcwd(), "data", "images"))

# --- Structural validation caps -----------------------------------------
# Raw file size on disk, checked before any parsing is attempted.
MAX_FILE_SIZE_BYTES = _env_int("SATQUERY_MAX_FILE_SIZE_BYTES", 2 * 1024 * 1024 * 1024)  # 2 GiB

# Pixel-count cap: width * height * band_count. Checked from a METADATA-ONLY
# read, before any pixel array is allocated — this is what blocks
# decompression-bomb files (tiny compressed file, enormous decoded array).
MAX_PIXEL_COUNT = _env_int("SATQUERY_MAX_PIXEL_COUNT", 400_000_000)  # e.g. ~20000x20000 px

# Wall-clock budget for the isolated metadata-probe subprocess. A corrupt or
# adversarial file that hangs the parser must not hang the caller.
PARSE_TIMEOUT_SECONDS = _env_float("SATQUERY_PARSE_TIMEOUT_SECONDS", 20.0)

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".geotiff"}

# --- Tiling ---------------------------------------------------------------
DEFAULT_TILE_SIZE = _env_int("SATQUERY_DEFAULT_TILE_SIZE", 1024)
DEFAULT_OVERLAP_PCT = _env_float("SATQUERY_DEFAULT_OVERLAP_PCT", 0.15)

# --- Co-registration --------------------------------------------------
# Three tiers, two boundaries:
#   offset <= NEGLIGIBLE                      -> aligned=True,  auto_corrected=False
#   NEGLIGIBLE < offset <= MAX_ALIGNMENT       -> aligned=True,  auto_corrected=True
#   offset > MAX_ALIGNMENT                     -> aligned=False, auto_corrected=False
#
# Below NEGLIGIBLE, two images are considered natively aligned — no
# correction needed. Between the two thresholds, Part 3 reports the pair as
# aligned but flags that a sub-pixel/pixel shift was detected and accounted
# for (`auto_corrected=True`). Above MAX_ALIGNMENT, `aligned=False` — Part 2
# is responsible for refusing the downstream task on that result; Part 3
# only computes the number.
COREG_NEGLIGIBLE_OFFSET_PX = _env_float("SATQUERY_COREG_NEGLIGIBLE_OFFSET_PX", 1.0)
COREG_MAX_ALIGNMENT_THRESHOLD_PX = _env_float("SATQUERY_COREG_MAX_ALIGNMENT_THRESHOLD_PX", 10.0)

# Long-side cap (px) for the downsampled overview used for phase correlation.
# A decimated/overview read at this size is memory-safe regardless of the
# source image's native resolution.
COREG_OVERVIEW_MAX_SIDE_PX = _env_int("SATQUERY_COREG_OVERVIEW_MAX_SIDE_PX", 1024)

# --- Stitching / NMS -----------------------------------------------------
STITCH_IOU_THRESHOLD = _env_float("SATQUERY_STITCH_IOU_THRESHOLD", 0.5)

# --- Normalization -----------------------------------------------------
OPTICAL_PERCENTILE_LOW = _env_float("SATQUERY_OPTICAL_PERCENTILE_LOW", 2.0)
OPTICAL_PERCENTILE_HIGH = _env_float("SATQUERY_OPTICAL_PERCENTILE_HIGH", 98.0)

# Typical dynamic range for calibrated SAR backscatter in dB. Values are
# clipped to this range before being scaled to [0, 1].
SAR_DB_MIN = _env_float("SATQUERY_SAR_DB_MIN", -30.0)
SAR_DB_MAX = _env_float("SATQUERY_SAR_DB_MAX", 5.0)

# --- Sanitization -------------------------------------------------------
MAX_SANITIZED_STRING_LEN = _env_int("SATQUERY_MAX_SANITIZED_STRING_LEN", 512)

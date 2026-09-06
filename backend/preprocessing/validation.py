"""
validate_and_prepare() — architecture.md Section 3.3.

The sequence, in order, and why:
  1. Fast structural checks (exists, extension, size on disk) — no parsing,
     so a bad path is rejected in microseconds.
  2. Content hash (image_id) — bounded work now that size is capped.
  3. Isolated, timeout-guarded metadata probe (utils.run_isolated running
     _probe_raster_metadata in a subprocess) — opens the file and reads
     ONLY header attributes, never pixel data. A corrupt or hostile file can
     only ever cost this one job its timeout budget.
  4. Pixel-count cap check using the metadata just read — strictly BEFORE
     any array is allocated. This is what blocks decompression-bomb files:
     a tiny compressed file that claims an enormous decoded size is
     rejected here, before anything tries to decode it.
  5. COG conversion, only for files that passed every check above.

Every exit path returns a schema-valid ImageMetadata (even for rejected
files — width/height/etc. are populated with whatever the probe legitimately
read, or zeroed out if rejection happened before a probe could run) and
persists it to the store, so a later tile_image()/check_coregistration()
call against the same image_id — valid or not — has something to look up.
"""

from __future__ import annotations

import os

from . import cog as cog_module
from . import config, store, utils


def _probe_raster_metadata(file_path: str) -> dict:
    """
    Runs INSIDE an isolated subprocess (see utils.run_isolated) — must stay
    a plain, picklable, module-level function.

    Opens the file and reads only header/metadata attributes. Never calls
    .read(), so no pixel array is ever allocated here no matter what
    dimensions the file claims — that's what makes the pixel-count cap
    check (done by the caller, using the numbers this returns) effective.
    """
    import rasterio

    with rasterio.open(file_path) as dataset:
        crs = dataset.crs.to_string() if dataset.crs else None
        bounds = list(dataset.bounds) if dataset.bounds else None
        res = dataset.res
        return {
            "width": dataset.width,
            "height": dataset.height,
            "band_count": dataset.count,
            "dtype": str(dataset.dtypes[0]) if dataset.dtypes else "unknown",
            "crs": crs,
            "bounds": bounds,
            "resolution_m": float(res[0]) if res else None,
        }


_EMPTY_PROBE = {
    "crs": None, "bounds": None, "width": 0, "height": 0,
    "band_count": 0, "dtype": "unknown", "resolution_m": None,
}


def check_pixel_count_cap(width: int, height: int, band_count: int, max_pixels: int | None = None) -> tuple[bool, int]:
    """
    Pure: whether width*height*band_count is within the configured cap.
    Returns (within_cap, total_px). Called using ONLY numbers already read
    by the metadata-only probe — never after an array has been allocated —
    which is what makes this check effective against decompression-bomb
    files (architecture.md hardening requirement).
    """
    if max_pixels is None:
        max_pixels = config.MAX_PIXEL_COUNT
    total_px = width * height * band_count
    return total_px <= max_pixels, total_px


def _build_metadata(image_id, declared_modality, declared_timestamp, errors, probed_meta=None, cog_path=""):
    """Builds, persists, and returns an ImageMetadata for any outcome —
    valid or not. is_valid is simply "no errors"."""
    from backend.shared.schemas import ImageMetadata

    meta = probed_meta or _EMPTY_PROBE
    result = ImageMetadata(
        image_id=image_id,
        modality=declared_modality,
        crs=meta["crs"],
        bounds=meta["bounds"],
        width=meta["width"],
        height=meta["height"],
        band_count=meta["band_count"],
        dtype=meta["dtype"],
        resolution_m=meta["resolution_m"],
        timestamp=declared_timestamp,
        cog_path=cog_path,
        is_valid=(len(errors) == 0),
        validation_errors=errors,
    )
    store.save_metadata(image_id, result.model_dump())
    return result


def validate_and_prepare(file_path: str, declared_modality, declared_timestamp: str | None):
    """
    architecture.md 3.3 literal interface:
        def validate_and_prepare(file_path, declared_modality, declared_timestamp) -> ImageMetadata
    """
    declared_timestamp = utils.sanitize_string(declared_timestamp)

    # --- 1. fast structural checks, no parsing ---
    if not os.path.exists(file_path):
        fallback_id = utils.compute_key_hash("missing-file", file_path)
        return _build_metadata(fallback_id, declared_modality, declared_timestamp, ["file not found at given path"])

    ext = os.path.splitext(file_path)[1].lower()
    file_size = os.path.getsize(file_path)

    structural_errors = []
    if ext not in config.SUPPORTED_EXTENSIONS:
        structural_errors.append(
            f"unsupported file extension {ext!r}; expected one of {sorted(config.SUPPORTED_EXTENSIONS)}"
        )
    if file_size > config.MAX_FILE_SIZE_BYTES:
        structural_errors.append(
            f"file exceeds maximum size cap ({file_size} bytes > {config.MAX_FILE_SIZE_BYTES} bytes)"
        )

    if structural_errors:
        # Still worth a real content hash if it's not absurdly oversized —
        # keeps the id meaningful (same bad bytes -> same id) without
        # spending real time hashing a multi-GB file we're rejecting anyway.
        if file_size <= config.MAX_FILE_SIZE_BYTES * 2:
            fallback_id = utils.compute_content_hash(file_path)
        else:
            fallback_id = utils.compute_key_hash("oversized-file", file_path, file_size)
        return _build_metadata(fallback_id, declared_modality, declared_timestamp, structural_errors)

    # --- 2. content hash ---
    image_id = utils.compute_content_hash(file_path)

    # --- 3. isolated, timeout-guarded metadata probe ---
    probe = utils.run_isolated(_probe_raster_metadata, args=(file_path,), timeout=config.PARSE_TIMEOUT_SECONDS)
    if not probe.ok:
        reason = (
            f"parsing timed out after {config.PARSE_TIMEOUT_SECONDS}s"
            if probe.timed_out
            else f"unreadable or corrupt file: {probe.error}"
        )
        return _build_metadata(image_id, declared_modality, declared_timestamp, [reason])

    probed_meta = probe.value

    # --- 4. pixel-count cap check — before any array is allocated ---
    within_cap, total_px = check_pixel_count_cap(
        probed_meta["width"], probed_meta["height"], probed_meta["band_count"]
    )
    if not within_cap:
        return _build_metadata(
            image_id, declared_modality, declared_timestamp,
            [f"pixel count exceeds cap ({total_px} px > {config.MAX_PIXEL_COUNT} px)"],
            probed_meta=probed_meta,
        )

    # --- 5. COG conversion ---
    store.ensure_image_dirs(image_id)
    cog_dst = store.cog_path_for(image_id)
    try:
        cog_module.convert_to_cog(file_path, cog_dst)
    except Exception as exc:  # noqa: BLE001 - any conversion failure must produce a clean validation error, not propagate
        return _build_metadata(
            image_id, declared_modality, declared_timestamp,
            [f"COG conversion failed: {type(exc).__name__}: {exc}"],
            probed_meta=probed_meta,
        )

    return _build_metadata(
        image_id, declared_modality, declared_timestamp, [], probed_meta=probed_meta, cog_path=cog_dst
    )

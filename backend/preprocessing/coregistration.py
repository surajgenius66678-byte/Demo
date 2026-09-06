"""
check_coregistration() — architecture.md Section 3.3.

  "check_coregistration returns aligned: false above the offset threshold;
   Part 2 is responsible for refusing the downstream task on that result,
   but Part 3 computes the number."

Design decision (the schema doesn't fully pin this down): `auto_corrected`
is an INFORMATIONAL flag, not a signal that Part 3 physically re-warped
either image. Part 3 detects and reports a registration offset; it never
produces a geometrically-corrected copy of either input, and nothing
downstream (Part 4's tiling input) consumes one. Actually resampling pixels
to correct alignment would be a real geometric-correction pipeline, which
Section 7 explicitly rules out building ("No general SAR terrain-correction
pipeline — consume already-corrected products"). This keeps that same
"don't overengineer" line for co-registration too.

Split the same way as tiling.py:
  - estimate_pixel_offset / offset_magnitude / classify_alignment: pure,
    numpy + scikit-image only, fully unit-testable.
  - check_coregistration: the literal public interface, rasterio + pydantic
    dependent, not executable in the build sandbox.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import config

if TYPE_CHECKING:
    from backend.shared.schemas import CoregistrationResult


# --------------------------------------------------------------------------
# Pure logic — no rasterio, no I/O
# --------------------------------------------------------------------------

def estimate_pixel_offset(
    array_a: np.ndarray,
    array_b: np.ndarray,
    upsample_factor: int = 10,
) -> tuple[float, float, float]:
    """
    Sub-pixel translational offset between two same-shape 2D arrays, via
    phase cross-correlation (skimage.registration.phase_cross_correlation).

    Returns (offset_row_px, offset_col_px, correlation_error). Only the
    magnitude of this is used elsewhere in this module (see
    offset_magnitude) — the sign/direction isn't meaningful here since Part
    3 never applies a correcting warp, only reports how misaligned the pair
    is (see module docstring).
    """
    if array_a.shape != array_b.shape:
        raise ValueError(f"arrays must be the same shape to compare, got {array_a.shape} vs {array_b.shape}")
    if array_a.ndim != 2:
        raise ValueError(f"expected 2D arrays, got shape {array_a.shape}")

    from skimage.registration import phase_cross_correlation

    shift, error, _diffphase = phase_cross_correlation(
        array_a.astype(np.float64),
        array_b.astype(np.float64),
        upsample_factor=upsample_factor,
    )
    return float(shift[0]), float(shift[1]), float(error)


def offset_magnitude(offset_row_px: float, offset_col_px: float) -> float:
    return float(np.hypot(offset_row_px, offset_col_px))


def classify_alignment(offset_px: float) -> tuple[bool, bool, str | None]:
    """
    Three-tier classification, per config.COREG_NEGLIGIBLE_OFFSET_PX /
    config.COREG_MAX_ALIGNMENT_THRESHOLD_PX:

        offset <= NEGLIGIBLE                -> (True,  False, None)
        NEGLIGIBLE < offset <= MAX_ALIGNMENT -> (True,  True,  <reason>)
        offset > MAX_ALIGNMENT               -> (False, False, <reason>)

    Returns (aligned, auto_corrected, reason).
    """
    if offset_px < 0:
        raise ValueError("offset_px must be non-negative")

    if offset_px <= config.COREG_NEGLIGIBLE_OFFSET_PX:
        return True, False, None

    if offset_px <= config.COREG_MAX_ALIGNMENT_THRESHOLD_PX:
        reason = (
            f"detected a {offset_px:.2f}px registration offset, within the "
            f"{config.COREG_MAX_ALIGNMENT_THRESHOLD_PX}px auto-correctable range; "
            f"flagged, no refusal needed"
        )
        return True, True, reason

    reason = (
        f"detected a {offset_px:.2f}px registration offset, exceeding the "
        f"{config.COREG_MAX_ALIGNMENT_THRESHOLD_PX}px alignment threshold; "
        f"Part 2 should refuse change-detection tasks on this pair"
    )
    return False, False, reason


# --------------------------------------------------------------------------
# I/O layer — rasterio-dependent, not executable in the build sandbox
# --------------------------------------------------------------------------

def _read_common_overview(dataset, max_side: int) -> np.ndarray:
    """Reads a single-band, decimated overview of `dataset`, downsampled so
    its longest side is at most `max_side`. This is a decoded-at-reduced-
    resolution read (GDAL decimates during decode), not a full-array read —
    memory use is bounded by max_side regardless of the source's native
    resolution."""
    from rasterio.enums import Resampling

    scale = min(1.0, max_side / max(dataset.width, dataset.height))
    out_height = max(1, int(round(dataset.height * scale)))
    out_width = max(1, int(round(dataset.width * scale)))
    band = dataset.read(1, out_shape=(out_height, out_width), resampling=Resampling.average)
    return band.astype(np.float64)


def check_coregistration(image_a_id: str, image_b_id: str) -> "CoregistrationResult":
    """
    architecture.md 3.3 literal interface:
        def check_coregistration(image_a_id, image_b_id) -> CoregistrationResult

    Brings both images into a common CRS via a virtual warp (WarpedVRT —
    never rewrites the source file), reads a bounded-size overview of each
    over their overlapping extent, and estimates the residual pixel offset
    via phase correlation.
    """
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds

    from backend.shared.schemas import CoregistrationResult
    from . import store

    meta_a = store.load_metadata(image_a_id)
    meta_b = store.load_metadata(image_b_id)

    for image_id, meta in ((image_a_id, meta_a), (image_b_id, meta_b)):
        if not meta.get("is_valid", False):
            raise ValueError(f"image_id={image_id!r} failed validation, cannot check co-registration: "
                              f"{meta.get('validation_errors')}")

    with rasterio.open(meta_a["cog_path"]) as src_a, rasterio.open(meta_b["cog_path"]) as src_b:
        # Virtual warp b into a's CRS — never rewrites either source file.
        target_crs = src_a.crs
        with WarpedVRT(src_b, crs=target_crs, resampling=1) as vrt_b:  # 1 == Resampling.bilinear
            bounds_a = src_a.bounds
            bounds_b = vrt_b.bounds
            overlap = (
                max(bounds_a.left, bounds_b.left),
                max(bounds_a.bottom, bounds_b.bottom),
                min(bounds_a.right, bounds_b.right),
                min(bounds_a.top, bounds_b.top),
            )
            if overlap[0] >= overlap[2] or overlap[1] >= overlap[3]:
                return CoregistrationResult(
                    aligned=False,
                    offset_px=float("inf"),
                    auto_corrected=False,
                    reason="no spatial overlap between the two images' footprints",
                )

            window_a = rasterio.windows.from_bounds(*overlap, transform=src_a.transform)
            window_b = rasterio.windows.from_bounds(*overlap, transform=vrt_b.transform)

            def _bounded_overview(dataset, window):
                from rasterio.enums import Resampling
                scale = min(1.0, config.COREG_OVERVIEW_MAX_SIDE_PX / max(window.width, window.height))
                out_h = max(1, int(round(window.height * scale)))
                out_w = max(1, int(round(window.width * scale)))
                return dataset.read(
                    1, window=window, out_shape=(out_h, out_w), resampling=Resampling.average
                ).astype(np.float64)

            arr_a = _bounded_overview(src_a, window_a)
            arr_b = _bounded_overview(vrt_b, window_b)

            # out_shape rounding can leave the two arrays 1px apart; clip to
            # the common shape rather than failing the whole check over it.
            h = min(arr_a.shape[0], arr_b.shape[0])
            w = min(arr_a.shape[1], arr_b.shape[1])
            arr_a, arr_b = arr_a[:h, :w], arr_b[:h, :w]

            offset_row, offset_col, _error = estimate_pixel_offset(arr_a, arr_b)

            # Scale the offset (measured in the downsampled overview) back up
            # to native pixel units of image A.
            native_scale = window_a.width / w if w else 1.0
            offset_px = offset_magnitude(offset_row, offset_col) * native_scale

    aligned, auto_corrected, reason = classify_alignment(offset_px)
    return CoregistrationResult(
        aligned=aligned, offset_px=offset_px, auto_corrected=auto_corrected, reason=reason
    )

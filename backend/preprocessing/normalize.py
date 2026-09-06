"""
Per-modality pixel normalization (architecture.md 3.3 "In scope").

Applied by tiling.py to each tile's array before it's written to disk, so
Part 4 always receives consistently-scaled float32 input in [0, 1] regardless
of the source sensor's native value range — Part 4 "doesn't touch raw files
or CRS" per its own spec, so normalization has to happen upstream, here.

Pure numpy, no rasterio/pydantic dependency — arrays in, arrays out. Assumes
band-first arrays, i.e. shape (bands, height, width), matching rasterio's
`dataset.read()` convention, so the I/O layer in tiling.py can pass rasterio
output straight through.

Design assumption (worth flagging, not specified by architecture.md): SAR
input is assumed to already be calibrated backscatter in dB, the common
delivery format for analysis-ready SAR products. If a source instead
delivers linear amplitude/power, pass assume_linear=True to convert first.
"""

from __future__ import annotations

import numpy as np

from . import config


def _clip_scale_to_unit(array: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip to [low, high] then linearly rescale to [0, 1] float32.
    Guards against a degenerate (constant-value) band where low == high."""
    span = high - low
    if span <= 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    clipped = np.clip(array, low, high)
    scaled = (clipped - low) / span
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def normalize_optical(
    array: np.ndarray,
    percentile_low: float | None = None,
    percentile_high: float | None = None,
) -> np.ndarray:
    """
    Per-band percentile stretch, e.g. 2nd-98th percentile by default. Each
    band is stretched independently, since different optical bands (e.g. RGB
    vs NIR) can have very different native value distributions.

    array: shape (bands, height, width), any numeric dtype.
    returns: float32 array, same shape, values in [0, 1].
    """
    if percentile_low is None:
        percentile_low = config.OPTICAL_PERCENTILE_LOW
    if percentile_high is None:
        percentile_high = config.OPTICAL_PERCENTILE_HIGH

    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"expected a (bands, height, width) array, got shape {array.shape}")

    out = np.empty(array.shape, dtype=np.float32)
    for b in range(array.shape[0]):
        band = array[b].astype(np.float64)
        low, high = np.nanpercentile(band, [percentile_low, percentile_high])
        out[b] = _clip_scale_to_unit(band, float(low), float(high))
    return out


def normalize_sar(
    array: np.ndarray,
    db_min: float | None = None,
    db_max: float | None = None,
    assume_linear: bool = False,
) -> np.ndarray:
    """
    Clip to a fixed dB dynamic range, then linearly rescale to [0, 1]. Unlike
    optical, SAR uses a fixed range rather than a per-image percentile
    stretch — backscatter dB values are physically comparable across scenes,
    so a fixed range keeps brightness meaningful and comparable image-to-image.

    array: shape (bands, height, width).
    assume_linear: if True, converts linear amplitude/power to dB
        (10*log10) before clipping. Leave False if the source already
        delivers calibrated dB backscatter.
    returns: float32 array, same shape, values in [0, 1].
    """
    if db_min is None:
        db_min = config.SAR_DB_MIN
    if db_max is None:
        db_max = config.SAR_DB_MAX

    array = np.asarray(array).astype(np.float64)
    if array.ndim != 3:
        raise ValueError(f"expected a (bands, height, width) array, got shape {array.shape}")

    if assume_linear:
        eps = 1e-10
        array = 10.0 * np.log10(np.clip(array, eps, None))

    return _clip_scale_to_unit(array, db_min, db_max)


def normalize_tile(array: np.ndarray, modality: str, **kwargs) -> np.ndarray:
    """Dispatches to normalize_optical / normalize_sar by modality string
    ("OPTICAL" | "SAR" — matches shared.schemas.Modality's values)."""
    modality = str(modality).upper()
    if modality == "OPTICAL":
        return normalize_optical(array, **kwargs)
    if modality == "SAR":
        return normalize_sar(array, **kwargs)
    raise ValueError(f"unknown modality {modality!r}, expected 'OPTICAL' or 'SAR'")

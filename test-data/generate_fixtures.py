#!/usr/bin/env python3
"""
Generates Part 3 test fixtures — architecture.md 3.3's mocking strategy:
    "provide 2-3 small sample GeoTIFFs (one optical, one SAR, one
     bi-temporal pair, at least one deliberately malformed)"

Requires rasterio (see requirements.txt). Run once after installing
dependencies, before running tests/test_integration.py:

    pip install -r requirements.txt
    python test-data/generate_fixtures.py

Produces, under test-data/:
    optical_scene.tif           4-band (R,G,B,NIR) optical scene, 1024x1024
    sar_scene.tif                1-band SAR backscatter in dB, 1024x1024
    bitemporal_before.tif        optical scene, "before"
    bitemporal_after_small.tif   same scene, shifted ~4px (within the
                                  default auto-correctable range)
    bitemporal_after_large.tif   same scene, shifted ~30px (beyond the
                                  default alignment threshold — should be
                                  refused for change-detection)
    malformed_truncated.tif      a valid GeoTIFF, truncated mid-file
    malformed_not_a_tiff.tif     garbage bytes with a .tif extension
    malformed_huge_dims.tif      a sparse file declaring ~4.1 billion
                                  pixels (64000x64000), used to exercise
                                  the pixel-count cap guard without
                                  actually needing gigabytes of real disk —
                                  same signature a real decompression-bomb
                                  file would have: tiny on-disk size, an
                                  enormous declared decoded size
"""

from __future__ import annotations

import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SIZE = 1024
RESOLUTION_M = 10.0
# UTM 43N, an arbitrary origin plausible for the ISRO/SIH context. Not tied
# to any real location -- purely a stand-in CRS + transform for testing.
CRS_EPSG = "EPSG:32643"
ORIGIN_X, ORIGIN_Y = 500000.0, 2000000.0

SMALL_SHIFT_PX = (2.5, -3.0)   # magnitude ~3.9px -- within the default 10px auto-correct range
LARGE_SHIFT_PX = (18.0, 24.0)  # magnitude 30px -- beyond the default 10px alignment threshold


def _transform():
    return from_origin(ORIGIN_X, ORIGIN_Y, RESOLUTION_M, RESOLUTION_M)


def _synthetic_scene(size: int, seed: int, n_blobs: int = 20) -> np.ndarray:
    """A base single-band scene with real spatial structure (overlapping
    Gaussian blobs, standing in for fields/structures/vegetation patches) —
    gives phase correlation and visual inspection something real to work
    with, unlike flat noise."""
    rng = np.random.default_rng(seed)
    scene = np.zeros((size, size), dtype=np.float64)
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n_blobs):
        cy, cx = rng.uniform(0, size, size=2)
        sigma = rng.uniform(15, 60)
        amp = rng.uniform(0.3, 1.0)
        scene += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))
    return scene


def _write_geotiff(path: str, array: np.ndarray, dtype: str) -> None:
    """array: shape (bands, height, width)."""
    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": dtype,
        "crs": CRS.from_string(CRS_EPSG),
        "transform": _transform(),
        "compress": "DEFLATE",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype))


def make_optical_scene(path: str, seed: int = 1) -> None:
    """4-band (R, G, B, NIR) scene, uint16, plausible reflectance-like
    values with per-band variation so a per-band percentile stretch
    (normalize.normalize_optical) has something meaningful to do."""
    base = _synthetic_scene(SIZE, seed=seed)
    rng = np.random.default_rng(seed)
    bands = []
    for band_weight, noise_std in [(0.8, 150), (1.0, 120), (0.6, 180), (1.4, 200)]:  # R,G,B,NIR-ish
        band = 1200 + base * 6000 * band_weight + rng.normal(0, noise_std, size=base.shape)
        bands.append(np.clip(band, 0, 12000))
    array = np.stack(bands, axis=0)
    _write_geotiff(path, array, "uint16")


def make_sar_scene(path: str, seed: int = 2) -> None:
    """1-band SAR backscatter, float32, expressed directly in dB
    (normalize.normalize_sar's default assumption), with speckle-like
    multiplicative noise layered on top of the structural scene."""
    base = _synthetic_scene(SIZE, seed=seed, n_blobs=30)
    rng = np.random.default_rng(seed)
    db = -22.0 + base * 20.0
    speckle = rng.gamma(shape=4.0, scale=1.0 / 4.0, size=base.shape)  # mean ~1, SAR-like speckle
    db = db + 3.0 * (speckle - 1.0)
    array = db[np.newaxis, :, :].astype(np.float32)
    _write_geotiff(path, array, "float32")


def make_bitemporal_pair(seed: int = 3) -> None:
    from scipy import ndimage

    base = _synthetic_scene(SIZE, seed=seed, n_blobs=25)
    rng = np.random.default_rng(seed)
    before = np.clip(1200 + base * 6000 + rng.normal(0, 150, size=base.shape), 0, 12000)
    before_array = np.stack([before] * 3 + [before * 1.3], axis=0)  # crude RGBN from one band
    _write_geotiff(os.path.join(OUT_DIR, "bitemporal_before.tif"), before_array, "uint16")

    for suffix, shift in [("small", SMALL_SHIFT_PX), ("large", LARGE_SHIFT_PX)]:
        shifted = ndimage.shift(before, shift=shift, mode="reflect")
        # small simulated real change (a synthetic "new structure" patch) so
        # this pair is also usable as a CHANGE_DETECTION fixture, not just coreg
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        change_patch = 4000 * np.exp(-(((yy - SIZE * 0.7) ** 2 + (xx - SIZE * 0.3) ** 2) / (2 * 40 ** 2)))
        after = np.clip(shifted + change_patch, 0, 12000)
        after_array = np.stack([after] * 3 + [after * 1.3], axis=0)
        _write_geotiff(os.path.join(OUT_DIR, f"bitemporal_after_{suffix}.tif"), after_array, "uint16")


def make_malformed_fixtures() -> None:
    # a valid file, truncated partway through -- exercises "corrupt file
    # fails only its own job" without hanging
    valid_tmp = os.path.join(OUT_DIR, "_tmp_valid_for_truncation.tif")
    _write_geotiff(valid_tmp, _synthetic_scene(256, seed=9)[np.newaxis, :, :].astype("uint8"), "uint8")
    with open(valid_tmp, "rb") as f:
        full_bytes = f.read()
    with open(os.path.join(OUT_DIR, "malformed_truncated.tif"), "wb") as f:
        f.write(full_bytes[: len(full_bytes) // 3])
    os.remove(valid_tmp)

    # garbage bytes with a .tif extension -- not a TIFF at all
    with open(os.path.join(OUT_DIR, "malformed_not_a_tiff.tif"), "wb") as f:
        f.write(b"this is not a geotiff, just text pretending to be one\x00\x01\x02" * 50)

    # sparse file declaring ~4.1 billion pixels without allocating real
    # space for them -- same signature a genuine decompression-bomb file
    # has (tiny on-disk size, enormous claimed decoded size). SPARSE_OK
    # tells GDAL not to materialize unwritten blocks.
    huge_path = os.path.join(OUT_DIR, "malformed_huge_dims.tif")
    profile = {
        "driver": "GTiff",
        "height": 64000,
        "width": 64000,
        "count": 1,
        "dtype": "uint8",
        "crs": CRS.from_string(CRS_EPSG),
        "transform": _transform(),
        "sparse_ok": "TRUE",
        "tiled": True,
    }
    with rasterio.open(huge_path, "w", **profile):
        pass  # deliberately never write any pixel data


def main() -> None:
    print(f"Writing fixtures to {OUT_DIR} ...")
    make_optical_scene(os.path.join(OUT_DIR, "optical_scene.tif"))
    print("  optical_scene.tif")
    make_sar_scene(os.path.join(OUT_DIR, "sar_scene.tif"))
    print("  sar_scene.tif")
    make_bitemporal_pair()
    print("  bitemporal_before.tif, bitemporal_after_small.tif, bitemporal_after_large.tif")
    make_malformed_fixtures()
    print("  malformed_truncated.tif, malformed_not_a_tiff.tif, malformed_huge_dims.tif")
    print("Done.")


if __name__ == "__main__":
    main()

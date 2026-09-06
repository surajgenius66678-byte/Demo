"""
Cloud-Optimized GeoTIFF conversion on ingest (architecture.md 3.3 "In scope").

Uses GDAL's native COG driver (GDAL >= 3.1) via rasterio.shutil.copy, rather
than adding rio-cogeo as an extra dependency — the COG driver already
handles internal tiling and overview generation in one pass. This is I/O
only and depends on rasterio, so it isn't executable in the build sandbox
(see PART3_BUILD_NOTES.md); the logic below follows rasterio's documented
COG-driver usage pattern.
"""

from __future__ import annotations


def convert_to_cog(src_path: str, dst_path: str, compress: str = "DEFLATE", blocksize: int = 512) -> None:
    """
    Converts the raster at src_path into a Cloud-Optimized GeoTIFF at
    dst_path. Never modifies src_path.
    """
    import rasterio
    from rasterio.shutil import copy as rio_copy

    creation_options = {
        "driver": "COG",
        "compress": compress,
        "blocksize": blocksize,
        "overview_resampling": "average",
        "bigtiff": "IF_SAFER",
    }
    with rasterio.open(src_path) as src:
        rio_copy(src, dst_path, **creation_options)

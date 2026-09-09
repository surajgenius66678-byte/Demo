"""
Thumbnail generation (frontend preview support — not in the original
architecture.md Section 3.3 contract, added afterward because browsers
can't inline-render GeoTIFF; see frontend/README.md "Known limitation").
 
Produces a small, viewable PNG from a (possibly multi-band, possibly
single-band SAR) COG, so the frontend has something real to show instead
of a dimensions-only placeholder.
 
Deliberately a single pure function with no caching/state — cheap enough
for a demo-scale deployment to regenerate per request. If thumbnail
requests become a hot path, wrap the call site in api/main.py with a
simple on-disk cache keyed by image_id, not this function itself.
"""
 
from __future__ import annotations
 
import io
 
 
def generate_thumbnail(cog_path: str, max_size: int = 512) -> bytes:
    """
    Reads the raster at cog_path and returns PNG-encoded bytes of a
    downsampled preview, longest side <= max_size, aspect ratio preserved.
 
    - Uses rasterio's decimated (out_shape) read so this never pulls the
      full-resolution array into memory, consistent with the "windowed
      reads only" rule applied elsewhere in Part 3.
    - Caps at the first 3 bands for the preview (RGB optical). A
      single-band raster (typical SAR) is repeated across R/G/B so it
      still renders as a normal grayscale image rather than failing.
    - Per-band 2nd-98th percentile stretch to 0-255, so a handful of
      extreme-value pixels (common in raw satellite data) don't wash the
      whole preview out to near-black or near-white.
 
    Raises whatever rasterio/PIL raise on a genuinely unreadable file —
    the caller (api/main.py) is expected to turn that into a 500, the same
    way it already turns validate_and_prepare failures into a 422.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from PIL import Image
 
    with rasterio.open(cog_path) as src:
        width, height = src.width, src.height
        scale = min(1.0, max_size / float(max(width, height)))
        out_width = max(1, round(width * scale))
        out_height = max(1, round(height * scale))
 
        band_count = min(src.count, 3)
        data = src.read(
            indexes=list(range(1, band_count + 1)),
            out_shape=(band_count, out_height, out_width),
            resampling=Resampling.average,
        )
 
    # (bands, h, w) -> (h, w, bands)
    data = np.transpose(data, (1, 2, 0))
 
    if band_count == 1:
        data = np.repeat(data, 3, axis=2)
 
    normalized = np.zeros(data.shape, dtype=np.uint8)
    for b in range(data.shape[2]):
        band = data[:, :, b].astype(np.float64)
        lo, hi = np.percentile(band, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        stretched = np.clip((band - lo) / (hi - lo) * 255.0, 0, 255)
        normalized[:, :, b] = stretched.astype(np.uint8)
 
    image = Image.fromarray(normalized, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
 
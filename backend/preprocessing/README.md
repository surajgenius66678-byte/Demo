# SatQuery AI — Part 3: Preprocessing & Geospatial Pipeline

>  **Merge note:** dependencies for the whole repo are consolidated at the root — run `pip install -r requirements.txt` from `satquery-ai/`, not a local one in this folder (this part no longer ships its own).

Turns a raw uploaded satellite image into safe, validated, analysis-ready
tiles. Pure geospatial/data engineering — this part never calls a model and
doesn't know what a VLM is, only pixels, CRS, and tiles (architecture.md
Section 3.3).

## Quick start

```bash
pip install -r requirements.txt
python test-data/generate_fixtures.py     # writes sample GeoTIFFs to test-data/
pytest tests/ -v                           # full suite (needs the fixtures above)
```

For a fast subset that needs nothing but numpy/scipy/scikit-image (no
GDAL/rasterio/pydantic at all):

```bash
python tests/test_core_logic.py
```

## Public interface

```python
from backend.preprocessing import validate_and_prepare, tile_image, check_coregistration, stitch_detections

meta  = validate_and_prepare(file_path, declared_modality, declared_timestamp)   # -> ImageMetadata
tiles = tile_image(meta.image_id, task, tile_size=1024, overlap_pct=0.15)        # -> list[Tile]
coreg = check_coregistration(image_a_id, image_b_id)                             # -> CoregistrationResult
merged_evidence = stitch_detections(tile_evidence, image_id)                     # -> Evidence
```

Everything else under `backend/preprocessing/` is an implementation detail —
Part 2 only needs the four functions above, re-exported from
`backend/preprocessing/__init__.py`.

## Layout

```
backend/
  shared/schemas.py       Section 4 data contracts, copied verbatim
  preprocessing/
    config.py              every tunable cap/threshold, in one place
    utils.py                content hashing, string sanitizing, isolated timeout runner
    store.py                 content-hash-addressed local storage (image_id -> COG/tiles/metadata)
    validation.py             validate_and_prepare()
    cog.py                     Cloud-Optimized GeoTIFF conversion
    normalize.py                per-modality (optical/SAR) pixel normalization
    tiling.py                    tile_image()
    coregistration.py             check_coregistration()
    stitching.py                   stitch_detections()
test-data/generate_fixtures.py   sample GeoTIFFs used by the integration tests
tests/
  test_core_logic.py       fast, dependency-light — the pure algorithmic core
  test_integration.py       full stack, needs the generated fixtures
```

Every module with I/O responsibilities is split into two layers: a **pure**
part (grid math, normalization, phase-correlation offset estimation, NMS,
RLE, merge/aggregation — numpy/scipy only) and an **I/O** part (rasterio
reads/writes, pydantic model construction). See `PART3_BUILD_NOTES.md` for
exactly which functions were executed and verified in the environment that
produced this code, versus written-correct-but-unexecuted pending the real
stack.

## Definition of Done, and where each part is covered

> "Correct metadata or a clear validation error for real and
> deliberately-broken files"
— `validate_and_prepare`, tested in `test_integration.py` against a valid
optical file, a valid SAR file, a truncated file, a non-TIFF file with a
`.tif` extension, and a missing path. Every path returns a schema-valid
`ImageMetadata`, never raises.

> "Tiles a large image without ever loading the full array (verify with a
> memory-profiled test)"
— `test_tile_image_never_loads_full_array_into_memory` in
`test_integration.py`: tiles a windowed-written 8192×8192×4-band image and
asserts the additional peak RSS during tiling stays well under half the
full array's size. `compute_tile_windows`'s coverage/no-gap/no-sliver
invariants are separately verified in `test_core_logic.py`.

> "Correctly reports offset on a misaligned pair"
— `check_coregistration`, tested against a small-shift pair (~3.9px,
expected `aligned=True, auto_corrected=True`) and a large-shift pair
(~30px, expected `aligned=False`), plus an image checked against itself
(`aligned=True`, near-zero offset). The underlying phase-correlation
estimator is separately verified in `test_core_logic.py` against known
synthetic sub-pixel and integer-pixel shifts.

## Hardening checklist (architecture.md 3.3)

- [x] Windowed reads only, never a full-array read — `tiling.py` reads via
      `rasterio.windows.Window`; `coregistration.py` reads decimated
      overviews via `out_shape`, both memory-bounded regardless of source size.
- [x] Metadata-only read for the pixel-count cap, before any array is
      allocated — `validation.py`'s `_probe_raster_metadata` never calls
      `.read()`; `check_pixel_count_cap` runs on those numbers alone.
- [x] COG conversion on ingest — `cog.py`.
- [x] `check_coregistration` returns `aligned=False` above the offset
      threshold; Part 3 computes the number, Part 2 owns the refusal.
- [x] File parsing runs isolated with a timeout — `utils.run_isolated`,
      used by `validation.py` around the metadata probe. A hung or crashed
      worker process is terminated; it costs its own timeout budget, never
      the caller's.

## Design decisions made where architecture.md left room

The spec is precise about the interface and the hardening list, less so
about a few implementation details. Documenting the choices made, since a
teammate picking this up should know they're choices, not spec:

- **`task` doesn't change tiling geometry.** `tile_image`'s grid is a pure
  function of `(width, height, tile_size, overlap_pct)` only. This
  guarantees two images of identical dimensions always tile identically —
  required for change-detection to keep before/after tiles aligned
  tile-for-tile — and means `tile_id` (and the cached tile file) is reused
  across different tasks run against the same image at the same grid
  parameters, rather than re-materialized per task.
- **`auto_corrected` is informational, not a physical correction.**
  `check_coregistration` detects and reports a registration offset; it
  never produces a geometrically re-warped copy of either image, and
  nothing downstream consumes one. Actually resampling pixels to correct
  alignment would be a real geometric-correction pipeline, which Section 7
  rules out building for SAR terrain correction — same reasoning applies
  here.
- **SAR input is assumed pre-calibrated in dB.** `normalize_sar` clips a
  fixed dB range by default. If a real source instead delivers linear
  amplitude/power, `assume_linear=True` converts via `10*log10(...)` first.
- **Mask RLE is a custom format**, since `mask_rle: Optional[str]` in the
  schema doesn't pin one down: `"{h}x{w}:{c0},{c1},..."`, alternating
  run-lengths starting with a background (0) run. `encode_mask_rle` /
  `decode_mask_rle` in `stitching.py`.
- **NMS uses mask IoU when both detections have masks, box IoU otherwise.**
  Suppression never crosses labels (a "building" box never suppresses a
  "road" box). An IoU exactly equal to the threshold is kept, not
  suppressed (strict `>`).
- **`Evidence.stats` merges by averaging.** It's a free-form
  `dict[str, float]` with no per-key semantics attached, so this is a
  deliberately simple default — a caller with sum-type stats should
  aggregate those upstream with key-specific logic.
- **Change-map raster merging avoids double-counting overlap margins**
  properly when possible: `stitch_detections` mosaics the actual per-tile
  probability rasters via `rasterio.merge` and recomputes
  `changed_area_px`/`pct` from the merged array, rather than trusting a
  plain sum of already-computed per-tile figures (which the pure
  `merge_change_maps` fallback does when no raster paths are available).
- **Storage is a simple content-hash-addressed directory tree**
  (`{DATA_DIR}/{image_id}/...`), not a database — `tile_image` and
  `check_coregistration` only receive an `image_id: str` per the interface,
  so Part 3 needs *some* way to resolve that back to a COG path and
  metadata across separate calls. A directory keyed by the content hash
  satisfies "never filename-keyed" without adding an external dependency.

## A note on scope

Per Section 1's steer against overengineering: this doesn't build a
database, a service layer, or an API around Part 3 — it's a pure Python
module called in-process by Part 2, exactly as specified ("Depends on:
nothing internal"). The storage layer above is the one addition beyond the
literal interface list, and it exists only because `image_id`-based lookup
across separate calls isn't possible without it.

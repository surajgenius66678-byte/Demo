# Build notes: what was actually run

Being upfront about this rather than letting it surface later as a surprise.

## The constraint

The environment this Part 3 build was produced in has no network access, so
`rasterio` / `GDAL` / `pydantic` / `pytest` couldn't be installed — only
`numpy`, `scipy`, `scikit-image`, `pillow`, and stdlib were available. Your
actual dev/deployment environment will have normal internet access and
`pip install -r requirements.txt` will just work; this only affected what
could be *verified by execution* while writing the code, not what the code
targets.

## What that shaped

Every module with real I/O work is split into two layers:

- **Pure logic** — grid math, per-modality normalization, phase-correlation
  offset estimation, threshold classification, box/mask IoU, NMS, RLE
  encode/decode, stats/change-map merging, confidence banding. numpy/scipy
  only, no rasterio or pydantic anywhere in the call path.
- **I/O layer** — opening rasters, windowed reads, COG conversion,
  `WarpedVRT` reprojection, `rasterio.merge`, and building the pydantic
  schema objects themselves.

This isn't a workaround bolted on for the sandbox — separating "the actual
algorithm" from "the file format it happens to be wrapped in" is good
practice regardless, and it's what made real execution-based testing
possible here at all instead of everything being unverified.

## What was actually executed here

All of `tests/test_core_logic.py` — 45 tests, all passing, covering:

- `utils.py`: content hashing matches `hashlib` and is chunk-size
  independent; key hashing is deterministic; string sanitization strips
  control characters, neutralizes a newline-prefixed role-spoofing attempt
  without touching a legitimate mid-line occurrence of the same word, and
  truncates correctly; `run_isolated` succeeds, captures an exception
  without raising, and enforces a timeout (verified by wall-clock, in a
  real subprocess, via a standalone script — multiprocessing's `spawn`
  needs a real importable module, which a `python -c` one-liner isn't).
- `normalize.py`: output shape/dtype/range, independent per-band stretch
  for optical, correct dB clipping for SAR (both ends), correct
  `10*log10` conversion for linear SAR input, no NaN/inf on a degenerate
  constant band, dispatcher routing and rejection of an unknown modality.
- `tiling.py`'s `compute_tile_windows`: full-coverage with no gaps across
  six width/height/tile_size/overlap combinations chosen to hit edge cases
  (image smaller than one tile, exact multiples, both-huge and both-tiny
  dimensions, very wide/short strips); the "no sliver tiles" invariant;
  determinism; input validation; and — the property that actually matters
  for change detection — two images of identical dimensions always
  produce identical grids.
- `coregistration.py`'s pure functions: `estimate_pixel_offset` recovers
  five known synthetic shifts (integer and sub-pixel) within 0.5px using
  real `skimage.registration.phase_cross_correlation`; shape/dimension
  validation; `classify_alignment`'s three-tier boundary behavior checked
  at, just above, and just below both configured thresholds, including
  confirming the boundary values themselves are inclusive.
- `stitching.py`'s pure functions: box IoU (identical/disjoint/known
  partial overlap/degenerate zero-area box) and mask IoU (identical/known
  overlap/both-empty/shape-mismatch rejection); exact RLE round-trips on
  six mask shapes including all-zero, all-one, checkerboard, and 1×1;
  NMS's same-label suppression, cross-label non-suppression, and a
  threshold-boundary case constructed with integer coordinates so the IoU
  is *exactly* 0.25 in float64 (avoiding the float noise a numeric search
  for the boundary hit on the first attempt — see git-blame-equivalent
  commentary in the test file); a three-box suppression chain;
  `merge_stats` averaging; `merge_change_maps`'s weighted confidence and
  its zero-changed-area fallback (no divide-by-zero); confidence banding
  at each tier.
- `validation.py`'s `check_pixel_count_cap`: within/exceeding/exact-boundary
  cases, plus a decompression-bomb-shaped case (200000×200000×3) against
  the real default cap.

Also verified, structurally: every module in `backend/preprocessing/`
(including `validation.py`, `tiling.py`, `coregistration.py`,
`stitching.py`, `cog.py` — the I/O-layer modules) imports cleanly with
*zero* rasterio/pydantic installed, and the whole package's public API
(`from backend.preprocessing import validate_and_prepare, tile_image,
check_coregistration, stitch_detections`) resolves correctly. That confirms
there are no syntax errors and the lazy-import structure (rasterio/pydantic
imported inside functions, not at module level) actually works as intended
— rasterio and pydantic are only ever imported once a function that needs
them is called, not merely defined.

`validate_and_prepare`'s field-assembly logic (`_build_metadata`) and its
fast-rejection control flow (missing file, bad extension, oversized file)
were additionally exercised end-to-end using a throwaway stand-in for
`ImageMetadata` (a plain object that stores kwargs and exposes
`.model_dump()`) — good enough to catch real control-flow bugs, including
one it did catch: an earlier version of the oversized-file branch tried to
hash a file that had already failed the existence check, which would have
raised on a missing path. That's fixed in the version in this zip.

## What was written but not executed

The rasterio-dependent I/O layer itself: `_probe_raster_metadata`'s actual
`rasterio.open()` call, `cog.convert_to_cog`, `tile_image`'s windowed reads
and per-tile GeoTIFF writes, `check_coregistration`'s `WarpedVRT` /
`rasterio.merge` usage, and `stitch_detections`'s raster mosaicking. These
were written carefully against rasterio's documented API (window reads,
`WarpedVRT`, the `COG` driver via `rasterio.shutil.copy`, `rasterio.merge`)
but the sandbox that produced this code had no way to run them.

`tests/test_integration.py` exists specifically to close this gap and is
written to run as-is once you have the real stack:

```bash
pip install -r requirements.txt
python test-data/generate_fixtures.py
pytest tests/test_integration.py -v
```

If something in the I/O layer has a bug, this is where it'll surface —
please treat a red test here as more informative than the green ones above,
not as a sign the whole part is unreliable; it's the one layer that
genuinely couldn't be checked before handing this off.

## Net assessment

The algorithms are real and verified — tiling never leaves gaps or slivers,
offset estimation recovers known shifts, NMS and RLE round-trip correctly,
merge logic doesn't divide by zero. The wiring of those algorithms into
rasterio's I/O calls is written correctly to the best of a careful reading
of rasterio's API, but is unverified by execution. Run
`pytest tests/test_integration.py -v` first thing after installing
dependencies — that's the one command that actually closes this gap.

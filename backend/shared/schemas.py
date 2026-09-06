# Copied verbatim from architecture.md Section 4 (single source of truth).
# Do not redefine these types locally anywhere else -- every part imports
# from here. (p2/p4/p5 each independently shipped their own copy of this
# file; diffed byte-identical on every non-comment line during merge, so
# this one canonical copy, taken straight from the architecture doc,
# replaces all three.)
#
# shared/schemas.py — single source of truth

from enum import Enum
from typing import Optional
from pydantic import BaseModel

class TaskType(str, Enum):
    SINGLE_IMAGE_VQA = "SINGLE_IMAGE_VQA"
    CAPTIONING = "CAPTIONING"
    GROUNDING = "GROUNDING"                # boxes and/or masks — see Detection.mask_rle
    CHANGE_DETECTION = "CHANGE_DETECTION"
    CHANGE_VQA = "CHANGE_VQA"
    OPTICAL_SAR_FUSION = "OPTICAL_SAR_FUSION"
    UNSUPPORTED = "UNSUPPORTED"
    # Trimmed from the original ontology: object-counting / spatial / statistical
    # queries are handled as VQA or GROUNDING plus a post-processing step, not
    # separate top-level tasks. Extend this enum only if a real query genuinely
    # doesn't fit one of the above.

class Modality(str, Enum):
    OPTICAL = "OPTICAL"
    SAR = "SAR"
    # Set explicitly at upload. Never inferred from pixels — see Part 3 hardening.

class ImageMetadata(BaseModel):
    image_id: str                 # content hash of the uploaded bytes
    modality: Modality
    crs: Optional[str]
    bounds: Optional[list[float]]     # [minx, miny, maxx, maxy]
    width: int
    height: int
    band_count: int
    dtype: str
    resolution_m: Optional[float]
    timestamp: Optional[str]
    cog_path: str
    is_valid: bool
    validation_errors: list[str] = []

class Tile(BaseModel):
    tile_id: str
    image_id: str
    col_off: int
    row_off: int
    width: int
    height: int
    affine_transform: list[float]     # [a, b, c, d, e, f]
    array_path: str                   # path to pixel data on disk — never inline raw arrays here

class CoregistrationResult(BaseModel):
    aligned: bool
    offset_px: float
    auto_corrected: bool
    reason: Optional[str]

class Detection(BaseModel):
    label: str
    box_px: list[float]               # [x0, y0, x1, y1], always full-image pixel space
    mask_rle: Optional[str]
    score: float

class ChangeMap(BaseModel):
    probability_raster_path: Optional[str]   # path, not inline pixels
    changed_area_px: int
    changed_area_pct: float
    mean_confidence: float

class Confidence(BaseModel):
    value: Optional[float]            # only set if a real calibrated number exists
    band: str                         # "LOW" | "MEDIUM" | "HIGH"
    basis: str                        # human-readable: what signal this came from

class Evidence(BaseModel):
    task: TaskType
    model_used: str
    modality_used: list[Modality]
    detections: list[Detection] = []
    change_map: Optional[ChangeMap] = None
    vqa_answer_raw: Optional[str] = None
    stats: dict[str, float] = {}
    confidence: Confidence
    warnings: list[str] = []

class FinalResponse(BaseModel):
    answer_text: str
    confidence: Confidence
    evidence: Evidence
    abstained: bool
    abstain_reason: Optional[str] = None

class ModelRegistryEntry(BaseModel):
    name: str
    version: str
    tasks: list[TaskType]
    modalities: list[Modality]
    checkpoint_path: str
    quantization: str                 # "none" | "8bit" | "4bit"
    requires_coregistration: bool
    max_input_px: int

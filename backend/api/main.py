"""
Part 2 — Backend Core: the FastAPI app implementing Part 1's contract
verbatim from Section 3.2:

    POST /api/upload   -> ImageMetadata
    POST /api/query    -> {"job_id": string}
    GET  /api/jobs/{id} -> {"status", "progress", "result", "error"}
    GET  /api/jobs/{id}/trace -> {"steps": [...]}
    GET  /api/models   -> list[ModelRegistryEntry]
    GET  /api/health   -> {"status": "ok", "models_loaded": [...]}

Everything below the endpoint bodies is stitching, not logic: intent
classification lives in agent/intent.py, sequencing in agent/planner.py,
concurrency/decoupling in agent/queue.py. This file's own job is the HTTP
surface and the chunked-upload bookkeeping, which is specific to this
endpoint and doesn't belong in agent/.

Integration (Section 5, step 3): `_funcs` below is the one place the mock
Part 3/4/5 imports get swapped for the real ones — everything downstream
(JobQueue, Planner) only knows about the Part345Functions/validate_and_prepare
call shape, not that they're mocks. As of this merge, all three of Part 3,
Part 4, and Part 5 are wired to their real implementations — none of Part 2's
own mocks in agent/mocks.py are on the live call path anymore. They're kept
in the tree for backend/tests/test_agent.py, which still exercises the
planner/queue logic against them directly.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Optional

# --- integration bootstrap (added while merging parts 1/2/4/5/6) -----------
# Part 2 (this file included) imports the shared schema bare, as
# `shared.schemas`, which resolves when `backend/` itself sits on
# sys.path. Parts 4 and 5 import the identical file as
# `backend.shared.schemas`, which resolves when the repo ROOT sits on
# sys.path instead. Both conventions are needed at once, so both roots
# go on sys.path here.
#
# That alone isn't quite enough: left alone, Python would load
# backend/shared/schemas.py twice, once under each name, producing two
# non-identical `Evidence`/`TaskType`/etc. classes. Pydantic validates
# nested-model fields by class identity, so a real Part 4 `Evidence`
# passed into a Part 2 `FinalResponse` would fail validation the
# instant real code replaced the mocks below. Importing the module
# once under its real name and aliasing the second name to the same
# module object keeps one identity everywhere. (Same block, for the
# test session, in /conftest.py at the repo root.)
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
for _p in (str(_REPO_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import shared.schemas as _schemas_mod  # noqa: E402
import shared as _shared_pkg  # noqa: E402
import backend as _backend_pkg  # noqa: E402

sys.modules.setdefault("backend.shared", _shared_pkg)
sys.modules.setdefault("backend.shared.schemas", _schemas_mod)
_backend_pkg.shared = _shared_pkg
# --- end integration bootstrap ----------------------------------------------

import asyncio

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from shared.schemas import Modality
from agent import mocks
from agent.planner import Part345Functions
from agent.queue import JobQueue
from agent.store import ImageStore, UploadSessionStore

# Real Part 4 (Section 3.4) and Part 5 (Section 3.5) — wired in during this
# merge. Both expose sync functions; Part 2's Part345Functions bundle wants
# Awaitable callables (see agent/mocks.py's own docstring: "if a real
# implementation ends up sync ... wrap it with asyncio.to_thread(...) at the
# call site rather than changing these signatures"), so each is wrapped
# below rather than changing either part's code.
from backend.model_registry.inference import (
    configure_engine as _configure_inference_engine,
)
from backend.model_registry.inference import health_check as _real_health_check
from backend.model_registry.inference import (
    list_available_models as _real_list_available_models,
)
from backend.model_registry.inference import run_inference as _real_run_inference
from backend.evidence.service import validate_and_respond as _real_validate_and_respond

# Real Part 3 (Section 3.3) — wired in once it was uploaded. Also plain sync
# functions (rasterio/GDAL calls), same asyncio.to_thread treatment as
# Part 4/5 above. Imported via the package's public re-export
# (backend/preprocessing/__init__.py), matching its own documented interface.
from backend.preprocessing import (
    validate_and_prepare as _real_validate_and_prepare,
)
from backend.preprocessing import check_coregistration as _real_check_coregistration
from backend.preprocessing import tile_image as _real_tile_image
from backend.preprocessing import generate_thumbnail as _real_generate_thumbnail

app = FastAPI(title="SatQuery AI — Backend Core (Part 2)")

# CORS — added during the merge. The frontend (Part 1) is a static file with
# no build step; depending on how it's served (opened directly as file://,
# or via a plain local file server on its own port — see start_program.bat)
# its origin won't match this API's http://localhost:8000, and browsers
# block cross-origin fetches by default. This is a local single-user
# hackathon deployment (Section 6: "runs entirely on local
# infrastructure... no live internet dependency"), not a multi-tenant
# public service, so a permissive allow-list is the right tradeoff here —
# tighten this before deploying it anywhere that isn't localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/tmp/satquery_uploads")
CHUNK_DIR = Path("/tmp/satquery_upload_chunks")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

image_store = ImageStore()
upload_sessions = UploadSessionStore(CHUNK_DIR)

# --- Part 3/4/5 wiring (Section 3.2 Mocking strategy / Section 5 step 3) --
# All five calls are real as of this merge (Part 3 landed last). Part 2's
# own mocks in agent/mocks.py stay in the tree, unused on this call path,
# for backend/tests/test_agent.py.
#
# Part 6 (Training) hasn't produced any real checkpoints yet — models.yaml's
# checkpoint_path entries are still placeholders (see backend/model_registry/
# config/models.yaml) — so Part 4's engine stays on its own mock inference
# engine per Section 5 step 4's fallback branch ("otherwise leave it on a
# pretrained-only fallback entry"). Flip use_mock=False here once real
# checkpoints land and this box has the GPU deps from
# backend/model_registry's requirements installed.
_configure_inference_engine(use_mock=False)


async def _run_inference_async(
    task,
    tiles,
    model_hint=None,
    *,
    query=None,
    image_modalities=None,
    image_order=None,
    upstream_evidence=None,
):
    return await asyncio.to_thread(
        _real_run_inference,
        task,
        tiles,
        model_hint,
        query=query,
        image_modalities=image_modalities,
        image_order=image_order,
        upstream_evidence=upstream_evidence,
    )


async def _validate_and_respond_async(query, evidence):
    return await asyncio.to_thread(_real_validate_and_respond, query, evidence)


async def _list_available_models_async():
    return await asyncio.to_thread(_real_list_available_models)


async def _health_check_async():
    return await asyncio.to_thread(_real_health_check)


async def _validate_and_prepare_async(file_path, declared_modality, declared_timestamp):
    return await asyncio.to_thread(
        _real_validate_and_prepare, file_path, declared_modality, declared_timestamp
    )


async def _tile_image_async(image_id, task, tile_size=1024, overlap_pct=0.15):
    return await asyncio.to_thread(_real_tile_image, image_id, task, tile_size, overlap_pct)


async def _generate_thumbnail_async(cog_path, max_size=512):
    return await asyncio.to_thread(_real_generate_thumbnail, cog_path, max_size)


async def _check_coregistration_async(image_a_id, image_b_id):
    return await asyncio.to_thread(_real_check_coregistration, image_a_id, image_b_id)


_validate_and_prepare = _validate_and_prepare_async  # Part 3 — real
_funcs = Part345Functions(
    tile_image=_tile_image_async,  # Part 3 — real
    check_coregistration=_check_coregistration_async,  # Part 3 — real
    run_inference=_run_inference_async,  # Part 4 — real
    validate_and_respond=_validate_and_respond_async,  # Part 5 — real
)
job_queue = JobQueue(funcs=_funcs, max_concurrent_inference=1)


class QueryRequest(BaseModel):
    query: str
    image_ids: list[str]


@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    modality: Modality = Form(...),
    timestamp: Optional[str] = Form(None),
    upload_id: Optional[str] = Form(None),
    chunk_index: Optional[int] = Form(None),
    total_chunks: Optional[int] = Form(None),
):
    chunk_bytes = await file.read()

    if total_chunks is not None and total_chunks > 1:
        # Chunked path (Section 3.2 hardening: resumable chunked handling).
        if chunk_index is None:
            raise HTTPException(400, "chunk_index is required when total_chunks > 1")
        session_id = upload_id or str(uuid.uuid4())
        session = upload_sessions.get_or_create(session_id, total_chunks, modality, timestamp)
        session.write_chunk(chunk_index, chunk_bytes)

        if not session.is_complete():
            return JSONResponse(status_code=202, content={
                "status": "chunk_received",
                "upload_id": session_id,
                "received_chunks": sorted(session.received),
                "total_chunks": total_chunks,
            })

        final_path = session.assemble()
        upload_sessions.discard(session_id)
    else:
        final_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename or 'upload'}"
        final_path.write_bytes(chunk_bytes)

    try:
        metadata = await _validate_and_prepare(str(final_path), modality, timestamp)
    except Exception as e:
        raise HTTPException(422, f"validation failed: {e}")

    image_store.put(metadata)
    return metadata.model_dump()


@app.get("/api/images/{image_id}/thumbnail")
async def get_thumbnail(image_id: str, max_size: int = 512):
    """
    Added post-merge: the frontend (Part 1) can't inline-render GeoTIFF in
    the browser, so it was falling back to a dimensions-only placeholder
    (see frontend/README.md "Known limitation"). This generates a small PNG
    preview server-side from the already-validated COG on disk instead.

    Not cached — see preprocessing/thumbnail.py's module docstring for why,
    and what to do if this ever needs to be.
    """
    meta = image_store.get(image_id)
    if meta is None:
        raise HTTPException(404, f"unknown image_id: {image_id}")
    try:
        png_bytes = await _generate_thumbnail_async(meta.cog_path, max_size)
    except Exception as e:
        raise HTTPException(500, f"thumbnail generation failed: {e}")
    return Response(content=png_bytes, media_type="image/png")


@app.post("/api/query")
async def submit_query(payload: QueryRequest):
    images = []
    for image_id in payload.image_ids:
        img = image_store.get(image_id)
        if img is None:
            raise HTTPException(404, f"unknown image_id: {image_id}")
        images.append(img)

    job_id = job_queue.submit(payload.query, images)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    record = job_queue.get(job_id)
    if record is None:
        raise HTTPException(404, "job not found")
    return {
        "status": record.status,
        "progress": record.progress,
        "result": record.result.model_dump() if record.result is not None else None,
        "error": record.error,
    }


@app.get("/api/jobs/{job_id}/trace")
async def get_job_trace(job_id: str):
    record = job_queue.get(job_id)
    if record is None:
        raise HTTPException(404, "job not found")
    return record.trace.as_dict()


@app.get("/api/models")
async def get_models():
    models = await _list_available_models_async()
    return [m.model_dump() for m in models]


@app.get("/api/health")
async def health():
    return await _health_check_async()
 
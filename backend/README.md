# Part 2 — Backend Core

>  **Merge note:** dependencies for the whole repo are consolidated at the root — run `pip install -r requirements.txt` from `satquery-ai/`, not a local one in this folder (this part no longer ships its own).

FastAPI app + agent (intent classifier, planner, bounded job queue) per
Section 3.2 of `architecture.md`. No image processing, no model inference,
no answer-text generation — those are Parts 3, 4, 5, which this talks to
through five functions, mocked for now per the Mocking strategy.

## Layout

```
backend/
├── shared/schemas.py     # Section 4, copied verbatim — every part imports from here
├── agent/
│   ├── intent.py          # classify_intent() — deterministic rules + LLM fallback layer
│   ├── planner.py         # sequences Part 3 -> Part 4 -> Part 5 per task type
│   ├── queue.py            # JobQueue — bounded concurrency, decoupled from HTTP
│   ├── trace.py             # per-stage audit trail
│   ├── store.py              # in-memory ImageStore + chunked-upload sessions
│   └── mocks.py               # stand-ins for the 5 Part 3/4/5 functions + models/health
├── api/main.py            # the 6 HTTP endpoints — thin wrapper around agent/
├── tests/test_agent.py    # pytest suite (see below)
├── pytest.ini
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Try it:
```bash
curl -F "file=@some_image.tif" -F "modality=OPTICAL" http://localhost:8000/api/upload
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" \
  -d '{"query": "describe this image", "image_ids": ["<image_id from upload>"]}'
curl http://localhost:8000/api/jobs/<job_id>
curl http://localhost:8000/api/jobs/<job_id>/trace
```

## Test it

```bash
pytest
```

25 tests covering intent classification (every task type, the ambiguous
single-image case, the LLM-layer handoff, the final UNSUPPORTED fallback),
planner sequencing (per task type, the co-registration refusal path, the
UNSUPPORTED short-circuit), the job queue (bounded concurrency verified by
tracking a shared counter inside a mock `run_inference`, decoupling
verified by sleeping with no polling and checking the job finished anyway,
failure isolation verified by making a mock raise and confirming the queue
keeps accepting jobs afterward), and the chunked-upload store.

This environment couldn't install FastAPI/Pydantic itself to run this
suite live (no network access to PyPI) — every module was instead
exercised against a minimal local stand-in for `pydantic.BaseModel` before
delivery, running these exact test functions. `pytest` in your own
environment is the real check; nothing above is a substitute for running
it there. `api/main.py` isn't covered by these tests since it's a thin
routing layer over `agent/` — worth adding `httpx.AsyncClient`-based tests
against it once Parts 3/4/5 are real and wiring mistakes become possible.

## Decisions the spec left open

**Co-registration gate extended to fusion.** Section 3.3 says Part 2 "is
responsible for refusing the downstream task" when two images aren't
aligned, in the context of change detection. `planner.py` applies the same
gate to `OPTICAL_SAR_FUSION`, since fusing pixel-misaligned optical/SAR is
exactly as unreliable as diffing misaligned before/after images. On
refusal, Part 4 is never called — Part 2 builds a minimal `Evidence` with
the failure in `warnings` and routes straight to `validate_and_respond`,
so Part 5's existing abstain path produces the decline, not a second
message format invented here.

**LLM intent layer is a pluggable interface, not a live call.** Section 1
rules out a live external API dependency, and Part 2 owns no inference. So
`LLMIntentClassifier` is an ABC; the default (`HeuristicFallbackClassifier`)
is a local stand-in for what a real small on-device model would resolve,
keeping Part 2 fully self-contained per its own Mocking strategy.
`NullLLMClassifier` (always abstains) is also here — pass it in to see the
literal "neither layer confident -> UNSUPPORTED" path. Swap in a real
model-backed implementation later via `IntentClassifier(llm_classifier=...)`;
nothing else changes.

**Chunked upload is intentionally basic**, per Section 7's "tus or basic
chunking" (not a custom protocol): client picks an `upload_id` on the first
chunk, sends `chunk_index`/`total_chunks` alongside each one, and the
`202` response after every partial chunk lists which indices have arrived
— that's enough for a client to resume by re-sending what's missing,
without a separate status endpoint or a full tus implementation.

**`GET /api/models` / `GET /api/health`** call two functions Part 4's own
interface (Section 3.4) defines but Section 3.2's abbreviated call list
doesn't repeat. Mocked here (`mock_list_available_models`,
`mock_health_check`) since Part 2's own endpoints need them regardless.

## Integration checklist (Section 5, step 3)

Everything mock-related is imported in exactly two places. To wire in real
Parts 3/4/5:
1. In `api/main.py`, replace `_validate_and_prepare` and the four
   `mocks.mock_*` names inside `_funcs` with the real imports.
2. Nothing in `agent/` needs to change — `Planner` and `JobQueue` only know
   about the `Part345Functions` call shape, not that they're mocks.
3. Every mock is `async def`. If a real implementation is sync (e.g. a
   rasterio or PyTorch call), wrap it at the call site with
   `asyncio.to_thread(...)` rather than changing these signatures.

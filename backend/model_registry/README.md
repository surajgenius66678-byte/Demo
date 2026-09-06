# SatQuery AI — Part 4: Model Registry & Specialist Inference Engine

>  **Merge note:** dependencies for the whole repo are consolidated at the root — run `pip install -r requirements.txt` from `satquery-ai/`, not a local one in this folder (this part no longer ships its own).

Built strictly against `architecture.md` Section 3.4, using Section 1 for
project context and Section 4 for the shared schema. Phase 0 rule
respected: this is the first product code written against that spec, and
it doesn't redefine anything Section 4 already owns.

## Quickstart (no GPU needed)

```bash
pip install -r requirements.txt   # only pydantic + pyyaml are required for this
pytest
```

```python
from backend.model_registry import inference
from backend.shared.schemas import TaskType

inference.configure_engine(use_mock=True)   # this is also the default if you skip this line
evidence = inference.run_inference(TaskType.CAPTIONING, my_tiles)
print(evidence.vqa_answer_raw, evidence.confidence.band)
```

## What's here

```
backend/
├── shared/schemas.py              Section 4, copied verbatim
└── model_registry/                Part 4
    ├── registry.py                 config-driven ModelRegistryEntry lookup
    ├── loader.py                   lazy load, LRU eviction, VRAM budget, quantization
    ├── inference.py                run_inference / list_available_models / health_check
    ├── mock_registry.py            zero-GPU fixture engine for Parts 2 & 5 to build against
    ├── adapters/                   one file per model family (Tile -> input, output -> Evidence)
    └── config/models.yaml          reference registry config (placeholder checkpoints)
tests/                              pytest, covers the Definition of Done below
```

This mirrors Section 8's repo layout exactly (`backend/model_registry/`,
`backend/shared/`) so dropping it into the real repo during Section 5
integration is a copy, not a restructure.

## How the pieces fit together

`registry.py` loads `models.yaml` into `ModelRegistryEntry` objects and
resolves `(task, modalities, model_hint)` to one entry. `loader.py` lazily
loads whatever `registry.py` selected, evicting the least-recently-used
resident model when a new load would exceed the VRAM budget — real model
construction is injected via a `model_factory` callable, so this file
never needs torch to be *importable*, only to actually load a real model.
Each file under `adapters/` isolates one model family's quirks: it turns
`Tile` objects into that model's input shape, calls it, and turns the
output back into a canonical `Evidence` — including shifting every box
from tile-local to full-image pixel space, per Section 4's coordinate
rule. `inference.py` wires all of that together behind the exact
functions Part 2 imports, and owns the OOM retry ladder.

## The OOM retry ladder

Section 3.4's hardening list: *"retry smaller batch → smaller/quantized
model → explicit entry in `Evidence.warnings`, never a raw process
crash."* `_run_with_oom_ladder` in `inference.py` implements that
literally: on an OOM it first halves the batch of tiles being processed,
and once batch size is already 1 it steps the model down through
`none → 8bit → 4bit`. If a single tile still OOMs at 4-bit, that one tile
is dropped (with a warning) and the rest continue — the function is
guaranteed to terminate because the remaining-tiles list strictly shrinks
every iteration. Whatever tiles *did* succeed get merged into one
`Evidence`; if literally nothing succeeded, `run_inference` still returns
a schema-valid `Evidence` with `confidence.band = "LOW"` and the failure
reasons in `warnings`, rather than raising. `tests/test_oom_handling.py`
drives this with a fake model that OOMs on command, including a
worst-case "OOMs no matter what" run, to prove it never crashes.

## Definition of Done → proof

| DoD criterion (Section 3.4) | Test |
|---|---|
| Schema-valid `Evidence` for every mandatory capability | `test_mock_registry.py::test_mock_returns_schema_valid_evidence_for_every_mandatory_task` |
| Survives a simulated OOM without crashing the process | `test_oom_handling.py` (all) |
| VRAM stays within budget across a session using 2+ models | `test_loader.py::test_lru_eviction_when_budget_exceeded`, `::test_touching_a_model_protects_it_from_eviction` |

## Switching from mock to real

Nothing calling `run_inference` / `list_available_models` / `health_check`
changes. Once Part 6 delivers a checkpoint:

1. Update the matching entry's `checkpoint_path` in `config/models.yaml`.
2. Fill in `default_model_factory` in `loader.py` — the seam is a single
   function, `(entry, device, quantization_override) -> handle`, dispatched
   per architecture. Each adapter's docstring specifies the exact handle
   interface it expects (e.g. `handle.generate(image_path, prompt)` for
   the VLM adapter).
3. Call `configure_engine(use_mock=False)` instead of `True` at process
   startup. `requirements.txt` splits out `torch` / `transformers` /
   `accelerate` / `bitsandbytes` as only needed from this point on.

## Open issues — flagging back to the shared contract

Built strictly to Section 3.2 and Section 4 as written, three gaps showed
up that are worth resolving centrally rather than each part silently
guessing differently:

1. **`run_inference(task, tiles, model_hint=None)` has no field for the
   user's natural-language query.** `SINGLE_IMAGE_VQA`, `GROUNDING`, and
   `CHANGE_VQA` are meaningless without it. Handled here as an additive
   keyword-only `query=` parameter (default `None`, degrades to a generic
   caption/no-op rather than crashing) — but Part 2 needs to actually pass
   it for those three task types to work correctly. Recommend adding
   `query` to Section 3.2's contract explicitly.
2. **`Tile` carries no `modality` field.** Populating
   `Evidence.modality_used` correctly, and pairing optical vs. SAR tiles
   in the fusion adapter, both need it. Handled here as an additive
   `image_modalities: dict[image_id, Modality]` kwarg with a best-effort
   fallback (assume the selected model's declared modalities) when it's
   omitted. Recommend adding `modality` to `Tile` in Section 4, since
   `ImageMetadata` already has it and Part 3 has it on hand when tiling.
3. **`Tile` carries no timestamp/ordering.** The change-detection adapter
   needs to know which of two images is "before" vs. "after"; without it,
   the fallback is sorting by `image_id` (a content hash), which has no
   relationship to time. Handled here as an additive
   `image_order: list[image_id]` kwarg. Recommend the same fix as #2 —
   Part 3 has `ImageMetadata.timestamp` available when it tiles.

All three are purely additive keyword args with safe fallbacks, so Part 2
can be built today against the literal Section 3.2 signature and adopt
them incrementally — nothing here is a breaking change.

Separately: `shared/schemas.py` is copied verbatim as instructed, but
several `Optional[X]` fields (`ImageMetadata.crs`, `Detection.mask_rle`,
`Confidence.value`, etc.) have no `= None` default. In Pydantic v2,
`Optional[X]` without an explicit default is still a *required* field —
callers must pass `None` explicitly, it won't default on its own. Every
constructor call in this part does that correctly; flagging it so Parts
2/3/5 don't get bitten by the same thing.

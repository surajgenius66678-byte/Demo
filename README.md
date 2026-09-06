# SatQuery AI

SIH 2026, problem 26167. Full spec: `docs/architecture.md`. What was actually
built, how it fits together, and every integration decision made while
merging it: `docs/SYSTEM_DESIGN.md`. This file is just: how to run it, how
to train it, and what to do if something goes wrong.

## Quick start (Windows)

Three double-click scripts, at the repo root:

1. **`Setup.bat`** — once. Creates a virtual environment, detects your GPU,
   installs the right PyTorch build for it, installs everything else.
2. **`Start_program.bat`** — runs the app. Opens two windows (backend API,
   frontend server) and your browser, pointed at the upload/query UI.
3. **`Start_training.bat`** — fine-tunes the model. Auto-detects your GPU,
   runs a quick safety check before committing to the real (long) run, and
   is resumable if interrupted. Read "Training" below before your first run.

Not on Windows, or prefer a terminal? Same three steps, manually:

```bash
python -m venv .venv && source .venv/bin/activate      # macOS/Linux
pip install torch --index-url https://download.pytorch.org/whl/cu121  # see note below
pip install -r requirements.txt

cd backend && python -m uvicorn api.main:app --port 8000    # terminal 1
cd frontend && python -m http.server 5500                    # terminal 2, then open http://localhost:5500

cd training && ./run_pipeline.sh                              # training, auto-detects GPU
```

**PyTorch + CUDA:** plain `pip install torch` gives you a CPU-only build on
Windows (works fine on Linux, where it bundles CUDA by default). `Setup.bat`
handles this automatically; doing it manually, use the `--index-url` above,
or check https://pytorch.org/get-started/locally/ for the current command —
that page is more likely to be up to date than this README by the time you
read it.

## What's real right now

| Part | Status |
|---|---|
| 1–3, Frontend/Backend/Preprocessing | Real. Upload, query routing, tiling, co-registration all run against actual code. |
| 4, Model Registry & Inference | Real code, running on its own **mock engine** — no fine-tuned checkpoint exists yet. |
| 5, Evidence & Response | Real, deterministic template mode (no LLM wired up — that's its zero-setup default). |
| 6, Training | Real, not yet run. `Start_training.bat` / `run_pipeline.sh` produces the checkpoint Part 4 is missing. |

The app works end-to-end today — upload images, ask questions, get
schema-valid responses with confidence scores and an execution trace — the
responses just aren't model-grounded until training has actually been run
once. See `docs/SYSTEM_DESIGN.md` for the full gap analysis.

## Training

### One command

```
Start_training.bat          (Windows)
./run_pipeline.sh            (macOS/Linux, from training/)
```

This auto-detects your GPU and picks the matching config, then runs the
whole pipeline — prepare → split → **dry run** → train → evaluate → export
— stopping immediately with a clear message if any stage fails, rather than
continuing on top of a broken stage.

### Which config gets picked

| Your GPU | VRAM | Config used |
|---|---|---|
| RTX 3050 6GB, RTX 2060 6GB, GTX 1660 Ti, similar laptop GPUs | 6–8GB | `config.laptop-6gb.yaml` |
| RTX 3060 12GB, RTX 4070, T4 16GB, RTX 2080 Ti | 10–20GB | `config.yaml` |
| RTX 3090/4090, A100 40GB/80GB, H100, or similar | 20GB+ | `config.a100.yaml` |

All three produce the *same* training dynamics (effective batch size 16,
LoRA rank 8, same epochs/learning rate) — only batch size, quantization,
and gradient checkpointing change, so only wall-clock time differs between
them, not what gets learned. Detection is automatic
(`training/select_config.py` — tries `torch.cuda`, falls back to
`nvidia-smi`, and if neither finds a GPU, says so plainly rather than
guessing) but you can always force one: `python train.py --config
config.a100.yaml`, or pass it as `run_pipeline.sh`'s first argument.

**Using an external/cloud GPU (A100 or otherwise) instead of this laptop?**
Nothing else changes — copy this repo (or just `training/`) to that
machine, run `Setup.bat`-equivalent steps there (or just `pip install -r
requirements.txt` if it already has CUDA set up), and run the same one
command. Detection re-runs on whatever GPU it finds there.

### Why a dry run first

`train.py --dry-run` runs a handful of real steps — same model, same
quantization, same batch size, same data — then exits without saving
anything. It answers one question in under a minute: *does this
config/environment/GPU combination actually work*, before you commit GPU
hours (yours, or a rented A100's) to the real run. `Start_training.bat` /
`run_pipeline.sh` run this automatically and stop before the real run if it
fails. It does **not** tell you whether the model will learn well — only
that the run is mechanically sound. Prints peak GPU memory used at the end,
which is the number to compare against your card's total VRAM.

### Resumable, on purpose, at every stage

- **Data prep** (`prepare_dataset.py`) writes records incrementally and
  skips whatever's already on disk on a rerun — Ctrl-C, a network drop, a
  killed process costs at most the one record in flight, not a restart from
  zero. Streams from the Hugging Face Hub when the dataset supports it
  (most do), so processing starts on the first records immediately rather
  than waiting for a full download first; falls back to a normal download
  automatically if it doesn't.
- **Training** (`train.py`) auto-resumes from the latest checkpoint under
  `training.output_dir` if one exists. Pass `--no-resume` to start fresh
  instead (needed after a config change the old checkpoint isn't compatible
  with — a different base model or adapter rank).

### Transparent, at every stage

`prepare_dataset.py` prints running totals as it goes — seen / kept /
filtered (by reason: empty question or answer, unreadable image, exact
duplicate) / resumed-skip — not just a final count. `train.py` prints peak
GPU memory at the end. Every stage that can fail prints *why*, not just
that it failed.

**A quality safeguard worth knowing about:** if an unusually large fraction
of records get filtered out, or too few end up on disk to train on
meaningfully, `prepare_dataset.py`/`splits.py` **stop with an error instead
of silently continuing** — a real fine-tuning run on a handful of
mis-filtered records would waste GPU time on something that can't possibly
produce a useful result. For the shipped default (RSVQA-LR), a healthy run
should filter close to nothing; if you see this error, it's almost always
`dataset.column_mapping` in the config pointing at the wrong columns, not
the data genuinely being that dirty — the error message says which config
line to check. Deliberately testing with a tiny dataset? Both scripts take
`--allow-low-yield` to skip the check.

### If you hit CUDA out of memory

In roughly the order to try them, cheapest first:

1. Confirm you're on the right config profile (table above) — `python
   select_config.py` prints what it detected.
2. Already on `config.laptop-6gb.yaml`? Lower `per_device_batch_size` to 1
   if it isn't already (it is, by default) and raise
   `gradient_accumulation_steps` to compensate (keep their product at 16 to
   match the other profiles' training dynamics).
3. Close other GPU-using applications (browser hardware acceleration, other
   apps/games) — a few hundred MB matters at the 6GB tier.
4. Lower `adapter.r` (LoRA rank) — try 4. Genuine quality/memory trade-off,
   not a free lunch.
5. As a last resort: a smaller base model. A bigger architectural change,
   not a config tweak — see `training/config.yaml`'s comments on
   `adapter`/`model` before going here.

### Wiring the trained checkpoint into the app

After a real training run:

1. `training/run_pipeline.sh` (or `Start_training.bat`) prints the adapter
   directory and where `checkpoint_metadata.json` landed.
2. Point the matching entry's `checkpoint_path` in
   `backend/model_registry/config/models.yaml` at that adapter directory.
3. In `backend/api/main.py`, change `configure_engine(use_mock=True)` to
   `configure_engine(use_mock=False)`.
4. Install the GPU inference extras if you haven't:
   `torch`/`transformers`/`peft`/`bitsandbytes`/`accelerate` are already in
   `requirements.txt`.
5. Restart `Start_program.bat` / the backend.

### Going further

`training/` ships one fine-tuned adapter (VQA/captioning, on RSVQA-LR) —
grounding and change-VQA still run on Part 4's mock engine. Two genuine,
scoped extensions if there's compute budget to spare (an A100, spare
laptop-GPU hours) — outlined, not built blind, since each is a real
undertaking: fine-tuning on VRSBench (grounding) or CDVQA (change-VQA)
following the same `train.py`/`config.yaml` pattern against a different
dataset, and/or a BigEarthNet-based adaptation stage before the current
RSVQA-LR fine-tune (the problem statement's suggested primary dataset for
general image-text representation learning, not strictly required — see
`docs/SYSTEM_DESIGN.md`'s "Known gaps" for the full reasoning on both).

## Tests

```bash
pip install -r requirements.txt
pytest                              # backend/tests/ + tests/ (incl. Part 3's) in one run
python tests/test_core_logic.py     # Part 3's pure-logic subset — no rasterio/GDAL/pydantic needed
```

`tests/test_core_logic.py`'s 45 tests were actually re-executed while
building this merge (numpy/scipy/scikit-image were available in that
sandbox) — genuine, current coverage. `pytest tests/test_integration.py -v`
is the single most valuable thing to run once dependencies are installed —
it's the one part of the system (Part 3's rasterio-dependent I/O layer)
that couldn't be executed at all while merging this (no network access
there to install rasterio). See `docs/SYSTEM_DESIGN.md`'s "Honest
verification" section for exactly what was and wasn't checked, and how.

## Troubleshooting

- **"Python was not found"** (Setup.bat) — install from
  python.org/downloads, ticking "Add python.exe to PATH" during install.
- **rasterio/GDAL fails to install** — see
  `backend/preprocessing/README.md`, or
  https://rasterio.readthedocs.io/en/stable/installation.html
- **Port 8000 or 5500 already in use** — `Start_program.bat` closes any
  leftover process on those ports automatically before starting; if you're
  running something else on one of them, close it first or edit the port
  numbers in `Start_program.bat` and `frontend/index.html`'s `BASE_URL`
  together.
- **Frontend loads but nothing happens when you upload/query** — check the
  "SatQuery AI Backend" window for errors; also confirm
  `frontend/index.html`'s `CONFIG.MODE` is `'real'`, not `'mock'` (it's set
  to `'real'` by default in this merged repo).
- **CUDA out of memory during training** — see the dedicated section above.
- Anything not covered here: `docs/SYSTEM_DESIGN.md`'s "Known gaps" section
  lists every limitation that's a known, deliberate scope decision rather
  than a bug.

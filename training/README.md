# Part 6 — Training & Fine-Tuning Pipeline

>  **Merge note:** dependencies for the whole repo are consolidated at the root — run `pip install -r requirements.txt` from `satquery-ai/`, not a local one in this folder (this part no longer ships its own).

Builds the remote-sensing-adapted VLM checkpoint that Part 4 loads. Entirely
offline/batch, per `architecture.md` Section 3.6 — nothing here ever runs at
request time or touches the live API. The only coupling point with the rest
of SatQuery AI is a file: `checkpoint_metadata.json`, described below.

## Open decisions this part inherits

Section 3.6 flags three choices as unresolved and says they deserve "a
dedicated research pass (candidate comparison, licensing check, benchmark
evidence)" before this part starts for real. This build makes a working
starting choice for each so the pipeline runs end-to-end today; none of them
are the result of that research pass, and none of them are load-bearing —
swap any of them in `config.yaml` and nothing else in this repo changes.

| Decision | Default here | Why |
|---|---|---|
| Base VLM | `google/paligemma-3b-pt-224` | Outputs bounding boxes as text (fits Part 4's `Detection` schema directly), Google already has a checkpoint fine-tuned on RSVQA-HR proving the domain transfer works, LoRA/QLoRA via `peft` is well documented |
| Dataset | RSVQA-LR ([rsvqa.sylvainlobry.com](https://rsvqa.sylvainlobry.com/), Lobry et al. 2020) | Canonical remote-sensing VQA benchmark, small enough for hackathon iteration speed, LoRA fine-tuning on it is validated in the literature (~87% exact-match accuracy reported by RS-LLaVA) |
| LoRA vs QLoRA | QLoRA (4-bit) | Lower VRAM floor — more likely to fit whatever GPU is actually available on the day |

Full reasoning and swap-in alternatives are commented directly in
`config.yaml` next to each field — that file is the actual source of truth,
this table is just the summary.

**Caveat:** `dataset.hf_dataset_id` defaults to a community re-upload of
RSVQA-LR (`exibings/rsvqa-lr`) so a first run needs no manual download step.
It is *not* the authoritative source. Spot-check it against
rsvqa.sylvainlobry.com before trusting numbers for an actual submission — or
set `dataset.local_path` to use the original files instead (see
`from_local()` in `data_prep/prepare_dataset.py`).

## Quick start

```bash
pip install -r requirements.txt
./run_pipeline.sh
```

Runs all five stages in order and stops at the first failure. Each stage is
also a standalone script if you want to inspect intermediate output or resume
after a crash:

```bash
python -m data_prep.prepare_dataset --config config.yaml   # -> data/processed/all_records.jsonl
python -m data_prep.splits          --config config.yaml   # -> data/splits/{train,val,test}.jsonl
python train.py                     --config config.yaml   # -> runs/.../adapter/
python evaluate.py                  --config config.yaml --adapter-dir runs/.../adapter
python export_checkpoint.py         --config config.yaml --adapter-dir runs/.../adapter
```

Everything — model, dataset, LoRA hyperparameters, training hyperparameters,
paths — is read from `config.yaml`. There is deliberately no second place any
of this is hardcoded.

## What "definition of done" looks like here

Section 3.6: *"at least one checkpoint + metadata file that Part 4 can load
and run inference with, plus an eval report with real numbers on a held-out
split."*

- **Held-out, not just split-off:** `data_prep/splits.py` seeds the shuffle
  from `config.yaml` and asserts zero id overlap across train/val/test before
  writing anything. `evaluate.py` only ever reads `test.jsonl`.
- **Real numbers, not invented:** `evaluate.py` runs actual generation against
  the test split and computes exact-match accuracy from the results. There is
  no code path that writes a metric value that didn't come from a real
  forward pass. `export_checkpoint.py` refuses to run if `eval_report.json`
  doesn't exist yet, so the contract can't be filled with a placeholder.
- **Loadable by Part 4:** `checkpoint_path` in the exported metadata points at
  a standard PEFT adapter directory — `PeftModel.from_pretrained(base_model,
  checkpoint_path)` loads it. Call `.merge_and_unload()` first if Part 4's
  registry ends up wanting a single merged model instead of a base+adapter
  pair.

## Handoff to Part 4 — `checkpoint_metadata.json`

The literal contract from Section 3.6, written by `export_checkpoint.py`:

```json
{
  "base_model": "google/paligemma-3b-pt-224",
  "adapter_type": "QLoRA",
  "dataset": "rsvqa-lr",
  "eval": {"accuracy": 0.0},
  "checkpoint_path": "/absolute/path/to/runs/rsvqa-lr-paligemma/adapter"
}
```

`tests/test_checkpoint_contract.py` checks a produced file against this shape
— run it standalone (pure stdlib, no GPU or ML libraries needed) as part of
the Section 5 integration acceptance test, before Part 4's registry config
ever points at the result:

```bash
python tests/test_checkpoint_contract.py runs/rsvqa-lr-paligemma/checkpoint_metadata.json
```

Per Section 5 step 4 of `architecture.md`: point Part 4's registry config
at this file's `checkpoint_path` once training has actually finished;
otherwise Part 4 falls back to a pretrained-only entry.

## Layout

```
training/
├── config.yaml                       # everything configurable lives here
├── requirements.txt
├── data_prep/
│   ├── prepare_dataset.py            # raw source -> canonical JSONL
│   └── splits.py                     # canonical JSONL -> train/val/test JSONL
├── train.py                          # LoRA/QLoRA fine-tune -> adapter/
├── evaluate.py                       # adapter + test split -> eval_report.json
├── export_checkpoint.py              # eval_report.json -> checkpoint_metadata.json
├── run_pipeline.sh                   # runs the five stages above in order
└── tests/
    └── test_checkpoint_contract.py   # validates the Part 4 handoff shape
```

## Honest status

This was built and syntax/logic-checked in an environment with no GPU and no
network access, so the ML-dependent stages (`prepare_dataset.py`'s
Hugging-Face path, `train.py`, `evaluate.py`) are complete and internally
consistent but have not been run end-to-end against real weights or real
data. What *was* actually executed here: the split logic (leakage-checked
against synthetic records), the local dataset-loading path, and the
contract validator (against both a passing and a deliberately broken
`checkpoint_metadata.json`). Run `./run_pipeline.sh` on real hardware before
treating its output as submission-ready — per the project's own
evidence-first principle, that first real run is what actually earns the
numbers in `checkpoint_metadata.json`, not this description of the code.

"""
Part 6 — dataset prep.

Loads the configured remote-sensing VQA dataset (default: RSVQA-LR, see
config.yaml's "dataset" section) and normalizes it into one canonical JSONL
schema that the rest of this pipeline never has to know the source format of:

    {"id": str, "image_path": str, "question": str, "answer": str}

Two loading paths, chosen by whether config.yaml's dataset.local_path is set:
  - from_huggingface(): turnkey, no manual download step, default path.
    Streams from the Hub when the dataset supports it (most simple
    image+text datasets do) so processing starts on the first records
    immediately rather than waiting for the entire dataset to download
    first, and falls back to a normal full download automatically if
    streaming isn't supported for this particular dataset.
  - from_local(): points at a manually-downloaded copy of the authoritative
    source (https://rsvqa.sylvainlobry.com/ for the default dataset).

Resumable by construction: output is appended to incrementally (one flush
per kept record, not one write at the end), and a rerun reads whatever
all_records.jsonl already has and skips those ids — an interrupted run
(Ctrl-C, killed process, network drop) picks up where it left off instead
of re-downloading/re-processing from zero. Progress prints every
--progress-every records (seen / kept / filtered-by-reason / resumed).

Quality filtering applied to every record before it's kept: blank question
or answer, and images that fail to open/decode, are dropped and counted
(not silently skipped) — an unreadable image or an empty answer is exactly
the kind of thing that would otherwise become confusing noise in an actual
training run. Exact-duplicate (image, question) pairs are dropped too.

Usage:
    python -m data_prep.prepare_dataset --config config.yaml
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml
from PIL import Image


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _record_id(question: str, image_ref: str) -> str:
    """Deterministic id so re-running this script never duplicates or
    reshuffles records — that determinism is what splits.py's reproducible
    split (and this script's own resumability) depends on."""
    h = hashlib.sha256(f"{image_ref}::{question}".encode("utf-8")).hexdigest()
    return h[:16]


def _save_image(img, out_dir: Path, rec_id: str) -> str:
    """Persists one image to disk under a content-derived filename and
    returns its path. Accepts either a PIL Image (the HF `datasets` path) or
    a path-like pointing at a file on disk (the local-manifest path).
    Raises on an unreadable/corrupt image — the caller counts and skips."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rec_id}.png"
    if not out_path.exists():
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img.convert("RGB").save(out_path)
    return str(out_path)


def iter_from_huggingface(cfg: dict):
    """Yields (rec_id, image_ref, question, answer) tuples — no image save,
    no filtering here, just source iteration. Tries streaming first (so
    processing starts before the whole dataset is downloaded); falls back
    to a normal full load if this dataset doesn't support streaming."""
    from datasets import load_dataset

    ds_cfg = cfg["dataset"]
    dataset_id = ds_cfg["hf_dataset_id"]
    mapping = ds_cfg["column_mapping"]

    try:
        print(f"Loading '{dataset_id}' from the Hugging Face Hub (streaming)...")
        ds = load_dataset(dataset_id, split="train", streaming=True)
        columns = ds.features.keys() if ds.features else None
        streaming = True
    except Exception as exc:  # noqa: BLE001 - streaming support varies per dataset; fall back rather than fail
        print(f"  streaming not available for this dataset ({type(exc).__name__}: {exc}); "
              f"falling back to a full (non-streaming) load.")
        ds = load_dataset(dataset_id, split="train")
        columns = ds.column_names
        streaming = False

    if columns is not None:
        missing = [c for c in mapping.values() if c not in columns]
        if missing:
            raise KeyError(
                f"dataset.column_mapping in config.yaml points at columns {missing} "
                f"which don't exist on '{dataset_id}'. Actual columns on this "
                f"dataset: {sorted(columns)}. Fix the three lines under "
                "dataset.column_mapping in config.yaml and rerun."
            )
    print(f"  {'streaming' if streaming else 'fully downloaded'} — iterating records now.")

    for i, row in enumerate(ds):
        question = str(row[mapping["question"]]).strip()
        answer = str(row[mapping["answer"]]).strip()
        image_ref = str(row.get("id", i))
        rec_id = _record_id(question, image_ref)
        yield rec_id, row[mapping["image"]], question, answer


def iter_from_local(cfg: dict):
    """
    Loads from a manually-downloaded copy of the authoritative dataset
    (https://rsvqa.sylvainlobry.com/ for the default RSVQA-LR case).

    RSVQA's own release format (separate question/answer/image JSON files,
    keyed by numeric ids) differs across the LR / HR / xBEN releases, so this
    function does not try to auto-parse it. Instead it expects a
    pre-normalized manifest at `<local_path>/records.json`:

        [
          {"image": "images/0001.tif", "question": "...", "answer": "..."},
          ...
        ]

    where "image" is a path relative to local_path. If your raw download is
    in RSVQA's native triplet format, write a short one-off script that joins
    the three files into this shape — it's normally under ~20 lines since the
    join key is just the image id.
    """
    ds_cfg = cfg["dataset"]
    local_path = Path(ds_cfg["local_path"])
    manifest_path = local_path / "records.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Expected {manifest_path} — see iter_from_local()'s docstring in "
            "this file for the manifest shape it expects."
        )
    with open(manifest_path) as f:
        raw = json.load(f)

    for row in raw:
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        rec_id = _record_id(question, row["image"])
        yield rec_id, local_path / row["image"], question, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--progress-every", type=int, default=200,
        help="print a running-totals line every N source records seen (default: 200)",
    )
    parser.add_argument(
        "--min-records", type=int, default=50,
        help="fail if fewer than this many records end up on disk (default: 50) — "
             "a real fine-tune on a handful of records would silently waste GPU "
             "time on a meaningless run rather than fail loudly here.",
    )
    parser.add_argument(
        "--max-filtered-fraction", type=float, default=0.5,
        help="fail if more than this fraction of seen records got filtered out "
             "(default: 0.5). A well-formed benchmark dataset should filter "
             "close to nothing; a high fraction almost always means a config "
             "problem (wrong column_mapping, wrong dataset id) rather than "
             "genuinely bad data — better to catch that here than downstream, "
             "after training already ran on whatever scraps were left.",
    )
    parser.add_argument(
        "--allow-low-yield", action="store_true",
        help="skip both checks above (e.g. for a deliberately tiny smoke-test dataset).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    images_dir = processed_dir / "images"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "all_records.jsonl"

    # --- resumability: which ids are already on disk from a prior run? ---
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])
        if done_ids:
            print(f"Resuming: {len(done_ids)} record(s) already in {out_path} — skipping those.")

    source = iter_from_local(cfg) if cfg["dataset"].get("local_path") else iter_from_huggingface(cfg)

    stats = {"seen": 0, "kept": 0, "resumed_skip": 0, "empty": 0, "unreadable_image": 0, "duplicate": 0}
    seen_ids: set[str] = set()

    # Append mode + a flush after every kept record: a crash mid-run loses at
    # most the one record being written, never anything already on disk.
    with open(out_path, "a") as out_f:
        for rec_id, image_ref, question, answer in source:
            stats["seen"] += 1

            if rec_id in done_ids:
                stats["resumed_skip"] += 1
            elif rec_id in seen_ids:
                stats["duplicate"] += 1
            elif not question or not answer:
                stats["empty"] += 1
            else:
                seen_ids.add(rec_id)
                try:
                    image_path = _save_image(image_ref, images_dir, rec_id)
                except Exception as exc:  # noqa: BLE001 - any decode failure -> filtered, not fatal
                    stats["unreadable_image"] += 1
                    print(f"  [filtered] id={rec_id}: unreadable image "
                          f"({type(exc).__name__}: {exc})", file=sys.stderr)
                else:
                    out_f.write(json.dumps({
                        "id": rec_id, "image_path": image_path,
                        "question": question, "answer": answer,
                    }) + "\n")
                    out_f.flush()
                    stats["kept"] += 1

            if stats["seen"] % args.progress_every == 0:
                print(f"  ... seen={stats['seen']} kept={stats['kept']} "
                      f"resumed_skip={stats['resumed_skip']} empty={stats['empty']} "
                      f"unreadable={stats['unreadable_image']} duplicate={stats['duplicate']}")

    total_on_disk = len(done_ids) + stats["kept"]
    total_filtered = stats["empty"] + stats["unreadable_image"] + stats["duplicate"]
    filtered_fraction = (total_filtered / stats["seen"]) if stats["seen"] else 0.0
    print(
        f"Done. seen={stats['seen']} newly_kept={stats['kept']} "
        f"total_on_disk={total_on_disk} filtered: empty={stats['empty']} "
        f"unreadable={stats['unreadable_image']} duplicate={stats['duplicate']} "
        f"({filtered_fraction:.0%} of seen) resumed_skip={stats['resumed_skip']}"
    )

    if total_on_disk == 0:
        print("No records produced — check the dataset config.", file=sys.stderr)
        sys.exit(1)

    if not args.allow_low_yield:
        # Quality/yield circuit breaker. Both defaults are deliberately loose
        # (a healthy run on the shipped RSVQA-LR config should sail past both
        # with room to spare) — the point isn't to be a strict data-quality
        # gate, it's to catch "something is actually broken" (wrong column
        # mapping, corrupted download, wrong dataset id entirely) before that
        # silently turns into a training run on whatever scraps survived.
        if total_on_disk < args.min_records:
            print(
                f"\nERROR: only {total_on_disk} record(s) on disk, below "
                f"--min-records={args.min_records}. Training on this few "
                f"records would burn GPU time on a run that can't possibly "
                f"produce a meaningful adapter. If this is deliberate (a "
                f"tiny smoke-test dataset), rerun with --allow-low-yield. "
                f"Otherwise, check dataset.hf_dataset_id and "
                f"dataset.column_mapping in {args.config} against the "
                f"messages printed above.",
                file=sys.stderr,
            )
            sys.exit(1)
        if stats["seen"] > 0 and filtered_fraction > args.max_filtered_fraction:
            print(
                f"\nERROR: {filtered_fraction:.0%} of records seen were filtered "
                f"out, above --max-filtered-fraction={args.max_filtered_fraction:.0%}. "
                f"For a benchmark dataset like the default (RSVQA-LR) this should "
                f"normally be close to 0% — a rate this high almost always means "
                f"dataset.column_mapping in {args.config} is pointing at the wrong "
                f"columns (mismatched question/answer/image fields silently produce "
                f"empty strings or unreadable image refs), or the wrong "
                f"hf_dataset_id entirely, rather than the source data genuinely "
                f"being this dirty. If this is expected for your dataset, rerun "
                f"with --allow-low-yield.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"-> {out_path}")


if __name__ == "__main__":
    main()

"""
Part 6 — deterministic train/val/test split.

Reads the canonical JSONL from prepare_dataset.py and writes three files —
train.jsonl, val.jsonl, test.jsonl — under paths.splits_dir. The split is a
seeded shuffle-then-slice, so it's exactly reproducible from config.yaml
alone.

This matters beyond tidiness: Part 6's "real numbers on a held-out split,
never invented" requirement (architecture.md Section 3.6) depends entirely on
evaluate.py only ever touching test.jsonl, which this script has to guarantee
was never seen during training. The leakage assertion at the bottom exists
because a silent split bug is exactly the kind of thing that would make an
"evidence-first" eval number quietly untrustworthy.

Usage:
    python -m data_prep.splits --config config.yaml
"""
import argparse
import json
import random
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_splits(records: list[dict], train_pct: float, val_pct: float,
                    test_pct: float, seed: int) -> dict[str, list[dict]]:
    if abs((train_pct + val_pct + test_pct) - 1.0) > 1e-6:
        raise ValueError(
            f"split percentages must sum to 1.0, got {train_pct + val_pct + test_pct}"
        )

    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_pct)
    n_val = int(n * val_pct)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def assert_no_leakage(splits: dict[str, list[dict]]) -> None:
    ids_by_split = {name: {r["id"] for r in rows} for name, rows in splits.items()}
    overlap = (
        (ids_by_split["train"] & ids_by_split["val"])
        | (ids_by_split["train"] & ids_by_split["test"])
        | (ids_by_split["val"] & ids_by_split["test"])
    )
    assert not overlap, f"Split leakage detected: {len(overlap)} id(s) appear in multiple splits"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--min-per-split", type=int, default=5,
        help="fail if any split ends up with fewer than this many records "
             "(default: 5) — an empty or near-empty val/test split makes "
             "evaluate.py's numbers meaningless without saying so.",
    )
    parser.add_argument("--allow-low-yield", action="store_true",
                         help="skip the --min-per-split check.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    splits_dir = Path(cfg["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    records_path = processed_dir / "all_records.jsonl"
    with open(records_path) as f:
        records = [json.loads(line) for line in f]

    split_cfg = cfg["split"]
    splits = compute_splits(
        records,
        split_cfg["train_pct"],
        split_cfg["val_pct"],
        split_cfg["test_pct"],
        split_cfg["seed"],
    )
    assert_no_leakage(splits)

    if not args.allow_low_yield:
        too_small = {name: len(rows) for name, rows in splits.items() if len(rows) < args.min_per_split}
        if too_small:
            print(
                f"ERROR: split(s) below --min-per-split={args.min_per_split}: {too_small}. "
                f"Out of {len(records)} total record(s) — if that total itself looks low, "
                f"the problem is upstream in prepare_dataset.py, not the split percentages "
                f"here. Adjust split.train_pct/val_pct/test_pct in {args.config}, get more "
                f"source data, or pass --allow-low-yield if this is a deliberate smoke test.",
                file=sys.stderr,
            )
            sys.exit(1)

    for name, rows in splits.items():
        out_path = splits_dir / f"{name}.jsonl"
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} records -> {out_path}")


if __name__ == "__main__":
    main()

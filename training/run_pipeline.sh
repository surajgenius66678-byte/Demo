#!/usr/bin/env bash
# Part 6 — end-to-end orchestration.
# Runs prepare -> split -> train -> evaluate -> export in order, stopping on
# the first failure (set -e) so a broken stage never silently produces a
# checkpoint_metadata.json downstream of bad data.
#
# Usage:
#   ./run_pipeline.sh                          # auto-detect GPU, full real run
#   ./run_pipeline.sh config.laptop-6gb.yaml   # explicit config, full real run
#   ./run_pipeline.sh --dry-run                # auto-detect GPU, dry-run the train stage only
#   ./run_pipeline.sh config.a100.yaml --dry-run
#
# --dry-run here means: run the real (fast) prepare + split stages, then
# train.py --dry-run (a handful of real steps, nothing saved — see train.py's
# own docstring), then stop — evaluate/export need a real saved adapter,
# which a dry run deliberately doesn't produce.
set -euo pipefail

CONFIG=""
DRY_RUN=""
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN="--dry-run"
    else
        CONFIG="$arg"
    fi
done

if [ -z "$CONFIG" ]; then
    echo "== [0/5] No config given — detecting GPU to pick a profile =="
    CONFIG=$(python select_config.py)
fi
echo "Using config: $CONFIG"
echo ""

SECONDS=0
echo "== [1/5] Preparing dataset =="
python -m data_prep.prepare_dataset --config "$CONFIG"
echo "   (took ${SECONDS}s so far)"

echo "== [2/5] Splitting train/val/test =="
python -m data_prep.splits --config "$CONFIG"
echo "   (took ${SECONDS}s so far)"

if [ -n "$DRY_RUN" ]; then
    echo "== [3/5] Training adapter (--dry-run: a few real steps, nothing saved) =="
    python train.py --config "$CONFIG" --dry-run
    echo ""
    echo "Dry run complete in ${SECONDS}s — the config/environment/GPU combination"
    echo "works. Re-run without --dry-run for the real thing; evaluate/export are"
    echo "skipped here since a dry run doesn't produce a saved adapter."
    exit 0
fi

echo "== [3/5] Training adapter =="
python train.py --config "$CONFIG"
echo "   (took ${SECONDS}s so far)"

OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['output_dir'])")
ADAPTER_DIR="$OUTPUT_DIR/adapter"

echo "== [4/5] Evaluating on held-out test split =="
python evaluate.py --config "$CONFIG" --adapter-dir "$ADAPTER_DIR"
echo "   (took ${SECONDS}s so far)"

echo "== [5/5] Exporting checkpoint_metadata.json for Part 4 =="
python export_checkpoint.py --config "$CONFIG" --adapter-dir "$ADAPTER_DIR"

echo ""
echo "Done in ${SECONDS}s. Point Part 4's registry config at:"
python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['checkpoint_metadata_path'])"
